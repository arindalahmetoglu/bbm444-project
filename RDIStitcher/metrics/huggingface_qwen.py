import os
import base64
import time
import re
import numpy as np
import requests
from .getprompt import get_eval_prompt, get_eval_compare_prompt

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_huggingface_api(prompt, image_base64, api_key, model="Qwen/Qwen2-VL-7B-Instruct", max_retries=3):
    """Call Hugging Face Inference API with Qwen-VL model"""
    
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for attempt in range(max_retries):
        try:
            # Prepare payload for Qwen-VL
            payload = {
                "inputs": {
                    "text": prompt,
                    "image": image_base64
                },
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.1,
                    "do_sample": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 503:
                # Model loading
                print(f"    Model loading, waiting 20 seconds... (attempt {attempt + 1})")
                time.sleep(20)
                continue
            elif response.status_code == 429:
                # Rate limit
                print(f"    Rate limited, waiting 10 seconds... (attempt {attempt + 1})")
                time.sleep(10)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            
            return None
            
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None
    
    return None

def get_qwen_siqs_hf(image_path, api_key, model="Qwen/Qwen2-VL-7B-Instruct", max_images=None):
    """Get SIQS scores using Hugging Face Qwen-VL"""
    if not os.path.exists(image_path):
        print(f"Directory {image_path} does not exist")
        return None
    
    # Get image files
    names = sorted([f for f in os.listdir(image_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if max_images and max_images > 0:
        names = names[:max_images]
        print(f"Limited to {max_images} images")
    
    if not names:
        print("No image files found")
        return None
    
    prompt = get_eval_prompt()
    scores = []
    
    print(f"Evaluating {len(names)} images with Qwen-VL on Hugging Face...")
    
    for i, name in enumerate(names):
        print(f"Processing {name} ({i+1}/{len(names)})")
        
        # Add delay to respect API rate limits
        if i > 0:
            time.sleep(3)  # 3 second delay between requests
        
        try:
            # Encode image
            base64_image = encode_image(os.path.join(image_path, name))
            
            # Call Hugging Face API
            response_text = call_huggingface_api(prompt, base64_image, api_key, model)
            
            if response_text:
                print(f"  Response: {response_text[:100]}...")
                
                # Extract score using multiple patterns
                score = None
                patterns = [
                    r'FINAL_SCORE:\s*\[?(\d+)\]?',
                    r'(?:overall|final|total)\s*score[:\s]*(\d+)',
                    r'score[:\s]*(\d+)\s*(?:points?|\/10)',
                ]
                
                for pattern in patterns:
                    score_match = re.search(pattern, response_text, re.IGNORECASE)
                    if score_match:
                        score = int(score_match.group(1))
                        break
                
                if score is not None:
                    scores.append(score)
                    print(f"  ✓ Score: {score}")
                else:
                    print(f"  ⚠ Could not extract score from response")
            else:
                print(f"  ✗ Failed to get response")
                
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            continue
    
    if scores:
        avg_score = np.mean(scores)
        print(f"\n✓ Successfully processed {len(scores)}/{len(names)} images")
        print(f"✓ Average score: {avg_score:.2f}")
        return avg_score
    else:
        print(f"\n✗ No valid scores obtained")
        return None

def get_qwen_micqs_hf(image_path1, image_path2, api_key, model="Qwen/Qwen2-VL-7B-Instruct", max_images=None):
    """Get MICQS comparison using Hugging Face Qwen-VL"""
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print("One or both directories do not exist")
        return None
    
    # Get image files
    names1 = sorted([f for f in os.listdir(image_path1) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    names2 = sorted([f for f in os.listdir(image_path2) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Find common image names (by numeric index)
    common_pairs = []
    for name1 in names1:
        try:
            # Extract numeric part from filename
            base1 = os.path.splitext(name1)[0]
            idx1 = int(re.search(r'(\d+)', base1).group(1))
            
            for name2 in names2:
                try:
                    base2 = os.path.splitext(name2)[0]
                    idx2 = int(re.search(r'(\d+)', base2).group(1))
                    if idx1 == idx2:
                        common_pairs.append((name1, name2, idx1))
                        break
                except (ValueError, AttributeError):
                    continue
        except (ValueError, AttributeError):
            continue
    
    common_pairs.sort(key=lambda x: x[2])  # Sort by index
    
    if max_images and max_images > 0:
        common_pairs = common_pairs[:max_images]
        print(f"Limited to {max_images} image pairs")
    
    if not common_pairs:
        print("No common image pairs found")
        return None
    
    prompt = get_eval_compare_prompt()
    scores = [0, 0]  # [image1_wins, image2_wins]
    
    print(f"Comparing {len(common_pairs)} image pairs with Qwen-VL on Hugging Face...")
    
    for i, (name1, name2, idx) in enumerate(common_pairs):
        print(f"Comparing {name1} vs {name2} ({i+1}/{len(common_pairs)})")
        
        # Add delay to respect API rate limits
        if i > 0:
            time.sleep(5)  # 5 second delay for comparison requests
        
        try:
            # For comparison, we need to encode both images and create a combined prompt
            base64_image1 = encode_image(os.path.join(image_path1, name1))
            base64_image2 = encode_image(os.path.join(image_path2, name2))
            
            # Create comparison prompt with both images
            comparison_prompt = f"""Compare these two stitched panorama images and determine which one has better quality.

Image 1: [First image]
Image 2: [Second image]

{prompt}

Please analyze both images and provide your final choice."""
            
            # Note: HF API typically handles one image at a time, so we'll need to make two calls
            # and then make a decision based on individual scores
            response1 = call_huggingface_api(get_eval_prompt(), base64_image1, api_key, model)
            time.sleep(2)
            response2 = call_huggingface_api(get_eval_prompt(), base64_image2, api_key, model)
            
            if response1 and response2:
                # Extract scores from both responses
                score1 = None
                score2 = None
                
                patterns = [
                    r'FINAL_SCORE:\s*\[?(\d+)\]?',
                    r'(?:overall|final|total)\s*score[:\s]*(\d+)',
                    r'score[:\s]*(\d+)\s*(?:points?|\/10)',
                ]
                
                for pattern in patterns:
                    if score1 is None:
                        match1 = re.search(pattern, response1, re.IGNORECASE)
                        if match1:
                            score1 = int(match1.group(1))
                    
                    if score2 is None:
                        match2 = re.search(pattern, response2, re.IGNORECASE)
                        if match2:
                            score2 = int(match2.group(1))
                
                if score1 is not None and score2 is not None:
                    if score1 > score2:
                        scores[0] += 1
                        print(f"  ✓ Winner: Image 1 (scores: {score1} vs {score2})")
                    elif score2 > score1:
                        scores[1] += 1
                        print(f"  ✓ Winner: Image 2 (scores: {score1} vs {score2})")
                    else:
                        print(f"  ⚠ Tie (scores: {score1} vs {score2})")
                else:
                    print(f"  ⚠ Could not extract scores")
            else:
                print(f"  ✗ Failed to get responses")
                
        except Exception as e:
            print(f"  ✗ Error comparing {name1} vs {name2}: {e}")
            continue
    
    total_comparisons = scores[0] + scores[1]
    print(f"\n✓ Successfully completed {total_comparisons}/{len(common_pairs)} comparisons")
    print(f"✓ Results: Image1={scores[0]}, Image2={scores[1]}")
    return scores 
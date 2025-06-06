import os
import base64
import time
import re
import numpy as np
from openai import OpenAI
from .getprompt import get_eval_prompt, get_eval_compare_prompt

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_gpt4v_siqs(image_path, api_key, model="gpt-4o", max_images=None):
    """Get SIQS scores using OpenAI GPT-4V"""
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
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    prompt = get_eval_prompt()
    scores = []
    
    print(f"Evaluating {len(names)} images with GPT-4V...")
    
    for i, name in enumerate(names):
        print(f"Processing {name} ({i+1}/{len(names)})")
        
        # Add delay to respect API rate limits
        if i > 0:
            time.sleep(2)  # 2 second delay between requests
        
        try:
            # Encode image
            base64_image = encode_image(os.path.join(image_path, name))
            
            # Call OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent scoring
            )
            
            response_text = completion.choices[0].message.content
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

def get_gpt4v_micqs(image_path1, image_path2, api_key, model="gpt-4o", max_images=None):
    """Get MICQS comparison using OpenAI GPT-4V"""
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
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    prompt = get_eval_compare_prompt()
    scores = [0, 0]  # [image1_wins, image2_wins]
    
    print(f"Comparing {len(common_pairs)} image pairs with GPT-4V...")
    
    for i, (name1, name2, idx) in enumerate(common_pairs):
        print(f"Comparing {name1} vs {name2} ({i+1}/{len(common_pairs)})")
        
        # Add delay to respect API rate limits
        if i > 0:
            time.sleep(3)  # 3 second delay for comparison requests
        
        try:
            # Encode both images
            base64_image1 = encode_image(os.path.join(image_path1, name1))
            base64_image2 = encode_image(os.path.join(image_path2, name2))
            
            # Call OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            
            response_text = completion.choices[0].message.content
            print(f"  Response: {response_text[:100]}...")
            
            # Extract choice using multiple patterns
            choice = None
            patterns = [
                r'FINAL_CHOICE[:\s]*image\s*(\d+)\s*is\s*better',
                r'FINAL_CHOICE[:\s]*(\d+)',
                r'(?:image|picture)\s*(\d+)\s*is\s*better',
            ]
            
            for pattern in patterns:
                choice_match = re.search(pattern, response_text, re.IGNORECASE)
                if choice_match:
                    choice = int(choice_match.group(1))
                    break
            
            if choice == 1:
                scores[0] += 1
                print(f"  ✓ Winner: Image 1")
            elif choice == 2:
                scores[1] += 1
                print(f"  ✓ Winner: Image 2")
            else:
                print(f"  ⚠ Could not determine winner")
                
        except Exception as e:
            print(f"  ✗ Error comparing {name1} vs {name2}: {e}")
            continue
    
    total_comparisons = scores[0] + scores[1]
    print(f"\n✓ Successfully completed {total_comparisons}/{len(common_pairs)} comparisons")
    print(f"✓ Results: Image1={scores[0]}, Image2={scores[1]}")
    return scores 
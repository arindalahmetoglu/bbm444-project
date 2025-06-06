import os
import base64
import requests
import time
import re
import numpy as np
from PIL import Image
import io
from .getprompt import get_eval_prompt, get_eval_compare_prompt

def resize_image_for_api(image_path, max_size=1024):
    """Resize image to reduce memory usage while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"    Error resizing image: {e}")
        # Fallback to original method
        return encode_image(image_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_ollama_api(prompt, images, api_url="http://localhost:11434/api/chat", model="qwen2.5vl:3b", max_retries=3):
    """Call Ollama API with image(s) and prompt, with retry logic"""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images
                    }
                ],
                "stream": False
            }
            
            response = requests.post(api_url, json=payload, timeout=180)
            
            if response.status_code == 500:
                print(f"    Server overloaded (attempt {attempt + 1}/{max_retries})")
                # Exponential backoff for server errors
                wait_time = (attempt + 1) ** 2 * 10  # 10, 40, 90 seconds
                print(f"    Waiting {wait_time} seconds for server to recover...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.Timeout:
            print(f"    Request timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(15)
            continue
        except Exception as e:
            print(f"    API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            else:
                return None
    
    return None

def get_qwen_siqs_local(image_path, api_url="http://localhost:11434/api/chat", max_images=None):
    """Get SIQS scores for images in a directory using local Ollama"""
    names = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if max_images and max_images > 0:
        names = names[:max_images]
        print(f"Limited to {max_images} images")
    
    prompt = get_eval_prompt()
    scores = []
    
    for i, name in enumerate(names):
        print(f"Processing {name} ({i+1}/{len(names)})")
        
        # Progressive delay - longer waits as we process more images
        if i > 0:
            base_delay = 5
            progressive_delay = min(i, 10)  # Cap at 10 extra seconds
            total_delay = base_delay + progressive_delay
            print(f"    Waiting {total_delay} seconds before next request...")
            time.sleep(total_delay)
            
        try:
            # Use resized images to reduce memory pressure
            print(f"    Resizing and encoding image...")
            base64_image = resize_image_for_api(os.path.join(image_path, name))
            
            print(f"    Sending API request...")
            response_text = call_ollama_api(prompt, [base64_image], api_url)
            
            if response_text:
                print(f"    Response received (first 200 chars): {response_text[:200]}...")
                
                # Multiple regex patterns to catch different score formats
                score_patterns = [
                    r'FINAL_SCORE:\s*\[?(\d+)\]?',  # FINAL_SCORE: [X] or FINAL_SCORE: X
                    r'(?:final|total|overall)\s*score[:\s]*(\d+)',  # Alternative score patterns
                    r'score[:\s]*(\d+)\s*(?:points?|\/10)',  # Score: X points or Score: X/10
                ]
                
                score = None
                for pattern in score_patterns:
                    score_match = re.search(pattern, response_text, re.IGNORECASE)
                    if score_match:
                        score = int(score_match.group(1))
                        break
                
                if score is not None:
                    scores.append(score)
                    print(f"    ✓ Extracted score: {score}")
                else:
                    print(f"    ⚠ Could not extract score. Response: {response_text[:300]}...")
            else:
                print(f"    ✗ Failed to get response after all retries")
                
        except Exception as e:
            print(f"    ✗ Error processing {name}: {e}")
            continue
    
    if scores:
        avg_score = np.mean(scores)
        print(f"\n✓ Successfully processed {len(scores)}/{len(names)} images")
        print(f"✓ Average score: {avg_score:.2f}")
        return avg_score
    else:
        print(f"\n✗ No valid scores obtained from {len(names)} images")
        return None

def get_qwen_micqs_local(image_path1, image_path2, api_url="http://localhost:11434/api/chat", max_images=None):
    """Get MICQS comparison between two directories using local Ollama"""
    names1 = sorted([f for f in os.listdir(image_path1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    names2 = sorted([f for f in os.listdir(image_path2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Find common image names (by numeric index)
    common_indices = []
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
                        common_indices.append((name1, name2, idx1))
                        break
                except (ValueError, AttributeError):
                    continue
        except (ValueError, AttributeError):
            continue
    
    common_indices.sort(key=lambda x: x[2])  # Sort by index
    
    if max_images and max_images > 0:
        common_indices = common_indices[:max_images]
        print(f"Limited to {max_images} image pairs")
    
    prompt = get_eval_compare_prompt()
    scores = [0, 0]  # [image1_wins, image2_wins]
    
    for i, (name1, name2, idx) in enumerate(common_indices):
        print(f"Comparing {name1} vs {name2} ({i+1}/{len(common_indices)})")
        
        # Even longer delays for comparison requests (processing 2 images)
        if i > 0:
            base_delay = 8
            progressive_delay = min(i, 15)  # Cap at 15 extra seconds
            total_delay = base_delay + progressive_delay
            print(f"    Waiting {total_delay} seconds before next comparison...")
            time.sleep(total_delay)
            
        try:
            print(f"    Resizing and encoding both images...")
            base64_image1 = resize_image_for_api(os.path.join(image_path1, name1))
            base64_image2 = resize_image_for_api(os.path.join(image_path2, name2))
            
            print(f"    Sending comparison request...")
            response_text = call_ollama_api(prompt, [base64_image1, base64_image2], api_url)
            
            if response_text:
                print(f"    Response received (first 150 chars): {response_text[:150]}...")
                
                # Multiple patterns to catch different choice formats
                choice_patterns = [
                    r'FINAL_CHOICE:\s*image\s*(\d+)\s*is\s*better',  # Standard format
                    r'FINAL_CHOICE:\s*(\d+)',  # Just the number
                    r'(?:image|picture)\s*(\d+)\s*is\s*better',  # Alternative phrasing
                    r'better.*?(?:image|picture)\s*(\d+)',  # Reverse order
                ]
                
                choice = None
                for pattern in choice_patterns:
                    choice_match = re.search(pattern, response_text, re.IGNORECASE)
                    if choice_match:
                        choice = int(choice_match.group(1))
                        break
                
                if choice == 1:
                    scores[0] += 1
                    print(f"    ✓ Winner: Image 1")
                elif choice == 2:
                    scores[1] += 1
                    print(f"    ✓ Winner: Image 2")
                else:
                    print(f"    ⚠ Could not determine winner. Response: {response_text[:300]}...")
            else:
                print(f"    ✗ Failed to get response after all retries")
                
        except Exception as e:
            print(f"    ✗ Error processing {name1} vs {name2}: {e}")
            continue
    
    total_comparisons = scores[0] + scores[1]
    print(f"\n✓ Successfully completed {total_comparisons}/{len(common_indices)} comparisons")
    return scores 
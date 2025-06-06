from openai import OpenAI
import os
import base64
from getprompt import get_eval_prompt, get_eval_compare_prompt
import time
import re
import numpy as np

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_qwen_siqs(args):
    imagepath = args.image_path
    names = sorted(os.listdir(imagepath))
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    prompt = get_eval_prompt()

    scores = []

    i = 0
    while i > len(names):
        if i > 0 :
            time.sleep(10)
        base64_image = encode_image(os.path.join(imagepath, names[i]))
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
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
        )
        response_text = completion.choices[0].message.content
        score_text = re.search(r'FINAL_SCORE: (\d+)', response_text)
        score = ''.join(filter(str.isdigit, score_text.group(1))) if score_text else None

        if score == None:
            continue
        else:
            i += 1
            scores.append(int(score))

    return np.mean(scores)

def get_qwen_micqs(args):
    imagepath = args.image_path
    imagepath2 = args.image_path2
    names = sorted(os.listdir(imagepath))
    scores = [0, 0]
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    prompt = get_eval_compare_prompt()

    i = 0
    while i > len(names):
        if i > 0 :
            time.sleep(10)
        base64_image1 = encode_image(os.path.join(imagepath, names[i]))
        base64_image2 = encode_image(os.path.join(imagepath2, names[i]))
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
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
        )
        response_text = completion.choices[0].message.content
        match = re.search(r'FINAL_CHOICE(.*)', response_text)
        final_choice_text = match.group(1).strip()
        if 'image 1 is better' in final_choice_text:
            scores[0] += 1
            i+=1
        elif 'image 2 is better' in final_choice_text:
            scores[1] += 1
            i+=1
        else:
            continue
    
    return scores

#!/usr/bin/env python3
"""
This script evaluates the quality of stitched images using OpenAI's GPT-4V/GPT-4o
models. It supports single-image scoring (SIQS) for both the UDIS-D and Beehive
datasets, using tailored, objective prompts for each.

"""

import os
import argparse
import base64
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import custom prompts
try:
    from siqs_openai_prompts import get_udis_d_prompt, get_beehive_prompt
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False

# --- Core Evaluation Logic ---

def encode_image(image_path: str) -> str:
    """Encodes the image at the given path to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_score(response: str, dataset: str) -> Optional[int]:
    """Parses the final score from the MLLM's text response."""
    if dataset == "beehive":
        # Beehive prompt has a very specific format
        match = re.search(r"FINAL_SCORE:\s*(\d+)", response, re.IGNORECASE)
    else:
        # UDIS-D prompt uses a more general format
        match = re.search(r"FINAL_SCORE:\s*\[?(\d+)\]?", response, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    
    # Fallback for less structured responses
    patterns = [
        r'(?:overall|final|total)\s*score[:\s]*(\d+)',
        r'score[:\s]*(\d+)\s*(?:points?|\/10)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
    return None

def evaluate_single_image(
    client: "OpenAI",
    image_path: str,
    prompt: str,
    model: str,
    output_dir: str
) -> Dict:
    """Sends a single image and prompt to the OpenAI API and returns the structured result."""
    image_name = os.path.basename(image_path)
    print(f"Processing {image_name}...")
    
    result = {
        "image_name": image_name,
        "score": None,
        "response": None,
        "error": None,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        base64_image = encode_image(image_path)
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1500,
            temperature=0.1,  # Low temperature for more deterministic, objective scoring
        )
        
        response_text = completion.choices[0].message.content
        result["response"] = response_text
        
        # Save the full text response
        response_file = output_dir / "detailed_responses" / f"{Path(image_name).stem}_response.txt"
        response_file.parent.mkdir(parents=True, exist_ok=True)
        response_file.write_text(f"Image: {image_name}\nModel: {model}\n{'-'*20}\n{response_text}")

        # Parse the score
        score = parse_score(response_text, "udis-d" if "distortions" in prompt else "beehive")
        if score is not None:
            result["score"] = score
            print(f"  Score: {score}")
        else:
            result["error"] = "Could not extract score from response."
            print(f"  {result['error']}")

    except Exception as e:
        error_message = f"API call failed: {e}"
        result["error"] = error_message
        print(f"  {error_message}")
        
    return result

def run_evaluation(
    results_dir: str,
    dataset: str,
    api_key: str,
    model: str,
    max_images: Optional[int],
    output_dir: str
):
    """Main function to run the MLLM evaluation for a given dataset."""
    
    # --- Setup and Validation ---
    
    print("🚀 Starting SIQS MLLM-Based Evaluation")
    print("=" * 50)
    
    if not OPENAI_AVAILABLE:
        print("OpenAI library not found. Please run 'pip install openai'.")
        return
        
    if not PROMPTS_AVAILABLE:
        print("'siqs_openai_prompts.py' not found in the current directory.")
        return

    client = OpenAI(api_key=api_key)
    
    if dataset == "udis-d":
        prompt = get_udis_d_prompt()
    elif dataset == "beehive":
        prompt = get_beehive_prompt()
    else:
        raise ValueError("Invalid dataset specified.")

    base_results_path = Path("results") / results_dir
    if not base_results_path.is_dir():
        print(f"Results directory not found: {base_results_path}")
        print(f"(Searched for '{results_dir}' inside the 'results/' directory)")
        return

    output_path = Path("results") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset:     {dataset.upper()}")
    print(f"Results Dir: {base_results_path}")
    print(f"Model:       {model}")
    print(f"Output Dir:  {output_path}")
    
    # --- Evaluation Loop ---
    
    all_results = {}
    method_dirs = [d for d in base_results_path.iterdir() if d.is_dir()]

    for method_dir in method_dirs:
        method_name = method_dir.name
        print(f"\n🔍 Evaluating Method: {method_name.upper()}")
        all_results[method_name] = []
        
        image_files = sorted([f for f in method_dir.glob("*.png")] + [f for f in method_dir.glob("*.jpg")])
        
        if not image_files:
            print("  No images found. Skipping.")
            continue
            
        if max_images and max_images > 0:
            print(f"  Found {len(image_files)} images, evaluating a sample of {max_images}.")
            image_files = image_files[:max_images]
        else:
            print(f"  Found {len(image_files)} images.")

        for i, image_path in enumerate(image_files):
            # Add a delay between API calls to respect rate limits
            if i > 0:
                time.sleep(2)
                
            single_result = evaluate_single_image(client, str(image_path), prompt, model, output_path)
            all_results[method_name].append(single_result)

    # --- Reporting ---
    
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    
    summary_stats = {}
    report_lines = [f"SIQS MLLM Evaluation Report - {dataset.upper()}\n{'='*40}\n"]

    for method, results in all_results.items():
        valid_scores = [r["score"] for r in results if r["score"] is not None]
        
        if not valid_scores:
            avg_score, min_score, max_score = "N/A", "N/A", "N/A"
        else:
            avg_score = sum(valid_scores) / len(valid_scores)
            min_score = min(valid_scores)
            max_score = max(valid_scores)

        summary_stats[method] = {
            "average_score": avg_score,
            "successful_evaluations": len(valid_scores),
            "total_images": len(results),
        }
        
        report_lines.append(f"\n--- {method.upper()} ---")
        report_lines.append(f"  Average Score: {avg_score:.2f}" if isinstance(avg_score, float) else f"  Average Score: {avg_score}")
        report_lines.append(f"  Evaluated:     {len(valid_scores)} / {len(results)} images")

    print("\n".join(report_lines))

    # Save detailed JSON results
    json_path = output_path / f"siqs_detailed_results_{dataset}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📋 Detailed JSON results saved to: {json_path}")
    
    # Save text report
    report_path = output_path / f"siqs_summary_report_{dataset}.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"📄 Summary report saved to: {report_path}")

# --- Argument Parser and Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIQS MLLM-Based Evaluation Script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate UDIS-D results (gets API key from environment)
  export OPENAI_API_KEY="sk-..."
  python run_siqs_evaluation.py udis_d_results --dataset udis-d

  # Evaluate Beehive results with a max of 5 images per method
  python run_siqs_evaluation.py beehive_results --dataset beehive --max-images 5

  # Evaluate using a different model and output directory
  python run_siqs_evaluation.py udis_d_results --dataset udis-d --model gpt-4o-2024-05-13 --output-dir mllm_results
"""
    )

    parser.add_argument("results_dir", type=str,
                        help="The base directory containing method subdirectories (e.g., 'udis_d_results').")
    parser.add_argument("--dataset", type=str, required=True, choices=["udis-d", "beehive"],
                        help="The dataset being evaluated ('udis-d' or 'beehive').")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="The OpenAI model to use for evaluation (e.g., 'gpt-4o').")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to evaluate per method. Defaults to all.")
    parser.add_argument("--output-dir", type=str, default="siqs_evaluation_results",
                        help="Directory to save detailed responses and summary reports (will be placed inside 'results/').")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("OpenAI API key must be provided via --api_key argument or OPENAI_API_KEY environment variable.")

    run_evaluation(
        results_dir=args.results_dir,
        dataset=args.dataset,
        api_key=args.api_key,
        model=args.model,
        max_images=args.max_images,
        output_dir=args.output_dir,
    ) 
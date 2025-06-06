import argparse
import json
import os
from metrics.qwen_local import get_qwen_siqs_local, get_qwen_micqs_local

def parse_args():
    parser = argparse.ArgumentParser(description="RDIStitcher Local MLLM Evaluation.")
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["qwen-siqs", "qwen-micqs"],
        help="type of metric",
        required=True
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="path to evaluation stitched images",
        required=True
    )
    parser.add_argument(
        "--image_path2",
        type=str,
        help="path2 to evaluation stitched images, only used for MICQS",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:11434/api/chat",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print(f"Running {args.metric_type} evaluation...")
    print(f"Image path: {args.image_path}")
    if args.image_path2:
        print(f"Image path 2: {args.image_path2}")
    
    result = None
    
    if args.metric_type == "qwen-siqs":
        result = get_qwen_siqs_local(
            args.image_path, 
            args.api_url, 
            args.max_images
        )
        if result is not None:
            print(f"Average SIQS score: {result:.2f}")
        else:
            print("Failed to get SIQS scores")
            
    elif args.metric_type == "qwen-micqs":
        if not args.image_path2:
            print("Error: --image_path2 is required for MICQS evaluation")
            return
            
        result = get_qwen_micqs_local(
            args.image_path, 
            args.image_path2, 
            args.api_url, 
            args.max_images
        )
        if result is not None:
            print(f"MICQS results: Path1 wins: {result[0]}, Path2 wins: {result[1]}")
        else:
            print("Failed to get MICQS scores")
    
    # Save results if output file specified
    if args.output and result is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'metric_type': args.metric_type,
                'result': result,
                'image_path': args.image_path,
                'image_path2': args.image_path2
            }, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 
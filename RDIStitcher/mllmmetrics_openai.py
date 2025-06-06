import argparse
import json
import os
from metrics.openai_gpt4v import get_gpt4v_siqs, get_gpt4v_micqs

def parse_args():
    parser = argparse.ArgumentParser(description="RDIStitcher OpenAI MLLM Evaluation.")
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["gpt4v-siqs", "gpt4v-micqs"],
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
        "--api_key",
        type=str,
        help="OpenAI API key",
        required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        help="Maximum number of images to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate arguments
    if args.metric_type == "gpt4v-micqs" and not args.image_path2:
        print("Error: --image_path2 is required for MICQS evaluation")
        return
    
    # Run evaluation
    if args.metric_type == "gpt4v-siqs":
        print(f"Running GPT-4V SIQS evaluation on {args.image_path}")
        result = get_gpt4v_siqs(
            image_path=args.image_path,
            api_key=args.api_key,
            model=args.model,
            max_images=args.max_images
        )
        
        if result is not None:
            print(f"\n🎯 Final SIQS Score: {result:.2f}/10")
            
            # Save results if output file specified
            if args.output:
                results = {
                    "metric_type": "siqs",
                    "model": args.model,
                    "image_path": args.image_path,
                    "average_score": result,
                    "max_images": args.max_images
                }
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"📁 Results saved to {args.output}")
        else:
            print("❌ Evaluation failed")
    
    elif args.metric_type == "gpt4v-micqs":
        print(f"Running GPT-4V MICQS comparison between {args.image_path} and {args.image_path2}")
        result = get_gpt4v_micqs(
            image_path1=args.image_path,
            image_path2=args.image_path2,
            api_key=args.api_key,
            model=args.model,
            max_images=args.max_images
        )
        
        if result is not None:
            wins1, wins2 = result
            total = wins1 + wins2
            if total > 0:
                win_rate1 = wins1 / total * 100
                win_rate2 = wins2 / total * 100
                print(f"\n🎯 Final MICQS Results:")
                print(f"   Image Path 1: {wins1} wins ({win_rate1:.1f}%)")
                print(f"   Image Path 2: {wins2} wins ({win_rate2:.1f}%)")
                
                # Save results if output file specified
                if args.output:
                    results = {
                        "metric_type": "micqs",
                        "model": args.model,
                        "image_path1": args.image_path,
                        "image_path2": args.image_path2,
                        "wins": [wins1, wins2],
                        "win_rates": [win_rate1, win_rate2],
                        "total_comparisons": total,
                        "max_images": args.max_images
                    }
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📁 Results saved to {args.output}")
            else:
                print("❌ No valid comparisons completed")
        else:
            print("❌ Evaluation failed")

if __name__ == "__main__":
    main() 
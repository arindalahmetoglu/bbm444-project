import argparse

from metrics.qwen import get_qwen_siqs, get_qwen_micqs 
from metrics.glm import get_glm_siqs, get_glm_micqs

def parse_args():
    parser = argparse.ArgumentParser(description="RDIStitcher.")
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["qwen-siqs", "qwen-micqs", "glm-siqs", "glm-micqs"],
        help="type of metric",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="path to evaluation stitched images",
    )
    parser.add_argument(
        "--image_path2",
        type=str,
        help="path2 to evaluation stitched images, only used for MICQS",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="api key can be obatined from the official website"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        help="base url can be obatined from the official website",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.metric_type == "qwen-siqs":
        print(get_qwen_siqs(args))
    elif args.metric_type == "qwen-micqs":
        print(get_qwen_micqs(args))
    elif args.metric_type == "glm-siqs":
        print(get_glm_siqs(args))
    elif args.metric_type == "glm-micqs":
        print(get_glm_micqs(args))

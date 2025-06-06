#!/usr/bin/env python3
"""
Implements NIQE, PIQE, and BRISQUE metrics for objective evaluation of image composition methods.
Supports both UDIS-D and Beehive dataset results with comprehensive statistical analysis.

All metrics use lower scores = better quality convention.

"""

import os
import json
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import NIQE implementation (BasicSR official)
try:
    from basicsr_niqe import calculate_niqe
    print("Official BasicSR NIQE implementation loaded successfully")
    NIQE_AVAILABLE = True
except ImportError as e:
    print(f" BasicSR NIQE implementation not available: {e}")
    NIQE_AVAILABLE = False

# Import official PIQE implementation
try:
    from pypiqe import piqe as pypiqe_calculate
    print("Official pypiqe implementation loaded successfully")
    PIQE_AVAILABLE = True
except ImportError as e:
    print(f" pypiqe not available: {e}")
    PIQE_AVAILABLE = False

# Import PIQ for BRISQUE
try:
    import piq
    import torch
    print("PIQ library for BRISQUE loaded successfully")
    BRISQUE_AVAILABLE = True
except ImportError as e:
    print(f"PIQ not available: {e}")
    BRISQUE_AVAILABLE = False

class NRIQAMetrics:
    """No-Reference Image Quality Assessment metrics implementation"""
    
    def __init__(self):
        """Initialize the metrics calculator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if BRISQUE_AVAILABLE else None
        if self.device:
            print(f"Using device: {self.device}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f" Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB (standard format for most processing)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Check if image is valid
            if img.size == 0:
                print(f" Empty image: {image_path}")
                return None
            
            return img
            
        except Exception as e:
            print(f" Error loading {image_path}: {e}")
            return None
    
    def compute_niqe(self, image: np.ndarray) -> Optional[float]:
        """
        Compute NIQE (Naturalness Image Quality Evaluator) using BasicSR implementation
        Lower scores indicate better quality
        """
        try:
            if not NIQE_AVAILABLE:
                return None
            
            # BasicSR NIQE expects BGR format
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            # Use the BasicSR NIQE implementation
            niqe_score = calculate_niqe(bgr_image, crop_border=0, input_order='HWC', convert_to='y')
            return float(niqe_score)
            
        except Exception as e:
            print(f" NIQE computation failed: {e}")
            return None
    
    def compute_piqe(self, image: np.ndarray) -> Optional[float]:
        """
        Compute PIQE (Perception based Image Quality Evaluator) using official pypiqe
        Lower scores indicate better quality
        """
        try:
            if not PIQE_AVAILABLE:
                return None
            
            # Ensure image is uint8 format as required by pypiqe
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Image is in [0,1] range
                    image_uint8 = (image * 255).astype(np.uint8)
                else:  # Image is in [0,255] range
                    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # Ensure we have 3 channels
            if len(image_uint8.shape) == 2:
                image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            elif len(image_uint8.shape) == 3 and image_uint8.shape[2] == 4:
                image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGBA2RGB)
            
            # pypiqe expects RGB format (which we already have)
            piqe_score, _, _, _ = pypiqe_calculate(image_uint8)
            
            return float(piqe_score)
            
        except Exception as e:
            print(f" PIQE computation failed: {e}")
            return None
    
    def compute_brisque(self, image: np.ndarray) -> Optional[float]:
        """
        Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) using PIQ
        Lower scores indicate better quality
        """
        try:
            if not BRISQUE_AVAILABLE:
                return None
            
            # Ensure image is in proper format
            if len(image.shape) == 2:
                # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Convert to tensor and normalize to [0,1]
            if image.dtype == np.uint8:
                img_tensor = torch.from_numpy(image).float() / 255.0
            else:
                img_tensor = torch.from_numpy(image).float()
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
            
            # Ensure we have the right tensor format: (1, C, H, W)
            if len(img_tensor.shape) == 3:  # (H, W, C)
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            elif len(img_tensor.shape) == 4:  # Already (N, C, H, W)
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}")
            
            img_tensor = img_tensor.to(self.device)
            
            # Compute BRISQUE using piq library
            brisque_score = piq.brisque(img_tensor, data_range=1.0)
            return float(brisque_score.cpu())
            
        except Exception as e:
            print(f" BRISQUE computation failed: {e}")
            return None
    
    def evaluate_single_image(self, image_path: str) -> Dict:
        """Evaluate a single image with all metrics"""
        start_time = time.time()
        
        # Initialize result dict
        result = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "niqe": None,
            "piqe": None,
            "brisque": None,
            "processing_time": 0,
            "error": None
        }

        # Load image
        try:
            image = self.load_image(image_path)
            if image is None:
                result["error"] = "Failed to load image"
                return result
        except Exception as e:
            result["error"] = f"Error loading image: {e}"
            return result

        # Compute all metrics
        try:
            niqe_score = self.compute_niqe(image)
            result["niqe"] = niqe_score
        except Exception as e:
            print(f"CRITICAL: NIQE failed catastrophically on {os.path.basename(image_path)}: {e}")

        try:
            piqe_score = self.compute_piqe(image)
            result["piqe"] = piqe_score
        except Exception as e:
            print(f"CRITICAL: PIQE failed catastrophically on {os.path.basename(image_path)}: {e}")

        try:
            brisque_score = self.compute_brisque(image)
            result["brisque"] = brisque_score
        except Exception as e:
            print(f"CRITICAL: BRISQUE failed catastrophically on {os.path.basename(image_path)}: {e}")
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Print progress indicator
        metrics_status = []
        if result['niqe'] is not None:
            metrics_status.append(f"NIQE: {result['niqe']:.3f}")
        if result['piqe'] is not None:
            metrics_status.append(f"PIQE: {result['piqe']:.3f}")
        if result['brisque'] is not None:
            metrics_status.append(f"BRISQUE: {result['brisque']:.3f}")
        
        status_str = " | ".join(metrics_status) if metrics_status else "No metrics computed"
        print(f"{os.path.basename(image_path)}: {status_str} ({processing_time:.2f}s)")
        
        return result

def evaluate_method_directory(method_name: str, directory_path: str, max_images: int = None) -> List[Dict]:
    """Evaluate all images in a method directory"""
    print(f"\n Evaluating {method_name} method")
    print(f" Directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f" Directory not found: {directory_path}")
        return []
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in Path(directory_path).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    image_files.sort()  # Sort for consistent ordering
    
    if not image_files:
        print(f" No image files found in {directory_path}")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Limit number of images if specified
    if max_images and max_images > 0:
        image_files = image_files[:max_images]
        print(f" Limiting evaluation to {max_images} images")
    
    # Initialize metrics calculator
    metrics = NRIQAMetrics()
    results = []
    
    # Evaluate each image
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i:3d}/{len(image_files)}] ", end="")
        result = metrics.evaluate_single_image(image_path)
        results.append(result)
    
    # Summary for this method
    successful = len([r for r in results if any(r[metric] is not None for metric in ['niqe', 'piqe', 'brisque'])])
    print(f" {method_name}: {successful}/{len(results)} images evaluated successfully")
    
    return results

def compute_aggregate_statistics(results: List[Dict], method_name: str) -> Dict:
    """Compute aggregate statistics for a method"""
    # Extract valid scores for each metric
    niqe_scores = [r['niqe'] for r in results if r['niqe'] is not None]
    piqe_scores = [r['piqe'] for r in results if r['piqe'] is not None]
    brisque_scores = [r['brisque'] for r in results if r['brisque'] is not None]
    
    def compute_stats(scores):
        if not scores:
            return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
        return {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "median": statistics.median(scores),
            "min": min(scores),
            "max": max(scores)
        }
    
    return {
        "method": method_name,
        "total_images": len(results),
        "successful_evaluations": len([r for r in results if r.get('niqe') is not None or r.get('piqe') is not None or r.get('brisque') is not None]),
        "niqe_stats": compute_stats(niqe_scores),
        "piqe_stats": compute_stats(piqe_scores),
        "brisque_stats": compute_stats(brisque_scores)
    }

def generate_comparison_report(all_results: Dict[str, List[Dict]], output_dir: str, dataset_name: str = ""):
    """Generate a comprehensive comparison report"""
    
    # Compute aggregate statistics for each method
    method_stats = {}
    for method, results in all_results.items():
        method_stats[method] = compute_aggregate_statistics(results, method)
    
    # Generate text report
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    report = f"""Quantitative Metrics Evaluation Report{dataset_suffix}
===============================================================================

Evaluation of No-Reference Image Quality Assessment metrics for image composition methods.
Lower scores indicate better perceived quality for all metrics.

Metrics Used:
- NIQE: Naturalness Image Quality Evaluator (BasicSR implementation)
- PIQE: Perception based Image Quality Evaluator (pypiqe implementation)  
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator (PIQ implementation)

"""
    
    # Add summary statistics
    report += "SUMMARY STATISTICS\n"
    report += "==================\n\n"
    
    for method, stats in method_stats.items():
        report += f"{method.upper()} Method:\n"
        report += f"  Total Images: {stats['total_images']}\n"
        report += f"  Successful Evaluations: {stats['successful_evaluations']}\n"
        
        if stats['successful_evaluations'] > 0:
            success_rate = (stats['successful_evaluations'] / stats['total_images']) * 100
            report += f"  Success Rate: {success_rate:.1f}%\n"
        report += "\n"
        
        for metric in ['niqe', 'piqe', 'brisque']:
            metric_stats = stats[f'{metric}_stats']
            if metric_stats['count'] > 0:
                report += f"  {metric.upper()} Statistics:\n"
                report += f"    Mean: {metric_stats['mean']:.4f}\n"
                report += f"    Std:  {metric_stats['std']:.4f}\n"
                report += f"    Median: {metric_stats['median']:.4f}\n"
                report += f"    Min: {metric_stats['min']:.4f}\n"
                report += f"    Max: {metric_stats['max']:.4f}\n"
                report += f"    Count: {metric_stats['count']}\n\n"
            else:
                report += f"  {metric.upper()}: No valid scores\n\n"
        
        report += "\n"
    
    # Add method comparison
    report += "METHOD COMPARISON (Lower is Better)\n"
    report += "====================================\n\n"
    
    for metric in ['niqe', 'piqe', 'brisque']:
        report += f"{metric.upper()} Comparison:\n"
        metric_means = []
        for method, stats in method_stats.items():
            mean_score = stats[f'{metric}_stats']['mean']
            if mean_score is not None:
                metric_means.append((method, mean_score))
        
        if metric_means:
            # Sort by score (lower is better)
            metric_means.sort(key=lambda x: x[1])
            for i, (method, score) in enumerate(metric_means, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                report += f"  {emoji} {i}. {method}: {score:.4f}\n"
        else:
            report += "  No valid scores available\n"
        report += "\n"
    
    # Add interpretation guide
    report += "INTERPRETATION GUIDE\n"
    report += "====================\n\n"
    report += "NIQE (Naturalness Image Quality Evaluator):\n"
    report += "  - Measures deviation from natural image statistics\n"
    report += "  - Range: [0, ∞), lower is better\n"
    report += "  - Good quality: < 5, Excellent: < 3\n\n"
    
    report += "PIQE (Perception based Image Quality Evaluator):\n"
    report += "  - Assesses spatial and spectral distortions\n"
    report += "  - Range: [0, 100], lower is better\n"
    report += "  - Good quality: < 45, Excellent: < 30\n\n"
    
    report += "BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator):\n"
    report += "  - Evaluates image naturalness based on spatial statistics\n"
    report += "  - Range: [0, 100], lower is better\n"
    report += "  - Good quality: < 40, Excellent: < 20\n\n"
    
    # Save report
    report_filename = f"nriqa_evaluation_report_{dataset_name.lower()}.txt" if dataset_name else "nriqa_evaluation_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save detailed JSON results
    json_results = {
        "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset_name,
        "metrics_info": {
            "niqe": "Naturalness Image Quality Evaluator (BasicSR) - Lower scores = better quality",
            "piqe": "Perception based Image Quality Evaluator (pypiqe) - Lower scores = better quality", 
            "brisque": "Blind/Referenceless Image Spatial Quality Evaluator (PIQ) - Lower scores = better quality"
        },
        "method_statistics": method_stats,
        "detailed_results": all_results
    }
    
    json_filename = f"nriqa_detailed_results_{dataset_name.lower()}.json" if dataset_name else "nriqa_detailed_results.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Report saved to: {report_path}")
    print(f"Detailed JSON: {json_path}")
    
    return method_stats

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Metrics Evaluation for Image Composition Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate UDIS-D results
  python evaluate_quantitative_metrics.py udis_d_results --dataset udis-d
  
  # Evaluate Beehive results  
  python evaluate_quantitative_metrics.py beehive_results --dataset beehive
  
  # Evaluate specific method only
  python evaluate_quantitative_metrics.py udis_d_results --method nis --max-images 50
  
  # Custom output directory
  python evaluate_quantitative_metrics.py udis_d_results --output-dir evaluation_results
        """
    )
    
    parser.add_argument("results_dir", type=str, 
                       help="Results directory containing method subdirectories (nis, udis, udis_plus_plus)")
    parser.add_argument("--dataset", type=str, choices=["udis-d", "beehive"], default="",
                       help="Dataset name for report organization (udis-d or beehive)")
    parser.add_argument("--max-images", type=int, default=None, 
                       help="Maximum number of images to evaluate per method (default: all)")
    parser.add_argument("--method", type=str, choices=["nis", "udis", "udis_plus_plus"], default=None,
                       help="Evaluate only specific method (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: results/<results_dir>_metrics)")
    
    args = parser.parse_args()
    
    # Prepend 'results/' to input directory and set default output directory
    results_dir_path = Path("results") / args.results_dir
    
    # Validate results directory
    if not results_dir_path.is_dir():
        print(f" Results directory not found: {results_dir_path}")
        print(f"   (Searched for '{args.results_dir}' inside the 'results/' directory)")
        return
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path("results") / f"{args.results_dir}_metrics"
    else:
        args.output_dir = Path(args.output_dir)
    
    # Define method directories based on the input directory
    method_mapping = {
        "nis": "nis",
        "udis": "udis", 
        "udis_plus_plus": "udis_plus_plus"
    }
    
    method_dirs = {}
    for method_key, method_subdir in method_mapping.items():
        method_path = results_dir_path / method_subdir
        if method_path.exists():
            method_dirs[method_key.upper().replace("_", "+")] = str(method_path)
        else:
            print(f" Method directory not found: {method_path}")
    
    if not method_dirs:
        print(f" No valid method directories found in: {results_dir_path}")
        print("Expected subdirectories: nis, udis, udis_plus_plus")
        return
    
    # Filter methods if specified
    if args.method:
        method_key = args.method.lower().replace("++", "_plus_plus")
        display_key = method_key.upper().replace("_", "+")
        
        found_method = None
        for key, path in method_dirs.items():
            if key.lower().replace("+", "_") == method_key:
                found_method = {key: path}
                break
        
        if found_method:
            method_dirs = found_method
        else:
            print(f" Unknown method: {args.method}")
            print(f"Available methods: {list(method_dirs.keys())}")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy required dependencies
    import shutil
    basicsr_source = "IMPORTANT_ALL_SCRIPTS/basicsr_niqe_standalone.py"
    if os.path.exists(basicsr_source):
        shutil.copy(basicsr_source, ".")
        print("✓ Copied basicsr_niqe_standalone.py")
    
    print("Quantitative Metrics Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper() if args.dataset else 'Unknown'}")
    print(f"Results directory: {results_dir_path}")
    print(f"Methods found: {list(method_dirs.keys())}")
    print(f"Metrics: NIQE (BasicSR), PIQE (pypiqe), BRISQUE (PIQ)")
    print(f"Output directory: {args.output_dir}")
    if args.max_images:
        print(f"Max images per method: {args.max_images}")
    print()
    
    # Evaluate each method
    all_results = {}
    for method_name, directory_path in method_dirs.items():
        results = evaluate_method_directory(method_name, directory_path, args.max_images)
        all_results[method_name] = results
    
    if not all_results:
        print(" No results obtained. Check your directory paths.")
        return
    
    # Generate comparison report
    generate_comparison_report(all_results, args.output_dir, args.dataset.upper() if args.dataset else "")

if __name__ == "__main__":
    main() 
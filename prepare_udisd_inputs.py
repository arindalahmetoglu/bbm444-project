#!/usr/bin/env python3
"""
Convert ORB+FLANN warps to format expected by UDIS/UDIS++
This script converts:
- .txt homography files to .npz format
- Copies warp and mask files to sift_preprocessed_data structure
"""

import os
import sys
import argparse
import numpy as np
import shutil
from pathlib import Path

def convert_homography_txt_to_npz(txt_file, output_dir, pair_idx):
    """
    Convert homography .txt file to .npz format expected by UDIS
    """
    try:
        # Load homography matrix from .txt file
        H = np.loadtxt(txt_file)
        
        # Create output directory for this pair
        pair_dir = os.path.join(output_dir, f"{pair_idx:06d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save as .npz file in the format UDIS expects
        npz_file = os.path.join(pair_dir, "h_params.npz")
        
        # UDIS expects the homography matrix in a specific format
        # Based on the working script, we need to save it as a structured array
        np.savez(npz_file, homography=H)
        
        return True
    except Exception as e:
        print(f"Error converting homography for pair {pair_idx}: {e}")
        return False

def copy_warp_and_mask_files(warps_dir, sift_data_dir, pair_idx):
    """
    Copy warp and mask files from ORB+FLANN output to sift_preprocessed_data structure
    """
    pair_idx_formatted = f"{pair_idx:06d}"
    
    # Source files (from ORB+FLANN preparation)
    warp1_src = os.path.join(warps_dir, "warped", "input1", f"{pair_idx_formatted}.png")
    warp2_src = os.path.join(warps_dir, "warped", "input2", f"{pair_idx_formatted}.png")
    mask1_src = os.path.join(warps_dir, "masks", "input1", f"{pair_idx_formatted}.png")
    mask2_src = os.path.join(warps_dir, "masks", "input2", f"{pair_idx_formatted}.png")
    
    # Destination files (for UDIS compatibility)
    warp1_dst = os.path.join(sift_data_dir, "warp1", f"{pair_idx_formatted}.png")
    warp2_dst = os.path.join(sift_data_dir, "warp2", f"{pair_idx_formatted}.png")
    mask1_dst = os.path.join(sift_data_dir, "mask1", f"{pair_idx_formatted}.png")
    mask2_dst = os.path.join(sift_data_dir, "mask2", f"{pair_idx_formatted}.png")
    
    # Check if all source files exist
    source_files = [warp1_src, warp2_src, mask1_src, mask2_src]
    for src_file in source_files:
        if not os.path.exists(src_file):
            print(f"Warning: Source file not found: {src_file}")
            return False
    
    # Copy files
    try:
        shutil.copy2(warp1_src, warp1_dst)
        shutil.copy2(warp2_src, warp2_dst)
        shutil.copy2(mask1_src, mask1_dst)
        shutil.copy2(mask2_src, mask2_dst)
        return True
    except Exception as e:
        print(f"Error copying warp/mask files for pair {pair_idx}: {e}")
        return False

def setup_sift_preprocessed_structure(sift_data_dir):
    """
    Create the directory structure expected by UDIS/UDIS++
    """
    subdirs = ["warp1", "warp2", "mask1", "mask2", "homography_params"]
    for subdir in subdirs:
        os.makedirs(os.path.join(sift_data_dir, subdir), exist_ok=True)

def convert_orb_flann_to_udis_format(warps_dir, output_dir, pair_indices=None):
    """
    Convert all ORB+FLANN warps to UDIS format
    """
    print(f"Converting ORB+FLANN warps to UDIS format...")
    print(f"Source: {warps_dir}")
    print(f"Output: {output_dir}")
    
    # Setup output directory structure
    setup_sift_preprocessed_structure(output_dir)
    
    # Get list of available homography files
    homography_dir = os.path.join(warps_dir, "homographies")
    if not os.path.exists(homography_dir):
        print(f"Error: Homography directory not found: {homography_dir}")
        return 0
    
    if pair_indices is None:
        # Process all available pairs
        homography_files = list(Path(homography_dir).glob("*.txt"))
        pair_indices = [int(f.stem) for f in homography_files]
    
    success_count = 0
    total_count = len(pair_indices)
    
    for pair_idx in sorted(pair_indices):
        pair_idx_formatted = f"{pair_idx:06d}"
        
        print(f"Converting pair {pair_idx_formatted}...")
        
        # Convert homography file
        txt_file = os.path.join(homography_dir, f"{pair_idx_formatted}.txt")
        if not os.path.exists(txt_file):
            print(f"  Warning: Homography file not found: {txt_file}")
            continue
        
        homography_success = convert_homography_txt_to_npz(
            txt_file, 
            os.path.join(output_dir, "homography_params"),
            pair_idx
        )
        
        # Copy warp and mask files
        warp_success = copy_warp_and_mask_files(warps_dir, output_dir, pair_idx)
        
        if homography_success and warp_success:
            success_count += 1
            print(f"  ✓ Successfully converted pair {pair_idx_formatted}")
        else:
            print(f"  ✗ Failed to convert pair {pair_idx_formatted}")
    
    print(f"\nConversion complete: {success_count}/{total_count} pairs converted successfully")
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="Convert ORB+FLANN warps to UDIS/UDIS++ compatible format.",
        epilog="This script takes the output from prepare_warp_homographies_udisd.sh and restructures it for UDIS and UDIS++."
    )
    parser.add_argument('--warps-dir', type=str, required=True,
                        help="Path to the directory containing ORB+FLANN warps (e.g., 'udis_d_warps').")
    parser.add_argument('--output-dir', type=str, default='processing_data/sift_preprocessed_data',
                        help="Path to the output directory (e.g., 'sift_preprocessed_data').")
    parser.add_argument('--pairs', nargs='+', type=int,
                        help='Specific pair indices to convert (default: all available)')
    parser.add_argument('--start', type=int, help='Start index for range conversion')
    parser.add_argument('--end', type=int, help='End index for range conversion')
    parser.add_argument('--stride', type=int, default=1, help='Stride for range conversion')
    
    args = parser.parse_args()
    
    # Determine which pairs to convert
    pair_indices = None
    if args.pairs:
        pair_indices = args.pairs
    elif args.start is not None and args.end is not None:
        pair_indices = list(range(args.start, args.end + 1, args.stride))
    
    # Convert warps
    success_count = convert_orb_flann_to_udis_format(
        args.warps_dir, 
        args.output_dir, 
        pair_indices
    )
    
    if success_count > 0:
        print(f"\n✓ Conversion successful! {success_count} pairs converted.")
        print(f"UDIS/UDIS++ can now use data from: {args.output_dir}")
    else:
        print("\n✗ Conversion failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
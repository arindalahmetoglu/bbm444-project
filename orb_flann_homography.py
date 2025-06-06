#!/usr/bin/env python3

"""
ORB+FLANN Homography Estimation for UDIS-D Dataset
==================================================

This script implements the standardized geometric warp generation pipeline
described in the paper's Appendix A using:
- ORB (Oriented FAST and Rotated BRIEF) feature detection
- FLANN (Fast Library for Approximate Nearest Neighbors) matching  
- RANSAC (Random Sample Consensus) homography estimation

"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ORBFLANNHomographyEstimator:
    """
    ORB+FLANN based homography estimator for image stitching
    """
    
    def __init__(self, max_features=5000, match_ratio=0.75, ransac_threshold=4.0, 
                 max_canvas_ratio=3.0):
        """
        Initialize the homography estimator
        
        Args:
            max_features: Maximum number of ORB features to detect
            match_ratio: Ratio threshold for Lowe's ratio test  
            ransac_threshold: RANSAC reprojection threshold
            max_canvas_ratio: Maximum canvas size ratio vs original image
        """
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.max_canvas_ratio = max_canvas_ratio
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        
        # Initialize FLANN matcher for binary descriptors
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,      
            key_size=12,         
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        logger.info(f"Initialized ORB+FLANN estimator:")
        logger.info(f"  Max features: {self.max_features}")
        logger.info(f"  Match ratio: {self.match_ratio}")
        logger.info(f"  RANSAC threshold: {self.ransac_threshold}")
        logger.info(f"  Max canvas ratio: {self.max_canvas_ratio}")

    def load_images(self, ref_path, tgt_path):
        """
        Load and validate input images
        
        Args:
            ref_path: Path to reference image
            tgt_path: Path to target image
            
        Returns:
            Tuple of (ref_image, tgt_image) as numpy arrays
        """
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target image not found: {tgt_path}")
        
        # Load images using OpenCV (BGR format)
        ref_img = cv2.imread(ref_path)
        tgt_img = cv2.imread(tgt_path)
        
        if ref_img is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        if tgt_img is None:
            raise ValueError(f"Could not load target image: {tgt_path}")
        
        logger.info(f"Loaded images:")
        logger.info(f"  Reference: {ref_img.shape} from {os.path.basename(ref_path)}")
        logger.info(f"  Target: {tgt_img.shape} from {os.path.basename(tgt_path)}")
        
        return ref_img, tgt_img

    def detect_and_match_features(self, ref_img, tgt_img):
        """
        Detect ORB features and match using FLANN
        
        Args:
            ref_img: Reference image (numpy array)
            tgt_img: Target image (numpy array)
            
        Returns:
            Tuple of (good_matches, kp1, kp2, des1, des2)
        """
        # Convert to grayscale for feature detection
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(tgt_gray, None)
        
        if des1 is None or des2 is None:
            raise ValueError("Failed to detect descriptors in one or both images")
        
        logger.info(f"ORB feature detection:")
        logger.info(f"  Reference: {len(kp1)} keypoints")
        logger.info(f"  Target: {len(kp2)} keypoints")
        
        # Match descriptors using FLANN
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        logger.info(f"FLANN matching:")
        logger.info(f"  Raw matches: {len(matches)}")
        logger.info(f"  Good matches (ratio test): {len(good_matches)}")
        
        if len(good_matches) < 4:
            raise ValueError(f"Insufficient matches for homography: {len(good_matches)} < 4")
        
        return good_matches, kp1, kp2, des1, des2

    def estimate_homography(self, good_matches, kp1, kp2):
        """
        Estimate homography using RANSAC
        
        Args:
            good_matches: List of good DMatch objects
            kp1: Keypoints from reference image
            kp2: Keypoints from target image
            
        Returns:
            Homography matrix (3x3 numpy array)
        """
        # Extract matched point coordinates
        # Points from reference image (kp1)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # Points from target image (kp2) 
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography: target -> reference
        # This will warp target image to align with reference
        H, mask = cv2.findHomography(
            dst_pts, src_pts, 
            cv2.RANSAC, 
            self.ransac_threshold
        )
        
        if H is None:
            raise ValueError("RANSAC failed to estimate homography")
        
        # Calculate inlier statistics
        inliers = np.sum(mask.ravel())
        inlier_ratio = inliers / len(good_matches)
        
        logger.info(f"RANSAC homography estimation:")
        logger.info(f"  Inliers: {inliers}/{len(good_matches)}")
        logger.info(f"  Inlier ratio: {inlier_ratio:.3f}")
        
        if inlier_ratio < 0.1:  # Less than 10% inliers
            logger.warning(f"Low inlier ratio: {inlier_ratio:.3f}")
        
        return H

    def calculate_canvas_size(self, ref_img, tgt_img, H):
        """
        Calculate the canvas size needed to fit both warped images
        
        Args:
            ref_img: Reference image
            tgt_img: Target image  
            H: Homography matrix (target -> reference)
            
        Returns:
            Tuple of (canvas_width, canvas_height, offset_x, offset_y)
        """
        h_ref, w_ref = ref_img.shape[:2]
        h_tgt, w_tgt = tgt_img.shape[:2]
        
        # Get corners of target image
        tgt_corners = np.float32([
            [0, 0], [w_tgt, 0], [w_tgt, h_tgt], [0, h_tgt]
        ]).reshape(-1, 1, 2)
        
        # Transform target corners to reference coordinate system
        tgt_corners_warped = cv2.perspectiveTransform(tgt_corners, H)
        
        # Get corners of reference image (already in reference coordinate system)
        ref_corners = np.float32([
            [0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]
        ]).reshape(-1, 1, 2)
        
        # Combine all corners
        all_corners = np.concatenate([ref_corners, tgt_corners_warped], axis=0)
        
        # Find bounding box
        x_coords = all_corners[:, 0, 0]
        y_coords = all_corners[:, 0, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Calculate canvas size
        canvas_width = int(np.ceil(max_x - min_x))
        canvas_height = int(np.ceil(max_y - min_y))
        
        # Calculate offsets (to handle negative coordinates)
        offset_x = -int(np.floor(min_x))
        offset_y = -int(np.floor(min_y))
        
        # Check canvas size ratio
        original_area = h_ref * w_ref
        canvas_area = canvas_height * canvas_width
        size_ratio = canvas_area / original_area
        
        if size_ratio > self.max_canvas_ratio:
            raise ValueError(f"Canvas too large: ratio {size_ratio:.2f} > {self.max_canvas_ratio}")
        
        logger.info(f"Canvas calculation:")
        logger.info(f"  Original size: {w_ref}x{h_ref}")
        logger.info(f"  Canvas size: {canvas_width}x{canvas_height}")
        logger.info(f"  Size ratio: {size_ratio:.2f}")
        logger.info(f"  Offset: ({offset_x}, {offset_y})")
        
        return canvas_width, canvas_height, offset_x, offset_y

    def create_warps_and_masks(self, ref_img, tgt_img, H, canvas_width, canvas_height, 
                               offset_x, offset_y):
        """
        Create warped images and masks
        
        Args:
            ref_img: Reference image
            tgt_img: Target image
            H: Homography matrix (target -> reference)
            canvas_width, canvas_height: Canvas dimensions
            offset_x, offset_y: Offset to handle negative coordinates
            
        Returns:
            Tuple of (warped_ref, warped_tgt, mask_ref, mask_tgt)
        """
        # Create translation matrix to handle offset
        T = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Transform reference image (identity transform + translation)
        H_ref = T  # Only translation for reference
        H_tgt = T @ H  # Translation + homography for target
        
        # Warp images
        warped_ref = cv2.warpPerspective(
            ref_img, H_ref, (canvas_width, canvas_height)
        )
        warped_tgt = cv2.warpPerspective(
            tgt_img, H_tgt, (canvas_width, canvas_height)
        )
        
        # Create masks (all-ones images warped the same way)
        h_ref, w_ref = ref_img.shape[:2]
        h_tgt, w_tgt = tgt_img.shape[:2]
        
        mask_ref_orig = np.ones((h_ref, w_ref), dtype=np.uint8) * 255
        mask_tgt_orig = np.ones((h_tgt, w_tgt), dtype=np.uint8) * 255
        
        mask_ref = cv2.warpPerspective(
            mask_ref_orig, H_ref, (canvas_width, canvas_height)
        )
        mask_tgt = cv2.warpPerspective(
            mask_tgt_orig, H_tgt, (canvas_width, canvas_height)
        )
        
        # Calculate coverage statistics
        total_pixels = canvas_width * canvas_height
        ref_coverage = np.sum(mask_ref > 0) / total_pixels
        tgt_coverage = np.sum(mask_tgt > 0) / total_pixels
        overlap_coverage = np.sum((mask_ref > 0) & (mask_tgt > 0)) / total_pixels
        
        logger.info(f"Warp statistics:")
        logger.info(f"  Reference coverage: {ref_coverage:.3f}")
        logger.info(f"  Target coverage: {tgt_coverage:.3f}")
        logger.info(f"  Overlap coverage: {overlap_coverage:.3f}")
        
        return warped_ref, warped_tgt, mask_ref, mask_tgt

    def process_image_pair(self, ref_path, tgt_path):
        """
        Complete pipeline to process an image pair
        
        Args:
            ref_path: Path to reference image
            tgt_path: Path to target image
            
        Returns:
            Dictionary containing all results
        """
        try:
            # Load images
            ref_img, tgt_img = self.load_images(ref_path, tgt_path)
            
            # Detect and match features
            good_matches, kp1, kp2, des1, des2 = self.detect_and_match_features(ref_img, tgt_img)
            
            # Estimate homography
            H = self.estimate_homography(good_matches, kp1, kp2)
            
            # Calculate canvas size
            canvas_width, canvas_height, offset_x, offset_y = self.calculate_canvas_size(
                ref_img, tgt_img, H
            )
            
            # Create warps and masks
            warped_ref, warped_tgt, mask_ref, mask_tgt = self.create_warps_and_masks(
                ref_img, tgt_img, H, canvas_width, canvas_height, offset_x, offset_y
            )
            
            return {
                'success': True,
                'homography': H,
                'canvas_size': (canvas_width, canvas_height),
                'offset': (offset_x, offset_y),
                'warped_ref': warped_ref,
                'warped_tgt': warped_tgt,
                'mask_ref': mask_ref,
                'mask_tgt': mask_tgt,
                'num_matches': len(good_matches),
                'num_inliers': np.sum(cv2.findHomography(
                    np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2),
                    np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2),
                    cv2.RANSAC, self.ransac_threshold
                )[1].ravel()) if len(good_matches) >= 4 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process image pair: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def save_results(results, output_dir, pair_idx):
    """
    Save processing results to files
    
    Args:
        results: Dictionary from process_image_pair
        output_dir: Output directory
        pair_idx: Pair index for file naming
    """
    if not results['success']:
        raise ValueError(f"Cannot save failed results: {results['error']}")
    
    # Create output directories
    homography_dir = os.path.join(output_dir, 'homographies')
    warped_dir = os.path.join(output_dir, 'warped')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(homography_dir, exist_ok=True)
    os.makedirs(os.path.join(warped_dir, 'input1'), exist_ok=True)
    os.makedirs(os.path.join(warped_dir, 'input2'), exist_ok=True)
    os.makedirs(os.path.join(masks_dir, 'input1'), exist_ok=True)
    os.makedirs(os.path.join(masks_dir, 'input2'), exist_ok=True)
    
    # Format pair index
    pair_idx_str = f"{pair_idx:06d}"
    
    # Save homography matrix
    homography_file = os.path.join(homography_dir, f"{pair_idx_str}.txt")
    np.savetxt(homography_file, results['homography'], fmt='%.10f')
    
    # Save warped images
    ref_warp_file = os.path.join(warped_dir, 'input1', f"{pair_idx_str}.png")
    tgt_warp_file = os.path.join(warped_dir, 'input2', f"{pair_idx_str}.png")
    cv2.imwrite(ref_warp_file, results['warped_ref'])
    cv2.imwrite(tgt_warp_file, results['warped_tgt'])
    
    # Save masks
    ref_mask_file = os.path.join(masks_dir, 'input1', f"{pair_idx_str}.png")
    tgt_mask_file = os.path.join(masks_dir, 'input2', f"{pair_idx_str}.png")
    cv2.imwrite(ref_mask_file, results['mask_ref'])
    cv2.imwrite(tgt_mask_file, results['mask_tgt'])
    
    logger.info(f"Saved results for pair {pair_idx}:")
    logger.info(f"  Homography: {homography_file}")
    logger.info(f"  Warped images: {ref_warp_file}, {tgt_warp_file}")
    logger.info(f"  Masks: {ref_mask_file}, {tgt_mask_file}")


def main():
    parser = argparse.ArgumentParser(
        description='ORB+FLANN Homography Estimation for UDIS-D Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--ref_image', required=True, 
                        help='Path to reference image')
    parser.add_argument('--tgt_image', required=True,
                        help='Path to target image')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--pair_idx', type=int, required=True,
                        help='Pair index for file naming')
    
    # ORB+FLANN parameters
    parser.add_argument('--max_features', type=int, default=5000,
                        help='Maximum number of ORB features')
    parser.add_argument('--match_ratio', type=float, default=0.75,
                        help='FLANN match ratio threshold')
    parser.add_argument('--ransac_threshold', type=float, default=4.0,
                        help='RANSAC reprojection threshold')
    parser.add_argument('--max_canvas_ratio', type=float, default=3.0,
                        help='Maximum canvas size ratio')
    
    # Other options
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    logger.info("ORB+FLANN Homography Estimation")
    logger.info(f"Reference: {args.ref_image}")
    logger.info(f"Target: {args.tgt_image}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Pair index: {args.pair_idx}")
    
    try:
        # Initialize estimator
        estimator = ORBFLANNHomographyEstimator(
            max_features=args.max_features,
            match_ratio=args.match_ratio,
            ransac_threshold=args.ransac_threshold,
            max_canvas_ratio=args.max_canvas_ratio
        )
        
        # Process image pair
        results = estimator.process_image_pair(args.ref_image, args.tgt_image)
        
        if not results['success']:
            logger.error(f"Processing failed: {results['error']}")
            return 1
        
        # Save results
        save_results(results, args.output_dir, args.pair_idx)
        
        logger.info("✓ Successfully completed ORB+FLANN homography estimation")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
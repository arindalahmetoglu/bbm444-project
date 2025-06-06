import os
import gc
import cv2
import yaml
import utils
import models
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from srwarp import transform
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import signal
import sys

# Custom timeout exception
class TimeoutException(Exception):
    pass

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

# Disable CUDA devices with less than required memory (in MB)
def check_gpu_memory(required_mb=4000):
    try:
        # Get available GPU memory
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            available_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
            if available_mb < required_mb:
                print(f"Warning: GPU has only {available_mb:.0f}MB memory, {required_mb}MB required")
                return False
            return True
        return False
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False

def estimate_homography_enhanced(ref_tensor, tgt_tensor, feature_type='sift', 
                                matcher_type='flann', min_match_count=10, 
                                lowe_ratio=0.75, ransac_reproj_threshold=5.0,
                                debug_mode=False, timeout_seconds=30):
    """
    Estimates homography using enhanced feature extraction and matching.
    
    Args:
        ref_tensor: Reference image tensor (CHW, 0-1 float)
        tgt_tensor: Target image tensor (CHW, 0-1 float)
        feature_type: Type of feature detector/descriptor ('sift', 'orb', 'akaze', 'brisk')
        matcher_type: Type of matcher ('bf' or 'flann')
        min_match_count: Minimum number of good matches
        lowe_ratio: Ratio threshold for Lowe's ratio test
        ransac_reproj_threshold: RANSAC reprojection threshold
        debug_mode: Whether to save debug visualization
        timeout_seconds: Maximum time allowed for homography estimation
        
    Returns:
        H_tgt2ref (3x3 NumPy array), success_flag (boolean), and timed_out (boolean)
    """
    start_time = time.time()
    timed_out = False
    
    # Set up timeout signal handler
    if timeout_seconds > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        # Convert tensors to OpenCV format (HWC, 0-255 uint8)
        ref_np = ref_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tgt_np = tgt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        ref_np = (ref_np * 255).astype(np.uint8)
        tgt_np = (tgt_np * 255).astype(np.uint8)

        # Convert to grayscale
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        tgt_gray = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2GRAY)
        
        # Create feature detector/descriptor
        if feature_type.lower() == 'sift':
            detector = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=20)
        elif feature_type.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, 
                                    nlevels=8, edgeThreshold=31, 
                                    firstLevel=0, WTA_K=2)
        elif feature_type.lower() == 'akaze':
            detector = cv2.AKAZE_create()
        elif feature_type.lower() == 'brisk':
            detector = cv2.BRISK_create()
        else:
            print(f"Warning: Unknown feature type '{feature_type}'. Using SIFT.")
            detector = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
        kp_tgt, des_tgt = detector.detectAndCompute(tgt_gray, None)
        
        if des_ref is None or des_tgt is None or len(kp_ref) < min_match_count or len(kp_tgt) < min_match_count:
            print(f"Warning: Not enough features detected. Ref: {len(kp_ref) if kp_ref else 0}, Tgt: {len(kp_tgt) if kp_tgt else 0}")
            return np.eye(3), False, False
        
        # Setup feature matcher
        if feature_type.lower() in ['sift', 'akaze']:
            if matcher_type.lower() == 'flann':
                # FLANN parameters for SIFT/SURF
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # Higher values = more accurate but slower
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:  # bf matcher
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # Binary descriptors (ORB, BRISK)
            if matcher_type.lower() == 'flann':
                # FLANN parameters for binary descriptors
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:  # bf matcher
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match features and apply ratio test
        try:
            if matcher_type.lower() == 'flann' or not matcher.crossCheck:
                # Use KNN match with ratio test
                matches = matcher.knnMatch(des_tgt, des_ref, k=2)
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
            else:
                # For cross-check matcher, use simple match
                matches = matcher.match(des_tgt, des_ref)
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
                # Take top matches
                good_matches = matches[:min(len(matches), 100)]
        except Exception as e:
            print(f"Error during matching: {e}")
            return np.eye(3), False, False
        
        # Check if we have enough good matches
        match_time = time.time() - start_time
        print(f"Found {len(good_matches)} good matches using {feature_type}/{matcher_type} in {match_time:.2f}s")
        
        if len(good_matches) < min_match_count:
            print(f"Warning: Not enough good matches - {len(good_matches)}/{min_match_count}")
            return np.eye(3), False, False
        
        # Optional: Debug visualization
        if debug_mode:
            try:
                os.makedirs("visualization", exist_ok=True)
                matches_img = cv2.drawMatches(tgt_np, kp_tgt, ref_np, kp_ref, good_matches, None, 
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite("visualization/matches.jpg", matches_img)
            except Exception as e:
                print(f"Warning: Could not save debug visualization: {e}")
        
        # Extract matched keypoints
        src_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H_tgt2ref, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_reproj_threshold)
        
        # Count inliers
        if mask is not None:
            inliers_count = np.sum(mask)
            inlier_ratio = inliers_count / len(good_matches)
            print(f"RANSAC found {inliers_count} inliers ({inlier_ratio:.2f}) out of {len(good_matches)} matches")
            
            # Optional: Debug visualization of inliers
            if debug_mode:
                try:
                    # Get only the inlier matches
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
                    inlier_img = cv2.drawMatches(tgt_np, kp_tgt, ref_np, kp_ref, inlier_matches, None, 
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imwrite("visualization/inliers.jpg", inlier_img)
                except Exception as e:
                    print(f"Warning: Could not save inlier visualization: {e}")
            
            # Check if enough inliers
            if inliers_count < 0.25 * min_match_count:
                print(f"Warning: Too few inliers: {inliers_count}")
                return np.eye(3), False, False
        else:
            print("Warning: RANSAC failed to find homography matrix")
            return np.eye(3), False, False
        
        if H_tgt2ref is None:
            print("Warning: RANSAC failed to find homography")
            return np.eye(3), False, False
        
        total_time = time.time() - start_time
        print(f"Homography estimation completed in {total_time:.2f}s")
        return H_tgt2ref, True, False
        
    except TimeoutException:
        print(f"Warning: Homography estimation timed out after {timeout_seconds} seconds")
        timed_out = True
        return np.eye(3), False, True
    
    except Exception as e:
        print(f"Error in homography estimation: {e}")
        return np.eye(3), False, False
    
    finally:
        # Cancel the alarm if it hasn't triggered
        if timeout_seconds > 0:
            signal.alarm(0)


def batched_predict(model, ref, ref_grid, ref_cell, ref_mask,
                    tgt, tgt_grid, tgt_cell, tgt_mask,
                    stit_grid, sizes, bsize):
    # Feature extraction
    fea_ref, fea_ref_grid, ref_coef, ref_freq = model.gen_feat(ref)
    fea_ref_w = model.NeuralWarping(
        ref, fea_ref, fea_ref_grid,
        ref_freq, ref_coef, ref_grid, ref_cell, sizes
    ).cpu()

    fea_tgt, fea_tgt_grid, tgt_coef, tgt_freq = model.gen_feat(tgt)
    fea_tgt_w = model.NeuralWarping(
        tgt, fea_tgt, fea_tgt_grid,
        tgt_freq, tgt_coef, tgt_grid, tgt_cell, sizes
    ).cpu()

    # Simple multiplication by masks (no seam cutting)
    fea_ref_w *= ref_mask.cpu()
    fea_tgt_w *= tgt_mask.cpu()
    fea = torch.cat([fea_ref_w, fea_tgt_w], dim=1).cuda()

    # Generate blended features
    stit_rep = model.gen_feat_for_blender(fea)

    # Process in batches
    ql = 0
    preds = []
    n = stit_grid.shape[1]

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(stit_rep, stit_grid[:, ql: qr, :])
        preds.append(pred)
        ql = qr

    # Concatenate results
    pred = torch.cat(preds, dim=1)
    return pred


def prepare_ingredients(tgt, ref, H_tgt2ref=None, load_preprocessed=False, pair_idx=None, 
                        preprocessed_dir=None, compute_on_the_fly=False, feature_type='sift',
                        matcher_type='flann', debug_mode=False, timeout_seconds=30,
                        max_canvas_ratio=1.8):
    b, c, h_ref, w_ref = ref.shape
    _, _, h_tgt, w_tgt = tgt.shape
    timed_out = False
    oversized_canvas = False

    # Get homography - prioritize the provided H_tgt2ref parameter
    if H_tgt2ref is not None:
        print("✓ Using provided IHN homography matrix - no feature computation needed")
    else:
        # Only try other sources if no homography was provided
        if load_preprocessed and pair_idx is not None and preprocessed_dir is not None and not compute_on_the_fly:
            try:
                pair_idx_str = f"{pair_idx:06d}"
                h_params_path = os.path.join(preprocessed_dir, "homography_params", pair_idx_str, "h_params.npz")
                
                if os.path.exists(h_params_path):
                    print(f"Loading precomputed homography from: {h_params_path}")
                    data = np.load(h_params_path)
                    H_tgt2ref = data['H_tgt2ref']
                    print("Successfully loaded precomputed H_tgt2ref.")
                else:
                    print(f"Warning: Precomputed homography file not found: {h_params_path}")
                    H_tgt2ref = None
            except Exception as e:
                print(f"Error loading precomputed homography: {e}")
                H_tgt2ref = None
        
        # Only compute on-the-fly if explicitly requested AND no homography was found
        if H_tgt2ref is None:
            if compute_on_the_fly:
                print(f"Computing homography using {feature_type}/{matcher_type} on-the-fly...")
                H_tgt2ref, success, timed_out = estimate_homography_enhanced(
                    ref, tgt, 
                    feature_type=feature_type, 
                    matcher_type=matcher_type,
                    debug_mode=debug_mode,
                    timeout_seconds=timeout_seconds
                )
                if not success:
                    if timed_out:
                        print("Homography estimation timed out. Skipping this pair.")
                        return None, None, None, None, None, None, None, None, None, timed_out, oversized_canvas
                    else:
                        print("Runtime homography estimation failed. Using identity matrix.")
                        H_tgt2ref = np.eye(3)
            else:
                print("No homography available and on-the-fly computation disabled. Using identity matrix.")
                H_tgt2ref = np.eye(3)

    # Determine canvas size based on warped dimensions
    corners_tgt = np.float32([
        [0, 0], [w_tgt-1, 0], [w_tgt-1, h_tgt-1], [0, h_tgt-1]
    ]).reshape(-1, 1, 2)
    
    # Transform target corners using homography
    corners_tgt_warped = cv2.perspectiveTransform(corners_tgt, H_tgt2ref)
    
    # Find extreme points
    x_coords = corners_tgt_warped[:, 0, 0]
    y_coords = corners_tgt_warped[:, 0, 1]
    
    # Include reference image dimensions
    w_min = min(0, np.min(x_coords))
    w_max = max(w_ref-1, np.max(x_coords))
    h_min = min(0, np.min(y_coords))
    h_max = max(h_ref-1, np.max(y_coords))
    
    # Version with ceil only (no +1)
    img_h = int(np.ceil(h_max - h_min))
    img_w = int(np.ceil(w_max - w_min))
    sizes = (img_h, img_w)
    
    # Check if canvas size is too large compared to original dimensions
    h_ratio = img_h / max(h_ref, h_tgt)
    w_ratio = img_w / max(w_ref, w_tgt)
    
    if h_ratio > max_canvas_ratio or w_ratio > max_canvas_ratio:
        print(f"Canvas size ({img_h}x{img_w}) exceeds {max_canvas_ratio}x of original dimensions.")
        print(f"Height ratio: {h_ratio:.2f}, Width ratio: {w_ratio:.2f}")
        oversized_canvas = True
        return None, None, None, None, None, None, None, None, None, timed_out, oversized_canvas
    
    print(f"Canvas size will be {img_h}x{img_w} (ratios: h={h_ratio:.2f}, w={w_ratio:.2f})")
    
    # Translation matrix to shift points to be positive
    T = np.array([
        [1, 0, -w_min],
        [0, 1, -h_min],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Convert matrices to PyTorch tensors
    H_tgt2ref_torch = torch.from_numpy(H_tgt2ref).double().cuda()
    T_torch = torch.from_numpy(T).double().cuda()
    
    # Homography for reference image (just translation)
    eye = torch.eye(3).double().cuda()
    H_ref = T_torch @ eye
    
    # Homography for target image (warp + translation)
    H_tgt = T_torch @ H_tgt2ref_torch

    # Create coordinate grids for sampling
    coord = utils.to_pixel_samples(None, sizes=sizes).cuda()
    cell = utils.make_cell(coord, None, sizes=sizes).cuda()

    # For target image: map stitched grid to target image coordinates
    coord1 = coord.clone()
    tgt_grid, tgt_mask = utils.gridy2gridx_homography(
        coord1.contiguous(), *sizes, h_tgt, w_tgt, H_tgt, cpu=False
    )

    cell1 = cell.clone()
    tgt_cell = utils.celly2cellx_homography(
        cell1.contiguous(), *sizes, h_tgt, w_tgt, H_tgt, cpu=False
    ).unsqueeze(0).repeat(b, 1, 1)

    # For reference image: map stitched grid to reference image coordinates
    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, h_ref, w_ref, H_ref, cpu=False
    )

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, h_ref, w_ref, H_ref, cpu=False
    ).unsqueeze(0).repeat(b, 1, 1)

    # Create stitching grid and mask
    stit_grid = utils.to_pixel_samples(None, sizes=sizes).cuda()
    stit_mask = (tgt_mask + ref_mask).clamp(0, 1)

    # Expand dimensions for batch processing
    ref_grid = ref_grid.unsqueeze(0).repeat(b, 1, 1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b, 1, 1)
    stit_grid = stit_grid.unsqueeze(0).repeat(b, 1, 1)

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, sizes, timed_out, oversized_canvas


def prepare_validation(config):
    # Load the blending model - same as original
    model_path = os.path.join(os.path.dirname(__file__), "pretrained", "NIS_blending.pth")
    sv_file = torch.load(model_path, map_location='cpu')
    model = models.make(sv_file['model'], load_sd=True).cuda()
    return model


def stitch_images(model, ref_tensor, tgt_tensor, config, args):
    """
    Main stitching function with timeout monitoring.
    """
    # Start timer for overall processing
    process_start_time = time.time()
    
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Get the dimensions
    b, c, h_ref, w_ref = ref_tensor.shape
    _, _, h_tgt, w_tgt = tgt_tensor.shape

    # Ensure the same size if different
    if ref_tensor.shape[-2:] != tgt_tensor.shape[-2:]:
        tgt_tensor = F.interpolate(tgt_tensor, size=(h_ref, w_ref), mode='bilinear')

    # Free memory before heavy operations
    torch.cuda.empty_cache()
    gc.collect()

    # Check for process timeout
    def check_timeout():
        current_time = time.time()
        elapsed_time = current_time - process_start_time
        if elapsed_time > args.process_timeout:
            print(f"Warning: Process timeout after {elapsed_time:.2f}s (limit: {args.process_timeout}s)")
            return True
        return False
    
    # Get precomputed homography if available
    H_tgt2ref = None
    if args.homography_path and os.path.exists(args.homography_path) and not args.compute_on_the_fly:
        try:
            H_tgt2ref = np.loadtxt(args.homography_path)
            print(f"✓ Successfully loaded IHN homography from {args.homography_path}")
            print(f"  Homography matrix shape: {H_tgt2ref.shape}")
            # Validate that it's a proper 3x3 homography matrix
            if H_tgt2ref.shape != (3, 3):
                print(f"Error: Invalid homography matrix shape {H_tgt2ref.shape}, expected (3, 3)")
                H_tgt2ref = None
            else:
                print("  Using pre-computed IHN homography - no SIFT+FLANN computation needed")
        except Exception as e:
            print(f"Error loading homography from {args.homography_path}: {e}")
            H_tgt2ref = None

    # Prepare ingredients using feature matching homography
    result = prepare_ingredients(
        tgt_tensor, ref_tensor, H_tgt2ref,
        load_preprocessed=args.sift_preprocessed_dir is not None,
        pair_idx=args.pair_idx, 
        preprocessed_dir=args.sift_preprocessed_dir,
        compute_on_the_fly=args.compute_on_the_fly,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        debug_mode=args.debug_mode,
        timeout_seconds=args.timeout_seconds,
        max_canvas_ratio=args.max_canvas_ratio
    )
    
    # Check if processing should be skipped or if timeout occurred
    if check_timeout() or result is None or len(result) < 9:
        print("Skipping this image pair due to errors in preparation or timeout.")
        # Create a black image to indicate skip
        if args.create_skip_image:
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image at {args.out}")
        return None
    
    tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, sizes, timed_out, oversized_canvas = result
    
    # Check if processing should be skipped due to timeout or oversized canvas
    if timed_out or oversized_canvas:
        print(f"Skipping this image pair. Timed out: {timed_out}, Oversized canvas: {oversized_canvas}")
        # Create a black image to indicate skip
        if args.create_skip_image:
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image at {args.out}")
        return None

    # Check for timeout again before heavy GPU operations
    if check_timeout():
        print("Skipping this image pair due to timeout after homography preparation.")
        if args.create_skip_image:
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image at {args.out}")
        return None

    # Free memory again after homography computation
    torch.cuda.empty_cache()
    gc.collect()

    # Normalize exactly like original
    ref_normalized = (ref_tensor - 0.5) * 2
    tgt_normalized = (tgt_tensor - 0.5) * 2

    # Reshape masks
    ref_mask = ref_mask.reshape(b, 1, *sizes)
    tgt_mask = tgt_mask.reshape(b, 1, *sizes)

    try:
        # Use neural blending without seam cut
        pred = batched_predict(
            model, ref_normalized, ref_grid, ref_cell, ref_mask,
            tgt_normalized, tgt_grid, tgt_cell, tgt_mask,
            stit_grid, sizes, config['eval_bsize']
        )
        
        # Check for timeout after GPU-intensive operations
        if check_timeout():
            print("Timeout after neural blending. Discarding results.")
            if args.create_skip_image:
                skip_img = Image.new('RGB', (100, 100), color='black')
                skip_img.save(args.out)
                print(f"Created placeholder skip image at {args.out}")
            return None
            
        # Post-process exactly as in original
        pred = pred.permute(0, 2, 1).reshape(b, c, *sizes)
        pred = ((pred + 1) / 2).clamp(0, 1) * stit_mask.reshape(b, 1, *sizes)

        # Save the result
        transforms.ToPILImage()(pred[0].cpu()).save(args.out)
        print(f"Stitched image saved to {args.out}")

        # Optionally save a visualization of the masks
        if args.save_masks:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            
            # Save masks
            ref_mask_vis = (ref_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            tgt_mask_vis = (tgt_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            stit_mask_vis = (stit_mask.reshape(b, 1, *sizes)[0, 0].cpu().numpy() * 255).astype(np.uint8)
            
            mask_out_path = args.out.replace('.png', '_masks.png')
            mask_vis = np.zeros((sizes[0], sizes[1]*3), dtype=np.uint8)
            mask_vis[:, :sizes[1]] = ref_mask_vis
            mask_vis[:, sizes[1]:sizes[1]*2] = tgt_mask_vis
            mask_vis[:, sizes[1]*2:] = stit_mask_vis
            
            Image.fromarray(mask_vis).save(mask_out_path)
            print(f"Mask visualization saved to {mask_out_path}")
        
        # Log total processing time
        process_time = time.time() - process_start_time
        print(f"Total stitching completed in {process_time:.2f}s")
        
        return pred
        
    except Exception as e:
        print(f"Error during stitching process: {e}")
        if args.create_skip_image:
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image due to error at {args.out}")
        return None


def main(config_, args):
    global config
    config = config_
    
    # Record start time for total processing
    start_time = time.time()
    
    # Check for sufficient GPU memory
    if args.check_gpu_memory:
        if not check_gpu_memory(args.min_gpu_memory):
            print(f"Insufficient GPU memory, minimum {args.min_gpu_memory}MB required")
            if args.create_skip_image:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                skip_img = Image.new('RGB', (100, 100), color='black')
                skip_img.save(args.out)
                print(f"Created placeholder skip image at {args.out}")
            return 1
    
    try:
        # Load only blending model
        model = prepare_validation(config)

        # Load input images
        ref_img_pil = Image.open(args.ref).convert('RGB')
        tgt_img_pil = Image.open(args.tgt).convert('RGB')
        
        # Scale if needed
        if args.scale != 1.0:
            print(f"Scaling input images by factor {args.scale}")
            ref_width, ref_height = ref_img_pil.size
            tgt_width, tgt_height = tgt_img_pil.size
            
            new_ref_width = int(ref_width * args.scale)
            new_ref_height = int(ref_height * args.scale)
            new_tgt_width = int(tgt_width * args.scale)
            new_tgt_height = int(tgt_height * args.scale)
            
            ref_img_pil = ref_img_pil.resize((new_ref_width, new_ref_height), Image.LANCZOS)
            tgt_img_pil = tgt_img_pil.resize((new_tgt_width, new_tgt_height), Image.LANCZOS)

        # Convert to tensors and move to GPU
        to_tensor = transforms.ToTensor()
        ref_tensor = to_tensor(ref_img_pil).cuda().unsqueeze(0)
        tgt_tensor = to_tensor(tgt_img_pil).cuda().unsqueeze(0)

        # Run inference with timeout monitoring
        with torch.no_grad():
            result = stitch_images(model, ref_tensor, tgt_tensor, config, args)
            success = result is not None
            
        # Log total processing time
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        if args.create_skip_image:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image due to error at {args.out}")
        return 1
        
    finally:
        # Perform cleanup regardless of success/failure
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='NIS/configs/test/NIS_blending.yaml', help='Path to config file')
    parser.add_argument('--gpu', default='0', help='GPU ID to use')
    parser.add_argument('--ref', default='left.jpg', help='Path to reference image')
    parser.add_argument('--tgt', default='right.jpg', help='Path to target image')
    parser.add_argument('--out', default='no_seam_results/result.png', help='Path to save output stitched image')
    parser.add_argument('--scale', type=float, default=1.0, 
                        help='Scale factor for input images (e.g., 0.5 for half resolution)')
    parser.add_argument('--sift_preprocessed_dir', type=str, default=None,
                        help='Path to the root directory of preprocessed SIFT data')
    parser.add_argument('--pair_idx', type=int, default=None,
                        help='1-based index of the image pair for loading precomputed SIFT data')
    parser.add_argument('--homography_path', type=str, default=None,
                        help='Path to a file containing a pre-computed homography matrix')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save visualization of the masks used in stitching')
    parser.add_argument('--compute_on_the_fly', action='store_true', default=False,
                        help='Compute feature matching homography on-the-fly instead of using precomputed data')
    parser.add_argument('--feature_type', type=str, default='sift', choices=['sift', 'orb', 'akaze', 'brisk'],
                        help='Type of feature detector/descriptor to use')
    parser.add_argument('--matcher_type', type=str, default='flann', choices=['bf', 'flann'],
                        help='Type of feature matcher to use')
    parser.add_argument('--debug_mode', action='store_true',
                        help='Save debug visualizations of feature matching process')
    parser.add_argument('--timeout_seconds', type=int, default=30,
                        help='Maximum time (in seconds) allowed for homography estimation before skipping')
    parser.add_argument('--process_timeout', type=int, default=30,
                        help='Maximum time (in seconds) allowed for the entire stitching process')
    parser.add_argument('--max_canvas_ratio', type=float, default=1.8,
                        help='Maximum allowed ratio between canvas size and original image dimensions')
    parser.add_argument('--create_skip_image', action='store_true',
                        help='Create a small black placeholder image when skipping')
    parser.add_argument('--check_gpu_memory', action='store_true',
                        help='Check if sufficient GPU memory is available before processing')
    parser.add_argument('--min_gpu_memory', type=int, default=4000,
                        help='Minimum required GPU memory in MB')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Warning: Config file {args.config} not found. Using default eval_bsize.")
        config_data = {'eval_bsize': 16384}
    else:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            if 'eval' in config_data and 'batch_size' in config_data['eval']:
                config_data['eval_bsize'] = config_data['eval']['batch_size']
            else:
                config_data['eval_bsize'] = 16384
                print("Using default eval_bsize of 16384")

    # Memory management
    torch.cuda.empty_cache()
    gc.collect()

    exit_code = main(config_data, args)
    sys.exit(exit_code) 
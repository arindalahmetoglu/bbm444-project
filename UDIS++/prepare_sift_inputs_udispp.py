import cv2
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
import glob
import argparse

# --- SIFT + RANSAC Homography Estimation (User Provided) ---
def estimate_homography_sift_ransac(ref_tensor, tgt_tensor, min_match_count=10, lowe_ratio=0.75):
    ref_np_hwc_rgb = ref_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tgt_np_hwc_rgb = tgt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ref_np_uint8 = (ref_np_hwc_rgb * 255).astype(np.uint8)
    tgt_np_uint8 = (tgt_np_hwc_rgb * 255).astype(np.uint8)
    ref_gray = cv2.cvtColor(ref_np_uint8, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(tgt_np_uint8, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    kp_tgt, des_tgt = sift.detectAndCompute(tgt_gray, None)
    if des_ref is None or des_tgt is None:
        print("Warning: SIFT descriptors not found.")
        return np.eye(3), False
    bf = cv2.BFMatcher()
    matches = []
    if des_tgt is not None and des_ref is not None:
        raw_matches = bf.knnMatch(des_tgt, des_ref, k=2)
        if raw_matches:
            for m_list in raw_matches:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.distance < lowe_ratio * n.distance:
                        matches.append(m)
                elif len(m_list) == 1:
                    matches.append(m_list[0])
    if len(matches) >= min_match_count:
        dst_pts = np.float32([kp_tgt[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H_tgt2ref, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H_tgt2ref is None:
            print("Warning: RANSAC failed. Returning identity.")
            return np.eye(3), False
        print("SIFT + RANSAC Homography estimated.")
        return H_tgt2ref, True
    else:
        print(f"Warning: Not enough matches - {len(matches)}/{min_match_count}. Returning identity.")
        return np.eye(3), False

# --- Canvas Size Calculation (User Provided) ---
def get_stitched_canvas_size(h_ref, w_ref, h_tgt, w_tgt, H_tgt2ref):
    corners_ref = np.float32([[0,0], [w_ref-1,0], [w_ref-1,h_ref-1], [0,h_ref-1]]).reshape(-1,1,2)
    corners_tgt = np.float32([[0,0], [w_tgt-1,0], [w_tgt-1,h_tgt-1], [0,h_tgt-1]]).reshape(-1,1,2)
    if H_tgt2ref is not None:
        corners_tgt_warped = cv2.perspectiveTransform(corners_tgt, H_tgt2ref)
    else:
        corners_tgt_warped = corners_tgt.copy()
    all_corners = np.concatenate((corners_ref, corners_tgt_warped), axis=0)
    x_coords, y_coords = all_corners[:,:,0], all_corners[:,:,1]
    w_min, w_max = np.min(x_coords), np.max(x_coords)
    h_min, h_max = np.min(y_coords), np.max(y_coords)
    out_w = int(np.ceil(w_max - w_min))
    out_h = int(np.ceil(h_max - h_min))
    T_canvas = np.array([[1,0,-w_min], [0,1,-h_min], [0,0,1]], dtype=np.float64)
    return out_h, out_w, T_canvas

# --- Image Loading & Saving Helpers ---
def load_image_as_tensor_and_numpy(image_path):
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_tensor_chw_float = TF.to_tensor(pil_img)
        img_tensor_bchw = img_tensor_chw_float.unsqueeze(0)
        img_np_hwc_rgb_uint8 = np.array(pil_img)
        return img_tensor_bchw, img_np_hwc_rgb_uint8
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

def create_and_save_output(img_np, H_transform, canvas_wh, output_path, is_mask=False):
    canvas_w, canvas_h = canvas_wh
    if img_np is None:
        print(f"Skipping save for {output_path} (input None).")
        # For masks, save a black image if original is None to avoid dataloader issues
        if is_mask and canvas_w > 0 and canvas_h > 0:
            dummy_img = np.zeros((canvas_h, canvas_w, 1), dtype=np.uint8) # Single channel for mask
            cv2.imwrite(output_path, dummy_img)
            print(f"Saved dummy black mask: {output_path}")
        return
    
    if canvas_w <= 0 or canvas_h <= 0:
        print(f"Error: Invalid canvas size ({canvas_w}x{canvas_h}) for {output_path}. Skipping save.")
        return

    if is_mask:
        mask_orig_gray = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255
        # Warp the single channel mask
        warped_mask_gray = cv2.warpPerspective(mask_orig_gray, H_transform, (canvas_w, canvas_h), borderValue=0)
        output_img_to_save = warped_mask_gray # Save as single channel grayscale
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        warped_img_bgr = cv2.warpPerspective(img_bgr, H_transform, (canvas_w, canvas_h), borderValue=(0,0,0))
        output_img_to_save = warped_img_bgr
    
    try:
        cv2.imwrite(output_path, output_img_to_save)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}. Image shape: {output_img_to_save.shape}, dtype: {output_img_to_save.dtype}")

# --- Main Processing Logic ---
def process_image_pair(ref_img_path, tgt_img_path, output_dir_base, pair_file_idx):
    print(f"Processing pair {pair_file_idx}: Ref: {ref_img_path}, Tgt: {tgt_img_path}")
    ref_tensor, ref_np = load_image_as_tensor_and_numpy(ref_img_path)
    tgt_tensor, tgt_np = load_image_as_tensor_and_numpy(tgt_img_path)

    base_name = f"{pair_file_idx:06d}.jpg"
    # Paths for saving
    path_warp1 = os.path.join(output_dir_base, "warp1", base_name)
    path_warp2 = os.path.join(output_dir_base, "warp2", base_name)
    path_mask1 = os.path.join(output_dir_base, "mask1", base_name)
    path_mask2 = os.path.join(output_dir_base, "mask2", base_name)

    if ref_tensor is None or tgt_tensor is None:
        print(f"Skipping pair {pair_file_idx} (image loading error).")
        # Save dummy black images for all outputs if loading fails
        dummy_canvas_wh = (128, 128); dummy_H = np.eye(3)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, path_warp1, False)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, path_warp2, False)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, path_mask1, True)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, path_mask2, True)
        return

    h_ref, w_ref, _ = ref_np.shape
    h_tgt, w_tgt, _ = tgt_np.shape
    H_tgt2ref, success = estimate_homography_sift_ransac(ref_tensor, tgt_tensor)
    if not success:
        print(f"Homography failed for {pair_file_idx}. Using identity.")
        H_tgt2ref = np.eye(3)
    
    canvas_h, canvas_w, T_canvas = get_stitched_canvas_size(h_ref, w_ref, h_tgt, w_tgt, H_tgt2ref)
    if canvas_w <=0 or canvas_h <=0:
        print(f"Invalid canvas for {pair_file_idx} ({canvas_w}x{canvas_h}). Using defaults.")
        canvas_w, canvas_h = max(256,w_ref,w_tgt), max(256,h_ref,h_tgt) # ensure it's somewhat sensible
        # If canvas is bad, just use identity warps into this default canvas
        # This requires images to be placed at origin, ensure T_canvas is identity if this path is taken
        # Or, more simply, warp with identity to original image size if T_canvas logic fails.
        # For simplicity with SIFT, if get_stitched_canvas_size fails, we might have bigger issues.
        # Let's assume identity warp to original image size and save them separately for now.
        # This part needs careful consideration for robust fallback.
        # Fallback to identity warping into their own original sizes if canvas calculation is problematic.
        T_canvas_ref = np.eye(3)
        T_canvas_tgt = np.eye(3)
        canvas_w_ref, canvas_h_ref = w_ref, h_ref
        canvas_w_tgt, canvas_h_tgt = w_tgt, h_tgt
        H_ref2canvas = T_canvas_ref
        H_tgt2canvas = T_canvas_tgt # H_tgt2ref is np.eye(3) if homography failed.
                                       # If homography succeeded but canvas failed, this might be wrong.
                                       # Let's simplify: if canvas is bad, output original images + blank masks.
        print(f"Fallback: Outputting original images for pair {pair_file_idx}.")
        create_and_save_output(ref_np, np.eye(3), (w_ref, h_ref), path_warp1, False)
        create_and_save_output(tgt_np, np.eye(3), (w_tgt, h_tgt), path_warp2, False)
        create_and_save_output(ref_np, np.eye(3), (w_ref, h_ref), path_mask1, True) # Will be full white mask of ref
        create_and_save_output(tgt_np, np.eye(3), (w_tgt, h_tgt), path_mask2, True) # Will be full white mask of tgt
        return
    else:
        H_ref2canvas = T_canvas
        H_tgt2canvas = T_canvas @ H_tgt2ref

    print(f"Canvas for pair {pair_file_idx}: {canvas_w}x{canvas_h}")
    create_and_save_output(ref_np, H_ref2canvas, (canvas_w, canvas_h), path_warp1, False)
    create_and_save_output(tgt_np, H_tgt2canvas, (canvas_w, canvas_h), path_warp2, False)
    create_and_save_output(ref_np, H_ref2canvas, (canvas_w, canvas_h), path_mask1, True)
    create_and_save_output(tgt_np, H_tgt2canvas, (canvas_w, canvas_h), path_mask2, True)

def main():
    parser = argparse.ArgumentParser(description="Prepare SIFT+RANSAC aligned images for UDIS++ Composition module.")
    parser.add_argument("--input_dir", type=str, default="raw_image_pairs/",
                        help="Dir with raw images. Expects pairs (e.g., img1_ref.jpg, img1_tgt.jpg or 01.jpg, 02.jpg).")
    parser.add_argument("--output_dir", type=str, default="sift_aligned_data_for_composition/",
                        help="Dir to save warp1, warp2, mask1, mask2.")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR_BASE = args.output_dir

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}"); return
    for sub in ["warp1", "warp2", "mask1", "mask2"]:
        os.makedirs(os.path.join(OUTPUT_DIR_BASE, sub), exist_ok=True)

    all_img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.[jp][pn]g")) +
                           glob.glob(os.path.join(INPUT_DIR, "*.[JP][PN]G")))
    if not all_img_paths:
        print(f"Error: No images found in {INPUT_DIR}"); return
    if len(all_img_paths) % 2 != 0:
        print(f"Warning: Odd number of images ({len(all_img_paths)}). Last image ignored.")
        all_img_paths = all_img_paths[:-1]

    num_pairs = len(all_img_paths) // 2
    print(f"Found {len(all_img_paths)} images, forming {num_pairs} pairs.")
    pair_idx = 1
    for i in range(0, len(all_img_paths), 2):
        ref_path, tgt_path = all_img_paths[i], all_img_paths[i+1]
        process_image_pair(ref_path, tgt_path, OUTPUT_DIR_BASE, pair_idx)
        pair_idx += 1

    abs_out_dir = os.path.abspath(OUTPUT_DIR_BASE)
    print("\n--- Processing Complete ---")
    print(f"Generated SIFT-aligned data in: {abs_out_dir}")
    print("\nNext Steps for UDIS++ Composition:")
    print(f"1. Ensure '{abs_out_dir}' contains 'warp1', 'warp2', 'mask1', 'mask2' subdirs.")
    print(f"2. In UDIS++/Composition/Codes/test.py (or test_other.py), update the data path to point to '{abs_out_dir}'.")
    print(f"   (Look for arguments like 'data_path', 'test_path', or hardcoded paths for loading test data)")
    print(f"3. Run the UDIS++/Composition test script, e.g., python UDIS++/Composition/Codes/test.py")

if __name__ == "__main__":
    main() 
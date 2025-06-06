import cv2
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
import glob
import argparse

# --- User-provided SIFT + RANSAC Homography Estimation ---
def estimate_homography_sift_ransac(ref_tensor, tgt_tensor, min_match_count=10, lowe_ratio=0.75):
    '''
    Estimates homography using SIFT and RANSAC.
    ref_tensor: Reference image tensor (B, C, H, W, 0-1 float, RGB).
    tgt_tensor: Target image tensor (B, C, H, W, 0-1 float, RGB).
    Returns H_tgt2ref (3x3 NumPy array) and success_flag (boolean).
    '''
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
        print("Warning: SIFT descriptors not found for one or both images.")
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

        H_tgt2ref, ransac_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H_tgt2ref is None:
            print("Warning: RANSAC failed to find homography. Returning identity.")
            return np.eye(3), False
        
        print("SIFT + RANSAC Homography estimated successfully.")
        return H_tgt2ref, True
    else:
        print(f"Warning: Not enough good matches found - {len(matches)}/{min_match_count}. Returning identity.")
        return np.eye(3), False

# --- User-provided Canvas Size Calculation ---
def get_stitched_canvas_size(h_ref, w_ref, h_tgt, w_tgt, H_tgt2ref):
    '''
    Calculates the dimensions of the output canvas needed to contain both
    the reference image and the warped target image.
    H_tgt2ref: Homography that maps points from target to reference coordinate system.
    '''
    corners_ref = np.float32([
        [0, 0], [w_ref - 1, 0], [w_ref - 1, h_ref - 1], [0, h_ref - 1]
    ]).reshape(-1, 1, 2)

    corners_tgt = np.float32([
        [0, 0], [w_tgt - 1, 0], [w_tgt - 1, h_tgt - 1], [0, h_tgt - 1]
    ]).reshape(-1, 1, 2)

    if H_tgt2ref is not None:
        corners_tgt_warped = cv2.perspectiveTransform(corners_tgt, H_tgt2ref)
    else: 
        corners_tgt_warped = corners_tgt.copy()

    all_corners = np.concatenate((corners_ref, corners_tgt_warped), axis=0)

    x_coords = all_corners[:, :, 0]
    y_coords = all_corners[:, :, 1]
    
    w_min, w_max = np.min(x_coords), np.max(x_coords)
    h_min, h_max = np.min(y_coords), np.max(y_coords)

    out_w = int(np.ceil(w_max - w_min))
    out_h = int(np.ceil(h_max - h_min))
    
    T_canvas = np.array([[1, 0, -w_min],
                         [0, 1, -h_min],
                         [0, 0, 1]], dtype=np.float64)

    return out_h, out_w, T_canvas

# --- Image Loading and Helper Functions ---
def load_image_as_tensor_and_numpy(image_path):
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_tensor_chw_float = TF.to_tensor(pil_img) 
        img_tensor_bchw = img_tensor_chw_float.unsqueeze(0)
        img_np_hwc_rgb_uint8 = np.array(pil_img) 
        return img_tensor_bchw, img_np_hwc_rgb_uint8
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def create_and_save_output(img_np, H_transform, canvas_wh, output_path, is_mask=False):
    canvas_w, canvas_h = canvas_wh
    
    if img_np is None:
        print(f"Skipping save for {output_path} as input image was None.")
        # Create a dummy black image if path is for mask and image is None, so dataloader doesn't break
        if is_mask:
            dummy_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            cv2.imwrite(output_path, dummy_img)
            print(f"Saved dummy black mask: {output_path}")
        return

    if is_mask:
        mask_orig_gray = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255
        warped_img_gray = cv2.warpPerspective(mask_orig_gray, H_transform, (canvas_w, canvas_h), borderValue=0)
        output_img_bgr = cv2.cvtColor(warped_img_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        warped_img_bgr = cv2.warpPerspective(img_bgr, H_transform, (canvas_w, canvas_h), borderValue=(0,0,0))
        output_img_bgr = warped_img_bgr

    cv2.imwrite(output_path, output_img_bgr)
    print(f"Saved: {output_path}")

# --- Main Processing ---
def process_image_pair(ref_img_path, tgt_img_path, output_dir_base, pair_file_idx):
    print(f"Processing pair {pair_file_idx}: Ref: {ref_img_path}, Tgt: {tgt_img_path}")

    ref_tensor, ref_np_uint8 = load_image_as_tensor_and_numpy(ref_img_path)
    tgt_tensor, tgt_np_uint8 = load_image_as_tensor_and_numpy(tgt_img_path)

    if ref_tensor is None or tgt_tensor is None:
        print(f"Skipping pair {pair_file_idx} due to image loading error.")
        # Save dummy black images so the ImageReconstruction dataloader doesn't fail if it expects files
        base_name = f"{pair_file_idx:06d}.jpg"
        dummy_canvas_wh = (128,128) # Arbitrary small size for dummy
        dummy_H = np.eye(3)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, os.path.join(output_dir_base, "warp1", base_name), is_mask=False)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, os.path.join(output_dir_base, "warp2", base_name), is_mask=False)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, os.path.join(output_dir_base, "mask1", base_name), is_mask=True)
        create_and_save_output(None, dummy_H, dummy_canvas_wh, os.path.join(output_dir_base, "mask2", base_name), is_mask=True)
        return

    h_ref, w_ref, _ = ref_np_uint8.shape
    h_tgt, w_tgt, _ = tgt_np_uint8.shape

    H_tgt2ref, success = estimate_homography_sift_ransac(ref_tensor, tgt_tensor)
    if not success:
        print(f"Homography estimation failed for pair {pair_file_idx}. Using identity matrix.")
        H_tgt2ref = np.eye(3)

    canvas_h, canvas_w, T_canvas = get_stitched_canvas_size(h_ref, w_ref, h_tgt, w_tgt, H_tgt2ref)
    if canvas_w <=0 or canvas_h <=0:
        print(f"Invalid canvas size ({canvas_w}x{canvas_h}) for pair {pair_file_idx}. Using default 256x256 and identity warps.")
        canvas_w, canvas_h = 256, 256
        H_ref2canvas = np.eye(3)
        H_tgt2canvas = np.eye(3)
    else:
        H_ref2canvas = T_canvas 
        H_tgt2canvas = T_canvas @ H_tgt2ref
    
    print(f"Canvas size for pair {pair_file_idx}: {canvas_w}x{canvas_h}")

    base_name = f"{pair_file_idx:06d}.jpg"
    path_warp1 = os.path.join(output_dir_base, "warp1", base_name)
    path_warp2 = os.path.join(output_dir_base, "warp2", base_name)
    path_mask1 = os.path.join(output_dir_base, "mask1", base_name)
    path_mask2 = os.path.join(output_dir_base, "mask2", base_name)

    create_and_save_output(ref_np_uint8, H_ref2canvas, (canvas_w, canvas_h), path_warp1, is_mask=False)
    create_and_save_output(tgt_np_uint8, H_tgt2canvas, (canvas_w, canvas_h), path_warp2, is_mask=False)
    create_and_save_output(ref_np_uint8, H_ref2canvas, (canvas_w, canvas_h), path_mask1, is_mask=True)
    create_and_save_output(tgt_np_uint8, H_tgt2canvas, (canvas_w, canvas_h), path_mask2, is_mask=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare SIFT+RANSAC aligned images for ImageReconstruction module.")
    parser.add_argument("--input_dir", type=str, default="test_images/", 
                        help="Directory containing test images. Images should be sortable into pairs (e.g., img1_ref, img1_tgt, img2_ref, img2_tgt or 01.jpg, 02.jpg, 03.jpg, 04.jpg).")
    parser.add_argument("--output_dir", type=str, default="sift_processed_data/", 
                        help="Directory to save the processed warp1, warp2, mask1, mask2 images.")
    args = parser.parse_args()

    INPUT_IMAGES_DIR = args.input_dir
    OUTPUT_DIR_BASE = args.output_dir

    if not os.path.isdir(INPUT_IMAGES_DIR):
        print(f"Error: Input directory not found: {INPUT_IMAGES_DIR}")
        return

    for sub_dir in ["warp1", "warp2", "mask1", "mask2"]:
        os.makedirs(os.path.join(OUTPUT_DIR_BASE, sub_dir), exist_ok=True)
    
    all_image_paths = sorted(glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.[jp][pn]g")) + \
                           glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.[JP][PN]G")))

    if not all_image_paths:
        print(f"Error: No images (jpg, png) found in {INPUT_IMAGES_DIR}")
        return
    
    if len(all_image_paths) % 2 != 0:
        print(f"Warning: Odd number of images ({len(all_image_paths)}) found in {INPUT_IMAGES_DIR}. The last image will be ignored.")
        all_image_paths = all_image_paths[:-1]

    num_pairs = len(all_image_paths) // 2
    print(f"Found {len(all_image_paths)} images, forming {num_pairs} pairs.")

    pair_file_idx = 1
    for i in range(0, len(all_image_paths), 2):
        ref_img_path = all_image_paths[i]
        tgt_img_path = all_image_paths[i+1]
        
        # Using pair_file_idx for 000001.jpg, 000002.jpg naming convention for output files
        process_image_pair(ref_img_path, tgt_img_path, OUTPUT_DIR_BASE, pair_file_idx)
        pair_file_idx += 1

    print("\n--- Processing Complete ---")
    abs_output_dir = os.path.abspath(OUTPUT_DIR_BASE)
    print(f"Generated SIFT-processed data in: {abs_output_dir}")
    print("\nNext Steps:")
    print(f"1. Ensure the directory '{abs_output_dir}' contains 'warp1', 'warp2', 'mask1', 'mask2' subdirectories with images.")
    print(f"2. Modify 'UnsupervisedDeepImageStitching/ImageReconstruction/Codes/constant.py':")
    print(f"   Set TEST_FOLDER = '{abs_output_dir}'")
    print(f"3. Navigate to 'UnsupervisedDeepImageStitching/ImageReconstruction/Codes/' and run:")
    print(f"   python inference.py")

if __name__ == "__main__":
    main() 
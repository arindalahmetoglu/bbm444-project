import os
import argparse
import numpy as np
import cv2
from PIL import Image
import sys

# Default constants for beehive dataset (can be overridden by command line args)
DEFAULT_HORIZONTAL_TRANSLATION = 490  # pixels
DEFAULT_VERTICAL_TRANSLATION = 330    # pixels
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def get_image_path(base_path, scan_idx, img_idx):
    """
    Constructs the path to an image file based on scan and image index.
    Based on the actual directory structure found:
    - Dataset has directories named horizontal_scan1, horizontal_scan2, ..., horizontal_scan16
    - Images are named img_1.jpg, img_2.jpg, ..., img_16.jpg
    """
    scan_folder_name = f"horizontal_scan{scan_idx}"  # e.g. horizontal_scan1, horizontal_scan2
    image_file_name = f"img_{img_idx}.jpg"  # e.g. img_1.jpg, img_2.jpg
    
    return os.path.join(base_path, scan_folder_name, image_file_name)


def calculate_homography_and_canvas(r1, c1, r2, c2, dx_translation, dy_translation, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    """
    Calculates the homography H_tgt2ref and canvas parameters.
    (r1, c1) are 1-based indices for the reference image.
    (r2, c2) are 1-based indices for the target image.
    dx_translation, dy_translation are the pixel translations between consecutive images.
    img_width, img_height are the scaled image dimensions.
    
    Note: Vertical order goes from bottom (scan 1) to top (scan 16).
    """
    dx_pixels = (c2 - c1) * dx_translation
    # Reverse the vertical calculation: higher scan numbers are above lower scan numbers
    # So if target scan > ref scan, target should be above (negative y)
    dy_pixels = -(r2 - r1) * dy_translation

    # Homography H_tgt2ref: transforms target image coordinates to reference image's coordinate system
    # If target is to the right (dx_pixels > 0), we need to shift it left by dx_pixels to align.
    # If target is above (dy_pixels < 0), we need to shift it down by |dy_pixels| to align.
    H_tgt2ref = np.array([
        [1, 0, -dx_pixels],
        [0, 1, -dy_pixels],
        [0, 0, 1]
    ], dtype=np.float64)

    # For the canvas layout: reference image on LEFT, target image on RIGHT
    # Reference image will be at canvas origin (0,0)
    # Target image will be at its natural offset position (dx_pixels, dy_pixels)
    
    # Reference image corners (at canvas origin)
    ref_corners = np.array([
        [0, 0],  # top-left
        [img_width - 1, 0],  # top-right
        [img_width - 1, img_height - 1],  # bottom-right
        [0, img_height - 1]  # bottom-left
    ])
    
    # Target image corners at their natural offset position
    target_corners = np.array([
        [dx_pixels, dy_pixels],  # top-left
        [img_width - 1 + dx_pixels, dy_pixels],  # top-right  
        [img_width - 1 + dx_pixels, img_height - 1 + dy_pixels],  # bottom-right
        [dx_pixels, img_height - 1 + dy_pixels]  # bottom-left
    ])
    
    # Find the bounding box of both images
    all_corners = np.vstack([ref_corners, target_corners])
    w_min = np.min(all_corners[:, 0])
    w_max = np.max(all_corners[:, 0])
    h_min = np.min(all_corners[:, 1])
    h_max = np.max(all_corners[:, 1])
    
    # Canvas size
    canvas_width = int(np.ceil(w_max - w_min)) + 1
    canvas_height = int(np.ceil(h_max - h_min)) + 1

    # Translation matrix T to shift the origin to (0,0) of the canvas
    T = np.array([
        [1, 0, -w_min],
        [0, 1, -h_min],
        [0, 0, 1]
    ], dtype=np.float64)

    # Homography for reference image on canvas: just apply the translation
    H_ref_on_canvas = T.copy()

    # Homography for target image on canvas: translate to offset position, then apply canvas translation
    H_tgt_offset = np.array([
        [1, 0, dx_pixels],
        [0, 1, dy_pixels],
        [0, 0, 1]
    ], dtype=np.float64)
    
    H_tgt_on_canvas = T @ H_tgt_offset

    return H_tgt2ref, H_ref_on_canvas, H_tgt_on_canvas, canvas_width, canvas_height

def create_masks(H_ref_on_canvas, H_tgt_on_canvas, canvas_width, canvas_height, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    """
    Creates simple rectangular masks for the warped images on the canvas.
    """
    ref_mask_np = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    tgt_mask_np = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # For reference image mask
    # Find the bounding box of the reference image on the canvas
    ref_corners_orig = np.float32([
        [0, 0], [img_width - 1, 0], [img_width - 1, img_height - 1], [0, img_height - 1]
    ]).reshape(-1, 1, 2)
    ref_corners_warped = cv2.perspectiveTransform(ref_corners_orig, H_ref_on_canvas)

    # Fill the rectangle for ref_mask. Using min/max of warped corners.
    # Add 0.5 before casting to int for proper rounding to nearest pixel for region filling.
    x_min_ref = int(np.min(ref_corners_warped[:,0,0]) + 0.5)
    x_max_ref = int(np.max(ref_corners_warped[:,0,0]) + 0.5)
    y_min_ref = int(np.min(ref_corners_warped[:,0,1]) + 0.5)
    y_max_ref = int(np.max(ref_corners_warped[:,0,1]) + 0.5)
    ref_mask_np[y_min_ref:y_max_ref+1, x_min_ref:x_max_ref+1] = 1.0


    # For target image mask
    tgt_corners_orig = np.float32([
        [0, 0], [img_width - 1, 0], [img_width - 1, img_height - 1], [0, img_height - 1]
    ]).reshape(-1, 1, 2)
    tgt_corners_warped = cv2.perspectiveTransform(tgt_corners_orig, H_tgt_on_canvas)

    x_min_tgt = int(np.min(tgt_corners_warped[:,0,0]) + 0.5)
    x_max_tgt = int(np.max(tgt_corners_warped[:,0,0]) + 0.5)
    y_min_tgt = int(np.min(tgt_corners_warped[:,0,1]) + 0.5)
    y_max_tgt = int(np.max(tgt_corners_warped[:,0,1]) + 0.5)
    tgt_mask_np[y_min_tgt:y_max_tgt+1, x_min_tgt:x_max_tgt+1] = 1.0
    
    # Ensure masks are clipped to canvas boundaries if necessary
    # (though with this direct calculation, they should be within)
    ref_mask_np = np.clip(ref_mask_np, 0.0, 1.0)
    tgt_mask_np = np.clip(tgt_mask_np, 0.0, 1.0)

    return ref_mask_np, tgt_mask_np

def main():
    parser = argparse.ArgumentParser(description="Generate homography, warps, and masks for the Beehive dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the Beehive dataset.")
    parser.add_argument("--ref_scan", required=True, type=int, help="Reference image scan index (1-based).")
    parser.add_argument("--ref_img", required=True, type=int, help="Reference image index within scan (1-based).")
    parser.add_argument("--tgt_scan", required=True, type=int, help="Target image scan index (1-based).")
    parser.add_argument("--tgt_img", required=True, type=int, help="Target image index within scan (1-based).")
    parser.add_argument("--output_dir_root", default=".", help="Root directory where shared_beehive_data and beehive_preprocessed_data will be created.")
    parser.add_argument("--dx", type=float, default=DEFAULT_HORIZONTAL_TRANSLATION, help=f"Horizontal translation between consecutive images in pixels (default: {DEFAULT_HORIZONTAL_TRANSLATION})")
    parser.add_argument("--dy", type=float, default=DEFAULT_VERTICAL_TRANSLATION, help=f"Vertical translation between consecutive scans in pixels (default: {DEFAULT_VERTICAL_TRANSLATION})")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for images and translations (default: 1.0)")

    args = parser.parse_args()

    # Apply scaling to image dimensions
    scaled_width = int(IMAGE_WIDTH * args.scale)
    scaled_height = int(IMAGE_HEIGHT * args.scale)

    # Automatically generate pair_id
    pair_id = f"r{args.ref_scan}c{args.ref_img}_r{args.tgt_scan}c{args.tgt_img}"
    print(f"Generated pair_id: {pair_id}")
    print(f"Using translations: dx={args.dx}, dy={args.dy}")
    print(f"Using scale factor: {args.scale} (scaled dimensions: {scaled_width}x{scaled_height})")

    # Define output directories relative to the provided root
    output_dir_root = args.output_dir_root
    shared_data_dir = os.path.join(output_dir_root, "processing_data", "shared_beehive_data")
    preprocessed_data_dir = os.path.join(output_dir_root, "processing_data", "beehive_preprocessed_data")
    
    # Create directories
    os.makedirs(os.path.join(shared_data_dir, "homography"), exist_ok=True)
    os.makedirs(os.path.join(shared_data_dir, "warps"), exist_ok=True)
    os.makedirs(os.path.join(shared_data_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_data_dir, "warp1"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_data_dir, "warp2"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_data_dir, "mask1"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_data_dir, "mask2"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_data_dir, "homography_params", pair_id), exist_ok=True)

    # Get image paths
    ref_image_path = get_image_path(args.dataset_path, args.ref_scan, args.ref_img)
    tgt_image_path = get_image_path(args.dataset_path, args.tgt_scan, args.tgt_img)

    if not os.path.exists(ref_image_path):
        print(f"Error: Reference image not found at {ref_image_path}")
        sys.exit(1)
    if not os.path.exists(tgt_image_path):
        print(f"Error: Target image not found at {tgt_image_path}")
        sys.exit(1)

    print(f"Reference image: {ref_image_path}")
    print(f"Target image: {tgt_image_path}")

    H_tgt2ref, H_ref_on_canvas, H_tgt_on_canvas, canvas_width, canvas_height = \
        calculate_homography_and_canvas(args.ref_scan, args.ref_img, args.tgt_scan, args.tgt_img, args.dx, args.dy, scaled_width, scaled_height)

    # Create masks using the calculated homographies and canvas size
    ref_mask_np, tgt_mask_np = create_masks(H_ref_on_canvas, H_tgt_on_canvas, canvas_width, canvas_height, scaled_width, scaled_height)

    # --- Save Homography for NIS (the only thing needed in shared_data) ---
    homography_txt_path = os.path.join(shared_data_dir, "homography", f"{pair_id}.txt")
    np.savetxt(homography_txt_path, H_tgt2ref, fmt='%.8f')
    print(f"Saved NIS homography to {homography_txt_path}")
    
    # --- Create and save data for UDIS/UDIS++ ---
    # Load original images to warp them
    try:
        ref_img_cv = cv2.imread(ref_image_path)
        tgt_img_cv = cv2.imread(tgt_image_path)
        if ref_img_cv is None: raise IOError(f"Could not read ref_img_cv: {ref_image_path}")
        if tgt_img_cv is None: raise IOError(f"Could not read tgt_img_cv: {tgt_image_path}")

        # Apply scaling to the images
        if args.scale != 1.0:
            ref_img_cv = cv2.resize(ref_img_cv, (scaled_width, scaled_height))
            tgt_img_cv = cv2.resize(tgt_img_cv, (scaled_width, scaled_height))
            print(f"Scaled images to {scaled_width}x{scaled_height}")

        # Warp images
        warped_ref_img = cv2.warpPerspective(ref_img_cv, H_ref_on_canvas, (canvas_width, canvas_height))
        warped_tgt_img = cv2.warpPerspective(tgt_img_cv, H_tgt_on_canvas, (canvas_width, canvas_height))

    except Exception as e:
        print(f"Error loading images for warping with OpenCV: {e}")
        sys.exit(1)

    # Warp images
    warped_ref_img = cv2.warpPerspective(ref_img_cv, H_ref_on_canvas, (canvas_width, canvas_height))
    warped_tgt_img = cv2.warpPerspective(tgt_img_cv, H_tgt_on_canvas, (canvas_width, canvas_height))

    # Convert masks to 8-bit images (0-255)
    ref_mask_img = (ref_mask_np * 255).astype(np.uint8)
    tgt_mask_img = (tgt_mask_np * 255).astype(np.uint8)

    # Save PNG warps and masks to shared directory
    ref_warp_pil = Image.fromarray(warped_ref_img)
    tgt_warp_pil = Image.fromarray(warped_tgt_img)
    ref_mask_pil = Image.fromarray(ref_mask_img)
    tgt_mask_pil = Image.fromarray(tgt_mask_img)
    ref_warp_pil.save(os.path.join(shared_data_dir, "warps", f"ref_{pair_id}.png"))
    tgt_warp_pil.save(os.path.join(shared_data_dir, "warps", f"tgt_{pair_id}.png"))
    ref_mask_pil.save(os.path.join(shared_data_dir, "masks", f"ref_{pair_id}.png"))
    tgt_mask_pil.save(os.path.join(shared_data_dir, "masks", f"tgt_{pair_id}.png"))
    print(f"Saved PNG warps and masks to {os.path.join(shared_data_dir, 'warps')} and {os.path.join(shared_data_dir, 'masks')}")

    # Save PNG warps and masks to preprocessed directory for UDIS/UDIS++
    ref_warp_pil.save(os.path.join(preprocessed_data_dir, "warp1", f"{pair_id}.png"))
    tgt_warp_pil.save(os.path.join(preprocessed_data_dir, "warp2", f"{pair_id}.png"))
    ref_mask_pil.save(os.path.join(preprocessed_data_dir, "mask1", f"{pair_id}.png"))
    tgt_mask_pil.save(os.path.join(preprocessed_data_dir, "mask2", f"{pair_id}.png"))
    print(f"Saved PNG warps and masks to {os.path.join(preprocessed_data_dir, 'warp1')}, {os.path.join(preprocessed_data_dir, 'warp2')}, {os.path.join(preprocessed_data_dir, 'mask1')}, {os.path.join(preprocessed_data_dir, 'mask2')}")
    
    # Save placeholder homography text file for UDIS
    udis_homography_path = os.path.join(preprocessed_data_dir, "homography_params", pair_id, "homography_0.txt")
    with open(udis_homography_path, "w") as f:
        f.write("placeholder")
    print(f"Saved placeholder homography to {udis_homography_path} for UDIS/UDIS++.")

    udis_h_path = os.path.join(preprocessed_data_dir, "homography_params", pair_id, "H.txt")
    np.savetxt(udis_h_path, H_tgt2ref, fmt='%.8f')
    print(f"Saved placeholder H.txt to {udis_h_path} for UDIS/UDIS++.")
    
    # Save npz file for UDIS++
    udis_plus_plus_h_params_path = os.path.join(preprocessed_data_dir, "homography_params", pair_id, "h_params.npz")
    np.savez(udis_plus_plus_h_params_path, h_matrix=H_tgt2ref)
    print(f"Saved h_params.npz to {udis_plus_plus_h_params_path} for UDIS++.")

    print(f"Successfully generated data for pair {pair_id}")

if __name__ == "__main__":
    main() 
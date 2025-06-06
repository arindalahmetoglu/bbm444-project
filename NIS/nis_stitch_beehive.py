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


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


# Global variables for seam cutting (from original)
c1 = [0, 0]; c2 = [0, 0]
finder = cv2.detail.SeamFinder.createDefault(2)


def seam_finder(ref, tgt, ref_m, tgt_m):
    """
    Seam cutting function from original NIS - performs seam cutting between overlapping regions
    """
    ref_ = ref.mean(dim=1, keepdim=True)
    ref_ -= ref_.min()
    ref_ /= ref_.max()

    tgt_ = tgt.mean(dim=1, keepdim=True)
    tgt_ -= tgt_.min()
    tgt_ /= tgt_.max()

    ref_ = (ref_.cpu().numpy() * 255).astype(np.uint8)
    tgt_ = (tgt_.cpu().numpy() * 255).astype(np.uint8)
    ref_m = (ref_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)
    tgt_m = (tgt_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)

    inp = np.concatenate([ref_, tgt_], axis=0).transpose(0,2,3,1)
    inp = np.repeat(inp, 3, -1)

    masks = np.stack([ref_m, tgt_m], axis=0)[..., None]
    corners = np.stack([c1, c2], axis=0).astype(np.uint8)

    ref_m, tgt_m = finder.find(inp, corners, masks)
    ref *= torch.Tensor(cv2.UMat.get(ref_m).reshape(1,1,*ref.shape[-2:])/255)
    tgt *= torch.Tensor(cv2.UMat.get(tgt_m).reshape(1,1,*tgt.shape[-2:])/255)

    stit_rep = ref + tgt

    return stit_rep


def batched_predict(model, ref, ref_grid, ref_cell, ref_mask,
                    tgt, tgt_grid, tgt_cell, tgt_mask,
                    stit_grid, sizes, bsize, seam_cut=False):
    """
    Batched prediction function from original NIS with seam cutting support
    """
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

    if seam_cut:
        print("Applying seam cutting...")
        fea = seam_finder(fea_ref_w, fea_tgt_w, ref_mask, tgt_mask).repeat(1,2,1,1).cuda()
    else:
        print("No seam cutting - using simple masking...")
        fea_ref_w *= ref_mask.cpu()
        fea_tgt_w *= tgt_mask.cpu()
        fea = torch.cat([fea_ref_w, fea_tgt_w], dim=1).cuda()

    stit_rep = model.gen_feat_for_blender(fea)

    ql = 0
    preds = []
    n = ref_grid.shape[1]

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(stit_rep, stit_grid[:, ql: qr, :])
        preds.append(pred)
        ql = qr

    pred = torch.cat(preds, dim=1)

    return pred


def load_beehive_homography(homography_path):
    """
    Load predetermined translational homography from beehive data.
    Returns H_tgt2ref (3x3 NumPy array) and success_flag (boolean).
    """
    try:
        if homography_path and os.path.exists(homography_path):
            H_tgt2ref = np.loadtxt(homography_path)
            print(f"Successfully loaded beehive translational homography from {homography_path}")
            print(f"H_tgt2ref:\n{H_tgt2ref}")
            return H_tgt2ref, True
        else:
            print(f"Warning: Homography file not found: {homography_path}")
            return np.eye(3), False
    except Exception as e:
        print(f"Error loading homography from {homography_path}: {e}")
        return np.eye(3), False


def prepare_ingredient_beehive(H_tgt2ref, tgt, ref, max_canvas_ratio=1.8):
    """
    Prepare ingredients for beehive stitching using predetermined homography.
    This replaces the original prepare_ingredient function that used IHN.
    Follows the same structure as the original but uses predetermined H_tgt2ref.
    """
    b, c, h, w = tgt.shape
    print("Using predetermined beehive translational homography (replacing IHN)")

    # Convert homography to tensor for processing
    H_tgt2ref_tensor = torch.from_numpy(H_tgt2ref).double()

    # Calculate canvas bounds using homography (same logic as original)
    shape = tgt.shape[-2:]
    corners_tgt = np.float32([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
    ]).reshape(-1, 1, 2)
    
    corners_tgt_warped = cv2.perspectiveTransform(corners_tgt, H_tgt2ref.astype(np.float32))
    
    x_coords = corners_tgt_warped[:, 0, 0]
    y_coords = corners_tgt_warped[:, 0, 1]
    
    # Include reference image dimensions
    w_min = min(0, np.min(x_coords))
    w_max = max(w-1, np.max(x_coords))
    h_min = min(0, np.min(y_coords))
    h_max = max(h-1, np.max(y_coords))

    img_h = int(np.ceil(h_max - h_min))
    img_w = int(np.ceil(w_max - w_min))
    sizes = (img_h, img_w)

    # Check canvas size limits
    h_ratio = img_h / max(h, h)
    w_ratio = img_w / max(w, w)
    
    if h_ratio > max_canvas_ratio or w_ratio > max_canvas_ratio:
        print(f"Canvas size ({img_h}x{img_w}) exceeds {max_canvas_ratio}x of original dimensions.")
        print(f"Height ratio: {h_ratio:.2f}, Width ratio: {w_ratio:.2f}")
        return None
    
    print(f"Canvas size will be {img_h}x{img_w} (ratios: h={h_ratio:.2f}, w={w_ratio:.2f})")

    # Create translation matrix (same as original)
    eye = torch.eye(3).double()
    T = utils.get_translation(h_min, w_min)

    H_tgt2ref_adjusted = T @ H_tgt2ref_tensor

    # Compensate reference matrix (same as original)
    eye, _, _ = transform.compensate_matrix(ref, eye)
    eye = T @ eye

    # Create coordinate grids (same as original)
    coord = utils.to_pixel_samples(None, sizes=sizes)
    cell = utils.make_cell(coord, None, sizes=sizes).cuda()
    coord = coord.cuda()

    # Target image grid mapping
    coord1 = coord.clone()
    tgt_grid, tgt_mask = utils.gridy2gridx_homography(
        coord1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref_adjusted.cuda(), cpu=False
    )

    cell1 = cell.clone()
    tgt_cell = utils.celly2cellx_homography(
        cell1.contiguous(), *sizes, *tgt.shape[-2:], H_tgt2ref_adjusted.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    # Reference image grid mapping
    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    # Stitching grid
    stit_grid = utils.to_pixel_samples(None, sizes).cuda()
    stit_mask = (tgt_mask + ref_mask).clamp(0,1)

    # Expand dimensions for batch processing
    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1)
    stit_grid = stit_grid.unsqueeze(0).repeat(b,1,1)

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, sizes


def prepare_validation(config):
    """
    Load only the blending model (no IHN needed for beehive)
    """
    model_path = os.path.join(os.path.dirname(__file__), "pretrained", "NIS_blending.pth")
    sv_file = torch.load(model_path, map_location='cpu')
    model = models.make(sv_file['model'], load_sd=True).cuda()
    
    print("Loaded NIS blending model (IHN not needed for beehive translational homography)")
    return model


def stitch_images_beehive_super_simple_fix(model, ref_tensor, tgt_tensor, config, args):
    """
    Main stitching function with SUPER SIMPLE FIX:
    Just use the images as they are without any swapping or complex logic.
    The fix is applied at the argument level by swapping the paths.
    """
    start_time = time.time()
    
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print(f"Processing with SUPER SIMPLE approach - no internal swapping")
    print(f"Ref: {args.ref}")
    print(f"Tgt: {args.tgt}")
    
    # Load predetermined beehive homography
    if not args.homography_path:
        print("Error: --homography_path is required for beehive stitching")
        return None
        
    H_tgt2ref, success = load_beehive_homography(args.homography_path)
    if not success:
        print("Error: Failed to load beehive homography")
        return None

    b, c, h, w = ref_tensor.shape

    # Ensure same size if different (from original)
    if ref_tensor.shape[-2:] != tgt_tensor.shape[-2:]:
        tgt_tensor = F.interpolate(tgt_tensor, size=(h, w), mode='bilinear')

    # Prepare ingredients using predetermined homography (replacing IHN)
    result = prepare_ingredient_beehive(H_tgt2ref, tgt_tensor, ref_tensor, args.max_canvas_ratio)
    
    if result is None:
        print("Error: Failed to prepare ingredients (oversized canvas)")
        if args.create_skip_image:
            skip_img = Image.new('RGB', (100, 100), color='black')
            skip_img.save(args.out)
            print(f"Created placeholder skip image at {args.out}")
        return None

    (tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, 
     stit_grid, stit_mask, sizes) = result

    # Normalize images (same as original)
    ref_tensor = (ref_tensor - 0.5) * 2
    tgt_tensor = (tgt_tensor - 0.5) * 2

    # Reshape masks
    ref_mask = ref_mask.reshape(b,1,*sizes)
    tgt_mask = tgt_mask.reshape(b,1,*sizes)

    try:
        # Use original pipeline with seam cutting
        pred = batched_predict(
            model, ref_tensor, ref_grid, ref_cell, ref_mask,
            tgt_tensor, tgt_grid, tgt_cell, tgt_mask,
            stit_grid, sizes, config['eval_bsize'], 
            seam_cut=args.seam_cut  # Allow control over seam cutting
        )
        
        # Post-process (same as original)
        pred = pred.permute(0, 2, 1).reshape(b, c, *sizes)
        pred = ((pred + 1)/2).clamp(0,1) * stit_mask.reshape(b, 1, *sizes)

        # Save result
        transforms.ToPILImage()(pred[0].cpu()).save(args.out)
        print(f"Stitched image saved to {args.out}")

        # Optionally save masks
        if args.save_masks:
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
        
        # Log processing time
        process_time = time.time() - start_time
        seam_status = "WITH seam cutting" if args.seam_cut else "WITHOUT seam cutting"
        print(f"Total stitching completed in {process_time:.2f}s ({seam_status})")
        
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
    
    start_time = time.time()
    
    try:
        # Load blending model (no IHN needed)
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

        # Convert to tensors
        to_tensor = transforms.ToTensor()
        ref_tensor = to_tensor(ref_img_pil).cuda().unsqueeze(0)
        tgt_tensor = to_tensor(tgt_img_pil).cuda().unsqueeze(0)

        # Run stitching with super simple approach
        with torch.no_grad():
            result = stitch_images_beehive_super_simple_fix(model, ref_tensor, tgt_tensor, config, args)
            success = result is not None
            
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
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIS beehive stitching with super simple fix')
    parser.add_argument('--ref', required=True, help='Reference image path')
    parser.add_argument('--tgt', required=True, help='Target image path')
    parser.add_argument('--out', required=True, help='Output image path')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for input images')
    parser.add_argument('--homography_path', required=True, help='Path to homography file')
    parser.add_argument('--seam_cut', action='store_true', default=False, help='Enable seam cutting')
    parser.add_argument('--max_canvas_ratio', type=float, default=1.8, help='Maximum canvas size ratio')
    parser.add_argument('--create_skip_image', action='store_true', help='Create placeholder image on failure')
    parser.add_argument('--save_masks', action='store_true', help='Save mask visualizations')
    
    args = parser.parse_args()
    
    # Default config for beehive processing
    config = {
        'eval_bsize': 1024 * 8,  # Batch size for evaluation
    }
    
    exit_code = main(config, args)
    sys.exit(exit_code) 
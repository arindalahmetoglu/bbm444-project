#!/usr/bin/env python3
"""
Direct homography extraction using original NIS code
This script directly loads and uses the original stitch.py functions
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

def extract_homography_direct(ref_path, tgt_path, device='cuda:0'):
    """
    Extract homography using the original NIS functions directly
    """
    print(f"Extracting homography using ORIGINAL NIS functions...")
    
    # Convert paths to absolute paths before changing directory
    ref_path = os.path.abspath(ref_path)
    tgt_path = os.path.abspath(tgt_path)
    
    # Change to the original NIS directory and add to path
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_nis_dir = os.path.join(script_dir, 'aaredundant', 'Neural-Image-Stitching_working_stitchpy')
    
    try:
        if not os.path.exists(original_nis_dir):
            raise ValueError(f"Original NIS directory not found: {original_nis_dir}")
            
        os.chdir(original_nis_dir)
        sys.path.insert(0, original_nis_dir)
        
        # Import the original modules
        import utils
        import models
        from srwarp import transform
        
        def prepare_ingredient(model, inp_tgt, inp_ref, tgt, ref):
            """Original prepare_ingredient function from stitch.py"""
            b, c, h, w = tgt.shape

            four_pred, _ = model(inp_tgt, inp_ref, iters_lev0=6, iters_lev1=3, test_mode=True)
            shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

            shape = tgt.shape[-2:]
            H_tgt2ref, w_max, w_min, h_max, h_min = utils.get_H(shift * w/128, shape)

            img_h = torch.ceil(h_max - h_min).int().item()
            img_w = torch.ceil(w_max - w_min).int().item()
            sizes = (img_h, img_w)

            h_max = h_max.item(); h_min = h_min.item()
            w_max = w_max.item(); w_min = w_min.item()

            eye = torch.eye(3).double()
            T = utils.get_translation(h_min, w_min)

            H_tgt2ref = H_tgt2ref[0].double().cpu()
            H_tgt2ref = T @ H_tgt2ref

            eye, _, _ = transform.compensate_matrix(ref, eye)
            eye = T @ eye

            return H_tgt2ref, eye, img_w, img_h
        
        def prepare_validation():
            """Original prepare_validation function"""
            sv_file = torch.load("pretrained/NIS_blending.pth")
            model = models.make(sv_file['model'], load_sd=True).cuda()

            H_model = models.IHN().cuda()
            sv_file = torch.load("pretrained/ihn.pth")
            H_model.load_state_dict(sv_file['model']['sd'])

            return model, H_model
        
        # Set device
        if device.startswith('cuda'):
            gpu_id = device.split(':')[-1] if ':' in device else '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        
        # Load models
        model, H_model = prepare_validation()
        H_model.eval()

        # Load images
        ref_img = Image.open(ref_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')
        
        to_tensor = transforms.ToTensor()
        ref = to_tensor(ref_img).unsqueeze(0).cuda()
        tgt = to_tensor(tgt_img).unsqueeze(0).cuda()

        b, c, h, w = ref.shape

        if ref.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt, size=(h, w), mode='bilinear')

        if h != 128 or w != 128:
            inp_ref = F.interpolate(ref, size=(128,128), mode='bilinear') * 255
            inp_tgt = F.interpolate(tgt, size=(128,128), mode='bilinear') * 255
        else:
            inp_ref = ref * 255
            inp_tgt = tgt * 255

        # Extract homography using original method
        H_tgt2ref, H_ref, canvas_width, canvas_height = prepare_ingredient(H_model, inp_tgt, inp_ref, tgt, ref)
        
        print(f"✓ Successfully extracted homography using ORIGINAL NIS")
        print(f"  Canvas size: {canvas_height}x{canvas_width}")
        
        return H_tgt2ref.detach().numpy(), H_ref.detach().numpy(), canvas_width, canvas_height
        
    except Exception as e:
        print(f"Error in original NIS processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    finally:
        # Restore working directory
        os.chdir(original_cwd)
        # Remove from path
        if original_nis_dir in sys.path:
            sys.path.remove(original_nis_dir)

def create_warps_and_masks(ref_path, tgt_path, H_tgt2ref, H_ref, canvas_width, canvas_height):
    """
    Create warps and masks using the homographies from original NIS
    """
    print(f"Creating warps and masks...")
    
    # Load original images
    ref_img = cv2.imread(ref_path)
    tgt_img = cv2.imread(tgt_path)
    
    if ref_img is None or tgt_img is None:
        raise ValueError(f"Could not load images: {ref_path}, {tgt_path}")
    
    # Convert homographies to OpenCV format
    H_tgt2ref_cv = H_tgt2ref.astype(np.float64)
    H_ref_cv = H_ref.astype(np.float64)
    
    # Warp images
    warped_ref = cv2.warpPerspective(ref_img, H_ref_cv, (canvas_width, canvas_height))
    warped_tgt = cv2.warpPerspective(tgt_img, H_tgt2ref_cv, (canvas_width, canvas_height))
    
    # Create masks
    ref_mask = np.ones((ref_img.shape[0], ref_img.shape[1]), dtype=np.uint8) * 255
    tgt_mask = np.ones((tgt_img.shape[0], tgt_img.shape[1]), dtype=np.uint8) * 255
    
    warped_ref_mask = cv2.warpPerspective(ref_mask, H_ref_cv, (canvas_width, canvas_height))
    warped_tgt_mask = cv2.warpPerspective(tgt_mask, H_tgt2ref_cv, (canvas_width, canvas_height))
    
    # Calculate mask coverage
    ref_coverage = np.sum(warped_ref_mask > 0) / (canvas_width * canvas_height)
    tgt_coverage = np.sum(warped_tgt_mask > 0) / (canvas_width * canvas_height)
    
    print(f"Reference mask coverage: {ref_coverage:.3f}")
    print(f"Target mask coverage: {tgt_coverage:.3f}")
    
    return warped_ref, warped_tgt, warped_ref_mask, warped_tgt_mask

def main():
    parser = argparse.ArgumentParser(description='Extract homography using original NIS functions directly')
    parser.add_argument('ref_image', help='Path to reference image')
    parser.add_argument('tgt_image', help='Path to target image')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--output_dir', default='shared_stitching_data_original_nis', 
                        help='Output directory for homography data')
    parser.add_argument('--pair_idx', type=int, required=True, help='Pair index for file naming')
    
    args = parser.parse_args()
    
    print(f"Original NIS Homography Extraction (Direct)")
    print(f"Reference: {args.ref_image}")
    print(f"Target: {args.tgt_image}")
    print(f"Device: {args.device}")
    
    try:
        # Extract homography using original NIS
        H_tgt2ref, H_ref, canvas_width, canvas_height = extract_homography_direct(
            args.ref_image, args.tgt_image, args.device
        )
        
        if H_tgt2ref is None:
            print("Failed to extract homography from original NIS")
            return 1
        
        # Create warps and masks
        warped_ref, warped_tgt, ref_mask, tgt_mask = create_warps_and_masks(
            args.ref_image, args.tgt_image, H_tgt2ref, H_ref, canvas_width, canvas_height
        )
        
        # Save outputs
        pair_idx_formatted = f"{args.pair_idx:06d}"
        
        # Create output directories
        os.makedirs(f"{args.output_dir}/homography", exist_ok=True)
        os.makedirs(f"sift_preprocessed_data/warp1", exist_ok=True) 
        os.makedirs(f"sift_preprocessed_data/warp2", exist_ok=True)
        os.makedirs(f"sift_preprocessed_data/mask1", exist_ok=True)
        os.makedirs(f"sift_preprocessed_data/mask2", exist_ok=True)
        
        # Save homography matrix
        homography_file = f"{args.output_dir}/homography/{pair_idx_formatted}.txt"
        np.savetxt(homography_file, H_tgt2ref, fmt='%.10f')
        
        # Save warps and masks
        cv2.imwrite(f"sift_preprocessed_data/warp1/{pair_idx_formatted}.png", warped_ref)
        cv2.imwrite(f"sift_preprocessed_data/warp2/{pair_idx_formatted}.png", warped_tgt)
        cv2.imwrite(f"sift_preprocessed_data/mask1/{pair_idx_formatted}.png", ref_mask)
        cv2.imwrite(f"sift_preprocessed_data/mask2/{pair_idx_formatted}.png", tgt_mask)
        
        print(f"✓ Successfully processed pair {args.pair_idx} using DIRECT ORIGINAL NIS")
        print(f"  Homography saved to: {homography_file}")
        print(f"  Warps and masks saved to sift_preprocessed_data/")
        
        return 0
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 
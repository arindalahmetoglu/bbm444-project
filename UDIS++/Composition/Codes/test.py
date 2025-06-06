# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
from network import Network, build_model
from dataset import TestDataset
import os
import cv2
import numpy as np
import glob


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')



def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    print(f"Loading test data from: {args.test_path}")
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
            
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        print(f"Loading model from: {model_path}")
        # Ensure map_location is set if loading a GPU-trained model on a CPU-only machine or different GPU setup
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model'])
        print('Successfully loaded model from {}!'.format(model_path))
    else:
        print('No checkpoint found in {}!'.format(MODEL_DIR))
        return

    print("##################start testing#######################")
    
    # Create folders for output using args.save_path as base
    # If args.save_path is not provided or empty, default to current directory for outputs
    base_output_dir = args.save_path if args.save_path else "."
    
    output_raw_comp_mask_path = os.path.join(base_output_dir, "raw_composition_mask")
    output_final_fusion_path = os.path.join(base_output_dir, "final_fusion")
    
    if not os.path.exists(output_raw_comp_mask_path):
        os.makedirs(output_raw_comp_mask_path, exist_ok=True)
    if not os.path.exists(output_final_fusion_path):
        os.makedirs(output_final_fusion_path, exist_ok=True)
    
    print(f"Saving raw composition masks to: {os.path.abspath(output_raw_comp_mask_path)}")
    print(f"Saving final fusion images to: {os.path.abspath(output_final_fusion_path)}")

    net.eval()
    for i, batch_value in enumerate(test_loader):
        warp1_tensor = batch_value[0].float()
        warp2_tensor = batch_value[1].float()
        mask1_tensor = batch_value[2].float()
        mask2_tensor = batch_value[3].float()

        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()

        with torch.no_grad():
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

        raw_comp_mask_tensor = batch_out['raw_comp_mask']
        stitched_image_tensor = batch_out['stitched_image']

        raw_comp_mask_np = raw_comp_mask_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
        stitched_image_np = ((stitched_image_tensor[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        
        # Save raw composition mask (direct U-Net output)
        path_raw_comp_mask = os.path.join(output_raw_comp_mask_path, str(i + 1).zfill(6) + ".png")
        cv2.imwrite(path_raw_comp_mask, (raw_comp_mask_np * 255).astype(np.uint8))

        # Save final fusion result (stitched image)
        path_final_fusion = os.path.join(output_final_fusion_path, str(i + 1).zfill(6) + ".jpg")
        cv2.imwrite(path_final_fusion, stitched_image_np.astype(np.uint8))

        print('Processed image i = {}'.format(i + 1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("##################end testing#######################")



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, 
                        default='/home/arin/Projects/bbm444-project/UDIS++/sift_aligned_data_for_composition/',
                        help='Path to the testing data, which should contain warp1, warp2, mask1, mask2 subfolders.')
    parser.add_argument('--save_path', type=str, default='.', 
                        help='Base directory where output subfolders (raw_composition_mask, final_fusion) will be created.')
    # The --name argument is not standard for this script's output naming, so it's omitted.
    # Output files are named numerically (e.g., 000001.png, 000002.png).

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
        
    test(args)
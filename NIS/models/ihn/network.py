import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from .update import GMA
from .extractor import BasicEncoderQuarter
from .corr import CorrBlock
from .utils import *

autocast = torch.cuda.amp.autocast

class IHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.hidden_dim = 128
        self.context_dim = 128
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')

        self.update_block_4 = GMA(32)
        self.update_block_2 = GMA(64)

    def get_flow_now_4(self, four_point_disp):
        four_point_disp = four_point_disp / 4
        device = four_point_disp.device
        batch_size = four_point_disp.shape[0]
        
        # Original four points (corners of the image)
        four_point_org = torch.zeros((2, 2, 2)).to(device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])
        
        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point_disp
        
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        
        # Manual calculation of perspective transform matrix
        A = torch.zeros(batch_size, 8, 8).to(device)
        b = torch.zeros(batch_size, 8, 1).to(device)
        
        for i in range(4):
            x, y = four_point_org[:, i, 0], four_point_org[:, i, 1]
            xp, yp = four_point_new[:, i, 0], four_point_new[:, i, 1]
            
            A[:, 2*i, 0] = x
            A[:, 2*i, 1] = y
            A[:, 2*i, 2] = 1
            A[:, 2*i, 6] = -x * xp
            A[:, 2*i, 7] = -y * xp
            
            A[:, 2*i+1, 3] = x
            A[:, 2*i+1, 4] = y
            A[:, 2*i+1, 5] = 1
            A[:, 2*i+1, 6] = -x * yp
            A[:, 2*i+1, 7] = -y * yp
            
            b[:, 2*i, 0] = xp
            b[:, 2*i+1, 0] = yp
        
        # Solve the system with torch.inverse instead of torch.gesv
        h = torch.matmul(torch.inverse(A), b).squeeze(-1)
        
        # Construct the homography matrix H
        H = torch.ones(batch_size, 9).to(device)
        H[:, :8] = h
        H = H.view(batch_size, 3, 3)
        
        # Create grid coordinates
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), 
                                     torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        
        points = torch.cat((gridx.flatten().unsqueeze(0), 
                          gridy.flatten().unsqueeze(0), 
                          torch.ones((1, self.sz[3] * self.sz[2]))),
                         dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(device)
        
        points_new = torch.bmm(H, points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                        points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        
        return flow

    def get_flow_now_2(self, four_point_disp):
        four_point_disp = four_point_disp / 2
        device = four_point_disp.device
        batch_size = four_point_disp.shape[0]
        
        # Original four points (corners of the image)
        four_point_org = torch.zeros((2, 2, 2)).to(device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])
        
        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point_disp
        
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        
        # Manual calculation of perspective transform matrix
        A = torch.zeros(batch_size, 8, 8).to(device)
        b = torch.zeros(batch_size, 8, 1).to(device)
        
        for i in range(4):
            x, y = four_point_org[:, i, 0], four_point_org[:, i, 1]
            xp, yp = four_point_new[:, i, 0], four_point_new[:, i, 1]
            
            A[:, 2*i, 0] = x
            A[:, 2*i, 1] = y
            A[:, 2*i, 2] = 1
            A[:, 2*i, 6] = -x * xp
            A[:, 2*i, 7] = -y * xp
            
            A[:, 2*i+1, 3] = x
            A[:, 2*i+1, 4] = y
            A[:, 2*i+1, 5] = 1
            A[:, 2*i+1, 6] = -x * yp
            A[:, 2*i+1, 7] = -y * yp
            
            b[:, 2*i, 0] = xp
            b[:, 2*i+1, 0] = yp
        
        # Solve the system with torch.inverse instead of torch.gesv
        h = torch.matmul(torch.inverse(A), b).squeeze(-1)
        
        # Construct the homography matrix H
        H = torch.ones(batch_size, 9).to(device)
        H[:, :8] = h
        H = H.view(batch_size, 3, 3)
        
        # Create grid coordinates
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), 
                                     torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        
        points = torch.cat((gridx.flatten().unsqueeze(0), 
                          gridy.flatten().unsqueeze(0), 
                          torch.ones((1, self.sz[3] * self.sz[2]))),
                         dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(device)
        
        points_new = torch.bmm(H, points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                        points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        
        return flow

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        return coords0, coords1

    def initialize_flow_2(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//2, W//2).to(img.device)
        coords1 = coords_grid(N, H//2, W//2).to(img.device)

        return coords0, coords1

    def forward(self, image1, image2, iters_lev0=6, iters_lev1=3, test_mode=False):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=False):
            fmap1_32, fmap1_64 = self.fnet1(image1)
            fmap2_32, _ = self.fnet1(image2)
        fmap1 = fmap1_32.float()
        fmap2 = fmap2_32.float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, sz=32)
        coords0, coords1 = self.initialize_flow_4(image1)
        sz = fmap1_32.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)

        disps = []
        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=False):
                delta_four_point = self.update_block_4(corr, flow)
            four_point_disp =  four_point_disp + delta_four_point
            coords1 = self.get_flow_now_4(four_point_disp)
            disps.append(four_point_disp)

        four_point_disp_med = four_point_disp
        flow_med = coords1 - coords0

        flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4
        image2 = warp(image2, flow_med)

        with autocast(enabled=False):
            _, fmap2_64 = self.fnet1(image2)
        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels = 2, radius= 4, sz = 32)
        coords0, coords1 = self.initialize_flow_2(image1)
        sz = fmap1.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)

        for itr in range(iters_lev1):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=False):
                delta_four_point = self.update_block_2(corr, flow)
            four_point_disp = four_point_disp + delta_four_point
            coords1 = self.get_flow_now_2(four_point_disp)

            four_point_disp_med += delta_four_point
            disps.append(four_point_disp_med)

        return disps[-1], disps

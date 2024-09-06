import torch
from torch import nn
import numpy as np
from ...render_utils.geometry_utils import getWorld2View, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, img_size, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, device = "cuda:0"):
        super(Camera, self).__init__()
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = img_size[1]
        self.image_height = img_size[0]

        zfar = 100.0
        znear = 0.01

        self.world_view_transform = torch.from_numpy(getWorld2View(R, T, trans, scale)).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

import torch
import torch.nn as nn
import numpy as np
import os
from .geometry import ADGaussianGeo
from .camera import Camera
from simple_knn._C import distCUDA2
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, so3_exp_map
from ...render_utils.geometry_utils import strip_symmetric, build_scaling_rotation, focal2fov, proj_pts, pts2obj
from ..neural_render import Neural_Renderer
import ipdb
import time

class GaussianScene(nn.Module):
    def __init__(self, data_dir, img_size, device = 'cuda:0'):
        super(GaussianScene, self).__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.device = device

        self.smplx_track_keys = ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'expr', 'cam_trans', 'cam_angle', 'shape', 'cam_para']
        self.load_smplx_track()

        self.geo_creator = ADGaussianGeo(device, self.shape, None)
        self.lap_indices = self.geo_creator.lap_indices
        self.init_attributes()
        self.setup_functions()

    def load_smplx_track(self):# 设置self.cameras
        smplx_track_dict = torch.load(os.path.join(self.data_dir, 'body_track', 'smplx_track.pth'))
        for key in self.smplx_track_keys:
            self.__setattr__(key, torch.from_numpy(smplx_track_dict[key]).to(self.device))
        self.cameras = []
        rots = so3_exp_map(self.cam_angle)
        transs = self.cam_trans.clone()
        cam_para = self.cam_para[0]
        rots[:, :, 1:] *= -1
        transs[:, 1:] *= -1
        fx, fy = float(cam_para[0]), float(cam_para[1])
        print(cam_para, self.img_size)
        FovX = focal2fov(fx, self.img_size[1])
        FovY = focal2fov(fy, self.img_size[0])
        for i in range(self.cam_angle.shape[0]):
            camera_indiv = Camera(rots[i].cpu().numpy(), transs[i].cpu().numpy(), FovX, FovY, self.img_size, device=self.device)
            self.cameras.append(camera_indiv)

    def load_driven_track(self, smplx_track_file):
        smplx_track_dict = torch.load(smplx_track_file)
        for key in self.smplx_track_keys:
            self.__setattr__('driven_' + key, torch.from_numpy(smplx_track_dict[key]).to(self.device))
        self.driven_cameras = []
        rots = so3_exp_map(self.driven_cam_angle)
        transs = self.driven_cam_trans.clone()
        cam_para = self.driven_cam_para[0]
        rots[:, :, 1:] *= -1
        transs[:, 1:] *= -1
        fx, fy = float(cam_para[0]), float(cam_para[1])
        driven_img_size = (int(cam_para[3]*2+.5), int(cam_para[2]*2+.5))
        print(cam_para, driven_img_size)
        FovX = focal2fov(fx, driven_img_size[1])
        FovY = focal2fov(fy, driven_img_size[0])
        for i in range(self.driven_cam_angle.shape[0]):
            camera_indiv = Camera(rots[i].cpu().numpy(), transs[i].cpu().numpy(), FovX, FovY, driven_img_size, device=self.device)
            self.driven_cameras.append(camera_indiv)
        return self.driven_cam_angle.shape[0]

    def cal_attr_lap_loss(self, attr):
        # ipdb.set_trace()
        attr_dis = attr[self.lap_indices[:, 0]] - torch.mean(attr[self.lap_indices[:, 1:]], dim=1)
        return torch.mean(attr_dis**2)

    def cal_dis_loss(self):
        return torch.mean(self._xyz_canonical_dis**2)
    
    def cal_scale_loss(self,arg_max = 6e-2):
        # This function will return nan if no scaling is larger than arg_max
        
        return torch.mean(torch.relu(self.scaling_activation(self._scaling)[self.scaling_activation(self._scaling)>arg_max] - arg_max))

    def opac_sparse_loss(self,sigma = 1e-8):
        opacity = self.get_opacity
        loss = (torch.log(opacity+sigma) + torch.log(1-opacity+sigma)).mean()
        return loss

    def init_attributes(self):
        pc_num = self.geo_creator.gaussian_pnum
        print('point number', pc_num)

        dist2 = torch.clamp_min(distCUDA2(self.geo_creator.canonical_points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        print('init scale', torch.exp(torch.min(scales)), torch.exp(torch.max(scales)), torch.exp(torch.mean(scales)))
        rots = torch.zeros((pc_num, 4), device=self.device)
        rots[:, 0] = 1
        opacities = torch.logit(0.1 * torch.ones((pc_num, 1), dtype=torch.float, device=self.device))

        self._body_pose_delta = nn.Parameter(torch.zeros_like(self.body_pose).requires_grad_(True))
        self._lhand_pose_delta = nn.Parameter(torch.zeros_like(self.lhand_pose).requires_grad_(True))
        self._rhand_pose_delta = nn.Parameter(torch.zeros_like(self.rhand_pose).requires_grad_(True))

        self._xyz_canonical_dis = nn.Parameter(torch.zeros_like(self.geo_creator.canonical_points).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation_base = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((pc_num), device=self.device)

        neural_feature = torch.zeros((pc_num, 32)).float().to(self.device)
        self._neural_features_op = nn.Parameter(neural_feature.contiguous().requires_grad_(True))
        self.neural_renderer = Neural_Renderer(input_dim=32, output_dim=3, network_capacity=32).to(self.device)

    def cal_lap_loss(self):
        return self.cal_attr_lap_loss(self._xyz_canonical_dis) + self.cal_attr_lap_loss(self._scaling) + self.cal_attr_lap_loss(self._opacity)

    def cal_lap_loss_neural(self):
        return self.cal_attr_lap_loss(self._xyz_canonical_dis) + self.cal_attr_lap_loss(self._scaling) + self.cal_attr_lap_loss(self._opacity) + self.cal_attr_lap_loss(self._neural_features_op)
    
    def save_updated_canonical(self, save_path):
        pts2obj(self.geo_creator.canonical_points + self._xyz_canonical_dis, save_path)

    def save_posed_updated_canonical(self, save_path):
        pts2obj(self._xyz, save_path)

    def capture(self):
        return (
            self._xyz_canonical_dis, self._scaling, self._rotation_base, self._opacity, self.optimizer.state_dict(), self._body_pose_delta, self._lhand_pose_delta, self._rhand_pose_delta, self._neural_features_op, self.neural_renderer.state_dict()
        )
    
    def restore(self, model_dict):
        (_xyz_canonical_dis, _scaling, _rotation_base, _opacity, opt_dict, _body_pose_delta, _lhand_pose_delta, _rhand_pose_delta, _neural_features_op, neural_renderer_dict) = model_dict
        self._xyz_canonical_dis.data = _xyz_canonical_dis.clone()
        self._scaling.data = _scaling.clone()
        self._rotation_base.data = _rotation_base.clone()
        self._opacity.data = _opacity.clone()
        self._body_pose_delta.data = _body_pose_delta.clone()
        self._lhand_pose_delta.data = _lhand_pose_delta.clone()
        self._rhand_pose_delta.data = _rhand_pose_delta.clone()
        self._neural_features_op.data = _neural_features_op.clone()
        self.neural_renderer.load_state_dict(neural_renderer_dict)
        # self.optimizer.load_state_dict(opt_dict)

        print('load scale', torch.exp(torch.min(_scaling)), torch.exp(torch.max(_scaling)), torch.exp(torch.mean(_scaling)))

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize 

    def update_xyz_rot_driven(self, idx):
        frame_smplx_dict = {}
        frame_idx = idx%(self.body_pose.shape[0])
        for key in self.smplx_track_keys[0:5]:
            frame_smplx_dict[key] = getattr(self, key)[frame_idx:frame_idx+1]
        frame_smplx_dict['body_pose'][:, 11*3:12*3] = self.driven_body_pose[idx:idx+1][:, 11*3:12*3]
        frame_smplx_dict['body_pose'][:, 14*3:15*3] = self.driven_body_pose[idx:idx+1][:, 14*3:15*3]
        pcs_rott, _ = self.geo_creator.compute_canonical_points_rott(frame_smplx_dict) #### (1, pc_num, 3, 4)
        pcs_canonical_homo = torch.cat((self.geo_creator.canonical_points.clone().detach() + self._xyz_canonical_dis, torch.ones_like(self._xyz_canonical_dis[..., :1])), dim=-1)
        self._xyz = torch.bmm(pcs_rott[0], pcs_canonical_homo.unsqueeze(-1)).squeeze(-1)
        self._rotation = quaternion_multiply(matrix_to_quaternion(pcs_rott[0, :, :3, :3]), self._rotation_base)

    def cal_pose_loss(self, frame_ids):
        return torch.mean(torch.abs(self._body_pose_delta[frame_ids].clone())) + torch.mean(torch.abs(self._lhand_pose_delta[frame_ids].clone())) + torch.mean(torch.abs(self._rhand_pose_delta[frame_ids].clone()))
    
    def get_xyz_rot_batch(self, indices):
        # indices must be a list of index

        bsz= len(indices)

        frame_smplx_dict = {}


        for key in self.smplx_track_keys[0:5]:
            frame_smplx_dict[key] = getattr(self, key)[indices].clone()

        pcs_rott, _ ,  pcs_canonical = self.geo_creator.compute_canonical_points_rott(frame_smplx_dict) #### (b, pc_num, 3, 4), _ ,(b, pc_num, 3)

        pcs_canonical_homo = torch.cat((pcs_canonical.clone().detach() + self._xyz_canonical_dis.unsqueeze(0), torch.ones_like(self._xyz_canonical_dis[..., :1]).unsqueeze(0).expand(bsz,-1,-1)), dim=-1)
        xyzs = torch.einsum("bvmn,bvnl->bvml",pcs_rott, pcs_canonical_homo.unsqueeze(-1)).squeeze(-1)
        
        rotations = quaternion_multiply(matrix_to_quaternion(pcs_rott[:, :, :3, :3]), self._rotation_base.unsqueeze(0).expand(bsz,-1,-1))

        return xyzs, rotations

    def register_xyz_rotation(self,xyz,rotation):
        self._xyz = xyz
        self._rotation = rotation

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_neural_features(self):
        return self._neural_features_op
         
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    def training_setup(self, training_args):
        optable_params = [
                {'params': [self._xyz_canonical_dis], 'lr': training_args.position_lr_init, "name": "xyz"},
                {'params': [self._neural_features_op], 'lr': training_args.neural_feature_lr, "name": "n_fea"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation_base], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': list(self.neural_renderer.parameters()), 'lr': training_args.neural_renderer_lr, "name": "neural_renderer"},
            ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.0, eps=1e-15)

    def forward(self, x):
        pass
    
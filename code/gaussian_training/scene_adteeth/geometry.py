from ...dmm_models import SMPLX, smplx_model_path
from ...render_utils.geometry_utils import bary_inerpolation
from ...dmm_models.adnerf_rendering import adnerf_rendering_part_ids_path, adnerf_teeth_mesh_file, adnerf_gaussian_info_file
# from ...dmm_models.smplx.smplx.lbs import vertices2landmarks
from pytorch3d.io import load_obj
import torch
import torch.nn as nn
import numpy as np
import time
import ipdb

class ADGaussianGeo(nn.Module):
    def __init__(self, device, shape_code, canonical_hair_centers = None):
        super(ADGaussianGeo, self).__init__()
        self.device = device
        
        shape_dim, expr_dim = 300, 100
        self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas=shape_dim, use_pca=False).to(self.device).eval()
        # portraitgen-code/code/dmm_models/adnerf_rendering/wholebody_sel.txt
        self.adnerf_rendering_part_ids = np.loadtxt(adnerf_rendering_part_ids_path, dtype=np.int64)

        # portraitgen-code/code/dmm_models/adnerf_rendering/wholebody_tris_info.pt
        adnerf_gaussian_info = torch.load(adnerf_gaussian_info_file)

        skin_part_bary_info = adnerf_gaussian_info['smplx_tris'].to(self.device) #### （*， 3)
        teeth_part_bary_info = adnerf_gaussian_info['teeth_tris'].to(self.device)
        self.lap_indices = adnerf_gaussian_info['lap_indices'].to(self.device)
        self.upper_teeth_vid = adnerf_gaussian_info['upper_vid']
        self.downer_teeth_vid = adnerf_gaussian_info['downer_vid']
        self.hair_vid = adnerf_gaussian_info['hair_vid']
        self.skin_part_pnum = skin_part_bary_info.shape[0]
        self.teeth_part_pnum = teeth_part_bary_info.shape[0] 
        if canonical_hair_centers is not None:
            self.hair_part_pnum = canonical_hair_centers.shape[0]
        else:
            self.hair_part_pnum = 0
            canonical_hair_centers = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.gaussian_pnum = self.skin_part_pnum + self.teeth_part_pnum*2 + self.hair_part_pnum

        self.skin_part_tids = skin_part_bary_info[:, 0].long() # which triangle the 3d gaussian locate in 
        self.skin_part_coords = torch.cat((skin_part_bary_info[:, 1:], 1. - torch.sum(skin_part_bary_info[:, 1:], dim=-1, keepdim=True)), dim=-1)
        self.teeth_part_tids = teeth_part_bary_info[:, 0].long()
        self.teeth_part_coords = torch.cat((teeth_part_bary_info[:, 1:], 1. - torch.sum(teeth_part_bary_info[:, 1:], dim=-1, keepdim=True)), dim=-1)
        
        _, faces, _ = load_obj(adnerf_teeth_mesh_file)
        self.tris = faces.verts_idx.to(self.device).long()
        self.shape_code = shape_code.to(self.device)

        self.canonical_hair_centers = canonical_hair_centers
        self.compute_canonical_points() # only compute one time to get self.canonical_teeth_centers


    def compute_canonical_points(self):
        smplx_out = self.smplx_model.forward(betas = self.shape_code)
        adnerf_part_verts = smplx_out.vertices[0, self.adnerf_rendering_part_ids, :] ### (nv, 3)

        canonical_skin_centers = bary_inerpolation(adnerf_part_verts.unsqueeze(0), self.tris, self.skin_part_tids.unsqueeze(0), self.skin_part_coords.unsqueeze(0)).squeeze(0)
        self.canonical_teeth_centers = bary_inerpolation(adnerf_part_verts.unsqueeze(0), self.tris, self.teeth_part_tids.unsqueeze(0), self.teeth_part_coords.unsqueeze(0)).squeeze(0)
        
        self.canonical_teeth_centers[:, 2] -= 0.02 #### teeth 1cm back 

        self.canonical_points = torch.cat((canonical_skin_centers, self.canonical_teeth_centers, self.canonical_teeth_centers, self.canonical_hair_centers), dim=0) # canonical_points is  necessary for Gaussians Initialization

    def compute_canonical_points_rott(self, frames_smplx_dict,register_canonical_points = False):
        b = frames_smplx_dict['expr'].shape[0]

        smplx_out, verts_rott, verts_canonical = self.smplx_model.forward(betas=self.shape_code.expand(b, -1), body_pose=frames_smplx_dict['body_pose'], left_hand_pose=frames_smplx_dict['lhand_pose'], right_hand_pose=frames_smplx_dict['rhand_pose'], expression=frames_smplx_dict['expr'], jaw_pose=frames_smplx_dict['jaw_pose'], with_rott_return=True)
        
        canonical_part_verts = verts_canonical[:, self.adnerf_rendering_part_ids, :]
        canonical_skin_centers = bary_inerpolation(canonical_part_verts, self.tris, self.skin_part_tids.unsqueeze(0).expand(b, -1), self.skin_part_coords.unsqueeze(0).expand(b,-1,-1))
        if register_canonical_points==True or b == 1: # only register canonical points when b == 1 or force it to register
            self.canonical_points = torch.cat((canonical_skin_centers.squeeze(0), self.canonical_teeth_centers, self.canonical_teeth_centers, self.canonical_hair_centers), dim=0)
        
        adnerf_part_verts_rott = verts_rott[:, self.adnerf_rendering_part_ids]
        skin_points_rott = bary_inerpolation(adnerf_part_verts_rott, self.tris, self.skin_part_tids.unsqueeze(0).expand(b, -1), self.skin_part_coords.unsqueeze(0).expand(b, -1, -1))
        
        return torch.cat((skin_points_rott, adnerf_part_verts_rott[:, self.upper_teeth_vid].unsqueeze(1).repeat(1, self.teeth_part_pnum, 1, 1), adnerf_part_verts_rott[:, self.downer_teeth_vid].unsqueeze(1).repeat(1, self.teeth_part_pnum, 1, 1), adnerf_part_verts_rott[:, self.hair_vid].unsqueeze(1).repeat(1, self.hair_part_pnum, 1, 1)), dim=1), smplx_out.vertices, torch.cat((canonical_skin_centers,self.canonical_teeth_centers.unsqueeze(0).expand(b,-1,-1),self.canonical_teeth_centers.unsqueeze(0).expand(b,-1,-1),self.canonical_hair_centers.unsqueeze(0).expand(b,-1,-1)),dim=1)
        
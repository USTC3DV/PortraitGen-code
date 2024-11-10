from code.render_utils import MeshRenderer
from pytorch3d.io import load_obj
import numpy as np
import torch
import cv2
import os

verts, faces, aux = load_obj("./code/dmm_models/smplx/SMPLX2020/smplx_uv.obj")
print(verts.shape)

uv_coords = aux.verts_uvs.float().cuda()
uv_coords[..., 1] = 1. - uv_coords[..., 1]
tris_uv = faces.textures_idx.cuda().int()
tris_verts = faces.verts_idx.cuda().long()
verts = verts.cuda().float()

mesh_renderer = MeshRenderer()

uv_size = 185
rast_out = mesh_renderer.rasterize_uv_img(uv_coords, tris_uv, uv_size) # rast_out's shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id)
print(rast_out.shape, float(torch.sum((rast_out[..., 3]>0).float())) / uv_size / uv_size)
# uv_mask_img = ((rast_out[0, :, :, 3] > 0).float()*255).byte().detach().cpu().numpy()
faces_num = tris_uv.shape[0]
teeth_faces_num = 36 # 确实撒了几千个高斯点，但是只有36个三角型，三角形是拿来缝高斯点的。
# uv_mask_img[((rast_out[0, :, :, 3]-1)>=(faces_num-teeth_faces_num)).cpu().numpy()] = 155
# cv2.imwrite('tmp/uv_mask.jpg', uv_mask_img)

##### save info, upper_teeth: 1164, downer_teeth: 1049
rast_out[..., 3] -= 1
smplx_part = rast_out[0, :, :, :][(rast_out[0, :, :, 3] > -.5)].reshape(-1, 4) # (rast_out[0, :, :, 3] > -.5) is the valid map [512,512], this is to select all the valid uv pixels.

valid_tris_verts = tris_verts[smplx_part[:, 3].long()] # smplx_part: pixel->triangle_idx, then triangle_idx->verts_idx
valid_verts = verts[valid_tris_verts[:,0]] * smplx_part[:, 0:1] + verts[valid_tris_verts[:,1]] * smplx_part[:, 1:2] + verts[valid_tris_verts[:,2]] * (1. - torch.sum(smplx_part[:, 0:2], dim=-1, keepdim=True))

valid_img_indices = torch.nonzero(rast_out[0, :, :, 3] > -.5, as_tuple=True)
img_valids = (torch.zeros_like(rast_out[0, :, :, 3]).long() - 1).cpu()
img_valids[valid_img_indices[0], valid_img_indices[1]] = torch.arange(valid_img_indices[0].shape[0])

neibor_indices = []
mindis = 2e-2
for y in range(1, img_valids.shape[0]-1):
    for x in range(1, img_valids.shape[1]-1):
        if img_valids[y,x] == -1:
            continue
        neibor_ids = []
        center_idx = img_valids[y,x]
        neibor_ids.append(center_idx)
        for dy in {-1, 1}:
            for dx in {-1, 1}:
                cur_idx = img_valids[y+dy, x+dx]
                if cur_idx == -1:
                    neibor_ids.append(center_idx)
                else:
                    neibor_ids.append(cur_idx)
        neibor_indices.append(neibor_ids)
neibor_indices = np.array(neibor_indices)
print(neibor_indices.shape, (img_valids.shape[0]-2) * (img_valids.shape[1]-2), neibor_indices.shape[0] / ((img_valids.shape[0]) * (img_valids.shape[1])))


teeth_info = np.loadtxt('code/dmm_models/adnerf_rendering-256/teeth_tris.txt', dtype=np.float32)
teeth_info[:, 0] += (faces_num + teeth_faces_num)

tris_info_dict = torch.load('code/dmm_models/adnerf_rendering/wholebody_tris_info.pt')
tris_info_dict['smplx_tris'] = smplx_part[..., [3,0,1]].cpu()
tris_info_dict['teeth_tris'] = torch.from_numpy(teeth_info)
tris_info_dict['lap_indices'] = torch.from_numpy(neibor_indices).long()
torch.save(tris_info_dict,f'./wholebody_{uv_size}_tris_info.pt')

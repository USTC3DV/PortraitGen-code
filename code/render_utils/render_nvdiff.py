
import torch
import numpy as np
import torch.nn as nn
import nvdiffrast.torch as dr
from .geometry_utils import SH, compute_vertex_normals, get_ndc_proj_matrix
import ipdb

class MeshRenderer(nn.Module):
    def __init__(self):
        super().__init__()

        self.glctx = None

        init_lit = torch.zeros((1, 9, 1)).float()
        # init_lit[0, 0, 0] = .5
        # init_lit[0, 2, 0] = .3
        init_lit[0, 0, 0] = 1.7
        init_lit[0, 2, 0] = 1.2
        self.register_buffer('init_lit', init_lit.cuda())
        self.SH = SH()

    def compute_shaded_color(self, texture, normal, gamma, img_size):
        '''
        texture: (B, h, w, 3), normal: (B, h, w, 3), gamma: (B, 9*3), img_size: [h, w] -> shade_img: (b, h, w, 3)
        '''

        h, w = img_size
        gamma = gamma.reshape(-1, 9, 3) + self.init_lit
        normal = normal.reshape(-1, h*w, 3)
        texture = texture.reshape(-1, h*w, 3)
        a, c = self.SH.a, self.SH.c
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(normal[..., :1]),
            -a[1] * c[1] * normal[..., 1:2],
             a[1] * c[1] * normal[..., 2:],
            -a[1] * c[1] * normal[..., :1],
             a[2] * c[2] * normal[..., :1] * normal[..., 1:2],
            -a[2] * c[2] * normal[..., 1:2] * normal[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * normal[..., 2:] ** 2 - 1),
            -a[2] * c[2] * normal[..., :1] * normal[..., 2:],
            0.5 * a[2] * c[2] * (normal[..., :1] ** 2  - normal[..., 1:2] ** 2)
        ], dim=-1)
        color = torch.bmm(Y, gamma)*texture
        return color.reshape(-1, h, w, 3)
    

    def forward_visualization_geo(self, vertices_cam, tris, cam_para, img_size, img_ori = None, return_rast_out = False):
        '''
        vertices_cam: (b, nv, 3), tris: (b, nf, 3)
        cam_para: (b, 4), img_size: [h, w] -> vis_img: (b, h, w, 3)
        '''

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=vertices_cam.device, output_db=True)

        with torch.inference_mode():
            vertices_normals = compute_vertex_normals(vertices_cam, tris)
            ndc_proj_mat = get_ndc_proj_matrix(cam_para, img_size)
            vertices_homo = torch.cat((vertices_cam, torch.ones_like(vertices_cam[..., :1])), dim=-1)
            vertices_ndc = torch.bmm(vertices_homo, ndc_proj_mat.permute(0, 2, 1))
            rast_out , _ = dr.rasterize(self.glctx, vertices_ndc.contiguous(), tris[0], resolution=img_size)
            normals = dr.interpolate(vertices_normals, rast_out, tris[0])[0]
            normals = torch.nn.functional.normalize(normals, dim=-1) 
            colors = torch.ones_like(normals)
            colors[:,:,:,0]=0.8
            colors[:,:,:,1]=0.8
            colors[:,:,:,2]=1
            vis_img = self.compute_shaded_color(colors, normals, normals.new_zeros(normals.shape[0], 27), img_size)
            if img_ori is not None:
                mask =  (rast_out[..., 3] > 0).float().unsqueeze(-1)
                vis_img = (1 - mask) * img_ori + mask * (vis_img*.6 + img_ori*.4)
            else:
                mask =  (rast_out[..., 3] > 0).float().unsqueeze(-1)
                vis_img = (1 - mask) * torch.ones_like(vis_img) + mask * vis_img
        if not return_rast_out:
            return torch.clamp((vis_img*255.).detach().byte(), 0, 255).cpu().numpy()
        else:
            return torch.clamp((vis_img*255.).detach().byte(), 0, 255).cpu().numpy(), rast_out
        
    def forward_rasterization(self, vertices_cam, cam_para, tris, img_size):
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=vertices_cam.device, output_db=True)
        ndc_proj_mat = get_ndc_proj_matrix(cam_para, img_size)
        vertices_homo = torch.cat((vertices_cam, torch.ones_like(vertices_cam[..., :1])), dim=-1)
        vertices_ndc = torch.bmm(vertices_homo, ndc_proj_mat.permute(0, 2, 1))
        rast_out, _ = dr.rasterize(self.glctx, vertices_ndc.contiguous(), tris[0], resolution=img_size, grad_db=False)
        return rast_out
    
    def forward_differentiable_mask(self, vertices_cam, tris, cam_para, img_size):
        '''
        vertices_cam: (b, nv, 3), tris: (b, nf, 3), cam_para: (b, 4), img_size: [h, w]
        '''
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=vertices_cam.device, output_db=True)

        ndc_proj_mat = get_ndc_proj_matrix(cam_para, img_size)
        vertices_homo = torch.cat((vertices_cam, torch.ones_like(vertices_cam[..., :1])), dim=-1)
        vertices_ndc = torch.bmm(vertices_homo, ndc_proj_mat.permute(0, 2, 1))
        rast_out , rast_db = dr.rasterize(self.glctx, vertices_ndc.contiguous(), tris[0], resolution=img_size)
        vertices_mask = torch.ones_like(vertices_cam[..., :1])
        rendered_mask = dr.interpolate(vertices_mask, rast_out, tris[0], rast_db=rast_db, diff_attrs='all')[0]
        rendered_mask = dr.antialias(rendered_mask, rast_out, vertices_ndc.contiguous(), tris[0])
        return rendered_mask
    

    def rasterize_uv_img(self, uvs, tris_uv, uv_size):
        '''
        uvs: (nv, 2), tris_uv: (nf, 3)
        '''
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=uvs.device, output_db=True)
        uvs_ndc = torch.cat((uvs*2.-1., torch.ones_like(uvs)), dim=-1).unsqueeze(0)
        uvs_ndc[:, :, 2] = .9
        rast_out, _ = dr.rasterize(self.glctx, uvs_ndc.contiguous(), tris_uv, resolution=(uv_size, uv_size))
        return rast_out
    
    def forward_rendering_uv(self, vertices_cam, tris, tex, uvs, tris_uv, cam_para, img_size, lights = None, return_rast_out = False):
        '''
        vertices_cam: (b, nv, 3), tris: (b, nf, 3), tex: (b, u, v, d), lights: (b, 27), uvs: (b, nv, 2), tris_uv: (b, nf, 3)
        cam_para: (b, 4), img_size: [h, w] -> render_img: (b, h, w, d)
        '''

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=vertices_cam.device, output_db=True)
        
        ndc_proj_mat = get_ndc_proj_matrix(cam_para, img_size)
        vertices_homo = torch.cat((vertices_cam, torch.ones_like(vertices_cam[..., :1])), dim=-1)
        vertices_ndc = torch.bmm(vertices_homo, ndc_proj_mat.permute(0, 2, 1))
        rast_out , rast_db = dr.rasterize(self.glctx, vertices_ndc.contiguous(), tris[0], resolution=img_size)
        uv_out, uv_d = dr.interpolate(uvs[0], rast_out, tris_uv[0])
        uv_d = None
        # uv_out, uv_d = dr.interpolate(uvs[0], rast_out, tris_uv[0], rast_db, diff_attrs='all')
        tex_img = dr.texture(tex.contiguous(), uv_out, uv_d)
        if lights is not None:
            vertices_normals = compute_vertex_normals(vertices_cam, tris)
            normals = dr.interpolate(vertices_normals, rast_out, tris[0])[0]
            normals = torch.nn.functional.normalize(normals, dim=-1)
            tex_img = self.compute_shaded_color(tex_img, normals, lights, img_size)
        tex_img = dr.antialias(tex_img, rast_out, vertices_ndc, tris[0])
        if return_rast_out:
            return tex_img, rast_out
        else:
            return tex_img
        
    def forward_neural_rendering(self, vertices_cam, tris, cam_para, img_size, neural_texture, img_ori = None, return_rast_out = False):
        '''
        vertices_cam: (b, nv, 3), tris: (b, nf, 3), neural_texture: (b, nv, 3)
        cam_para: (b, 4), img_size: [h, w] -> vis_img: (b, h, w, 3)
        '''
        # ipdb.set_trace()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=vertices_cam.device, output_db=True)
            # self.glctx = dr.RasterizeCudaContext(device=vertices_cam.device)
        # ipdb.set_trace()
            
        vertices_normals = compute_vertex_normals(vertices_cam, tris)
        ndc_proj_mat = get_ndc_proj_matrix(cam_para, img_size)
        vertices_homo = torch.cat((vertices_cam, torch.ones_like(vertices_cam[..., :1])), dim=-1)
        vertices_ndc = torch.bmm(vertices_homo, ndc_proj_mat.permute(0, 2, 1))
        rast_out , _ = dr.rasterize(self.glctx, vertices_ndc.contiguous(), tris[0], resolution=img_size)
        
        
        rast_neural_texture = dr.interpolate(neural_texture.contiguous(), rast_out.contiguous(), tris[0].contiguous())[0]
        rast_neural_texture_anti = dr.antialias(rast_neural_texture, rast_out, vertices_ndc.contiguous(), tris[0])
        # ipdb.set_trace()
        
        vertices_mask = torch.ones_like(vertices_cam[..., :1])
        rendered_mask = dr.interpolate(vertices_mask, rast_out.contiguous(), tris[0].contiguous())[0]
        rendered_mask = dr.antialias(rendered_mask, rast_out, vertices_ndc.contiguous(), tris[0])

        if not return_rast_out:
            return rast_neural_texture_anti, rendered_mask
        else:
            return rast_neural_texture_anti, rendered_mask, rast_out
import torch
import math
from icecream import ic

from diff_gaussian_rasterization_fea import GaussianRasterizationSettings as GaussianRasterizationSettingsFea
from diff_gaussian_rasterization_fea import GaussianRasterizer as GaussianRasterizerFea

def render_fea(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, mask=None, use_fix_opacity=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    bg_color = torch.tensor([1.0] * 32).float().to(screenspace_points.device)
    raster_settings = GaussianRasterizationSettingsFea(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerFea(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if use_fix_opacity:
       opacity = torch.ones_like(pc.get_opacity,dtype=pc.get_opacity.dtype, requires_grad=True, device="cuda")*0.9

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = pc.get_neural_features

    if mask is not None:
        means3D = means3D[mask]
        means2D = means2D[mask]
        shs = shs[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]

    rendered_image, radii, _, rendered_mask = rasterizer(
        means3D = means3D,
        means2D = means2D,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return {"render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


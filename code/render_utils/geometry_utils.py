import numpy as np
import torch
from pytorch3d.ops import efficient_pnp
import math
import cv2

class SH:
    def __init__(self):
        # self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        # self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]
        self.a = [1.0, 2.0 / 3.0, 1.0 / 4.0]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


def compute_vertex_normals(vertices, faces):
    '''
    vertices: (b, nv, 3), faces: (b, nf, 3) -> normals: (b, nv, 3)
    '''
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = vertices.new_zeros((bs * nv, 3))
    ex_faces = (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    faces = faces + ex_faces 
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]
    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)
    normals.index_add_(0, faces[:, 1].long(), torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(), torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))
    normals = torch.nn.functional.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals

def get_ndc_proj_matrix(cam_paras, img_size, n=.1, f=50.):
    '''
    cam_paras: (b, 4), img_size: [h, w] -> ndc_proj_mat: (b, 4, 4)
    '''
    batch_size = cam_paras.shape[0]
    height, width = img_size
    ndc_proj_mat = cam_paras.new_zeros((batch_size, 4, 4))
    fx, fy, cx, cy = cam_paras[:, 0], cam_paras[:, 1], cam_paras[:, 2], cam_paras[:, 3]
    ndc_proj_mat[:, 0, 0] = 2*fx/(width-1)
    ndc_proj_mat[:, 0, 2] = 1-2*cx/(width-1)
    ndc_proj_mat[:, 1, 1] = -2*fy/(height-1)
    ndc_proj_mat[:, 1, 2] = 1-2*cy/(height-1)
    ndc_proj_mat[:, 2, 2] = -(f+n)/(f-n)
    ndc_proj_mat[:, 2, 3] = -(2*f*n)/(f-n)
    ndc_proj_mat[:, 3, 2] = -1.
    return ndc_proj_mat

def euler2rot(euler_angle):
    '''
    euler_angle: (b, 3) -> (b, 3, 3)
    '''

    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def unproj_pts(cam_para, proj_xys, Zs):
    '''
    cam_para: (b, 4), proj_xys: (b, n, 2), zs: (b, n) -> (b, n, 3)
    '''
    fx, fy, cx, cy = cam_para[:, 0:1], cam_para[:, 1:2], cam_para[:, 2:3], cam_para[:, 3:4]
    proj_x = proj_xys[..., 0]
    proj_y = proj_xys[..., 1]
    Xs = -(proj_x-cx)*Zs/fx
    Ys = (proj_y-cy)*Zs/fy
    return torch.stack((Xs, Ys, Zs), dim=-1)

def proj_pts(rott_geo, cam_para):
    '''
    rott_geo: (b, n, 3), cam_para: (b, 4) -> (b, n, 2)
    '''

    fx, fy, cx, cy = cam_para[:, 0:1], cam_para[:, 1:2], cam_para[:, 2:3], cam_para[:, 3:4]
    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]
    fxX = fx*X
    fyY = fy*Y
    proj_x = -fxX/Z + cx
    proj_y = fyY/Z + cy
    return torch.cat((proj_x[:, :, None], proj_y[:, :, None]), 2)

def rot_trans_pts(geometry, rot, trans):
    return torch.bmm(geometry, rot.permute(0,2,1)) + trans.unsqueeze(1)

def forward_rott(geometry, euler_angle, trans):
    '''
    geometry: (b, n, 3), euler_angle: (b, 3), trans: (b, 3) -> (b, n, 3)
    '''

    rot = euler2rot(euler_angle)
    return rot_trans_pts(geometry, rot, trans)

def forward_rott_proj(geometry, euler_angle, trans, cam_para):
    '''
    geometry: (b, n, 3), euler_angle: (b, 3), trans: (b, 3), cam_para: (b, 4) -> (b, n, 2)
    '''

    return proj_pts(forward_rott(geometry, euler_angle, trans), cam_para)

### note: this function reture much poorer results than solve_pnp_cv
def solve_pnp_mix(pts3d, pts2d, camera_para):
    '''
    pts3d: (b, n, 3), pts2d: (b, n, 2), camera_para: (b, 4) -> rot: (b, 3, 3), trans: (b, 1, 3)
    '''

    b, n, _ = pts3d.shape
    normalized_pts2d = pts2d.clone()
    normalized_pts2d[:, :, 0] = (normalized_pts2d[:, :, 0] - camera_para[:, 2:3]) / camera_para[:, 0:1]
    normalized_pts2d[:, :, 1] = (normalized_pts2d[:, :, 1] - camera_para[:, 3:]) / camera_para[:, 1:2]

    transformed_pts3d = pts3d.clone()
    transformed_pts3d[..., 1:] *= -1
    
    _, rots_epnp, ts_epnp, _, _ = efficient_pnp(transformed_pts3d, normalized_pts2d)

    rots, ts = [], []
    for i in range(b):
        camera_mat = np.array((camera_para[i, 0].item(), 0, camera_para[i, 2].item(), 0, camera_para[i, 1].item(), camera_para[i, 3].item(), 0, 0, 1), dtype=np.float32).reshape(3, 3)
        rot_epnp = rots_epnp[i].permute(1, 0).cpu().numpy()
        rvec_init = cv2.Rodrigues(rot_epnp)[0].reshape(3, 1)
        t_init = ts_epnp[i].cpu().numpy().reshape(3, 1)
        _, r_vec, t = cv2.solvePnP(transformed_pts3d[i].cpu().numpy(), pts2d[i].cpu().numpy(), camera_mat, np.zeros((4, 1), dtype=np.float32), rvec_init, t_init, True)
        rot, _ = cv2.Rodrigues(r_vec)
        rots.append(rot)
        ts.append(t.reshape(1, 3))
    rots = torch.as_tensor(np.array(rots), dtype=torch.float32, device=pts3d.device)
    ts = torch.as_tensor(np.array(ts), dtype=torch.float32, device=pts3d.device)
    rots[:, 1:] *= -1
    rots[..., 1:] *= -1
    ts [..., 1:] *= -1
    return rots, ts


def solve_pnp_cv(pts3d, pts2d, camera_para, confidence = None):
    '''
    pts3d: (b, n, 3), pts2d: (b, n, 2), camera_para: (b, 4), confidence: (b, n, 1) -> rot: (b, 3, 3), trans: (b, 1, 3)
    '''

    b, n, _ = pts3d.shape

    transformed_pts3d = pts3d.clone()
    transformed_pts3d[..., 1:] *= -1
    
    rots, ts = [], []
    for i in range(b):
        camera_mat = np.array((camera_para[i, 0].item(), 0, camera_para[i, 2].item(), 0, camera_para[i, 1].item(), camera_para[i, 3].item(), 0, 0, 1), dtype=np.float32).reshape(3, 3)
        rot_epnp = np.eye(3).astype(np.float32)
        rvec_init = cv2.Rodrigues(rot_epnp)[0].reshape(3, 1)
        t_init = np.array((0., 0., 1.5), dtype=np.float32).reshape(3, 1) ### assume init distance 1.5m
        if confidence is not None:
            valid_ids = confidence[i, :, 0].cpu().numpy() > 0.8
        else:
            valid_ids = np.ones(n, dtype=np.bool)
        _, r_vec, t = cv2.solvePnP(transformed_pts3d[i, valid_ids].cpu().numpy(), pts2d[i, valid_ids].cpu().numpy(), camera_mat, np.zeros((4, 1), dtype=np.float32), rvec_init, t_init, True)
        rot, _ = cv2.Rodrigues(r_vec)
        rots.append(rot)
        ts.append(t.reshape(1, 3))
    rots = torch.as_tensor(np.array(rots), dtype=torch.float32, device=pts3d.device)
    ts = torch.as_tensor(np.array(ts), dtype=torch.float32, device=pts3d.device)
    rots[:, 1:] *= -1
    rots[..., 1:] *= -1
    ts [..., 1:] *= -1
    return rots, ts

def bary_inerpolation(
    vertices,
    faces,
    lmk_faces_idx,
    lmk_bary_coords
):
    ''' General barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx(...), dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor BxL, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor BxLx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx(...), dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    ori_last_dims = vertices.shape[2:]
    vertices = vertices.reshape(batch_size, num_verts, -1)
    dim_feature = vertices.shape[-1]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.reshape(-1)).reshape(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).reshape(-1, 1, 1) * num_verts

    lmk_vertices = vertices.reshape(-1, dim_feature)[lmk_faces].reshape(
        batch_size, -1, 3, dim_feature)
    # print(lmk_vertices.shape, lmk_bary_coords.shape)
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks.reshape((batch_size, lmk_bary_coords.shape[1]) + ori_last_dims)

def getWorld2View(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def pts2obj(pts, filepath):
    with open(filepath, 'w') as f:
        for pt in pts:
            f.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))

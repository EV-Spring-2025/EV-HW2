from copy import deepcopy
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.constants import colormap
from src.utils import build_rotation, setup_camera, quat_mult
from diff_gaussian_rasterization import GaussianRasterizer


w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25
traj_length = 15


def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array(
        [
            [np.cos(ry), 0., -np.sin(ry), 0.],
            [0.,         1., 0.,          cam_height],
            [np.sin(ry), 0., np.cos(ry),  center_dist],
            [0.,         0., 0.,          1.]
        ]
    )
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, seg_as_col=False, remove_background=False):
    params = dict(np.load(f"./output/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': F.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if remove_background:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if remove_background:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    # TODO: Implement calculate_trajectories to visualize 3D motion trajectories of foreground points over time.
    #
    # Input:
    # – scene_data: a list of dictionaries, each containing 3D Gaussian data per frame (e.g., means3D)
    # – is_fg: a boolean mask indicating which points are foreground
    #
    # Output:
    # – A LineSet object representing trajectories over the past traj_length frames, with consistent color assignment per trajectory
    #
    # Steps:
    # 1. Subsample foreground points from each frame using traj_frac.
    # 2. Stack past traj_length points to form trajectories.
    # 3. Apply coloring and return a LineSet via make_lineset.
    return


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = GaussianRasterizer(raster_settings=cam)(**timestep_data)
    return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None, d_near=1.5, d_far=6):
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]

    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]

    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]

    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)

    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(seq, render_mode, line_type, remove_background):
    scene_data, is_fg = load_scene_data(seq, remove_background)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    w2c, k = init_camera()
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(render_mode == 'depth'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if line_type is not None:
        if line_type == 'trajectories':
            linesets = calculate_trajectories(scene_data, is_fg)
        elif line_type == 'rotations':
            linesets = calculate_rot_vec(scene_data, is_fg)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(
        cparams,
        allow_arbitrary=True,
    )

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    start_time = time.time()
    num_timesteps = len(scene_data)

    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        if line_type == 'trajectories':
            t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        else:
            t = int(passed_frames % num_timesteps)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if render_mode == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data[t]['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth = render(w2c, k, scene_data[t])
            pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(render_mode == 'depth'))
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if line_type is not None:
            if line_type == 'trajectories':
                lt = t - traj_length
            else:
                lt = t
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    visualize(
        config.data.sport_type,
        render_mode=config.visualize.render_mode,
        line_type=config.visualize.line_type,
        remove_background=config.visualize.remove_background,
    )

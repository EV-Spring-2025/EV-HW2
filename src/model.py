from pathlib import Path

import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import (
    build_rotation,
    inverse_sigmoid,
)


class Dynamic3DGaussiansModel:
    def __init__(self, data_dir, md):
        self.md = md
        self.init_pt_cld = np.load(Path(data_dir, "init_pt_cld.npz"))["data"]
        self.seg = self.init_pt_cld[:, 6]
        self.params = self._init_params()
        self.variables = self._init_variables()
        self.optimizer = self._initialize_optimizer()

    def _knn(self, pts, num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        indices = []
        sq_dists = []
        for p in pcd.points:
            _, i, d = pcd_tree.search_knn_vector_3d(p, num + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
        return np.array(sq_dists), np.array(indices)

    def _init_params(self):
        sq_dist, _ = self._knn(self.init_pt_cld[:, :3], 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=1e-7)
        max_cams = 50
        params_np = {
            'means3D': self.init_pt_cld[:, :3],
            'rgb_colors': self.init_pt_cld[:, 3:6],
            'seg_colors': np.stack((self.seg, np.zeros_like(self.seg), 1 - self.seg), -1),
            'unnorm_rotations': np.tile([1, 0, 0, 0], (self.seg.shape[0], 1)),
            'logit_opacities': np.zeros((self.seg.shape[0], 1)),
            'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
            'cam_m': np.zeros((max_cams, 3)),
            'cam_c': np.zeros((max_cams, 3)),
        }

        return {
            k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            for k, v in params_np.items()
        }

    def _initialize_optimizer(self):
        lrs = {
            'means3D': 0.00016 * self.variables['scene_radius'],
            'rgb_colors': 0.0025,
            'seg_colors': 0.0,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
            'cam_m': 1e-4,
            'cam_c': 1e-4,
        }
        param_groups = [
            {
                'params': [v],
                'name': k,
                'lr': lrs[k]
            }
            for k, v in self.params.items()
        ]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def _init_variables(self):
        cam_centers = np.linalg.inv(self.md['w2c'][0])[:, :3, 3]
        scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, axis=0)[None], axis=-1))
        N = self.params['means3D'].shape[0]
        return {
            'scene_radius': scene_radius,
            'max_2D_radius': torch.zeros(N, device="cuda"),
            'means2D_gradient_accum': torch.zeros(N, device="cuda"),
            'denom': torch.zeros(N, device="cuda"),
        }

    def initialize_per_timestep(self):
        new_params = self._compute_new_positions()
        self._store_previous_variables()
        self._update_optimizer_params(new_params)

    def _compute_new_positions(self):
        """
        Predicts the initial 3D position and rotation for each Gaussian
        at the current timestep using linear extrapolation from the previous two frames.
        """
        pts = self.params['means3D']  # position at t-1
        rot = F.normalize(self.params['unnorm_rotations'])  # rotation at t-1
        prev_pts = self.variables["prev_pts"]  # position at t-2
        prev_rot = self.variables["prev_rot"]  # rotation at t-2
    
        # TODO: implement linear extrapolation
        # Predict current mean and rotation based on previous motion
        # new_pts = ...
        # new_rot = ...    
        # return {
        #     'means3D': new_pts,
        #     'unnorm_rotations': F.normalize(new_rot)
        # }

    def _store_previous_variables(self):
        pts = self.params['means3D']
        rot = F.normalize(self.params['unnorm_rotations'])
        is_fg = self.params['seg_colors'][:, 0] > 0.5
        fg_pts = pts[is_fg]

        prev_inv_rot_fg = rot[is_fg]
        prev_inv_rot_fg[:, 1:] *= -1
        prev_offset = fg_pts[self.variables["neighbor_indices"]] - fg_pts[:, None]

        self.variables.update(
            {
                "prev_pts": pts.detach(),
                "prev_rot": rot.detach(),
                "prev_inv_rot_fg": prev_inv_rot_fg.detach(),
                "prev_offset": prev_offset.detach(),
                "prev_col": self.params['rgb_colors'].detach(),
            }
        )

    def initialize_post_first_timestep(self, num_knn=20):
        is_fg = self.params['seg_colors'][:, 0] > 0.5
        init_fg_pts = self.params['means3D'][is_fg]
        init_bg_pts = self.params['means3D'][~is_fg]
        init_bg_rot = F.normalize(self.params['unnorm_rotations'][~is_fg])

        # TODO: Use KNN to find the nearest neighbors of init_fg_pts,
        # then compute the corresponding neighbor distances and apply a Gaussian-like weighting using the squared distances.
        # neighbor_sq_dist, neighbor_indices = ...
        # neighbor_weight = ...
        # neighbor_dist = ...
        # self.variables.update(
        #     {
        #         "neighbor_indices": torch.tensor(neighbor_indices).cuda().long().contiguous(),
        #         "neighbor_weight": torch.tensor(neighbor_weight).cuda().float().contiguous(),
        #         "neighbor_dist": torch.tensor(neighbor_dist).cuda().float().contiguous(),
        #         "init_bg_pts": init_bg_pts.detach(),
        #         "init_bg_rot": init_bg_rot.detach(),
        #         "prev_pts": self.params['means3D'].detach(),
        #         "prev_rot": F.normalize(self.params['unnorm_rotations']).detach(),
        #     }
        # )

        params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in params_to_fix:
                param_group['lr'] = 0.0

    def _update_optimizer_params(self, new_params):
        for k, v in new_params.items():
            group = [x for x in self.optimizer.param_groups if x["name"] == k][0]
            stored_state = self.optimizer.state.pop(group["params"][0], None)
            if stored_state:
                stored_state["exp_avg"] = torch.zeros_like(v)
                stored_state["exp_avg_sq"] = torch.zeros_like(v)

            param = torch.nn.Parameter(v.requires_grad_(True))
            group["params"][0] = param
            if stored_state:
                self.optimizer.state[param] = stored_state
            self.params[k] = param

    def _cat_params_to_optimizer(self, new_params):
        for k, v in new_params.items():
            group = [g for g in self.optimizer.param_groups if g['name'] == k][0]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                self.params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                self.params[k] = group["params"][0]

    def _remove_points(self, to_remove):
        to_keep = ~to_remove
        keys = [k for k in self.params.keys() if k not in ['cam_m', 'cam_c']]
        for k in keys:
            group = [g for g in self.optimizer.param_groups if g['name'] == k][0]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
                del self.optimizer.state[group['params'][0]]

                group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                self.params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
                self.params[k] = group["params"][0]

        self.variables.update(
            {
                'means2D_gradient_accum': self.variables['means2D_gradient_accum'][to_keep],
                'denom': self.variables['denom'][to_keep],
                'max_2D_radius': self.variables['max_2D_radius'][to_keep]
            }
        )

    def _accumulate_mean2d_gradient(self):
        self.variables['means2D_gradient_accum'][self.variables['seen']] += torch.norm(
            self.variables['means2D'].grad[self.variables['seen'], :2],
            dim=-1,
        )
        self.variables['denom'][self.variables['seen']] += 1
    
    def _get_normalized_grads(self):
        grads = self.variables['means2D_gradient_accum'] / self.variables['denom']
        grads[grads.isnan()] = 0.0
        return grads

    def _get_clone_mask(self, grads, threshold=0.0002):
        max_scale = torch.exp(self.params['log_scales']).max(dim=1).values
        return torch.logical_and(
            grads >= threshold,
            max_scale <= 0.01 * self.variables['scene_radius'],
        )

    def _clone_points(self, mask):
        new_params = {k: v[mask] for k, v in self.params.items() if k not in ['cam_m', 'cam_c']}
        self._cat_params_to_optimizer(new_params)

    def _get_split_mask(self, grads, threshold=0.0002):
        num_pts = self.params['means3D'].shape[0]
        padded_grad = torch.zeros(num_pts, device="cuda")
        padded_grad[:grads.shape[0]] = grads
        max_scale = torch.exp(self.params['log_scales']).max(dim=1).values
        return torch.logical_and(
            padded_grad >= threshold,
            max_scale > 0.01 * self.variables['scene_radius'],
        )

    def _split_points(self, mask, n=2):
        new_params = {k: v[mask].repeat(n, 1) for k, v in self.params.items() if k not in ['cam_m', 'cam_c']}
        stds = torch.exp(self.params['log_scales'])[mask].repeat(n, 1)
        samples = torch.normal(mean=0.0, std=1.0, size=stds.shape, device="cuda") * stds
        rots = build_rotation(self.params['unnorm_rotations'][mask]).repeat(n, 1, 1)
        new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
        self._cat_params_to_optimizer(new_params)

    def _reset_accumulators(self):
        num_pts = self.params['means3D'].shape[0]
        self.variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
        self.variables['denom'] = torch.zeros(num_pts, device="cuda")
        self.variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

    def _get_removal_mask(self, i, threshold=None):
        if threshold is None:
            threshold = 0.25 if i == 5000 else 0.005
        return (torch.sigmoid(self.params['logit_opacities']) < threshold).squeeze()

    def _refresh_opacities(self):
        new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(self.params['logit_opacities']) * 0.01)}
        self._update_optimizer_params(new_params)

    def densify(self, i):
        if i > 5000:
            return

        # i <= 5000
        self._accumulate_mean2d_gradient()

        if i >= 500 and i % 100 == 0:
            n = 2

            grads = self._get_normalized_grads()

            to_clone = self._get_clone_mask(grads)
            self._clone_points(to_clone)

            to_split = self._get_split_mask(grads)
            self._split_points(to_split, n)
            self._reset_accumulators()
            
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            self._remove_points(to_remove)

            to_remove = self._get_removal_mask(i)

            if i >= 3000:
                big_points_ws = torch.exp(self.params['log_scales']).max(dim=1).values > 0.1 * self.variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)

            self._remove_points(to_remove)
            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            self._refresh_opacities()

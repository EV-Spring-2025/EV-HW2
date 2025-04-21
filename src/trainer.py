import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from src.data import get_dataset, get_batch, load_metadata
from src.loss import calc_loss, calc_psnr
from src.model import Dynamic3DGaussiansModel
from src.utils import (
    params2rendervar,
    params2cpu,
)
from diff_gaussian_rasterization import GaussianRasterizer


class Trainer:
    def __init__(self, data_dir, output_dir):
        self.md = load_metadata(Path(data_dir, "train_meta.json"))
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model = Dynamic3DGaussiansModel(data_dir, self.md)

    def train_single_timestep(self, t, every_i):
        dataset = get_dataset(t, self.md, self.data_dir)
        todo_dataset = []
        is_initial = (t == 0)
        if not is_initial:
            self.model.initialize_per_timestep()

        num_iters = 10000 if is_initial else 2000
        progress_bar = tqdm(range(num_iters), desc=f"Timestep {t}")
        for i in range(num_iters):
            curr_data = get_batch(todo_dataset, dataset)
            loss, self.model.variables = calc_loss(self.model.params, curr_data, self.model.variables, is_initial)
            loss.backward()

            with torch.no_grad():
                if is_initial:
                    self.model.densify(i)
                if i % every_i == 0:
                    self.report_progress(self.model.params, dataset[0], progress_bar, every_i=every_i)

                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)

        progress_bar.close()

        if is_initial:
            self.model.initialize_post_first_timestep()
        return params2cpu(self.model.params, is_initial)

    def report_progress(self, params, data, progress_bar, every_i=100):
        im, _, _, = GaussianRasterizer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"psnr": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)

    def save_params(self, output_params):
        to_save = {}
        for k in output_params[0].keys():
            if k in output_params[1].keys():
                to_save[k] = np.stack([params[k] for params in output_params])
            else:
                to_save[k] = output_params[0][k]
        os.makedirs(self.output_dir, exist_ok=True)
        np.savez(Path(self.output_dir, "params"), **to_save)

    def fit(self, every_i=100):
        os.makedirs(self.output_dir, exist_ok=True)
        num_timesteps = len(self.md['fn'])
        output_params = []
        for t in range(num_timesteps):
            out_param = self.train_single_timestep(t, every_i)
            output_params.append(out_param)

        self.save_params(output_params)
        torch.cuda.empty_cache()

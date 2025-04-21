import math

import torch
import torch.nn.functional as F

from src.utils import (
    build_rotation,
    quat_mult,
    params2rendervar,
)

from diff_gaussian_rasterization import GaussianRasterizer


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_mse(x, y):
    return ((x - y) ** 2).view(x.shape[0], -1).mean(dim=1, keepdim=True)


def calc_psnr(x, y):
    return 20 * torch.log10(1.0 / torch.sqrt(calc_mse(x, y)))


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def calc_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = GaussianRasterizer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = GaussianRasterizer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

        # TODO: Implement three loss terms to enforce geometric consistency:
        # – rigid: Encourage the current offset to match the previous offset using a weighted L2 loss.
        # – rot: Enforce local rotational consistency between neighboring relative rotations.
        # – iso: Encourage the magnitude of the current offset to match the neighbor distances for isotropy.
        # losses['rigid'] = weighted_l2_loss_v2()
        # losses['rot'] = weighted_l2_loss_v2()
        # losses['iso'] = weighted_l2_loss_v1()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {
        'im': 1.0,
        'seg': 3.0,
        'rigid': 4.0,
        'rot': 4.0,
        'iso': 2.0,
        'floor': 2.0,
        'bg': 20.0,
        'soft_col_cons': 0.01,
    }

    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['seen'] = seen
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])

    return loss, variables

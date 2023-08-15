import torch
from einops import rearrange
import torch.nn.functional as F

BLOCK = 8
EXPANSION = 4


def qindex_patch(a, b):
    eps = 1e-7
    E_a = torch.mean(a, dim=-1)
    E_a2 = torch.mean(a * a, dim=-1)
    E_b = torch.mean(b, dim=-1)
    E_b2 = torch.mean(b * b, dim=-1)
    E_ab = torch.mean(a * b, dim=-1)

    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return 4 * cov_ab * E_a * E_b / (var_a + var_b + eps) / (E_a ** 2 + E_b ** 2 + eps)


def qindex_torch(img1, img2, block):
    img1 = rearrange(img1, 'b (h h1) (w w1) -> b (h w) (h1 w1)', h1=block, w1=block)
    img2 = rearrange(img2, 'b (h h1) (w w1) -> b (h w) (h1 w1)', h1=block, w1=block)

    qindex = qindex_patch(img1, img2)
    return qindex.mean(1)


def D_lambda_torch(pr, ms):
    """

    Args:
        pr: predicted multi-spectral image. [b c h w]
        ms: low resolution multi-spectral image. [b c h//r w//r]

    Returns:

    """
    NC = pr.size(1)

    block = BLOCK
    expansion = EXPANSION

    d_lambda_list = []
    for i in range(NC):
        for j in range(i + 1, NC):
            band1 = pr[:, i]
            band2 = pr[:, j]
            q_hr = qindex_torch(band1, band2, block=block * expansion)

            band1 = ms[:, i]
            band2 = ms[:, j]
            q_lr = qindex_torch(band1, band2, block=block)

            diff = (q_hr - q_lr).abs()
            d_lambda_list.append(diff)

    d_lambda = torch.stack(d_lambda_list, -1)
    return d_lambda.mean(1)


def D_s_torch(pr, ms, pan):
    block = BLOCK
    expansion = EXPANSION

    NC = pr.size(1)
    pan_lr = F.interpolate(pan, scale_factor=(1 / expansion, 1 / expansion), mode='bicubic', align_corners=True)

    d_s_list = []
    for i in range(NC):
        band1 = pr[:, i]
        band2 = pan[:, 0]
        q_hr = qindex_torch(band1, band2, block=block * expansion)

        band1 = ms[:, i]
        band2 = pan_lr[:, 0]
        q_lr = qindex_torch(band1, band2, block=block)

        diff = (q_hr - q_lr).abs()
        d_s_list.append(diff)

    d_s = torch.stack(d_s_list, -1)
    return d_s.mean(1)


def no_ref_evaluate(pr, ms, pan):
    d_lambda = D_lambda_torch(pr, ms)
    d_s = D_s_torch(pr, ms, pan)
    qnr = (1 - d_lambda) * (1 - d_s)
    return [d_lambda, d_s, qnr]


if __name__ == '__main__':
    pr = torch.rand(10, 4, 256, 256) * 500
    ms = torch.rand(10, 4, 64, 64) * 500
    pan = torch.rand(10, 1, 256, 256) * 500

    metric_pr = no_ref_evaluate(pr, ms, pan)

    import full_res_PGMAN as pg

    metric_gt = pg.no_ref_evaluate(pr, ms, pan)

    for p, g in zip(metric_pr, metric_gt):
        diff = p - g
        print(diff)

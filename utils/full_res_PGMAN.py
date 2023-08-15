import torch
import torch.nn.functional as F
# from torch.nn.functional import in

def Q_torch(a, b):  # N x H x W
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))

    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return 4 * cov_ab * E_a * E_b / (var_a + var_b) / (E_a ** 2 + E_b ** 2)


def D_lambda_torch(ps, l_ms):  # N x C x H x W
    L = ps.shape[1]
    N = ps.size(0)
    total = torch.Tensor([0]*N).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        for j in range(L):
            if j != i:
                total += torch.abs(Q_torch(ps[:, i, :, :], ps[:, j, :, :]) - Q_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))

    return total / L / (L - 1)


def downsample(imgs, r=4):
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h // r, w // r], mode='bicubic', align_corners=True)


def D_s_torch(ps, l_ms, pan):  # N x C x H x W
    L = ps.shape[1]
    N = ps.size(0)
    l_pan = downsample(pan)

    total = torch.Tensor([0]*N).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        total += torch.abs(Q_torch(ps[:, i, :, :], pan[:, 0, :, :]) - Q_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))

    return total / L

def QNR_torch(ps, l_ms, pan):
    d_lambda = D_lambda_torch(ps, l_ms)
    d_s = D_s_torch(ps, l_ms, pan)
    qnr = (1-d_lambda)*(1-d_s)
    return qnr

def no_ref_evaluate(ps, l_ms, pan):
    # no reference metrics
    c_D_lambda = D_lambda_torch(ps, l_ms)
    c_D_s = D_s_torch(ps, l_ms, pan)
    c_qnr = QNR_torch(ps, l_ms, pan)
    return [c_D_lambda, c_D_s, c_qnr]


if __name__ == '__main__':
    ps = torch.randn(3, 4, 256, 256)
    pan = torch.randn(3, 1, 256, 256)
    lr = torch.randn(3, 4, 64, 64)

    print(Q_torch(ps[:, 0], pan[:, 0]))

    print(D_s_torch(ps, lr, pan))
    print(D_lambda_torch(ps, lr))

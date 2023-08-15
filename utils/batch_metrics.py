import torch
import numpy as np

eps = 1e-10


# Cross-correlation matrix
def batch_cross_correlation(H_fuse, H_ref):
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, -1, keepdim=True)
    mean_ref = torch.mean(H_ref_reshaped, -1, keepdim=True)

    CC = torch.sum((H_fuse_reshaped - mean_fuse) * (H_ref_reshaped - mean_ref), -1) / torch.sqrt(
        torch.sum((H_fuse_reshaped - mean_fuse) ** 2, -1) * torch.sum((H_ref_reshaped - mean_ref) ** 2, -1))

    CC = torch.nansum(CC, dim=-1) / N_spectral
    CC = CC.sum()
    return CC


# Spectral-Angle-Mapper (SAM)
def batch_SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)
    N_pixels = H_fuse_reshaped.size(-1)

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse_reshaped * H_ref_reshaped, 1)
    fuse_norm = torch.nansum(H_fuse_reshaped ** 2, dim=1).sqrt()
    ref_norm = torch.nansum(H_ref_reshaped ** 2, dim=1).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm)), dim=1) / N_pixels)
    return SAM.sum()


# Root-Mean-Squared Error (RMSE)
def batch_RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    batch = H_fuse.size(0)
    H_fuse_reshaped = H_fuse.view(batch, -1)
    H_ref_reshaped = H_ref.view(batch, -1)

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / H_fuse_reshaped.size(1))
    return RMSE.sum()


# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def batch_ERGAS(H_fuse, H_ref, beta):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)
    N_pixels = H_fuse_reshaped.size(-1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=-1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / beta) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2, dim=-1) / N_spectral)
    return ERGAS.sum()


# Peak SNR (PSNR)
def batch_PSNR(H_fuse, H_ref, Bit, reduction="sum"):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / H_fuse_reshaped.size(-1))

    # Calculating max pixel
    Max_pixel = 2**Bit - 1

    # Calculating PSNR
    PSNR = torch.nansum(10 * torch.log10(torch.div(Max_pixel, rmse) ** 2), dim=1) / N_spectral

    if reduction == "sum":
        return PSNR.sum()
    elif reduction == "none":
        return PSNR
    else:
        raise NotImplementedError("No such fucking reduction method")

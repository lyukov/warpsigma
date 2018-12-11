import numpy as np
from scipy.ndimage.filters import gaussian_gradient_magnitude

def get_energy(img, sigma=3.0):
    return gaussian_gradient_magnitude(img, sigma)

def get_cumulative_energy(img, sigma=3.0):
    energy = get_energy(img, sigma)
    return energy.cumsum(axis=0).cumsum(axis=1)

# Energy of subimage [r1, r2) x [c1, c2)
def get_energy_of_subimage(cum_energy, r1, c1, r2, c2):
    enBig = cum_energy[r2 - 1, c2 - 1]
    enSmall = 0
    if r1 > 0 and c1 > 0:
        enSmall = cum_energy[r1 - 1, c1 - 1]
    enLeft = 0
    if c1 > 0:
        enLeft = cum_energy[r2 - 1, c1 - 1]
    enUp = 0
    if r1 > 0:
        enUp = cum_energy[r1 - 1, c2 -1]
    return enBig + enSmall - enLeft - enUp

# (i, j) element have an energy value of subimage [i*s, i*s + H] x [j*s, j*s + W]
def get_patch_energy_map(img, W, H, s=3, sigma=3.0):
    cum_energy = get_cumulative_energy(img, sigma)
    enBig   = cum_energy[H - 1::s, W - 1::s]
    enSmall = cum_energy[:-(H - 1):s, :-(W - 1):s]
    enLeft  = cum_energy[H - 1::s, :-(W - 1):s]
    enUp    = cum_energy[:-(H - 1):s, W - 1::s]
    return enBig + enSmall - enLeft - enUp

# (i, j) element have an energy value of subimage [i, i + H] x [j, j + W]
def get_patch_map(img, W, H, s=3):
    cumimg  = img.cumsum(axis=0).cumsum(axis=1)
    enBig   = cumimg[H - 1::s, W - 1::s]
    enSmall = cumimg[:-(H - 1):s, :-(W - 1):s]
    enLeft  = cumimg[H - 1::s, :-(W - 1):s]
    enUp    = cumimg[:-(H - 1):s, W - 1::s]
    return enBig + enSmall - enLeft - enUp

# Yields top left coords of patches
def create_patch_coords_generator(ref_img, W, H, ratio=0.05, step=3, sigma=3.0):
    pem = get_patch_energy_map(ref_img, W, H, step, sigma)
    pem_flat = pem.flatten()
    pem_flat.sort()
    thrs = pem_flat[-int(len(pem_flat) * ratio)]
    for i in range(pem.shape[0]):
        for j in range(pem.shape[1]):
            if pem[i, j] < thrs:
                continue
            yield (i * step, j * step)

# Yields top left coords of patches
def create_patch_coords_generator_from_mse_dispersion(ref_img, rest_imgs,
        W, H, ratio=0.05, step=3):
    squared_error = (rest_imgs - ref_img) ** 2
    res = []
    for img in squared_error:
        res.append(get_patch_map(img, 31, 31, 3))
    mse_deviation = np.array(res).std(axis=0)
    pem_flat = mse_deviation.flatten()
    pem_flat.sort()
    thrs = pem_flat[-int(len(pem_flat) * ratio)]
    for i in range(mse_deviation.shape[0]):
        for j in range(mse_deviation.shape[1]):
            if mse_deviation[i, j] < thrs:
                continue
            yield (i * step, j * step)
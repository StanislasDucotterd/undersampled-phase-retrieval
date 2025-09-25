import os
import math
import time
import argparse
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from deepinv.models import DRUNet
from deepinv.models.complex import to_complex_denoiser
from phase_retrieval_operators import RandomPhaseRetrieval
from models.optimization import transform, inverse, correct_global_phase, cosine_similarity
torch.set_grad_enabled(False)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

# Parameter of the experiment
exp = 1
model_name = 'mfoe' 
# noiseless experiments
sampling_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_sigmas = [0.0]
# noisy experiments
sampling_ratios = [1.0]
n_sigmas = [0.001, 0.01, 0.1, 1.0]

nb_layers = 1

n_channels = 1
denoiser = DRUNet(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="download",  
    device=device,
)
denoiser_complex = to_complex_denoiser(denoiser, mode="abs_angle")

psnr = PSNR(data_range=1.).to(device)
data_folder = 'test_data'
images = os.listdir(data_folder)

diags = torch.load(os.path.join('saved_operators', data_folder, f'diffuser{exp}_{nb_layers}_layer.pth'), weights_only=True)
img_shape = diags[0].shape
physics = RandomPhaseRetrieval(
    p=1.0,
    img_shape=img_shape,
    nb_layers=nb_layers
)
physics.diags = diags

for ratio in sampling_ratios:
    mask = torch.load(os.path.join('saved_operators', data_folder, f'mask_{ratio}.pth'), weights_only=True)
    physics.mask = mask
    physics = physics.to(device)
    for n_sigma in n_sigmas:
        filename = f'{model_name}{exp}_{ratio}_{nb_layers}_layer'
        if n_sigma > 0: filename += f'_sigma_{n_sigma}'
        filename += '.pth'

        mean_psnr = 0.
        mean_cosines = 0.

        start = time.time()
        for image in images:
            img = torch.load(os.path.join(data_folder, image, 'img.pth'), weights_only=True).to(device)
            flipped = img.shape[2] != 321
            if flipped:
                img = img.permute(0, 1, 3, 2)
            if img.shape[2] == 512 :
                img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            img_phase = torch.exp(1j * (img - 0.5) * torch.pi)
            x_init = torch.randn_like(img_phase)
            y = physics(img_phase)
            y = torch.clip(y + n_sigma * torch.randn_like(y), min=0.)

            x = x_init.clone()
            K = 1000
            final_sigma = 0.001 / ratio
            first_sigma = 1.0
            lamb = 0.23

            sigma = 1.0
            log_sigmas = torch.linspace(math.log(first_sigma), math.log(final_sigma), K)

            for k in range(K):
                alpha = lamb * n_sigma**2 / torch.exp(log_sigmas[k])**2
                x = transform(x, k)
                x = denoiser_complex(x, torch.exp(log_sigmas[k]))
                x = inverse(x, k)
                x = physics.prox(x, y, alpha)

            mean_cosines += cosine_similarity(x, img_phase).item() / len(images)
            x = correct_global_phase(x, img_phase)
            x = torch.angle(x) / torch.pi + 0.5
            mean_psnr += psnr(x, img).item() / len(images)
            if flipped:
                x = x.permute(0, 1, 3, 2)
            torch.save(x.cpu(), os.path.join(data_folder, image, filename))

        print('Model:', model_name, 'Sampling ratio:', ratio, 'Noise level:', n_sigma)
        print(f'Total time: {time.time() - start:.2f} seconds')
        print(f'Mean cosine similarity: {mean_cosines:.4f}')
        print(f'Mean PSNR: {mean_psnr:.2f} dB')

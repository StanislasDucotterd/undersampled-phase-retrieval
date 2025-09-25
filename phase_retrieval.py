import os
import time
import argparse
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from models.mfoe import MFoE
from models.moreau_tv import MoreauTV
from models.optimization import APGDR, correct_global_phase, cosine_similarity
from phase_retrieval_operators import RandomPhaseRetrieval
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

# Load the model for the Regularizer
if model_name == 'mfoe':
    infos = torch.load('MFoE_weights.pth', map_location='cpu', weights_only=True)
    config = infos['config']
    model = MFoE(param_model=config['model_params'], param_multi_conv=config['multi_convolution'],
                param_fw=config['optimization']['fixed_point_solver_fw_params'],
                param_bw=config['optimization']['fixed_point_solver_bw_params'])
    model.load_state_dict(infos['state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    model.conv_layer.spectral_norm(mode="power_method", n_steps=500)
    model.lamb.data = torch.log(torch.tensor(1e4, device=device))
elif model_name == 'tv':
    model = MoreauTV(lamb=torch.log(torch.tensor(1e3)))
    model = model.to(device)
    model.eval()
else:
    raise ValueError('Unknown model')

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

        if ratio > 0.25:
            p = 0.25 / ratio
            aux_mask = mask * torch.bernoulli(p * torch.ones_like(mask))

        mean_psnr = 0.
        mean_cosines = 0.

        start = time.time()
        for image in images:
            img = torch.load(os.path.join(data_folder, image, 'img.pth'), weights_only=True).to(device)
            flipped = img.shape[2] != 321 and data_folder == 'test_data'
            if flipped:
                img = img.permute(0, 1, 3, 2)
            if img.shape[2] == 512 :
                img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
                
            img_phase = torch.exp(1j * (img - 0.5) * torch.pi)
            x_init = torch.randn_like(img_phase)
            y = physics(img_phase)
            y = torch.clip(y + n_sigma * torch.randn_like(y), min=0.)

            x = x_init.clone()
            sigma = 1.0 * torch.ones(1, 1, 1, 1, device=device)
            for k in range(3):
                if k == 0 and model_name == 'mfoe' and ratio > 0.25: physics.mask = aux_mask.to(device)
                x = APGDR(x, y, model, sigma, physics, n_sigma, max_iter=10000, tol=1e-5)
                if k == 0 and model_name == 'mfoe' and ratio > 0.25: physics.mask = mask.to(device)
                sigma = sigma * 0.25

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

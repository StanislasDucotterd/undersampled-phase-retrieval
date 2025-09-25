# Undersampled Phase Retrieval with Image Priors
Implementation of experiments done in : https://arxiv.org/abs/2509.15026v1

![alt text](https://github.com/StanislasDucotterd/undersampled-phase-retrieval/blob/main/phase_retrieval_results.png?raw=true)

#### Description
Phase retrieval seeks to recover a complex signal from amplitude-only measurements, a challenging nonlinear inverse problem. Current theory and algorithms often ignore signal priors. By contrast, we evaluate here a variety of image priors in the context of severe undersampling with structured random Fourier measurements. Our results show that those priors significantly improve reconstruction, allowing accurate reconstruction even below the weak recovery threshold.

#### Requirements
The required packages:
- `pytorch`
- `deepinv` (to load DRUNet and its parameters)

#### Training

You can run the reconstruction code with TV or MFoE using

```bash
python phase_retrieval.py --device cpu or cuda:n
```

and with DRUNet using

```bash
python phase_retrieval_drunet.py --device cpu or cuda:n
```
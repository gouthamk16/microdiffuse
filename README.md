## MicroDiffuse: A Single-File Diffusion Model from Scratch
A ~400 line pure Python script (no dependencies) that implements a denoising diffusion probabilistic model (DDPM) from scratch. Analogous to Karpathy's microgpt: the complete diffusion algorithm in one readable file.

### Design Philosophy
Like microgpt, we strip diffusion to its bare algorithmic core:
- Sub 400 lines of pure Python code
- Scalar autograd (reuse the same Value class approach)
- A tiny neural network that learns to denoise
- DDPM forward (noising) and reverse (denoising) processes
- Generates new samples by iteratively denoising pure noise

### Data
Instead of images (which need large models and pixel grids), we generate 2D point clouds — points sampled from a learned distribution. This is the simplest possible continuous data domain for diffusion:
- Dataset: Points arranged in simple geometric patterns (e.g. a circle, a spiral, concentric rings). Procedurally generated.
- Output: After training, the model turns random Gaussian noise into points that match the training distribution.
This is the 2D equivalent of what Stable Diffusion does in high-dimensional pixel space.
# Deep Models

Public objects from `aberrant.model.deep`:

- `Autoencoder`
- `OnlineAutoencoderEnsemble`

Notes:
- `Autoencoder` requires `torch` (`aberrant[dl]`).
- `OnlineAutoencoderEnsemble` uses an online, phased warm-up (`feature_map_warmup`,
  `detector_warmup`, `ready`) and returns a continuous anomaly score.
- `OnlineAutoencoderEnsemble` is inspired by KitNET but does not reproduce the
  author's normalized denoising-autoencoder implementation.
- Seeded PyTorch architectures initialize through model-owned generators and do
  not replace the process-wide PyTorch RNG state.

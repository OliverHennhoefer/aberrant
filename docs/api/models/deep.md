# Deep Models

Public objects from `aberrant.model.deep`:

- `Autoencoder`
- `OnlineAutoencoderEnsemble`

Notes:
- `Autoencoder` requires `torch` (`aberrant[dl]`).
- `OnlineAutoencoderEnsemble` uses an online, phased warm-up (`feature_map_warmup`,
  `detector_warmup`, `ready`) and returns a continuous anomaly score.
- `KitNET` remains a compatibility alias; this lightweight ensemble does not
  reproduce the author's normalized denoising-autoencoder implementation.

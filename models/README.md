# CNN-SWS supplied model weights

`CNN_XKS_numpy_weights.npz` is a lossless float32 tensor export of the
pre-trained `CNN_XKS.h5` model supplied by the authors of
[YW-Zhang94/CNN-SWS](https://github.com/YW-Zhang94/CNN-SWS). It is included so
inference can use the official weights with NumPy alone, without TensorFlow.
The exporter is `../export_cnn_numpy_weights.py`.

The inference script mirrors the official Keras `load/2_load/load.py`
architecture: nine channels-last Conv1D layers with stride 2, the same five
zero-padding layers, ReLU activations, Flatten, and a two-class softmax dense
layer. The four input columns are original radial/transverse and corrected
radial/transverse waveforms. No amplitude normalization is applied, matching
the upstream loader.

Source model citation: Zhang & Gao (2022), *Geophysical Research Letters*,
49, e2021GL097101, https://doi.org/10.1029/2021GL097101.

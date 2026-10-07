# Third-party notices

`models/wan_encdec.py` is a modified, bidirectional 1D adaptation based on the
Wan2.2 VAE architecture from the Alibaba Wan Team:

- Source: https://github.com/Wan-Video/Wan2.2
- Upstream license: Apache License 2.0
- Included license text: `THIRD_PARTY_LICENSES/WAN2.2_LICENSE.txt`
- Canonical license URL: https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt

The adaptation changes the dimensionality and convolution behavior for motion
sequences and does not include Wan2.2 model weights.

PyTorch, NumPy, SciPy, PyYAML, EasyDict, and tqdm remain subject to their own
licenses. This notice does not replace the need to add a license for the
InterPet4D project itself.

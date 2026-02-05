# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from torchvision import transforms
try:
    from torch_utils import persistence
except ImportError:
    from .torch_utils import persistence


warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass
#----------------------------------------------------------------------------
# Pre-trained InVAE encoder.

@persistence.persistent_class
class InvaeEncoder(Encoder):
    def __init__(self,
        vae_name    = 'REPA-E/e2e-invae',  # Name of the VAE to use.
        batch_size  = 8,                    # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_invae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        # invae.encode() now returns sampled latents directly
        return self._vae.encode(x).sample()

    def encode(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 127.5 - 1
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

#----------------------------------------------------------------------------
# Pre-trained VAVAE encoder.

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量 (0-1范围)
])

@persistence.persistent_class
class VavaeEncoder(Encoder):
    def __init__(self,
        config_path = None,                      # Path to VAVAE config yaml file.
        batch_size  = 8,                         # Batch size to use when running the VAE.
        img_size    = 256,                       # Image size for preprocessing.
    ):
        super().__init__()
        self.config_path = config_path
        self.batch_size = int(batch_size)
        self.img_size = img_size
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_vavae(self.config_path, device=device)
        else:
            self._vae.model.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def encode(self, x): # raw pixels => raw latents
        """Encode images to latent representations using VA_VAE.encode_images().
        
        Args:
            x: Input image tensor [N, C, H, W], uint8, 0-255
        
        Returns:
            torch.Tensor: Encoded latent representation
        """
        self.init(x.device)
        # VA_VAE.encode_images expects images normalized to [-1, 1]
        # x = x.to(torch.float32) / 255
        # x = transform(x)
        # x = x.to(torch.float32)
        x = x.to(torch.float32) / 127.5 - 1
        # Use VA_VAE's encode_images method which handles batching internally
        latents = self._vae.encode_images(x)
        return latents

#----------------------------------------------------------------------------

def load_invae(vae_name="REPA-E/e2e-invae", device=torch.device('cpu')):
    import os, sys
    try:
        # If encoders.py is imported as part of a package (e.g., REG.preprocessing.encoders)
        from ..models.invae import VAE_F16D32
    except ImportError:
        try:
            # If running with project root on sys.path
            from models.invae import VAE_F16D32
        except ImportError:
            # Fallback: add project root (one level up from this file) to sys.path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from models.invae import VAE_F16D32

    # Get cache dir from hf
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
    # Download https://huggingface.co/REPA-E/e2e-invae/resolve/main/e2e-invae-400k.pt to cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    vae_path = os.path.join(cache_dir, "e2e-invae-400k.pt")
    if not os.path.exists(vae_path):
        import requests
        url = "https://huggingface.co/REPA-E/e2e-invae/resolve/main/e2e-invae-400k.pt"
        r = requests.get(url)
        with open(vae_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {vae_path}")

    vae = VAE_F16D32().to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    return vae.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------

def load_vavae(config_path, device=torch.device('cpu')):
    """Load VAVAE (Vision Aligned VAE) model using VA_VAE class from models/vavae.py.
    
    Args:
        config_path: Path to VAVAE config yaml file (e.g., 'tokenizer/configs/vavae_f16d32_vfdinov2.yaml').
        device: Device to load the model on.
    
    Returns:
        VA_VAE instance with model in eval mode.
    """
    import os, sys
    try:
        # If encoders.py is imported as part of a package
        from ..models.vavae import VA_VAE
    except ImportError:
        try:
            # If running with project root on sys.path
            from models.vavae import VA_VAE
        except ImportError:
            # Fallback: add project root (one level up from this file) to sys.path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from models.vavae import VA_VAE

    if config_path is None:
        raise ValueError("config_path is required for loading VAVAE. "
                        "Please provide path to VAVAE config yaml file.")
    
    # Load VA_VAE using the config file
    # VA_VAE.__init__ automatically calls load() which sets model to eval mode
    vavae = VA_VAE(config=config_path)
    # Move model to specified device and disable gradients
    vavae.model = vavae.model.to(device)
    vavae.model.requires_grad_(False)
    return vavae

#----------------------------------------------------------------------------
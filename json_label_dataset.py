import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class JsonLabelDataset(Dataset):
    """Dataset that reads text captions from a JSON file with format [{"id":"", "en":""}].
    
    The label JSON file should have the following structure (JSON array):
    [
        {"id": "data_000000/flux_512_100k_00000014", "en": "A small sailboat with a red sail..."},
        {"id": "data_000000/flux_512_100k_00000015", "en": "A beautiful sunset over the ocean..."},
        ...
    ]
    
    Where "id" is the file path/name (without extension) and "en" is the English text caption.
    
    Args:
        data_dir: Root directory containing 'images' and 'vae-in' subdirectories
        label_file: Path to the JSON file containing labels (default: 'labels.json' in data_dir)
    """
    
    def __init__(self, data_dir, label_file=None):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-in')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )

        # Load labels from JSON file with format [{"id":"", "en":""}]
        if label_file is None:
            label_file = os.path.join(data_dir, 'labels.json')
        
        if os.path.exists(label_file):
            print(f"Using label file: {label_file}")
        else:
            raise FileNotFoundError(f"Label file not found: {label_file}")

        # Parse JSON array format
        with open(label_file, 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        
        # Convert list format [{"id":"", "en":""}] to dict {id: en}
        label_dict = {item['id']: item['en'] for item in label_list}
        
        # Extract labels for each feature file based on file id
        self.captions = []
        for fname in self.feature_fnames:
            # Extract id from filename (e.g., "data_000000/img-latents-flux_512_100k_00000014.npy" 
            # -> "data_000000/flux_512_100k_00000014")
            file_id = self._extract_id(fname)
            if file_id in label_dict:
                self.captions.append(label_dict[file_id])
            else:
                raise KeyError(f"Caption not found for file id: {file_id} (from {fname})")

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()
    
    def _extract_id(self, fname):
        """Extract id from filename.
        
        Examples:
            "data_000000/img-latents-flux_512_100k_00000014.npy" -> "data_000000/flux_512_100k_00000014"
            "data_000000/flux_512_100k_00000014.png" -> "data_000000/flux_512_100k_00000014"
        """
        # Remove extension
        fname_no_ext = os.path.splitext(fname)[0]
        # Replace backslashes with forward slashes for consistency
        fname_no_ext = fname_no_ext.replace('\\', '/')
        # Remove "img-latents-" prefix if present
        parts = fname_no_ext.split('/')
        if len(parts) >= 2:
            # Handle case: "data_000000/img-latents-flux_512_100k_00000014"
            basename = parts[-1]
            if basename.startswith('img-latents-'):
                basename = basename[len('img-latents-'):]
            elif basename.startswith('img'):
                basename = basename[3:]  # Remove 'img' prefix
            parts[-1] = basename
            return '/'.join(parts)
        return fname_no_ext

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            f"Number of feature files and label files should be same. " \
            f"image_fnames: {len(self.image_fnames)}, feature_fnames: {len(self.feature_fnames)}"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        caption = self.captions[idx]
        return torch.from_numpy(image), torch.from_numpy(features), caption

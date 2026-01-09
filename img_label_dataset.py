import os
import json
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from safetensors import safe_open
from torchvision import transforms


class ImgLabelDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        json_path: str,
        image_dir: str,
        transform=None,
    ):
        self.data_dir = data_dir
        self.json_path = json_path
        self.image_dir = image_dir
        self.transform = transform

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.img_ids = [item["img_id"] for item in json_data]  # list of str

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.index_to_file_info = self._build_index_mapping()

        if len(self.index_to_file_info) > len(self.img_ids):
            self.index_to_file_info = self.index_to_file_info[:len(self.img_ids)]
        elif len(self.index_to_file_info) < len(self.img_ids):
            raise ValueError("safetensors has fewer samples than JSON!")

        assert len(self.index_to_file_info) == len(self.img_ids)

    def _build_index_mapping(self):
        mapping = []
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels_shape = f.get_slice('labels').get_shape()
                num_imgs = labels_shape[0]
                for i in range(num_imgs):
                    mapping.append({
                        'safe_file': safe_file,
                        'idx_in_file': i
                    })
        return mapping

    def __len__(self):
        return len(self.index_to_file_info)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        info = self.index_to_file_info[idx]
        with safe_open(info['safe_file'], framework="pt", device="cpu") as f:
            label_slice = f.get_slice('labels')
            label = label_slice[info['idx_in_file']:info['idx_in_file']+1].squeeze(0)  # scalar tensor

        return image, label

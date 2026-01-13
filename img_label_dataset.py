import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from tokenizer.vavae import VA_VAE

from typing import Any, Callable, Dict, List, Optional, Union
from modelscope import CLIPModel, CLIPProcessor, CLIPConfig
from torchvision import transforms
from datasets import load_dataset

from torch.utils.data import DataLoader

def get_clip_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = 0,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # if isinstance(self, TextualInversionLoaderMixin):
    #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    # untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
    #    removed_text = tokenizer.batch_decode(untruncated_ids[:, 77 - 1: -1])
    #    print("The following part of your input was truncated because CLIP can only handle sequences up to"
    #        f" {77} tokens: {removed_text}")
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(device=device)
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量 (0-1范围)
])

# 定义转换函数
def transform_fn(sample):
    # 处理图像
    image = sample['jpg']
    if hasattr(image, 'resize'):
        image = image.resize((256, 256))
    x = transform(image)
    
    # 获取prompt
    prompt = sample.get('json', {}).get('prompt', "")
    
    return {'image': x, 'prompt': prompt}

def collate_fn(batch):
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    
    # 如果图像已经是tensor，可以直接stack
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    
    return images, prompts

def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # local
    rank = 0
    device = 0
    world_size = 1
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0],
                              f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    tokenizer = VA_VAE(
        args.config
    )

    # CLIP
    model_id = "AI-ModelScope/CLIP-GmP-ViT-L-14"
    clip_config = CLIPConfig.from_pretrained(model_id)
    max_tokens = 77
    clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32, config=clip_config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=max_tokens,
                                                   return_tensors="pt", truncation=True)

    base_url = "/root/lanyun-tmp/dataset/data_{i:06d}.tar"
    num_shards = 1  # Number of webdataset tar files
    urls = [base_url.format(i=i) for i in range(num_shards)]
    print(urls)
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)

    dataset = dataset.map(transform_fn)

    flip_transform = transforms.RandomHorizontalFlip(p=1.0)

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    for batch_idx, (images, prompts) in enumerate(dataloader):
        run_images += images.shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} images')
    
        # 处理批次数据
        x = images
        x1 = flip_transform(x) if flip_transform else x
    
        z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

        # print(prompts)
        y = get_clip_prompt_embeds(clip_processor.tokenizer, clip_model.text_model, prompts).detach().cpu()
        z1 = tokenizer.encode_images(x1).detach().cpu()

        latents.append(z)
        labels.append(y)
        latents_flip.append(z1)

        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}',
                          'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')

            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/your/output")
    parser.add_argument("--config", type=str, default="config_details.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)

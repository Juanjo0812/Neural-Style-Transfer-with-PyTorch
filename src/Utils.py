import torch
from torchvision import transforms
import numpy as np
from PIL import Image


# Upload content and style images
def load_image(img_path, max_size=400, shape=None, device="cpu"):
    image = Image.open(img_path).convert("RGB")

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)


# Invert image transformation to show
def im_convert(tensor):
    image = tensor.detach().cpu().clone().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

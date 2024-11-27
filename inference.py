from uuid import uuid4

import torch
from torch import Tensor

from config import CONFIG
from model import DiffusionReverseProcess
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def generate(cfg: CONFIG) -> Tensor:
    """
    Generate Image using trained model.
    :param cfg: config
    :return: image tensor
    """

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # print(f'Device: {device}\n')

    # Initialize Diffusion Reverse Process
    drp = DiffusionReverseProcess()

    # Set model to eval mode
    model = torch.load(cfg.model_path).to(device)
    model.eval()

    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

    # Convert the image to proper scale
    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2

    return xt


def generate_and_show_random_images(cfg: CONFIG):
    """
    Generate and show random images using trained model.
    :param cfg: config
    :return: None
    """


    # Generate
    generated_imgs = []
    for _ in tqdm(range(cfg.num_img_to_generate)):
        xt = generate(cfg)
        xt = 255 * xt[0].numpy()
        generated_imgs.append(xt.astype(np.uint8))

    fig, axes = plt.subplots(2, 6, figsize=(6, 2))

    # Plot each image in the corresponding subplot
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(generated_imgs[i], (1, 2, 0)))
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(f'result_{uuid4()}.png')
    plt.show()
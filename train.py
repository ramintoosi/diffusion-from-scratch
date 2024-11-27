import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data import AnimeDataset
from model import Unet, DiffusionForwardProcess


def train(cfg):
    # Dataset and Dataloader
    mnist_ds = AnimeDataset()
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Initiate Model
    if os.path.isfile(cfg.model_path):
        print('load checkpoint')
        model = torch.load(cfg.model_path).to(device)
    else:
        model = Unet(im_channels=3).to(device)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    # Diffusion Forward Process to add noise
    dfp = DiffusionForwardProcess()

    # Best Loss
    best_eval_loss = float('inf')

    # Train
    for epoch in range(cfg.num_epochs):

        # For Loss Tracking
        losses = []

        # Set model to train mode
        model.train()

        # Loop over dataloader
        for imgs in tqdm(mnist_dl):
            imgs = imgs.to(device)

            # Generate noise and timestamps
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)

            # Add noise to the images using Forward Process
            noisy_imgs = dfp.add_noise(imgs, noise, t)

            # Avoid Gradient Accumulation
            optimizer.zero_grad()

            # Predict noise using U-net Model
            noise_pred = model(noisy_imgs, t)

            # Calculate Loss
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            # Backprop + Update model params
            loss.backward()
            optimizer.step()

        # Mean Loss
        mean_epoch_loss = np.mean(losses)

        # Display
        print('Epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            mean_epoch_loss,
        ))

        # Save based on train-loss
        if mean_epoch_loss < best_eval_loss:
            best_eval_loss = mean_epoch_loss
            torch.save(model, cfg.model_path)

    print(f'Done training.....')
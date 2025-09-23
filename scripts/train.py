import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os

from models.encryption import DMSEEncryption
from models.global_features import DCTGlobalFeatureExtractor
from models.local_features import LocalFeatureExtractor
from models.fusion_model import FeatureFusionModel
from utils.loss_functions import ContrastiveLoss, CenterLoss, InterLevelAlignmentLoss
from data.datasets import EncryptedImageDataset


def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    global_extractor = DCTGlobalFeatureExtractor(
        num_bins=config['global']['num_bins'],
        feature_dim=config['global']['feature_dim']
    ).to(device)

    local_extractor = LocalFeatureExtractor(
        in_channels=config['local']['in_channels'],
        feature_dim=config['local']['feature_dim']
    ).to(device)

    fusion_model = FeatureFusionModel(
        global_feat_dim=config['fusion']['global_feat_dim'],
        local_feat_dim=config['fusion']['local_feat_dim'],
        num_patches=config['fusion']['num_patches'],
        hidden_dim=config['fusion']['hidden_dim']
    ).to(device)

    # Initialize loss functions
    contrastive_loss = ContrastiveLoss(margin=config['loss']['margin'])
    center_loss = CenterLoss(
        num_classes=config['loss']['num_classes'],
        feat_dim=config['local']['feature_dim'],
        device=device
    )
    ila_loss = InterLevelAlignmentLoss()

    # Optimizer
    optimizer = optim.SGD(
        list(global_extractor.parameters()) +
        list(local_extractor.parameters()) +
        list(fusion_model.parameters()),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    # Data loader
    train_dataset = EncryptedImageDataset(
        root_dir=config['data']['train_path'],
        transform=None  # Add data augmentation as needed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    # Training loop
    for epoch in range(config['training']['epochs']):
        global_extractor.train()
        local_extractor.train()
        fusion_model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            global_features = global_extractor(images)
            local_features = local_extractor(images)
            fused_features = fusion_model(global_features, local_features)

            # Compute losses
            loss_global = contrastive_loss(global_features, global_features, labels)  # Simplified example
            loss_local = center_loss(local_features, labels)
            loss_ila = ila_loss([global_features, local_features, fused_features])

            total_loss = config['loss']['alpha'] * loss_global + \
                         (1 - config['loss']['alpha']) * loss_local + \
                         config['loss']['beta'] * loss_ila

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % config['training']['log_interval'] == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}')

        # Save checkpoint
        if epoch % config['training']['checkpoint_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'global_extractor_state_dict': global_extractor.state_dict(),
                'local_extractor_state_dict': local_extractor.state_dict(),
                'fusion_model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item()
            }
            torch.save(checkpoint, os.path.join(config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPIR model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
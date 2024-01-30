import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
import random
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class MIMVisionModel(nn.Module):
    def __init__(
            self, 
            hidden_dim,
            num_heads,
            num_layers_encode,
            num_layers_decode,
            device
        ):
        super(MIMVisionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers_encode = num_layers_encode
        self.num_layers_decode = num_layers_decode
        self.device = device
        self.image_size = 32
        self.patch_size = 4

        self.vision_model_encode = VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=hidden_dim,
            depth=num_layers_encode,
            num_heads=num_heads,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            class_token=False,
            num_classes=hidden_dim,
            global_pool='map'
        )

        self.decode_mask_token = nn.Parameter(torch.rand((1, 1, hidden_dim)))
        self.vision_model_decode = VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=hidden_dim,
            depth=num_layers_decode,
            num_heads=num_heads,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            class_token=False,
            num_classes=self.patch_size*self.patch_size*3,
            global_pool='map'
        )
        self.vision_model_encode.attn_pool = nn.Identity()
    
    def mask_patches(self, x, mask_ratio):
        B, N, C = x.shape
        
        mask = torch.zeros((B, N), dtype=torch.bool).to(self.device)
        num_patches_to_mask = int(N * mask_ratio)

        for i in range(B):
            indices = random.sample(range(N), num_patches_to_mask)
            mask[i, indices] = True

        masked_patches = torch.masked_select(x, mask.unsqueeze(-1))
        masked_patches = masked_patches.view(B, -1, C)

        return masked_patches, mask
    
    def forward_encode(self, x, mask_ratio):
        # x has shape (B, C, H, W)
        x = self.vision_model_encode.patch_embed(x)
        x = self.vision_model_encode._pos_embed(x)

        if mask_ratio < 1:
            x, mask = self.mask_patches(x, mask_ratio)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool).to(self.device)

        x = self.vision_model_encode.blocks(x)
        x = self.vision_model_encode.norm(x)
        x = self.vision_model_encode.head(x)

        return x, mask

    def forward_decode(self, x, mask):
        B, N = mask.shape

        full_x = self.decode_mask_token.repeat(B, N, 1)
        full_x[mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim)] = x.flatten()
        x = full_x

        x = self.vision_model_decode._pos_embed(x)

        x = self.vision_model_decode.blocks(x)
        x = self.vision_model_decode.norm(x)

        x = self.vision_model_decode.head(x)

        return x
    
    def forward(self, x, mask_ratio=0.5):
        B, C, H, W = x.shape
        x, mask = self.forward_encode(x, mask_ratio)
        x = self.forward_decode(x, mask)

        x = x.view(B, self.image_size // self.patch_size, self.image_size // self.patch_size, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, 3, self.image_size, self.image_size)

        mask = mask.reshape(B, mask.shape[1], 1, 1, 1).repeat(1, 1, self.patch_size, self.patch_size, 3).view(B, self.image_size // self.patch_size, self.image_size // self.patch_size, self.patch_size, self.patch_size, 3)
        mask = mask.permute(0, 5, 1, 3, 2, 4).contiguous()
        mask = mask.view(B, 3, self.image_size, self.image_size)

        return x, mask

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        # RandAugment(2, 9),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ])

    # Define your training dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # Define your model
    model = MIMVisionModel(hidden_dim=192, num_heads=3, num_layers_encode=6, num_layers_decode=6, device='cuda')
    model = nn.DataParallel(model)
    model = model.to('cuda')

    # print out number of parameters in the model
    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    max_lr = 1e-3
    warmup_steps = 10000
    optimizer = optim.AdamW(model.parameters(), lr=0)

    plt.ion()
    fig, axes = plt.subplots(1, 2)

    # Training loop
    num_epochs = 1000
    step = 0
    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(train_loader):
            step += 1
            images = images.to('cuda')
            # Forward pass
            reconstructions, mask = model(images, mask_ratio=0.5)

            # Compute the loss
            loss = criterion(reconstructions[~mask], images[~mask])

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

            if step <= warmup_steps:
                lr = max_lr * step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                # Visualize
                images = images * 0.25 + 0.5
                reconstructions = reconstructions * 0.25 + 0.5
                images = images.float().clamp(0, 1)
                reconstructions = reconstructions.float().clamp(0, 1)

                # apply mask
                images[~mask] = 0.5
                reconstructions[mask] = 0.5

                axes[0].imshow(images[0].cpu().permute(1, 2, 0))
                axes[1].imshow(reconstructions[0].detach().cpu().permute(1, 2, 0))
                plt.pause(0.1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


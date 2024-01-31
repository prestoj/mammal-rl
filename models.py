import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
import random
from torchvision.transforms import RandAugment
import math
import matplotlib.pyplot as plt

class MIMVisionModel(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
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
        self.image_size = image_size
        self.patch_size = patch_size

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
    
    def forward(self, x, mask_ratio):
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

class VisionManager:
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_heads,
        num_layers_encode,
        num_layers_decode,
        device,
        max_lr=1e-4,
        batch_size=32,
        warmup_steps=10000,
        total_steps=100000,
        ema_decay=0.999
    ):
        self.device = device

        self.model = MIMVisionModel(
            image_size,
            patch_size,
            hidden_dim,
            num_heads,
            num_layers_encode,
            num_layers_decode,
            device
        )
        self.model = self.model.to(device)
        self.model.train()

        self.ema_model = MIMVisionModel(
            image_size,
            patch_size,
            hidden_dim,
            num_heads,
            num_layers_encode,
            num_layers_decode,
            device
        )
        self.ema_model = self.ema_model.to(device)
        self.ema_model.eval()

        self.ema_decay = ema_decay

        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RandAugment(2, 9),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)
        self.criterion = nn.MSELoss()

        self.image_queue = torch.zeros((batch_size, 3, image_size, image_size)).to(device)
        self.image_queue_marker = 0

        self.step = 0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.cosine_decay_steps = self.total_steps - self.warmup_steps

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

    def update_learning_rate(self):
        self.step += 1

        if self.step <= self.warmup_steps:
            lr = self.max_lr * self.step / self.warmup_steps
        else:
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * (self.step - self.warmup_steps) / self.cosine_decay_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def process_image_for_learning(self, image):
        image = Image.fromarray(image)
        image = self.train_transforms(image)

        self.image_queue[self.image_queue_marker] = image
        self.image_queue_marker += 1
        if self.image_queue_marker == self.image_queue.shape[0]:
            self.learn()
    
    def learn(self):
        self.update_learning_rate()

        self.optimizer.zero_grad()

        reconstructions, mask = self.model(self.image_queue, mask_ratio=0.25)
        loss = self.criterion(reconstructions[~mask], self.image_queue[~mask])

        print(self.step, self.step * self.image_queue.shape[0], loss.item())

        if self.step % 5 == 0:
            # visualize
            self.images_queue = self.image_queue * 0.25 + 0.5
            reconstructions = reconstructions * 0.25 + 0.5
            self.images_queue = self.images_queue.float().clamp(0, 1)
            reconstructions = reconstructions.float().clamp(0, 1)

            self.images_queue[~mask] = 0.5
            reconstructions[mask] = 0.5

            self.ax1.imshow(self.images_queue[0].permute(1, 2, 0).cpu().numpy())
            self.ax2.imshow(reconstructions[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.pause(0.1)

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        self.image_queue_marker = 0
        self.image_queue = torch.zeros_like(self.image_queue).to(self.device)

    def input_image(self, image):
        self.process_image_for_learning(image)

        image = Image.fromarray(image)
        image = self.test_transforms(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructions, _ = self.ema_model.forward_encode(image, mask_ratio=1.0)
            reconstructions = reconstructions.squeeze(0)
        
        return reconstructions

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.model = self.model.to(self.device)
        self.ema_model = self.ema_model.to(self.device)
        self.model.train()
        self.ema_model.eval()

class WorldModel(nn.Module):
    def __init__(
        self,
        image_size,
        hidden_dim,
        num_heads,
        num_layers,
        actions_dict,
        device
    ):
        super(WorldModel, self).__init__()
        self.device = device
        self.action_dict = actions_dict

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
            ),
            num_layers
        )

        self.action_embeddings = nn.ModuleDict({
            f'{key}_{value}': nn.Parameter(torch.rand(hidden_dim)) for key in actions_dict for value in range(actions_dict[key])
        })

        self.action_heads = nn.ModuleDict({
            key: nn.Linear(hidden_dim, value) for key, value in actions_dict.items()
        })

        self.vision_head = nn.Linear(hidden_dim, image_size*image_size*3)

    def forward(self, state, actions):
        B, N, C = state.shape

        action_embeddings = torch.zeros((B, N, C)).to(self.device)
        for key in actions:
            action_embeddings += actions[key].unsqueeze(1).repeat(1, N, 1) * self.action_embeddings[key]
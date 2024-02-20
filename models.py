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
import numpy as np

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

    def update_learning_rate(self):
        self.step += 1

        if self.step <= self.warmup_steps:
            lr = self.max_lr * self.step / self.warmup_steps
        else:
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * (self.step - self.warmup_steps) / self.cosine_decay_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def process_image_for_learning(self, image):
        if self.step == self.total_steps:
            return

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

        loss.backward()
        self.optimizer.step()

        if self.step % 10 == 0:
            print(f"VISION | Step {self.step}: {loss.item():.4f}")

        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        self.image_queue_marker = 0
        self.image_queue = torch.zeros_like(self.image_queue).to(self.device)

    def encode_image(self, image, learn_from_image=True):
        """
        expects a numpy array of shape (3, H, W)
        """
        if learn_from_image:
            self.process_image_for_learning(image)

        # check if image is a numpy array
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
            image = self.test_transforms(image)
            image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructions, _ = self.ema_model.forward_encode(image, mask_ratio=1.0)

        return image, reconstructions

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
        hidden_dim,
        num_heads,
        num_layers,
        actions_dict,
        device
    ):
        super(WorldModel, self).__init__()
        self.device = device
        self.action_dict = actions_dict
        self.hidden_dim = hidden_dim

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
            ),
            num_layers
        )


        self.action_embeddings = nn.Parameter(torch.rand((len(actions_dict), hidden_dim)))
        self.action_encoder = nn.ModuleDict({
            key: nn.Linear(value, hidden_dim) for key, value in actions_dict.items()
        })
        self.action_heads = nn.ModuleDict({
            key: nn.Linear(hidden_dim, value) for key, value in actions_dict.items()
        })

        self.value_embedding = nn.Parameter(torch.rand((1, hidden_dim)))
        self.value_head = nn.Linear(hidden_dim, 1)

    def actions_to_vec(self, batched_actions):
        """
        batched_actions: List[Dict[str, int]]

        Returns: torch.Tensor of shape (B, len(actions_dict), hidden_dim)
        """
        B = len(batched_actions)
        actions_tensor = torch.zeros(B, self.action_embeddings.shape[0], self.hidden_dim).to(self.device)

        for i_batch, actions in enumerate(batched_actions):
            for i_action, (action_name, action_value) in enumerate(sorted(actions.items())):
                action_one_hot = torch.zeros(self.action_dict[action_name]).to(self.device)
                action_one_hot[action_value] = 1
                action_vector = self.action_encoder[action_name](action_one_hot)
                actions_tensor[i_batch, i_action] = action_vector
        
        return actions_tensor


    def forward(self, state, actions):
        B, N, C = state.shape

        action_tokens = self.action_embeddings.unsqueeze(0).repeat(B, 1, 1)
        if actions is not None:
            action_tokens = action_tokens + self.actions_to_vec(actions)

        value_tokens = self.value_embedding.unsqueeze(0).repeat(B, 1, 1)

        all_tokens = torch.cat([state, action_tokens, value_tokens], dim=1)

        all_tokens = self.transformer(all_tokens)
        vision_tokens = all_tokens[:, :N]
        action_tokens = all_tokens[:, N:-1]
        value_tokens = all_tokens[:, -1]

        actions = {}
        for i_action, (action_name, action_head) in enumerate(sorted(self.action_heads.items())):
            actions[action_name] = action_head(action_tokens[:, i_action])

        value = self.value_head(value_tokens)

        return vision_tokens, actions, value


class WorldManager:
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_heads,
        num_layers,
        num_vision_encode_layers,
        num_vision_decode_layers,
        actions_dict,
        device,
        max_lr=1e-4,
        batch_size=32,
        warmup_steps=10000,
        trajectory_length=256,
        dataset_size=1024,
        reward_discount=0.98
    ):
        self.device = device

        self.vision_manager = VisionManager(
            image_size,
            patch_size,
            hidden_dim,
            num_heads,
            num_vision_encode_layers,
            num_vision_decode_layers,
            device,
            max_lr,
            batch_size,
            warmup_steps,
            total_steps=100000,
            ema_decay=0.999
        )

        self.model = WorldModel(
            hidden_dim,
            num_heads,
            num_layers,
            actions_dict,
            device
        )
        self.model = self.model.to(device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)
        self.batch_size = batch_size

        self.step = 0
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr

        self.most_recent_SARS = [None, None, None, None]
        self.SARS_queue = []
        self.SARS_dataset = []
        self.trajectory_length = trajectory_length
        self.dataset_size = dataset_size
        self.data_points_added_since_last_step = 0
        self.reward_discount = reward_discount

    def update_learning_rate(self):
        self.step += 1

        if self.step <= self.warmup_steps:
            lr = self.max_lr * self.step / self.warmup_steps
        else:
            lr = self.max_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def learn(self):
        self.update_learning_rate()

        self.optimizer.zero_grad()

        batched_SARS = [SARS_priority['SARS'] for SARS_priority in self.SARS_dataset[:self.batch_size]]
        batched_states = torch.cat([SARS[0] for SARS in batched_SARS], dim=0).to(self.device)
        batched_actions = [SARS[1] for SARS in batched_SARS]
        batched_rewards = torch.tensor([SARS[2] for SARS in batched_SARS]).to(self.device)
        batched_next_states = torch.cat([SARS[3] for SARS in batched_SARS], dim=0).to(self.device)

        _, vision_states = self.vision_manager.encode_image(batched_states, learn_from_image=False)
        _, vision_next_states = self.vision_manager.encode_image(batched_next_states, learn_from_image=False)

        predicted_next_states, _, predicted_values = self.model(vision_states, batched_actions)
        _, action_logits, _ = self.model(vision_next_states, None)
        action_probs = {key: F.softmax(value, dim=-1) for key, value in action_logits.items()}

        value_loss_itemized = F.mse_loss(predicted_values.squeeze(1), batched_rewards, reduction='none')
        value_loss = value_loss_itemized.mean()

        state_loss = F.mse_loss(predicted_next_states, vision_next_states, reduction='mean')

        action_loss = 0
        action_entropy = 0
        advantage = batched_rewards - predicted_values.squeeze(1).detach()
        total_num_actions = 0
        for action_name, action_prob in action_probs.items():
            for i in range(len(batched_SARS)):
                action_loss += -advantage[i] * torch.log(action_prob[i][batched_actions[i][action_name]])
                action_entropy += -torch.mean(action_prob[i] * torch.log(action_prob[i]))
                total_num_actions += 1
        action_loss = action_loss / total_num_actions
        action_entropy = action_entropy / total_num_actions

        loss = (value_loss + state_loss + action_loss - 1e-3 * action_entropy) / 3

        loss.backward()
        self.optimizer.step()

        if self.step % 10 == 0:
            print(f"WORLD | Step {self.step}: {loss.item():.4f} | Value: {value_loss.item():.4f} | State: {state_loss.item():.4f} | Action: {action_loss.item():.4f} | Entropy: {action_entropy.item():.4f}")

        for i in range(len(batched_SARS)):
            self.SARS_dataset[i]['priority'] = value_loss_itemized[i].detach().item()

        self.SARS_dataset = sorted(self.SARS_dataset, key=lambda x: x['priority'], reverse=True)

    def add_most_recent_SARS_to_queue(self):
        self.SARS_queue.append(self.most_recent_SARS)
        if all(element is not None for element in self.most_recent_SARS):
            self.most_recent_SARS = [None, None, None, None]

        if len(self.SARS_queue) == self.trajectory_length:
            # TODO instead of calculating this every time, you can do a cool little math trick to subtract the first and add the last
            # you'll just need to keep track of some things.

            # calculate the discounted reward
            discounted_reward = 0
            for SARS in reversed(self.SARS_queue):
                discounted_reward = SARS[2] + self.reward_discount * discounted_reward
            SARS = self.SARS_queue[0]
            SARS[2] = discounted_reward
            self.SARS_dataset = [{'SARS': SARS, 'priority': None}] + self.SARS_dataset

            # remove the lowest priority SARS from dataset and the oldest SARS from the queue
            if len(self.SARS_dataset) > self.dataset_size:
                self.SARS_dataset = self.SARS_dataset[:-1]
            self.SARS_queue = self.SARS_queue[1:]

            self.data_points_added_since_last_step += 1
            # only looking at batch_size // 2 because I want half the data to come from new data and the other half to be the highest priority data
            if self.data_points_added_since_last_step == self.batch_size // 2:
                self.learn()
                self.data_points_added_since_last_step = 0

    def get_action_from_environment(self, image, reward):
        image, image_encoded = self.vision_manager.encode_image(image, learn_from_image=True)

        self.most_recent_SARS[2] = reward
        self.most_recent_SARS[3] = image
        if all(element is not None for element in self.most_recent_SARS):
            self.add_most_recent_SARS_to_queue()

        with torch.no_grad():
            # TODO maybe use an ema model here?
            _, actions, _ = self.model(image_encoded, None)

        chosen_actions = {}
        for action_name, action_values in actions.items():
            action_values = F.softmax(action_values, dim=-1)
            action_value = torch.multinomial(action_values, 1).item()
            chosen_actions[action_name] = action_value
        
        self.most_recent_SARS[0] = image
        self.most_recent_SARS[1] = chosen_actions
        return chosen_actions

    def save(self, path, vision_path):
        torch.save({
            'world_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sars_dataset': self.SARS_dataset,
            'step': self.step
        }, path)

        self.vision_manager.save(vision_path)

    def load(self, path, vision_path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['world_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.SARS_dataset = checkpoint['sars_dataset']
        self.step = checkpoint['step']
        self.model = self.model.to(self.device)
        self.model.train()

        self.vision_manager.load(vision_path)
        self.vision_manager.model = self.vision_manager.model.to(self.device)
        
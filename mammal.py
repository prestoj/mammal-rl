import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import uuid
import timm
from PIL import Image


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def drop_path(x, drop_prob
: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# class MammalVision(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim, 
#             num_heads,
#             patch_size, 
#             image_size, 
#             num_input_layers, 
#             num_output_layers, 
#     ):
#         super(MammalVision, self).__init__()

#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.vision_num_patches = (image_size // patch_size) ** 2
#         self.hidden_dim = hidden_dim
#         self.vision_patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
#         self.pos_embed = nn.Parameter(torch.randn(1, self.vision_num_patches, hidden_dim) * .02)
#         self.vision_input_blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=4,
#                 qkv_bias=True,
#                 attn_drop=0.0,
#                 drop_path=0.0,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#             ) for _ in range(num_input_layers)
#         ])
#         self.vision_output_blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=4,
#                 qkv_bias=True,
#                 attn_drop=0.0,
#                 drop_path=0.0,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#             ) for _ in range(num_output_layers)
#         ])

#         self.vision_output_head = nn.Linear(hidden_dim, 3 * patch_size * patch_size)

#     def forward_input(self, x):
#         x = self.vision_patch_embed(x)
#         x = x.flatten(2).transpose(1, 2)
#         x = x + self.pos_embed
#         x = self.vision_input_blocks(x)
#         return x
    
#     def forward_output(self, x):
#         x = self.vision_output_blocks(x)
#         x = self.vision_output_head(x)
#         x = x.reshape(-1, self.vision_num_patches, 3, self.patch_size, self.patch_size)
#         x = x.permute(0, 2, 1, 3, 4)
#         x = x.reshape(-1, 3, self.image_size, self.image_size).contiguous()
#         return x

#     def forward(self, x):
#         x = self.forward_input(x)
#         x = self.forward_output(x)
#         return x

# class OldMammalCentral(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim, 
#             num_heads,
#             num_state_tokens,
#             num_layers,
#             vision_patch_size,
#             vision_image_size,
#             vision_num_input_layers,
#             vision_num_output_layers,
#     ):
#         super(OldMammalCentral, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_state_tokens = num_state_tokens

#         self.vision = MammalVision(
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             patch_size=vision_patch_size,
#             image_size=vision_image_size,
#             num_input_layers=vision_num_input_layers,
#             num_output_layers=vision_num_output_layers,
#         )

#         self.state_embed = nn.Parameter(torch.randn(num_state_tokens, hidden_dim))

#         self.central_pre_action_blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=4,
#                 qkv_bias=True,
#                 attn_drop=0.0,
#                 drop_path=0.0,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#             ) for _ in range(num_layers)
#         ])
#         self.central_post_action_blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=4,
#                 qkv_bias=True,
#                 attn_drop=0.0,
#                 drop_path=0.0,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, image, state):
#         image_tokens = self.vision.forward_input(image)
#         state = state + self.state_embed.unsqueeze(0)
#         x = torch.cat((state, image_tokens), dim=1)
#         x = self.central_pre_action_blocks(x)
#         return x


# class OldMammal(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim, 
#             num_heads,
#             num_layers,
#             vision_patch_size,
#             vision_image_size,
#             vision_num_input_layers,
#             vision_num_output_layers,
#             action_dict,
#             rollout_length=8,
#     ):
#         super(OldMammal, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.action_dict = action_dict
#         self.num_state_tokens = len(action_dict) + 3 # recurrent state, env value, inner value, actions

#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#         ])

#         self.central = MammalCentral(
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_state_tokens=self.num_state_tokens,
#             num_layers=num_layers,
#             vision_patch_size=vision_patch_size,
#             vision_image_size=vision_image_size,
#             vision_num_input_layers=vision_num_input_layers,
#             vision_num_output_layers=vision_num_output_layers,
#         )

#         self.action_heads = nn.ModuleDict({
#             f'action_{key}': nn.Linear(hidden_dim, value) for key, value in action_dict.items()
#         })
#         self.env_value_head = nn.Linear(hidden_dim, 1)
#         self.inner_value_head = nn.Linear(hidden_dim, 1)

#         self.rollout_length = rollout_length
#         self.rollout_steps_so_far = 0
#         self.rollout_rewards = []
#         self.rollout_actions = []
#         self.rollout_env_values = []
#         self.rollout_inner_values = []
#         self.rollout_images = []
#         self.rollout_predicted_images = []
#         self.vision_losses = []

#         self.decay = 0.99

#         self.recurrent_cell = nn.LSTMCell(hidden_dim, hidden_dim)
#         self.recurrent_state = (torch.zeros(1, hidden_dim), torch.zeros(1, hidden_dim))
#         self.recurrent_layer_norm = nn.LayerNorm(hidden_dim)
#         self.reward_history = []

#         self.actions_enc = nn.ModuleDict({
#             f'action_enc_{key}': nn.Linear(value, hidden_dim) for key, value in action_dict.items()
#         })
#         self.env_reward_enc = nn.Linear(1, hidden_dim)
#         self.inner_reward_enc = nn.Linear(1, hidden_dim)

#         self.max_lr = 1e-5
#         self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
#         self.n_optimizer_steps = 0
#         self.vision_criterion = nn.MSELoss(reduction='mean')

#         self.n_trajectories_evaluated = 1
        
#         plt.ion()
#         self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 4))

#     def forward(self, image, reward, debug=False, device='cpu'):
#         image = self.transform(image)
#         image = image.unsqueeze(0)
#         image = image.to(device)

#         if len(self.rollout_predicted_images) > 0:
#             target = image[0].detach()
#             current_vision_loss = self.vision_criterion(target, self.rollout_predicted_images[-1][0])
#         else:
#             current_vision_loss = torch.tensor(0.0).to(device)
#         self.vision_losses.append(current_vision_loss)

#         self.recurrent_state = (self.recurrent_state[0].to(device), self.recurrent_state[1].to(device))
#         env_reward_encoded = self.env_reward_enc(torch.tensor([reward]).float().to(device)).unsqueeze(0).unsqueeze(0)
#         inner_reward_encoded = self.inner_reward_enc(torch.tensor([current_vision_loss]).float().to(device)).unsqueeze(0).unsqueeze(0)
#         actions_placeholder = torch.stack([torch.zeros((1, self.hidden_dim)) for _ in range(len(self.action_dict))], dim=1).to(device)
#         state = torch.cat((self.recurrent_state[0].unsqueeze(0), env_reward_encoded, inner_reward_encoded, actions_placeholder), dim=1)

#         x = self.central(image, state)

#         actions_vecs = {key: head(x[:, i+3]) for i, (key, head) in enumerate(self.action_heads.items())}
#         best_actions = {}
#         best_score = None
#         best_x = None
#         for _ in range(self.n_trajectories_evaluated):

#             actions = {}
#             for action in actions_vecs: # should clean this up so it's no evaluation n_trajectories_evaluated times
#                 distribution = torch.softmax(actions_vecs[action], dim=-1)
#                 if torch.min(distribution) < 0.01 or torch.max(distribution) > 0.99:
#                     sampled_distribution = (distribution - torch.min(distribution)) / (torch.max(distribution) - torch.min(distribution))
#                     sampled_distribution = sampled_distribution * 0.98 + 0.01
#                     actions[action] = (torch.multinomial(sampled_distribution, 1)[0][0].cpu().item(), distribution)
#                 else:
#                     actions[action] = (torch.multinomial(distribution, 1)[0][0].cpu().item(), distribution)

#             actions_one_hot_enc = torch.stack([self.actions_enc[f'action_enc_{action[7:]}'](torch.eye(self.action_dict[action[7:]])[actions[action][0]].unsqueeze(0).to(device)) for action in actions], dim=1)
#             x_ = torch.cat((x[:, :3], actions_one_hot_enc, x[:, self.num_state_tokens:]), dim=1)
#             x_ = self.central.central_post_action_blocks(x_)
#             cur_score = self.env_value_head(x_[:, 1])[0] * 0.99 + self.inner_value_head(x_[:, 2])[0] * 0.01
#             if best_score is None or cur_score > best_score:
#                 best_x = x_
#                 best_actions = actions
#                 best_score = cur_score

#         actions = best_actions
#         x = best_x

#         if len(self.rollout_predicted_images) > 0:
#             self.rollout_rewards.append(reward)
#             self.reward_history.append(reward)
#             self.reward_history = self.reward_history[-10000:]

#             self.rollout_actions.append(sorted(actions.items()))
#             self.rollout_env_values.append(self.env_value_head(x[:, 1])[0])
#             self.rollout_inner_values.append(self.inner_value_head(x[:, 2])[0])
#             if debug:
#                 print(self.rollout_env_values[-1].item(), self.rollout_inner_values[-1].item())
#             self.rollout_steps_so_far += 1

#         if self.rollout_steps_so_far >= self.rollout_length:

#             # REINFORCE
#             env_returns = []
#             # R = self.decay * self.rollout_env_values[-1].detach().item()
#             R = 0
#             for r in self.rollout_rewards[::-1]:
#                 R = r + self.decay * R
#                 env_returns.insert(0, R)
#             env_returns = torch.tensor(env_returns).to(device)

#             inner_returns = []
#             # R = self.decay * self.rollout_inner_values[-1].detach().item()
#             R = 0
#             for i in range(len(self.rollout_rewards)):
#                 R = self.vision_losses[-i-1].detach().item() * 0.05 + self.decay * R # scaled to be comparable to the other rewards
#                 inner_returns.insert(0, R)
#             inner_returns = torch.tensor(inner_returns).to(device)

#             policy_losses = []
#             entropy_losses = []
#             for rollout_action, env_R, inner_R in zip(self.rollout_actions[:1], env_returns[:1], inner_returns[:1]):
#                 R = 0.01 * inner_R
#                 if sum(self.rollout_rewards) > 0:
#                     R += 0.99 * env_R 

#                 for action in rollout_action:
#                     policy_losses.append(-torch.log(action[1][1][0, action[1][0]] + 1e-5) * R)
#                     entropy_losses.append(torch.mean(-torch.log(action[1][1][0] + 1e-5)))

#             policy_loss = torch.stack(policy_losses).mean()
#             vision_loss = torch.stack(self.vision_losses).mean()
#             env_value_loss = F.mse_loss(self.rollout_env_values[0].squeeze().to(device), env_returns[0], reduction='mean')
#             inner_value_loss = F.mse_loss(self.rollout_inner_values[0].squeeze().to(device), inner_returns[0], reduction='mean')
#             entropy_loss = torch.stack(entropy_losses).mean()
#             loss = 0.2 * policy_loss + 0.2 * vision_loss + 0.2 * env_value_loss + 0.2 * inner_value_loss + 0.2 * entropy_loss
#             print(f'After {self.n_optimizer_steps} steps, loss: {loss.item():.3f}, policy_loss: {policy_loss.item():.3f}, vision_loss: {vision_loss.item():.3f}, env_value_loss: {env_value_loss.item():.3f}, inner_value_loss: {inner_value_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}')

#             # warmup the optimizer over the first 1000 steps
#             if self.n_optimizer_steps <= 1000:
#                 lr = self.max_lr * (self.n_optimizer_steps / 1000)
#                 for param_group in self.optimizer.param_groups:
#                     param_group['lr'] = lr

#             self.optimizer.zero_grad()
#             loss.backward()

#             self.optimizer.step()
#             self.n_optimizer_steps += 1

#             if debug:
#                 for action in actions:
#                     print(action, [round(x, 3) for x in actions[action][1].detach().cpu().numpy().tolist()[0]])

#                 channel_means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#                 channel_stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

#                 predicted_image_ = self.rollout_predicted_images[-1][0].detach().cpu().numpy()
#                 predicted_image_ = predicted_image_ * channel_stds + channel_means
#                 predicted_image_ = np.clip(predicted_image_, 0, 1)
#                 predicted_image_ = np.transpose(predicted_image_, (1, 2, 0))

#                 prev_image_ = self.rollout_images[-1][0].detach().cpu().numpy()
#                 prev_image_ = prev_image_ * channel_stds + channel_means
#                 prev_image_ = np.clip(prev_image_, 0, 1)
#                 prev_image_ = np.transpose(prev_image_, (1, 2, 0))

#                 cur_image_ = image[0].detach().cpu().numpy()
#                 cur_image_ = cur_image_ * channel_stds + channel_means
#                 cur_image_ = np.clip(cur_image_, 0, 1)
#                 cur_image_ = np.transpose(cur_image_, (1, 2, 0))

#                 self.ax[0].cla()
#                 self.ax[0].imshow(predicted_image_)
#                 self.ax[1].cla()
#                 self.ax[1].imshow(prev_image_)
#                 self.ax[2].cla()
#                 self.ax[2].imshow(cur_image_)
#                 self.fig.canvas.draw()
#                 self.fig.canvas.flush_events()

#             self.rollout_steps_so_far = 0
#             self.rollout_rewards = []
#             self.rollout_actions = []
#             self.rollout_env_values = []
#             self.rollout_inner_values = []
#             self.rollout_images = []
#             self.rollout_predicted_images = []
#             self.vision_losses = []
#             x = x.detach()
#             self.recurrent_state = (self.recurrent_state[0].detach(), self.recurrent_state[1].detach())

#         predicted_image = self.central.vision.forward_output(x[:, self.num_state_tokens:]) / 10
#         self.rollout_predicted_images.append(predicted_image)
#         self.rollout_images.append(image)
#         recurrent_input = self.recurrent_state[0] + x[:, 0]
#         recurrent_input = self.recurrent_layer_norm(recurrent_input)
#         self.recurrent_state = self.recurrent_cell(recurrent_input, self.recurrent_state)

#         actions_to_return = {}
#         for action in actions:
#             # remove 'action_' from the action name
#             action_name = action[7:]
#             actions_to_return[action_name] = actions[action][0]
#             if 'camera' in action_name:
#                 actions_to_return[action_name] -= 5
#         return actions_to_return

#     def reset(self, device):
#         self.recurrent_state = (torch.zeros(1, self.hidden_dim).to(device), torch.zeros(1, self.hidden_dim).to(device))
#         self.rollout_steps_so_far = 0
#         self.rollout_rewards = []
#         self.rollout_actions = []
#         self.rollout_env_values = []
#         self.rollout_inner_values = []
#         self.rollout_images = []
#         self.rollout_predicted_images = []
#         self.vision_losses = []


# class MammalCentral(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim, 
#             num_heads,
#             num_state_tokens,
#             num_layers,
#     ):
#         super(MammalCentral, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_state_tokens = num_state_tokens

#         self.vision = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True)
#         data_config = timm.data.resolve_model_data_config(self.vision)
#         self.transforms = timm.data.create_transform(**data_config, is_training=False)

#         # make sure the vision model doesn't get updated
#         for param in self.vision.parameters():
#             param.requires_grad = False
#         self.vision.eval()

#         self.state_embed = nn.Parameter(torch.rand(num_state_tokens, hidden_dim) * 2 - 1)

#         self.central_blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, image, state, device='cpu'):
#         # first convert from np array to pil image
#         image = Image.fromarray(image)
#         image = self.transforms(image).unsqueeze(0).to(device)
#         image_tokens = self.vision.forward_features(image)

#         state[:, 0] = image_tokens[:, 0]
#         image_tokens = image_tokens[:, 1:]

#         state = state + self.state_embed.unsqueeze(0)
#         x = torch.cat((state, image_tokens), dim=1)
#         x = self.central_blocks(x)

#         return x


# class EpisodicMemory(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim, 
#             num_heads,
#             num_layers,
#             memories_fetched_per_step,
#             max_memory_size,
#             action_dict,
#     ):
#         super(EpisodicMemory, self).__init__()

#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads

#         self.action_dict = action_dict

#         self.blocks = nn.Sequential(*[
#             Block(
#                 dim=hidden_dim,
#                 num_heads=num_heads,
#             ) for _ in range(num_layers)
#         ])
#         self.actions_enc = nn.ModuleDict({
#             f'action_enc_{key}': nn.Linear(value, hidden_dim) for key, value in action_dict.items()
#         })
#         self.reward_enc = nn.Linear(1, hidden_dim)
#         self.vectors_enc = nn.Linear(hidden_dim, hidden_dim)

#         self.memory = {}
#         self.max_memory_size = max_memory_size
#         self.memories_fetched_per_step = memories_fetched_per_step
    
#     def forward(self, x, device='cpu'):
#         current_time = time.time()
#         if not self.memory:
#             return x.unsqueeze(0)

#         with torch.no_grad():
#             memory_keys, memory_values = zip(*self.memory.items())
#             memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
#             memory_similarities = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]
#             memory_times = torch.tensor([mv['time'] - current_time for mv in memory_values]).to(device)
#             memory_suprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
#             memory_rewards = torch.tensor([mv['reward'] for mv in memory_values]).to(device)
            
#             recency_scores = EpisodicMemory._get_scores(memory_times) ** 2
#             surprise_scores = EpisodicMemory._get_scores(memory_suprises) ** 2
#             similarity_scores = (1 - memory_similarities) ** 2

#             scores = recency_scores + surprise_scores + similarity_scores
#             scores = scores.to(device)

#             _, indices = torch.topk(scores, min(self.memories_fetched_per_step, scores.shape[0]), dim=0)
#             selected_vectors = memory_vectors[indices]
#             selected_vectors = selected_vectors.squeeze(1)

#         selected_vectors = selected_vectors.detach()
#         selected_vectors = self.vectors_enc(selected_vectors)

#         reward_vectors = self.reward_enc(memory_rewards[indices].unsqueeze(1))
#         selected_vectors = selected_vectors + reward_vectors

#         action_vectors = []
#         for i, memory_key in enumerate(memory_keys):
#             if i in indices:
#                 actions = self.memory[memory_key]['actions']
#                 for action, count in actions.items():
#                     action_vector = self.actions_enc[f'action_enc_{action}'](torch.eye(self.action_dict[action])[count].to(device))
#                     action_vectors.append(action_vector)
#         if action_vectors:
#             action_vectors = torch.stack(action_vectors, dim=1).mean(dim=1).unsqueeze(0)
#             selected_vectors = selected_vectors + action_vectors

#         selected_vectors = selected_vectors.unsqueeze(0)
#         target_vector = self.vectors_enc(x).unsqueeze(0)

#         vectors = torch.cat((target_vector, selected_vectors), dim=1)
#         vectors = self.blocks(vectors)

#         return vectors[:, 0]
    
#     def highest_similarity_score(self, x, device='cpu'):
#         if not self.memory:
#             return None

#         with torch.no_grad():
#             _, memory_values = zip(*self.memory.items())
#             memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
#             memory_similarities = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]

#         max_similarity, index = torch.max(memory_similarities, dim=0)
#         max_similarity = max_similarity.item()
#         memory_uid = list(self.memory.keys())[index.item()]
#         return max_similarity, memory_uid
    
#     def get_intrinsic_reward(self, x, surprise, device='cpu'):
#         if not self.memory:
#             return 0.0
        
#         with torch.no_grad():
#             _, memory_values = zip(*self.memory.items())
#             memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
#             max_similarity = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]
#             memory_surprise = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
#             memory_max_similarities = torch.tensor([mv['max_similarity'] for mv in memory_values]).to(device)

#             score = (memory_max_similarities > max_similarity).float().mean() + (memory_surprise < surprise).float().mean()
#             score = score / 2

#         return float(score.item())

#     def add_memory(self, state_vector, actions, reward, expected_reward, device='cpu'):
#         current_time = time.time()

#         reward = float(reward)
#         expected_reward = float(expected_reward)
#         surprise = float(abs(expected_reward - reward))

#         if not self.memory:
#             memory_uid = str(uuid.uuid4())
#             print('adding memory', memory_uid)
#             self.memory[memory_uid] = {
#                 'vector': state_vector,
#                 'time': current_time,
#                 'reward': reward,
#                 'actions': actions,
#                 'surprise': surprise,
#                 'max_similarity': 0.0,
#             }
#             return memory_uid

#         if np.random.rand() < 0.01:
#             self.remove_lowest_priority_memory(device=device)

#         _, memory_values = zip(*self.memory.items())

#         memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
#         surprise_cutoff = torch.quantile(memory_surprises, 0.5)

#         max_similarity, _ = self.highest_similarity_score(state_vector, device=device)
#         memory_similarities = torch.tensor([mv['max_similarity'] for mv in memory_values]).to(device)
#         similarity_cutoff = torch.quantile(memory_similarities, 0.5)
#         if len(self.memory) < self.max_memory_size or (max_similarity < similarity_cutoff and surprise > surprise_cutoff):
#             memory_uid = str(uuid.uuid4())
#             print('adding memory', surprise, max_similarity)
#             self.memory[memory_uid] = {
#                 'vector': state_vector,
#                 'time': current_time,
#                 'reward': reward,
#                 'actions': actions,
#                 'surprise': surprise,
#                 'max_similarity': max_similarity,
#             }
#         else:
#             memory_uid = None

#         while len(self.memory) > self.max_memory_size:
#             self.remove_lowest_priority_memory(device=device)

#         return memory_uid

#     def remove_lowest_priority_memory(self, device='cpu'):
#         current_time = time.time()
#         with torch.no_grad():
#             _, memory_values = zip(*self.memory.items())
#             memory_times = torch.tensor([mv['time'] - current_time for mv in memory_values])
#             memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values])
#             memory_max_similarities = torch.tensor([mv['max_similarity'] for mv in memory_values])

#             recency_scores = EpisodicMemory._get_scores(memory_times) ** 2
#             surprise_scores = EpisodicMemory._get_scores(memory_surprises) ** 2
#             similarity_scores = (1 - EpisodicMemory._get_scores(memory_max_similarities)) ** 2

#             scores = recency_scores + surprise_scores + similarity_scores
#             _, index = torch.min(scores, dim=0)
#             print('deleting memory', recency_scores[index].item(), surprise_scores[index].item(), similarity_scores[index].item())
#             del self.memory[list(self.memory.keys())[index]]
    
#     def _get_scores(vector):
#         _, sorted_indices = torch.sort(vector)
#         sorted_rewards = vector[sorted_indices]
#         _, inverse_indices, counts = torch.unique_consecutive(sorted_rewards, return_inverse=True, return_counts=True)
#         cumulative_counts = counts.cumsum(0)
#         start_ranks = cumulative_counts - counts
#         average_ranks = (start_ranks + cumulative_counts - 1).float() / 2
#         scores = average_ranks[inverse_indices]
#         scores = scores / len(vector)
#         original_order_scores = torch.empty_like(scores)
#         original_order_scores[sorted_indices] = scores
#         scores = original_order_scores

#         return scores

# class Mammal(nn.Module):
#     def __init__(
#             self, 
#             hidden_dim,
#             num_heads,
#             num_layers,
#             action_dict,
#             rollout_length=8,
#     ):
#         super(Mammal, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.action_dict = action_dict
#         self.num_state_tokens = 2

#         self.central = MammalCentral(
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_state_tokens=self.num_state_tokens,
#             num_layers=num_layers,
#         )
#         self.episodic_memory = EpisodicMemory(
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             memories_fetched_per_step=64,
#             max_memory_size=1024,
#             action_dict=action_dict,
#         )

#         self.action_heads = nn.ModuleDict({
#             f'action_{key}': nn.Linear(hidden_dim, value) for key, value in action_dict.items()
#         })
#         self.env_value_head = nn.Linear(hidden_dim, 1)
#         self.inner_value_head = nn.Linear(hidden_dim, 1)

#         self.rollout_length = rollout_length
#         self.rollout_length_so_far = 0
#         self.rollouts = {
#             'rewards': [],
#             'exploration_bonuses': [],
#             'actions': [],
#             'env_values': [],
#             'states': [],
#         }

#         self.decay = 0.99

#         # self.recurrent_cell = nn.LSTMCell(hidden_dim, hidden_dim)
#         self.recurrent_state = (torch.zeros(1, hidden_dim), torch.zeros(1, hidden_dim))
#         self.recurrent_layer_norm = nn.LayerNorm(hidden_dim)
#         self.env_reward_enc = nn.Linear(1, hidden_dim)

#         self.max_lr = 1e-4
#         self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
#         self.n_optimizer_steps = 0

#         self.n_forwards = 0
#         self.losses_accumulated = 0 

#     def forward(self, image, reward, debug=False, device='cpu'):
#         self.recurrent_state = (self.recurrent_state[0].to(device), self.recurrent_state[1].to(device))
#         env_reward_encoded = self.env_reward_enc(torch.tensor([reward]).float().to(device)).unsqueeze(0).unsqueeze(0)
#         state = torch.cat((torch.zeros(1, 1, self.hidden_dim).to(device), env_reward_encoded), dim=1)

#         x = self.central(image, state, device=device)
#         output = x[:, 0]
#         # reccurent_outputs = self.recurrent_cell(x[:, 0], self.recurrent_state)
#         # output = reccurent_outputs[0] + self.recurrent_state[0]
#         # output = self.recurrent_layer_norm(output)
#         # self.recurrent_state = reccurent_outputs
#         if len(self.episodic_memory.memory) > 0:
#             output = self.episodic_memory(output, device=device)

#         actions_vecs = {key: head(output) for key, head in self.action_heads.items()}

#         actions = {}
#         for action in actions_vecs:
#             distribution = torch.softmax(actions_vecs[action], dim=-1)
#             if torch.min(distribution) < 0.01 or torch.max(distribution) > 0.99:
#                 sampled_distribution = (distribution - torch.min(distribution)) / (torch.max(distribution) - torch.min(distribution))
#                 sampled_distribution = sampled_distribution * 0.98 + 0.01
#                 actions[action] = (torch.multinomial(sampled_distribution, 1)[0][0].cpu().item(), distribution)
#             else:
#                 actions[action] = (torch.multinomial(distribution, 1)[0][0].cpu().item(), distribution)

#             # # just take the argmax
#             # actions[action] = (torch.argmax(distribution, dim=-1)[0].cpu().item(), distribution)
        
#         self.rollouts['rewards'].append((self.n_forwards, reward))
#         self.rollouts['actions'].append((self.n_forwards, sorted(actions.items())))
#         self.rollouts['env_values'].append((self.n_forwards, self.env_value_head(output)[0]))
#         self.rollouts['states'].append((self.n_forwards, output.detach()))

#         exploration_bonus = 0.0
#         if len(self.rollouts['rewards']) >= self.rollout_length:
#             env_returns = []
#             R = 0
#             for r in self.rollouts['rewards'][:0:-1]:
#                 R = r[1] + self.decay * R
#                 env_returns.insert(0, R)
#             env_returns = torch.tensor(env_returns).to(device)

#             surprise = float(abs(self.rollouts['env_values'][0][1].item() - env_returns[0].item()))
#             exploration_bonus = self.episodic_memory.get_intrinsic_reward(self.rollouts['states'][0][1], surprise, device=device)
            
#             if self.n_optimizer_steps >= 100:
#                 self.episodic_memory.add_memory(self.rollouts['states'][0][1], {action[0][7:]: action[1][0] for action in self.rollouts['actions'][0][1]}, env_returns[0], self.rollouts['env_values'][0][1], device=device)
        
#         if debug:
#             print(self.rollouts['env_values'][-1][1].item())
#             print('exploration_bonus', exploration_bonus)
            
#         self.rollouts['exploration_bonuses'].append((self.n_forwards, exploration_bonus))
#         self.rollout_length_so_far += 1

#         stepped = False
#         if self.rollout_length_so_far >= self.rollout_length:
#             stepped = self.learn(device=device)
#             if stepped:
#                 output = output.detach()
#                 self.rollout_length_so_far = 0

#         if len(self.rollouts['rewards']) >= self.rollout_length:
#             self.rollouts['rewards'] = self.rollouts['rewards'][1:]
#             self.rollouts['exploration_bonuses'] = self.rollouts['exploration_bonuses'][1:]
#             self.rollouts['actions'] = self.rollouts['actions'][1:]
#             self.rollouts['states'] = self.rollouts['states'][1:]
#             self.rollouts['env_values'] = self.rollouts['env_values'][1:]

#         if debug:
#             for action in actions:
#                 print(action, [round(x, 3) for x in actions[action][1].detach().cpu().numpy().tolist()[0]])

#         self.n_forwards += 1

#         actions_to_return = {}
#         for action in actions:
#             # remove 'action_' from the action name
#             action_name = action[7:]
#             actions_to_return[action_name] = actions[action][0]
#             if 'camera' in action_name:
#                 actions_to_return[action_name] -= 5
#         return actions_to_return

#     def learn(self, device='cpu'):
#         # REINFORCE
#         env_returns = []
#         R = 0
#         for r in self.rollouts['rewards'][:0:-1]:
#             R = r[1] + self.decay * R
#             env_returns.insert(0, R)
#         env_returns = torch.tensor(env_returns).to(device)

#         internal_returns = []
#         R = 0
#         for r in self.rollouts['exploration_bonuses'][:0:-1]:
#             R = r[1] + self.decay * R
#             internal_returns.insert(0, R)
#         internal_returns = torch.tensor(internal_returns).to(device)

#         policy_losses = []
#         entropy_losses = []
#         for rollout_action, env_R, internal_R in zip(self.rollouts['actions'][:1], env_returns[:1], internal_returns[:1]):
#             R = env_R + 0.1 * internal_R
#             for action in rollout_action[1]:
#                 policy_losses.append(-torch.log(action[1][1][0, action[1][0]] + 1e-5) * R)
#                 entropy_losses.append(torch.mean(-torch.log(action[1][1][0] + 1e-5)))

#         policy_loss = torch.stack(policy_losses).mean()
#         env_value_loss = F.mse_loss(self.rollouts['env_values'][0][1].squeeze().to(device), env_returns[0], reduction='mean')
#         entropy_loss = torch.stack(entropy_losses).mean()
#         loss = 1 * policy_loss + 1 * env_value_loss + 0.1 * entropy_loss
#         print(f'After {self.n_optimizer_steps} steps and {self.n_forwards} forwards, loss: {loss.item():.3f}, policy_loss: {policy_loss.item():.3f}, env_value_loss: {env_value_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}')

#         # warmup the optimizer over the first 1000 steps
#         if self.n_optimizer_steps <= 1000:
#             lr = self.max_lr * (self.n_optimizer_steps / 1000)
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr

#         stepped = False
#         if self.losses_accumulated == 7:
#             loss.backward()
#             print('stepping')
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#             torch.cuda.empty_cache()
#             self.n_optimizer_steps += 1
#             self.recurrent_state = (self.recurrent_state[0].detach(), self.recurrent_state[1].detach())
#             stepped = True
#             self.losses_accumulated = 0
#         else:
#             loss.backward(retain_graph=True)
#             self.losses_accumulated += 1
        
#         return stepped

#     def reset(self, device):
#         self.optimizer.zero_grad()
#         self.recurrent_state = (torch.zeros(1, self.hidden_dim).to(device), torch.zeros(1, self.hidden_dim).to(device))
#         self.rollouts['rewards'] = []
#         self.rollouts['actions'] = []
#         self.rollouts['env_values'] = []
#         self.rollouts['states'] = []


class MammalCentral(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads,
            num_state_tokens,
            num_layers,
    ):
        super(MammalCentral, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_state_tokens = num_state_tokens

        self.vision = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(self.vision)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # make sure the vision model doesn't get updated
        for param in self.vision.parameters():
            param.requires_grad = False
        self.vision.eval()

        self.state_embed = nn.Parameter(torch.rand(num_state_tokens, hidden_dim) * 2 - 1)

        self.central_blocks = nn.Sequential(*[
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])

    def forward(self, image, state, device='cpu'):
        # first convert from np array to pil image
        image = Image.fromarray(image)
        image = self.transforms(image).unsqueeze(0).to(device)
        image_tokens = self.vision.forward_features(image)

        state[:, 0] = image_tokens[:, 0]
        image_tokens = image_tokens[:, 1:]

        state = state + self.state_embed.unsqueeze(0)
        x = torch.cat((state, image_tokens), dim=1)
        x = self.central_blocks(x)

        return x


class LimbicSystem(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads,
            num_layers,
            memories_fetched_per_step,
            max_memory_size,
            action_dict,
    ):
        super(LimbicSystem, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.action_dict = action_dict

        self.blocks = nn.Sequential(*[
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])
        self.actions_enc = nn.ModuleDict({
            f'action_enc_{key}': nn.Linear(value, hidden_dim) for key, value in action_dict.items()
        })
        self.reward_enc = nn.Linear(1, hidden_dim)
        self.vectors_enc = nn.Linear(hidden_dim, hidden_dim)

        self.value_head = nn.Linear(hidden_dim, 1)

        self.memory = {}
        self.max_memory_size = max_memory_size
        self.memories_fetched_per_step = memories_fetched_per_step
    
    def forward(self, x, device='cpu'):
        current_time = time.time()
        if not self.memory:
            return x.unsqueeze(0)
        
        with torch.no_grad():
            memory_keys, memory_values = zip(*self.memory.items())
            memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
            memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
            memory_similarities = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]
            memory_rewards = torch.tensor([mv['reward'] for mv in memory_values]).to(device)

            # high_reward_scores = LimbicSystem._get_scores(memory_rewards)
            # low_reward_scores = (1 - LimbicSystem._get_scores(memory_rewards))
            # extreme_score = 1 - (high_reward_scores * low_reward_scores)
            # extreme_score = LimbicSystem._get_scores(extreme_score) ** 2
            surprise_score = LimbicSystem._get_scores(memory_surprises) ** 2
            similarity_scores = LimbicSystem._get_scores(memory_similarities) ** 2
            stochasticity_scores = LimbicSystem._get_scores(torch.rand_like(memory_similarities)) ** 2

            scores = surprise_score + similarity_scores + stochasticity_scores
            scores = scores.to(device)

            _, indices = torch.topk(scores, min(self.memories_fetched_per_step, scores.shape[0]), dim=0)

            for index in indices:
                memory_key = memory_keys[index]
                self.memory[memory_key]['time_last_recall'] = current_time

            selected_vectors = memory_vectors[indices]
            selected_vectors = selected_vectors.squeeze(1)

        selected_vectors = selected_vectors.detach()
        selected_vectors = self.vectors_enc(selected_vectors).unsqueeze(0)
        target_vector = self.vectors_enc(x).unsqueeze(0)

        vectors = torch.cat((target_vector, selected_vectors), dim=1)
        vectors = self.blocks(vectors)

        value_prediction = self.value_head(vectors[0])

        memory_losses = F.mse_loss(value_prediction[1:], memory_rewards[indices].unsqueeze(1), reduction='none').squeeze(1)

        memory_actions = []
        for i, index in enumerate(indices):
            memory_key = memory_keys[index]
            self.memory[memory_key]['surprise'] = memory_losses[i].detach().item()
            memory_actions.append(self.memory[memory_key]['actions'])

        return {
            'output_vectors': vectors[0],
            'value_prediction': value_prediction[0],
            'memory_losses': memory_losses,
            'memory_actions': memory_actions,
            'memory_rewards': memory_rewards[indices],
        }
    
    def highest_similarity_score(self, x, device='cpu'):
        if not self.memory:
            return None

        with torch.no_grad():
            _, memory_values = zip(*self.memory.items())
            memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
            memory_similarities = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]

        max_similarity, index = torch.max(memory_similarities, dim=0)
        max_similarity = max_similarity.item()
        memory_uid = list(self.memory.keys())[index.item()]
        return max_similarity, memory_uid

    def get_intrinsic_reward(self, x, surprise, device='cpu'):
        if not self.memory:
            return 0.0

        with torch.no_grad():
            _, memory_values = zip(*self.memory.items())
            memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)

            memory_similarities = torch.cosine_similarity(x.unsqueeze(0), memory_vectors, dim=-1)[:, 0]

            memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)

            memory_rewards = torch.tensor([mv['reward'] for mv in memory_values]).to(device)
            memory_reward_scores = LimbicSystem._get_scores(memory_rewards)

            if len(memory_similarities) > 100:
                high_reward_scores = memory_reward_scores[torch.topk(memory_similarities, int(0.01 * len(memory_similarities)), dim=0)[1]]
            else:
                high_reward_scores = torch.tensor([0.0]).to(device)

            score = (memory_surprises < surprise).float().mean()
            score = score + high_reward_scores.mean()
            score = score / 2

        return float(score.item())

    def add_memory(self, state_vector, actions, reward, expected_reward, device='cpu'):
        current_time = time.time()

        reward = float(reward)
        expected_reward = float(expected_reward)
        surprise = F.mse_loss(torch.tensor([reward]).to(device), torch.tensor([expected_reward]).to(device), reduction='none').item()

        if not self.memory:
            memory_uid = str(uuid.uuid4())
            print('adding memory', memory_uid)
            self.memory[memory_uid] = {
                'vector': state_vector,
                'time': current_time,
                'time_last_recall': current_time,
                'reward': reward,
                'actions': actions,
                'surprise': surprise,
                'max_similarity': 0.0,
            }
            return memory_uid

        if np.random.rand() < 0.01:
            self.remove_lowest_priority_memory(device=device)

        _, memory_values = zip(*self.memory.items())

        memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
        surprise_cutoff = torch.quantile(memory_surprises, 0.5)

        max_similarity, _ = self.highest_similarity_score(state_vector, device=device)
        if len(self.memory) < self.max_memory_size or surprise > surprise_cutoff:
            memory_uid = str(uuid.uuid4())
            print('adding memory', surprise, max_similarity)
            self.memory[memory_uid] = {
                'vector': state_vector,
                'time': current_time,
                'time_last_recall': current_time,
                'reward': reward,
                'actions': actions,
                'surprise': surprise,
                'max_similarity': max_similarity,
            }
        else:
            memory_uid = None

        while len(self.memory) > self.max_memory_size:
            self.remove_lowest_priority_memory(device=device)

        return memory_uid

    def remove_lowest_priority_memory(self, device='cpu'):
        current_time = time.time()
        with torch.no_grad():
            _, memory_values = zip(*self.memory.items())
            memory_times = torch.tensor([mv['time_last_recall'] - current_time for mv in memory_values])

            recency_scores = LimbicSystem._get_scores(memory_times)

            scores = recency_scores
            _, index = torch.min(scores, dim=0)
            print('deleting memory', memory_times[index].item())
            del self.memory[list(self.memory.keys())[index]]
    
    def _get_scores(vector):
        _, sorted_indices = torch.sort(vector)
        sorted_rewards = vector[sorted_indices]
        _, inverse_indices, counts = torch.unique_consecutive(sorted_rewards, return_inverse=True, return_counts=True)
        cumulative_counts = counts.cumsum(0)
        start_ranks = cumulative_counts - counts
        average_ranks = (start_ranks + cumulative_counts - 1).float() / 2
        scores = average_ranks[inverse_indices]
        scores = scores / len(vector)
        original_order_scores = torch.empty_like(scores)
        original_order_scores[sorted_indices] = scores
        scores = original_order_scores

        return scores

class Mammal(nn.Module):
    def __init__(
            self, 
            hidden_dim,
            num_heads,
            num_layers,
            action_dict,
            rollout_length=20,
    ):
        super(Mammal, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dict = action_dict
        self.num_state_tokens = 2

        self.central = MammalCentral(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_state_tokens=self.num_state_tokens,
            num_layers=num_layers,
        )
        self.limbic_system = LimbicSystem(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            memories_fetched_per_step=128,
            max_memory_size=8192,
            action_dict=action_dict,
        )

        self.action_heads = nn.ModuleDict({
            f'action_{key}': nn.Linear(hidden_dim, value) for key, value in action_dict.items()
        })
        self.env_value_head = nn.Linear(hidden_dim, 1)
        self.inner_value_head = nn.Linear(hidden_dim, 1)

        self.rollout_length = rollout_length
        self.rollout_length_so_far = 0
        self.rollouts = {
            'rewards': [],
            'actions': [],
            'env_values': [],
            'states': [],
        }

        self.decay = 0.95

        # self.recurrent_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.recurrent_state = (torch.zeros(1, hidden_dim), torch.zeros(1, hidden_dim))
        self.recurrent_layer_norm = nn.LayerNorm(hidden_dim)
        self.env_reward_enc = nn.Linear(1, hidden_dim)

        self.max_lr = 1e-3
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        self.n_optimizer_steps = 0

        self.n_forwards = 0
        self.losses_accumulated = 0 

    def forward(self, image, reward, debug=False, device='cpu'):
        self.recurrent_state = (self.recurrent_state[0].to(device), self.recurrent_state[1].to(device))
        env_reward_encoded = self.env_reward_enc(torch.tensor([reward]).float().to(device)).unsqueeze(0).unsqueeze(0)
        state = torch.cat((torch.zeros(1, 1, self.hidden_dim).to(device), env_reward_encoded), dim=1)

        x = self.central(image, state, device=device)
        output = None
        value_prediction = None
        memory_losses = None
        memory_actions = None
        memory_rewards = None
        if len(self.limbic_system.memory) > 0:
            limbic_output = self.limbic_system(x[:, 0], device=device)
            output = limbic_output['output_vectors']
            value_prediction = limbic_output['value_prediction']
            memory_losses = limbic_output['memory_losses']
            memory_actions = limbic_output['memory_actions']
            memory_rewards = limbic_output['memory_rewards']

        if output is None:
            output = x[:, 0]
            value_prediction = self.limbic_system.value_head(output)

        actions_vecs = {key: head(output) for key, head in self.action_heads.items()}

        actions = {}
        all_actions_with_distributions = []
        for action in actions_vecs:
            distributions = torch.softmax(actions_vecs[action], dim=-1)
            actions[action] = torch.multinomial(distributions[0], 1)[0].cpu().item()

            actions_taken = [actions[action]]
            if memory_actions is not None:
                for memory in memory_actions:
                    actions_taken.append(memory[action])
            all_actions_with_distributions.append((action, distributions, actions_taken))
        
        self.rollouts['rewards'].append((self.n_forwards, reward))
        self.rollouts['actions'].append((self.n_forwards, sorted(actions.items())))
        self.rollouts['env_values'].append((self.n_forwards, value_prediction))
        self.rollouts['states'].append((self.n_forwards, output[:1].detach()))
        self.rollout_length_so_far += 1

        all_returns = []
        if len(self.rollouts['rewards']) >= self.rollout_length:
            env_returns = []
            R = 0
            for r in self.rollouts['rewards'][:0:-1]:
                R = r[1] + self.decay * R
                env_returns.insert(0, R)
            env_returns = torch.tensor(env_returns).to(device)

            surprise = F.mse_loss(self.rollouts['env_values'][0][1].squeeze().to(device), env_returns[0], reduction='none').item()

            exploration_bonus = 0.1 * self.limbic_system.get_intrinsic_reward(self.rollouts['states'][0][1], surprise, device=device)

            self.rollouts['rewards'][-1] = (self.rollouts['rewards'][-1][0], self.rollouts['rewards'][-1][1] + exploration_bonus)
            env_returns[0] = env_returns[0] + (self.decay ** (self.rollout_length - 1)) * exploration_bonus
            
            self.limbic_system.add_memory(self.rollouts['states'][0][1], {action[0]: action[1] for action in self.rollouts['actions'][0][1]}, env_returns[0], self.rollouts['env_values'][0][1], device=device)

            all_returns.append(env_returns[0])
            if memory_rewards is not None:
                for memory_reward in memory_rewards:
                    all_returns.append(memory_reward)
        if len(all_returns) > 0:
            all_returns = torch.stack(all_returns).to(device)

        stepped = False
        if self.rollout_length_so_far >= self.rollout_length:
            stepped = self.learn(all_actions_with_distributions, all_returns, memory_losses, device=device)
            if stepped:
                output = output.detach()
                self.rollout_length_so_far = 0

        if len(self.rollouts['rewards']) >= self.rollout_length:
            self.rollouts['rewards'] = self.rollouts['rewards'][1:]
            self.rollouts['actions'] = self.rollouts['actions'][1:]
            self.rollouts['states'] = self.rollouts['states'][1:]
            self.rollouts['env_values'] = self.rollouts['env_values'][1:]

        if debug:
            for action in actions_vecs:
                distributions = torch.softmax(actions_vecs[action], dim=-1)
                # print(action, distributions[0].detach().cpu().numpy().tolist())
                # round to nearest thousandth
                print(action, [round(x, 3) for x in distributions[0].detach().cpu().numpy().tolist()])

        self.n_forwards += 1

        actions_to_return = {}
        for action in actions:
            # remove 'action_' from the action name
            action_name = action[7:]
            actions_to_return[action_name] = actions[action]
            if 'camera' in action_name:
                actions_to_return[action_name] -= 5
        return actions_to_return

    def learn(self, all_actions, all_returns, memory_losses=None, device='cpu'):
        
        policy_losses = []
        entropy_losses = []
        if all_actions is not None:
            for memory_action in all_actions:
                policy_losses.append(-torch.log(memory_action[1][0, memory_action[2]] + 1e-5) * all_returns)
                entropy_losses.append(torch.mean(-torch.log(memory_action[1] + 1e-5)))
        
        policy_loss = torch.cat(policy_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()

        value_loss = F.mse_loss(self.rollouts['env_values'][0][1].squeeze().to(device), all_returns[0], reduction='none')
        if memory_losses is not None:
            value_loss = torch.cat((value_loss.unsqueeze(0), memory_losses)).mean()

        loss = 1 * policy_loss + 1 * value_loss + 0.001 * entropy_loss
        print(f'After {self.n_optimizer_steps} steps and {self.n_forwards} forwards, loss: {loss.item():.3f}, policy_loss: {policy_loss.item():.3f}, value_loss: {value_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}')

        # warmup the optimizer over the first 10 steps
        if self.n_optimizer_steps <= 10:
            lr = self.max_lr * (self.n_optimizer_steps / 10)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        stepped = False
        if self.losses_accumulated == self.rollout_length - 1:
            loss.backward()
            print('stepping')
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            self.n_optimizer_steps += 1
            self.recurrent_state = (self.recurrent_state[0].detach(), self.recurrent_state[1].detach())
            stepped = True
            self.losses_accumulated = 0
        else:
            loss.backward(retain_graph=True)
            self.losses_accumulated += 1

        return stepped

    def reset(self, device):
        self.optimizer.zero_grad()
        self.recurrent_state = (torch.zeros(1, self.hidden_dim).to(device), torch.zeros(1, self.hidden_dim).to(device))
        self.rollouts['rewards'] = []
        self.rollouts['actions'] = []
        self.rollouts['env_values'] = []
        self.rollouts['states'] = []

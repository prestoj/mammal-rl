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


class Vision(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads,
            num_state_tokens,
            num_layers,
    ):
        super(Vision, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_state_tokens = num_state_tokens

        self.vision = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
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
                proj_drop=0.2,
                attn_drop=0.2,
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


class NextTokenModel(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads,
            num_layers,
            memories_fetched_per_step,
            max_memory_size,
            action_dict,
    ):
        super(NextTokenModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.action_dict = action_dict

        self.start_token = nn.Parameter(torch.rand(1, hidden_dim) * 2 - 1)
        self.positional_embeddings = nn.Parameter(torch.rand(memories_fetched_per_step * 2 + 2, hidden_dim) * 2 - 1)

        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation='gelu',
                norm_first=True,
                dropout=0.2,
            ),
            num_layers=num_layers,
            mask_check=False,
        )

        self.actions_enc = nn.ModuleDict({
            f'action_enc_{key}': nn.Sequential(*[
                nn.Linear(value, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            ]) for key, value in action_dict.items()
        })
        self.value_head = nn.Sequential(*[
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * hidden_dim, 1),
        ])

        self.memory = {}
        self.max_memory_size = max_memory_size
        self.memories_fetched_per_step = memories_fetched_per_step
    
    def forward(self, x, device='cpu'):
        current_time = time.time()
        if not self.memory:
            return x.unsqueeze(0)

        with torch.no_grad():
            memory_keys, memory_values = zip(*sorted(self.memory.items(), key=lambda x: x[1]['time']))
            memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
            memory_next_vectors = torch.stack([mv['next_vector'] for mv in memory_values]).to(device)
            memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)
            memory_similarities = torch.cosine_similarity(x, memory_vectors, dim=-1)
            memory_rewards = torch.tensor([mv['reward'] for mv in memory_values]).to(device)

            surprise_score = NextTokenModel._get_scores(memory_surprises) ** 2
            similarity_scores = NextTokenModel._get_scores(memory_similarities) ** 2
            stochasticity_scores = NextTokenModel._get_scores(torch.rand_like(memory_similarities)) ** 2

            scores = surprise_score + similarity_scores + stochasticity_scores
            scores = scores.to(device)

            _, indices = torch.topk(scores, min(self.memories_fetched_per_step, scores.shape[0]), dim=0)

            for index in indices:
                memory_key = memory_keys[index]
                self.memory[memory_key]['time_last_recall'] = current_time

        selected_vectors = memory_vectors[indices]
        selected_vectors = selected_vectors.squeeze(1).squeeze(1).detach()

        target_vector = x

        # convert actions to vectors
        action_vectors = []
        for action in self.action_dict:
            action_vector = torch.zeros(selected_vectors.shape[0], self.action_dict[action]).to(device)
            for i, index in enumerate(indices):
                memory_key = memory_keys[index]
                action_vector[i, self.memory[memory_key]['actions'][f'action_{action}']] = 1
            action_vectors.append(self.actions_enc[f'action_enc_{action}'](action_vector))

        action_vectors = torch.stack(action_vectors).mean(0)

        vectors = torch.stack((selected_vectors, action_vectors), dim=1)
        vectors = vectors.reshape(-1, vectors.shape[-1])
        vectors = torch.cat((self.start_token, vectors, target_vector), dim=0)
        vectors = vectors + self.positional_embeddings[:vectors.shape[0]]

        mask = nn.Transformer.generate_square_subsequent_mask(vectors.shape[0], device=device)

        vectors = self.transformer_blocks(vectors, mask=mask)

        value_indices = torch.arange(0, len(selected_vectors) + 1) * 2 + 1
        value_prediction = self.value_head(vectors[value_indices])
        memory_losses = F.mse_loss(value_prediction[:-1].flatten(), memory_rewards[indices].flatten(), reduction='none')

        memory_actions = []
        for i, index in enumerate(indices):
            memory_key = memory_keys[index]
            self.memory[memory_key]['surprise'] = memory_losses[i].detach().item()
            memory_actions.append(self.memory[memory_key]['actions'])

        return {
            'output_vectors': vectors[1:],
            'memory_next_vectors': memory_next_vectors[indices],
            'value_prediction': value_prediction[-1],
            'memory_losses': memory_losses.flatten(),
            'memory_actions': memory_actions,
            'memory_rewards': memory_rewards[indices].flatten(),
        }
    
    def highest_similarity_score(self, x, device='cpu'):
        if not self.memory:
            return None

        with torch.no_grad():
            _, memory_values = zip(*self.memory.items())
            memory_vectors = torch.stack([mv['vector'] for mv in memory_values]).to(device)
            memory_similarities = torch.cosine_similarity(x, memory_vectors, dim=-1)

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

            memory_similarities = torch.cosine_similarity(x, memory_vectors, dim=-1)
            memory_max_similarities = torch.tensor([mv['max_similarity'] for mv in memory_values]).to(device)

            # memory_surprises = torch.tensor([mv['surprise'] for mv in memory_values]).to(device)

            # score = (memory_surprises < surprise).float().mean()
            score = (memory_max_similarities > memory_similarities.max().item()).float().mean()
            # score = score / 2

        return float(score.item())

    def add_memory(self, state_vector, next_state_vector, actions, reward, expected_reward, device='cpu'):
        current_time = time.time()

        reward = float(reward)
        expected_reward = float(expected_reward)
        reward_surprise = F.mse_loss(torch.tensor([reward]).to(device), torch.tensor([expected_reward]).to(device), reduction='none').item()
        # state_surprise = F.mse_loss(state_vector, next_state_vector, reduction='none').mean().item()
        # surprise = (reward_surprise + state_surprise) / 2
        surprise = reward_surprise

        if not self.memory:
            memory_uid = str(uuid.uuid4())
            print('adding memory', memory_uid)
            self.memory[memory_uid] = {
                'vector': state_vector,
                'next_vector': next_state_vector,
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
                'next_vector': next_state_vector,
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

            recency_scores = NextTokenModel._get_scores(memory_times)

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

class AnimalGuy(nn.Module):
    def __init__(
            self, 
            hidden_dim,
            num_heads,
            num_layers,
            action_dict,
            rollout_length=16,
            device='cpu',
    ):
        super(AnimalGuy, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dict = action_dict
        self.num_state_tokens = 2

        self.vision = Vision(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_state_tokens=self.num_state_tokens,
            num_layers=num_layers,
        )

        self.vision_to_next = nn.Sequential(*[
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * hidden_dim, hidden_dim),
        ])

        self.next_token_model = NextTokenModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            memories_fetched_per_step=64,
            max_memory_size=1024,
            action_dict=action_dict,
        )

        self.action_heads = nn.ModuleDict({
            f'action_{key}': nn.Sequential(*[
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, value),
            ]) for key, value in action_dict.items()
        })
         # TODO split this into a Linear(hidden_dim, 2*hidden_dim) to handle next state probabilistically (i.e. with a mean and variance)
        self.state_head = nn.Sequential(*[
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * hidden_dim, hidden_dim),
        ])

        self.rollout_length = rollout_length
        self.rollout_length_so_far = 0
        self.rollouts = {
            'rewards': [],
            'actions': [],
            'env_values': [],
            'states': [],
        }

        self.decay = 0.99

        self.env_reward_enc = nn.Linear(1, hidden_dim)

        self.max_lr = 1e-4
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=1e-5)
        self.n_optimizer_steps = 0

        self.n_forwards = 0
        self.losses_accumulated = 0 

    def forward(self, image, reward, debug=False, device='cpu'):
        reward = float(reward)
        encoded_reward = self.env_reward_enc(torch.tensor([reward]).to(device)).unsqueeze(0).unsqueeze(0)
        state = torch.cat([torch.zeros(1, 1, self.hidden_dim).to(device), encoded_reward], dim=1)

        x = self.vision(image, state, device=device)
        x = self.vision_to_next(x[:, 0])

        output = None
        value_prediction = None
        memory_losses = None
        memory_actions = None
        memory_rewards = None
        if len(self.next_token_model.memory) > 0:
            model_output = self.next_token_model(x, device=device)
            output = model_output['output_vectors']
            value_prediction = model_output['value_prediction']
            memory_next_vectors = model_output['memory_next_vectors']
            memory_losses = model_output['memory_losses']
            memory_actions = model_output['memory_actions']
            memory_rewards = model_output['memory_rewards']

            state_indices = torch.arange(1, len(output), 2)
            action_indices = torch.arange(0, len(output), 2)

            state_inputs = output[state_indices]
            action_inputs = output[action_indices]

        if output is None:
            output = x
            value_prediction = self.next_token_model.value_head(output)[0]

            action_inputs = output

            memory_next_vectors = None
            state_inputs = None

        actions_vecs = {key: head(action_inputs) for key, head in self.action_heads.items()}

        actions = {}
        all_actions_with_distributions = []
        for action in actions_vecs:
            distributions = torch.softmax(actions_vecs[action], dim=-1)
            actions[action] = torch.multinomial(distributions[-1], 1)[0].cpu().item()

            actions_taken = [actions[action]]
            if memory_actions is not None:
                for memory in memory_actions:
                    actions_taken.append(memory[action])
            all_actions_with_distributions.append((action, distributions, actions_taken))
        
        self.rollouts['rewards'].append((self.n_forwards, reward))
        self.rollouts['actions'].append((self.n_forwards, sorted(actions.items())))
        self.rollouts['env_values'].append((self.n_forwards, value_prediction))
        self.rollouts['states'].append((self.n_forwards, output[-1].detach()))
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

            exploration_bonus = 0.0 * self.next_token_model.get_intrinsic_reward(self.rollouts['states'][0][1], surprise, device=device)

            self.rollouts['rewards'][-1] = (self.rollouts['rewards'][-1][0], self.rollouts['rewards'][-1][1] + exploration_bonus)
            env_returns[0] = env_returns[0] + (self.decay ** (self.rollout_length - 1)) * exploration_bonus
            
            self.next_token_model.add_memory(self.rollouts['states'][0][1], self.rollouts['states'][1][1], {action[0]: action[1] for action in self.rollouts['actions'][0][1]}, env_returns[0], self.rollouts['env_values'][0][1], device=device)

            if memory_rewards is not None:
                for memory_reward in memory_rewards:
                    all_returns.append(memory_reward)
            all_returns.append(env_returns[0])

        if len(all_returns) > 0:
            all_returns = torch.stack(all_returns).to(device)

        stepped = False
        if self.rollout_length_so_far >= self.rollout_length:
            stepped = self.learn(all_actions_with_distributions, all_returns, memory_losses, state_inputs, memory_next_vectors, device=device)
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

    def learn(self, all_actions, all_returns, memory_losses, state_inputs, target_next_states, device='cpu'):
        
        policy_losses = []
        entropy_losses = []
        if all_actions is not None:
            for memory_action in all_actions:
                policy_losses.append(-torch.log(memory_action[1][0, memory_action[2]] + 1e-5) * all_returns)
                entropy_losses.append(torch.mean(-torch.log(memory_action[1] + 1e-5)))

        policy_loss = torch.cat(policy_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()

        value_loss = F.mse_loss(self.rollouts['env_values'][0][1].to(device), all_returns[-1:], reduction='none')
        if memory_losses is not None:
            value_loss = torch.cat((value_loss, memory_losses)).mean()

        if state_inputs is not None:
            predicted_states = self.state_head(state_inputs)
            state_loss = F.mse_loss(predicted_states, target_next_states.detach(), reduction='none').mean()
        else:
            state_loss = torch.tensor([0.0]).to(device)

        loss = 1 * policy_loss + 1 * value_loss + 1 * state_loss + 1e-5 * entropy_loss
        # print(f'After {self.n_optimizer_steps} steps and {self.n_forwards} forwards, loss: {loss.item():.3f}, policy_loss: {policy_loss.item():.3f}, value_loss: {value_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}')
        print(f'After {self.n_optimizer_steps} steps and {self.n_forwards} forwards, loss: {loss.item():.3f}, policy_loss: {policy_loss.item():.3f}, value_loss: {value_loss.item():.3f}, state_loss: {state_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}')

        loss = loss / self.rollout_length

        # warmup the optimizer over the first 1000 steps
        if self.n_optimizer_steps <= 1000:
            lr = self.max_lr * (self.n_optimizer_steps / 1000)
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
            stepped = True
            self.losses_accumulated = 0
        else:
            loss.backward(retain_graph=True)
            self.losses_accumulated += 1

        return stepped

    def reset(self, device):
        self.optimizer.zero_grad()
        self.rollouts['rewards'] = []
        self.rollouts['actions'] = []
        self.rollouts['env_values'] = []
        self.rollouts['states'] = []

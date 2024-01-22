import time
import numpy as np
import gym
import minerl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mammal import Mammal
import gc
import multiprocessing
import time
import pandas as pd

action_space = {
    "forward": 2,
    "back": 2,
    "left": 2,
    "right": 2,
    "jump": 2,
    "sneak": 2,
    "sprint": 2,
    "attack": 2,
    "camera_0": 11,
    "camera_1": 11,
}

def worker(env_name, queue):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    model = Mammal(
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
        action_dict=action_space,
    )
    model = model.to(device)
    checkpoint = torch.load('model_and_optimizer.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.n_optimizer_steps = checkpoint['n_optimizer_steps']
    model.limbic_system.memory = checkpoint['memory']
    # for uid in model.limbic_system.memory:
    #     print(f"{uid}, {model.limbic_system.memory[uid]['reward']:.3f}, {model.limbic_system.memory[uid]['surprise']:.3f}, {model.limbic_system.memory[uid]['max_similarity']:.3f}")
    # convert limbic system memory attributes to pd dataframe and sort by reward, only looking at reward surprise and max_similarity
    df = pd.DataFrame.from_dict(model.limbic_system.memory, orient='index')
    df = df.sort_values(by=['reward'], ascending=False)
    df = df[['reward', 'surprise', 'max_similarity']]
    print(df.head(20))

    # count the number of rewards less than 0.001
    print((df['reward'] < 0.001).sum())
    
    model = model.to(device)

    # Assuming 'model' is your neural network model
    mean_weights = []
    max_weights = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Flatten the parameter tensor
        param_flat = param.view(-1)

        # Compute mean and max
        mean_weights.append(param_flat.abs().mean().item())
        max_weights.append(param_flat.abs().max().item())

    # Calculate the overall mean and max across all parameters
    overall_mean = sum(mean_weights) / len(mean_weights)
    overall_max = max(max_weights)

    print(f"Mean Weight: {overall_mean}")
    print(f"Max Weight: {overall_max}")

    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    env = gym.make(env_name)
    obs = env.reset()
    done = False
    total_reward = 0
    step_reward = 0
    i = 0
    while not done:

        # time.sleep(0.05)
        env.render()
        i += 1
        if i % 5 != 1:
            obs, reward, done, info = env.step(actions)
            # if np.random.rand() < 0.01:
            #     if actions['forward'] == 0:
            #         reward += 1
            #     if actions['back'] == 0:
            #         reward += 1
            #     if actions['left'] == 0:
            #         reward += 1
            #     if actions['right'] == 0:
            #         reward += 1
            #     if actions['jump'] == 0:
            #         reward += 1
            #     if actions['sneak'] == 0:
            #         reward += 1
            #     if actions['sprint'] == 0:
            #         reward += 1
            #     if actions['attack'] == 0:
            #         reward += 1
            #     if actions['camera'][0] == 0:
            #         reward += 1
            #     if actions['camera'][1] == 0:
            #         reward += 1
            step_reward += reward
            continue


        total_reward += step_reward
        
        # get the image from the observation
        image = obs['pov']
        # print(step_reward)

        t0 = time.time()
        actions = model(image, step_reward, debug=True, device=device)

        # combine camera_0 and camera_1 into a single action
        actions['camera'] = [actions['camera_0'], actions['camera_1']]
        # print(i, actions, time.time() - t0)

        # take a step in the environment
        obs, reward, done, info = env.step(actions)

        # # reward the model for seeing red, making sure to exclude "white" where green and blue are also high
        # reward += float((image[:, :, 0] / (image.sum(axis=2) + 1e-5)).mean())

        # if np.random.rand() < 0.01:
        #     if actions['forward'] == 0:
        #         reward += 1
        #     if actions['back'] == 0:
        #         reward += 1
        #     if actions['left'] == 0:
        #         reward += 1
        #     if actions['right'] == 0:
        #         reward += 1
        #     if actions['jump'] == 0:
        #         reward += 1
        #     if actions['sneak'] == 0:
        #         reward += 1
        #     if actions['sprint'] == 0:
        #         reward += 1
        #     if actions['attack'] == 0:
        #         reward += 1
        #     if actions['camera'][0] == 0:
        #         reward += 1
        #     if actions['camera'][1] == 0:
        #         reward += 1
        # if step_reward > 0:
        #     print('Reward:', step_reward)

        step_reward = reward

    # Send rewards or other data back to the main process if needed
    queue.put(total_reward)

    env.close()

def main():
    n_games = 0
    while True:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=("MineRLTreechop-v0", queue))
        p.start()
        p.join()  # Wait for the process to complete

        # Retrieve data from the queue if needed
        total_reward = queue.get()
        print(n_games, total_reward)

        n_games += 1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
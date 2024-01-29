import time
import numpy as np
import gym
import minerl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mammal import Mammal
from next_token_rl import AnimalGuy
import gc
import multiprocessing

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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = AnimalGuy(
        hidden_dim=384,
        num_heads=6,
        num_layers=3,
        action_dict=action_space,
        device=device,
    )
    model = model.to(device)
    checkpoint = torch.load('model_and_optimizer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.n_optimizer_steps = checkpoint['n_optimizer_steps']
    model.next_token_model.memory = checkpoint['memory']
    model = model.to(device)
    model.train()

    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    env = gym.make(env_name)
    obs = env.reset()
    done = False
    total_reward = 0
    step_reward = 0
    i = 0
    while not done:
        i += 1
        if i % 5 != 1:
            obs, reward, done, info = env.step(actions)
            step_reward += reward
            continue
        # get the image from the observation
        image = obs['pov']

        # get the action from the model
        actions = model(image, step_reward, device=device)

        # combine camera_0 and camera_1 into a single action
        actions['camera'] = [actions['camera_0'], actions['camera_1']]

        # take a step in the environment
        obs, reward, done, info = env.step(actions)
        if step_reward > 0:
            print('Reward:', step_reward)
        total_reward += step_reward
        step_reward = reward

    # Send rewards or other data back to the main process if needed
    queue.put(total_reward)

    env.close()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'n_optimizer_steps': model.n_optimizer_steps,
        'memory': model.next_token_model.memory,
    }, 'model_and_optimizer.pth')


def main():
    n_games = 0
    all_rewards = []
    while True:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=("MineRLTreechop-v0", queue))
        p.start()
        p.join()  # Wait for the process to complete

        # Retrieve data from the queue if needed
        total_reward = queue.get()
        print(n_games, total_reward)

        all_rewards.append(total_reward)
        print(all_rewards)

        n_games += 1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # model = AnimalGuy(
    #     hidden_dim=384,
    #     num_heads=6,
    #     num_layers=3,
    #     action_dict=action_space,
    #     device='cuda:0',
    # )
    # model = model.to('cuda:0')
    # print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': model.optimizer.state_dict(),
    #     'n_optimizer_steps': model.n_optimizer_steps,
    #     'memory': model.next_token_model.memory,
    # }, 'model_and_optimizer.pth')

    main()

import gym
import os
import numpy as np
from tqdm import tqdm
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_tspolicy
from envs.bags_games_cifar import BagsGamesCIFAR
from envs.hsja_games_cifar import HsjaGamesCIFAR
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10(os.getcwd() + '/data', train=False, transform=transform, download=True)

eval_steps = 5000
adaptive = 0 # non-adaptive, just stateful defense 
ratio = 0.5
defended = True
cont = 0
seed = 2

# Make evaluation env
env = gym.make("BagsGamesCIFAR-v0",
               steps=eval_steps,
               ratio_benign=ratio,
               adaptive=adaptive,
               dataset=dataset,
               defended=defended,
               train=True,
               rint=1,
               radv=1,
               intercept=1)
    
interceptor = RPPO.load("mods/games/bags4int_14.pt" , env, "interceptor", seed)
adversary = RPPO.load("mods/games/bags4adv.pt", env, "adversary", seed)
benign = RandomAgent(env=env)

mean_rint, std_rint, stl1, orl1, epsilons, acc1, mean_eps, start1, mean_acc1 = evaluate_tspolicy(interceptor, adversary, benign, env, act_size=4, n_eval_episodes=100)



# a1 = np.array(stl1)
# b1 = np.array(orl1)
# c1 = np.array(start1)
# e1 = np.array(acc1)
# d1 = np.column_stack((a1,b1,c1,e1))

# # Make evaluation env
# env = gym.make("HsjaGamesCIFAR-v0",
#                steps=eval_steps,
#                ratio_benign=ratio,
#                adaptive=adaptive,
#                dataset=dataset,
#                defended=defended,
#                cont=cont,
#                train=True,
#                rint=1,
#                radv=1,
#                intercept=1)
    
# interceptor = RPPO.load("mods/games/hsja4int_0.pt" , env, "interceptor", seed)
# adversary = RPPO.load("mods/games/hsja4adv.pt", env, "adversary", seed)

# benign = RandomAgent(env=env)

# mean_rint, std_rint, stl2, orl2, epsilons, acc2, mean_eps, start2, mean_acc2 = evaluate_tspolicy(interceptor, adversary, benign, env, n_eval_episodes=100)

# a2 = np.array(stl2)
# b2 = np.array(orl2)
# c2 = np.array(start2)
# e2 = np.array(acc2)
# d2 = np.column_stack((a2,b2,c2,e2))

# d3 = np.column_stack((a1,a2,b1,b2,c1,c2,e1,e2))


# print(mean_acc1, mean_acc2)

# for i, d in enumerate(d1):
#     if (d1[i] != d2[i]).any():
#         print("Episode:", i)
#         print("V1:", d1[i])
#         print("V2:", d2[i])
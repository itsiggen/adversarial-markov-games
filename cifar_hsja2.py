import gym
import os
import numpy as np
from agents.rppo import RPPO
from agents.benign import RandomAgent
from torchvision import datasets, transforms
from utils.evaluation import evaluate_rdpolicy, evaluate_rtpolicy
from envs.hsja_games_cifar import HsjaGamesCIFAR
os.environ['CUDA_VISIBLE_DEVICES'] = ''

transform = transforms.ToTensor()
dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)

eval_steps = 5000
adaptive = 0 # non-adaptive, just stateful defense 
ratio = 0.5
defended = True
cont = 0
seed = 2

# Make evaluation env
env = gym.make("HsjaGamesCIFAR-v0",
               steps=eval_steps,
               ratio_benign=ratio,
               adaptive=adaptive,
               dataset=dataset,
               defended=defended,
               cont=cont,
               train=False,
               rint=1,
               radv=1,
               intercept=1)
    
interceptor = RPPO.load("mods/games/hsja4int_0.pt" , env, "interceptor", seed)
adversary = RPPO.load("mods/games/hsja4adv.pt", env, "adversary", seed)

benign = RandomAgent(env=env)

mean_rint, std_rint, mean_radv, std_radv, epsilons, iters, mean_eps, start_eps, mean_acc = evaluate_rtpolicy(interceptor, adversary, benign, env, n_eval_episodes=100)


res = [mean_eps, start_eps, mean_acc]

z = list(zip(iters,epsilons))
a = [np.interp(1000, i[0], i[1]) for i in z]
b = [np.interp(2000, i[0], i[1]) for i in z]
c = np.mean(a)
d = np.mean(b)
print('chsja2:', res, c, d)
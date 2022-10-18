import pickle5 as pickle
# import cloudpickle as cp

with open('hsjaskip_best2.pt', 'rb') as pickle_file:
    a = pickle.load(pickle_file)

# a = pickle.load('hsjaskip_best2.pt')
# pickle.dump(obj, file, protocol=None
pickle.dump(a, 'hshsh', protocol=4)

# from stable_baselines3 import PPO

# model = PPO.load('hsjaskip_best3.pt')
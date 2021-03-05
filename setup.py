from setuptools import setup, find_packages

long_description = """
[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Do
# Adversarial Markov Games
Adversarial Markov Games is a set of reinforcement learning environments where adaptive attacks
and defenses against machine learning models can be performed and evaluated.
## Requirements
## Links
Repository:
https://github.com/itsiggen/adversarial-markov-games
Documentation:
## Example
"""

setup(name='adversarial_markov_games',
      install_requires=[
          'gym[atari,classic_control]>=0.11',
          'scipy',
          'numpy',
          'pandas',
          'ray'
          'matplotlib'],
      description='Adversarial Markov Games Package',
      author='Ilias Tsingenopoulos',
      url='https://github.com/itsiggen/adversarial-markov-games',
      author_email='ashley.hill@u-psud.fr',
      keywords="adversarial-machine-learning, reinforcement-learning gym openai data-science",
      license="MIT",
      long_description='Adversarial Markov Games Package',
      long_description_content_type='text/markdown',
      version="0.0.1",
      )
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Project 3: Collaboration and Competition - Report

### DDPG ('Deep Deterministic Policy Gradient') Algorithm and Hyperparameters

I used DDPG for the assignment, although I had originally wanted to use PPO, which apparently tends to converge faster with better solutions. Unfortunately, I couldn't get PPO to function properly. DDPG is a policy gradient method. Policy gradient methods are a subset of poliy-based methods. While the latter search directly for the optimal policy, policy gradient methods estimate the best weights by gradient descent. Essentially, they estimate the gradient via a neural network rather than making direct guesses. It is similar to actor-critic methods, except the actor maps states to actions directly rather than stochastically. The target networks are time-delayed to avoid interdependence on the outputs of the original networks.

I used the following hyperparameters as constants in ddpg_classes.py for the PyTorch DDPG agent class:

exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.3
actor_learning_rate = 1e-4
discount = 0.99
self.tau = 1e-3 (for soft update of target parameters)

For experience replay:

buffer_size = 100000
batch_size = 64
seed = 2

The actor network consists of the following structure:

- 512 linear layers with size of 256 + 33 (for state space size).
- A linear output layer with 512 + 33 inputs and outputs corresponding to the number of actions.

It uses torch.nn.BatchNorm1d for normalization.

The critic network consists of the following structure:

- 1 linear layer with size of 256 + 33 (for state space size) + number of actions .
- 511 linear layers with size of 256 + 33 (for state space size).
- A linear output layer with 512 + 33 inputs and outputs corresponding to the number of actions.

It uses torch.nn.BatchNorm1d for normalization and dropout with a value of 0.2.

### Room for improvement

The main thing I would improve is simply to get PPO working, as it should converge faster with a smoother curve.

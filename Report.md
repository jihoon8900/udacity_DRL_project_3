[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Report for Project 3: Collaboration and Competition 

## Introduction

In this `Report.md`, you can see an implementation for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Trained Agent][image1]
## Implementation specs

### 1. Summary

I implement the [Twin Delayed Deep Deterministic policy gradient algorithm (TD3)](https://arxiv.org/abs/1802.09477) and apply TD3 to multi-agent.

--------

### 2. Details

#### 2-1. Concepts

- Clipped Double-Q Learning
- Delayed Policy Updates for TD3(Twin Delayed DDPG)
- Multi-agent actor-critic

- Actor: Actor uses only local states & local ations.
- Critic: Critic uses states of all agents & actions of all agents.


------------
#### 2-2. Networks

##### 2-2-1. Actor

| Layer | Type | Dimension |
|:---:|:---:|:---:|
| `Input` | Input (state) | 8 |
| `BatchNorm` | Batch Normalization | - |
| `1st hidden layer` | Fully connected layer | 256 |
| `BatchNorm` | Batch Normalization | - |
| `Activation function` | LeakyReLu function | - |
| `2nd hidden layer` | Fully connect    ed layer | 128 |
| `BatchNorm` | Batch Normalization | - |
| `Activation function` | LeakyReluReLu function | - |
| `3rd hidden layer` | Fully connected layer | 2 |
| `Activation function` | tanh function | - |
| `(Output)` | Output (action) | 2 |

##### 2-2-2. Critic

| Layer | Type | Dimension |
|:---:|:---:|:---:|
| `Input` | Input (state_size * agent_num) | 8 * 2 |
| `BatchNorm` | Batch Normalization | - |
| `1st hidden layer` | Fully connected layer | 256 |
| `Activation function` | ReLu function | - |
| `Concat` | Concatenate with action(256+action_size * agent_num) | (256+2*2) |
| `2nd hidden layer` | Fully connected layer | 128 |
| `Activation function` | ReLu function | - |
| `3rd hidden layer` | Fully connected layer | 1 |

#### 2-3. Hyperparameters

| parameter    | value  | description                                                                   |
|--------------|--------|-------------------------------------------------------------------------------|
| BUFFER_SIZE  | 1e6    | Number of experiences to keep on the replay memory for the TD3                |
| BATCH_SIZE   | 256    | Minibatch size used at each learning step                                     |
| GAMMA        | 0.99   | Discount applied to future rewards                                            |
| TAU          | 4e-2   | Scaling parameter applied to soft update                                      |
| LR_ACTOR     | 6e-4   | Learning rate for actor used for the Adam optimizer                           |
| LR_CRITIC    | 2e-3   | Learning rate for critic used for the Adam optimizer                          |
| NUM_LEARN    | 8      | Number of learning at each step                                               |
| NUM_TIME_STEP| 10     | Every NUM_TIME_STEP do update                                                 |
| EPSILON      | 1.0    | Epsilon to noise of action                                                    |
| EPSILON_DECAY| 2e-5   | Epsilon decay to noise epsilon of action                                      |
| POLICY_DELAY | 3      | Delay for policy update (TD3)                                                 |
| AGENT_NUM    | 2      | Number of agents for multi-agent                                              |
| n_episodes   | 3000   | Maximum number of training episodes (Training hyperparameters for `train` function)                                    |
| max_t        | 3000   | Maximum number of steps per episode (Training hyperparameters for `train` function)                                     |


-----------

### 3. Result and Future works

#### 3-1. Reward

![output](https://user-images.githubusercontent.com/53895034/137690289-5421bf1c-76b1-4a41-b4c4-577fbd82faaa.png)

| axis     | value   |
|:--------:|:-------:|
| x-axis   | episode | 
| y-axis   | reward  | 

Environment solved in 884 episodes. You can see relatively stable learning since TD3 is the improved version of DDPG.


#### 3-2. Future works

- Optimizing the hyperparameters
- Deepening the network model 
- Implement with other algorithms (**PPO(Proximal Policy Optimization)**, **SAC(Soft Actor Critic)**)

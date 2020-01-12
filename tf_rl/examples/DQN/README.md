## Introduction

DQN implementation of TensorFlow/PyTorch




## Usage

- TensorFlow
    ```shell script
    python tf_rl/examples/DQN/main.py --gin_file=./tf_rl/examples/DQN/config/dopamine.gin \
                                      --gin_params="train_eval.env_name='Pong'"
    ```

- PyTorch
    ```shell script
    python tf_rl/examples/DQN/pytorch/main.py --env_name="Pong"
    ```

- To override the existing `gin_file` by command line args, pls check [here](https://github.com/google/gin-config/blob/master/docs/index.md#experiments-with-multiple-gin-files-and-extra-command-line-bindings)


## Goals
- Ant-v2: 3500
- HalfCheetah-v2: 7000
- Hopper-v2: 1500
- Humanoid-v2: 2000
- InvertedDoublePendulum-v2: 6000
- InvertedPendulum-v2: 800
- Reacher-v2: -6
- Swimmer-v2: 40
- Walker2d-v2: 2500


## Hyper-params

- `dopamine.gin`: Dopamine's hyper-params, [[Ref]](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin)
- `nature_dqn.gin`: Mnih et al. (2015)
- Based on the common setting of training as in `domapine.gin`, I have examined some possibilities below
    - `experimental/adam_huber.gin`
    - `experimental/adam_mse.gin`
    - `experimental/rmsprop_mse.gin`
    - Since `dopamine.gin` uses `RMSProp` with Huber Loss func, I didn't add it here!
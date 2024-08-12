from gym import register as gym_register

from irl_environments import register_all

IBC_TASKS = ["REACH", "PUSH", "INSERT", "PARTICLE", "PUSH_DISCONTINUOUS", "PUSH_MULTIMODAL"]


ADROIT_TASKS = [
    "pen-human-v0",
    "hammer-human-v0",
    "door-human-v0",
    "relocate-human-v0",
]


# NOTE: For IBC, We disguise the gymnasium environment as a gym environment,
# relying on the backwards compatability since tf-agents still uses gym
register_all(gym_register)

GYM_TASKS = [
    "quad_insert_a0o0eval",
    "quad_insert_aLoLeval",
    "quad_insert_aMoMeval",
    # "quad_insert_a0o0eval_chunking",
    # "quad_insert_aLoLeval_chunking",
    # "quad_insert_aMoMeval_chunking",
]

D4RL_TASKS = [
    "antmaze-large-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-medium-play-v0",
    "halfcheetah-expert-v0",
    "halfcheetah-medium-expert-v0",
    "halfcheetah-medium-replay-v0",
    "halfcheetah-medium-v0",
    "hopper-expert-v0",
    "hopper-medium-expert-v0",
    "hopper-medium-replay-v0",
    "hopper-medium-v0",
    "kitchen-complete-v0",
    "kitchen-mixed-v0",
    "kitchen-partial-v0",
    "walker2d-expert-v0",
    "walker2d-medium-expert-v0",
    "walker2d-medium-replay-v0",
    "walker2d-medium-v0",
] + ADROIT_TASKS

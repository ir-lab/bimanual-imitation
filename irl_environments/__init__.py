import warnings

from gymnasium.envs.registration import register as gymnasium_register

warnings.filterwarnings(action="ignore", module="gym", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", module="gym", category=UserWarning)


def register_all(register_fn):

    register_fn(
        id="path_2d_v1",
        entry_point="irl_environments.path2D:BasePath2DEnv",
        max_episode_steps=600,
    )

    register_fn(
        id="path_3d_v1",
        entry_point="irl_environments.path3D:BasePath3DEnv",
        max_episode_steps=600,
    )

    register_fn(
        id="quad_insert_a0o0",
        entry_point="irl_environments.bimanual_quad_insert:QuadInserta0o0",
        max_episode_steps=1600,
    )

    # NOTE: eval environments can have a longer # of steps (just to evaluate the agent)
    register_fn(
        id="quad_insert_a0o0eval",
        entry_point="irl_environments.bimanual_quad_insert:QuadInserta0o0",
        max_episode_steps=2400,
    )

    register_fn(
        id="quad_insert_aLoL",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaLoL",
        max_episode_steps=1600,
    )

    register_fn(
        id="quad_insert_aLoLeval",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaLoL",
        max_episode_steps=2400,
    )

    register_fn(
        id="quad_insert_aMoM",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaMoM",
        max_episode_steps=1600,
    )

    register_fn(
        id="quad_insert_aMoMeval",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaMoM",
        max_episode_steps=2400,
    )

    # TODO: Make max_episode_steps depend on action_horizon
    # For now, assume all methods have an action_horizon of 4

    register_fn(
        id="quad_insert_a0o0eval_chunking",
        entry_point="irl_environments.bimanual_quad_insert:QuadInserta0o0Chunking",
        max_episode_steps=600,
    )

    register_fn(
        id="quad_insert_aLoLeval_chunking",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaLoLChunking",
        max_episode_steps=600,
    )

    register_fn(
        id="quad_insert_aMoMeval_chunking",
        entry_point="irl_environments.bimanual_quad_insert:QuadInsertaMoMChunking",
        max_episode_steps=600,
    )


register_all(gymnasium_register)

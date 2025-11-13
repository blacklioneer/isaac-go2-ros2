import gymnasium as gym

gym.register(
    id="Isaac-Dronep-Play-Direct-v0",
    entry_point=f"{__name__}.cf_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cf_env:QuadcopterEnvCfg",
    },
)
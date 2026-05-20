# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import numpy as np

from handover.gym_compat import gym
from handover.config import get_config_from_args

scene_id = 105


def main():
    cfg = get_config_from_args()

    env = gym.make(cfg.ENV.ID, cfg=cfg)
    step_count = 0
    episode_count = 0

    print("Starting handover demo. Press Ctrl+C to stop.", flush=True)

    while True:
        episode_count += 1
        env.reset(scene_id=scene_id)
        print("episode:", episode_count, "scene_id:", scene_id, flush=True)

        for _ in range(int(3.0 / cfg.SIM.TIME_STEP)):
            action = np.array(cfg.ENV.PANDA_INITIAL_POSITION)
            action += np.random.uniform(low=-1.0, high=+1.0, size=len(action))
            obs, reward, done, info = env.step(action)
            step_count += 1

            if step_count % 10 == 0:
                print(
                    "step:",
                    step_count,
                    "frame:",
                    obs["frame"],
                    "done:",
                    done,
                    "reward:",
                    reward,
                    "info:",
                    info,
                    flush=True,
                )


if __name__ == "__main__":
    main()

from pathlib import Path

import irl_data

IRL_DATA_BASE_DIR = Path(irl_data.__file__).parent
EXPERT_TRAJS_DIR = IRL_DATA_BASE_DIR / "expert_trajectories"
EXPERT_VAL_TRAJS_DIR = IRL_DATA_BASE_DIR / "expert_validation_trajectories"

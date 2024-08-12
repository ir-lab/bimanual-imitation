from pathlib import Path

import bimanual_imitation

BIMANUAL_IMITATION_BASE_DIR = Path(bimanual_imitation.__file__).parent
RESULTS_DIR = BIMANUAL_IMITATION_BASE_DIR.parent / "results"  # for official results
TEST_RESULTS_DIR = BIMANUAL_IMITATION_BASE_DIR.parent / "test_results"  # for experimenting

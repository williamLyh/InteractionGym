import os
from pathlib import Path

# tau2 (optional extra) reads its domain data from $TAU2_DATA_DIR when installed from git; use a checkout in
# third_party/tau2-bench if there is one. Must run before anything imports tau2.
_data = Path(__file__).resolve().parents[1] / "third_party" / "tau2-bench" / "data"
if not os.environ.get("TAU2_DATA_DIR") and (_data / "tau2" / "domains").is_dir():
    os.environ["TAU2_DATA_DIR"] = str(_data)

# The default sound bank (interaction_gym.noisebank) would fetch DEMAND on first use: tests are offline and
# deterministic, so they use the synthetic stand-ins unless a test sets these itself.
os.environ["IG_SOUNDBANK"] = "synthetic"
os.environ["IG_SOUNDBANK_FETCH"] = "0"

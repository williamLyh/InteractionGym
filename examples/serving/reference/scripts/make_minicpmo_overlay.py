"""Overlay of an installed vLLM-Omni for the MiniCPM-o 4.5 audio server: a patched copy, the base install untouched.

    python scripts/make_minicpmo_overlay.py <vllm_omni package dir> <overlay root>

e.g. ``$IG_OMNI_ENV/bin/python scripts/make_minicpmo_overlay.py $(python -c 'import vllm_omni, os;
print(os.path.dirname(vllm_omni.__file__))') /data/overlays/minicpmo``. The server then runs with the overlay first on
``PYTHONPATH`` (``IG_MINICPMO_OVERLAY=<overlay root>`` in serving.env, used by scripts/run_minicpmo.sh). Rebuild it
after reinstalling vLLM-Omni.

Edit: Code2Wav's HiFT CUDA-graph wrapper captures one extra graph lazily for every new (batch, mel frames, cache)
shape up to a hard-coded 8 (``HiFTGraphWrapper.max_lazy_graphs``), each one more allocation that never goes back.
The connector ``extra`` key ``hift_max_lazy_graphs`` (default 8, the stock behaviour) sets that cap; 0 keeps only the
two graphs captured at start-up and runs other shapes eagerly.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

def sub(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text()
    n = text.count(old)
    assert n == count, f"{path.name}: expected {count} match(es), found {n} for:\n{old}"
    path.write_text(text.replace(old, new))


def main(base: Path, root: Path) -> None:
    dst = root / "vllm_omni"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst, ignore=shutil.ignore_patterns("__pycache__"))
    mdir = dst / "model_executor/models/minicpmo_4_5"
    sub(mdir / "minicpmo_4_5_code2wav.py",
        """            "enabled": bool(extra.get("enable_hift_graph", False)),
            "capture_batch_sizes": capture_batch_sizes,""",
        """            "enabled": bool(extra.get("enable_hift_graph", False)),
            "capture_batch_sizes": capture_batch_sizes,
            "max_lazy_graphs": int(extra.get("hift_max_lazy_graphs", 8)),""")
    sub(mdir / "batched_token2wav.py",
        """                with torch.inference_mode(), _autocast_disabled(hift_parameter.device):
                    self.hift_graph_wrapper.capture()
""",
        """                self.hift_graph_wrapper.max_lazy_graphs = max(0, int(graph_config.get("max_lazy_graphs", 8)))
                logger.info("HiFT lazy CUDA graph cap: %d", self.hift_graph_wrapper.max_lazy_graphs)
                with torch.inference_mode(), _autocast_disabled(hift_parameter.device):
                    self.hift_graph_wrapper.capture()
""")
    print("overlay at", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))

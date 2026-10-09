# Import shim for the MiniCPM-o 4.5 server: its Token2wav flow.yaml references top-level "cosyvoice2.*" classes
# (e.g. cosyvoice2.flow.flow.CausalMaskedDiffWithXvec). vLLM-Omni loads flow.yaml with hyperpyyaml directly
# without registering the alias that the stepaudio2-minicpmo package sets up, so expose stepaudio2.cosyvoice2
# under that name. Only on PYTHONPATH for the MiniCPM-o server (scripts/run_minicpmo.sh).
import stepaudio2.token2wav as _t2w

_t2w._setup_cosyvoice2_alias()  # registers cosyvoice2.flow.flow etc. in sys.modules
import stepaudio2.cosyvoice2 as _real  # noqa: E402

__path__ = list(_real.__path__)

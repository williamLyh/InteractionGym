"""GPU tuner (serving-layout auto-tuner), empirical first: run real episodes on the current layout (ramping env
concurrency), record per-service GPU util / queues / env-side waits, rebalance GPUs toward the
measured load, optionally apply and re-measure (``rebalance.loop``). A predictive fallback
(per-service probes + ``solve``) ranks layouts when episodes cannot be run. See docs/GPU_TUNER.md."""

from .empirical import Level, MockCluster, run_batch
from .launch import launch_config, launch_script, load_host
from .meter import Meter, MeteredAgent, MeteredChat, MeteredSpeech
from .monitor import GpuSampler, Monitor
from .profile import Curve, DemandProfile, ServiceDemand, SupplyProfile
from .rebalance import Layout, Proposal, Round, allocate, analyze, loop, measure, propose
from .solve import Hardware, Plan, ServiceSpec, evaluate, place, solve

__all__ = ["Curve", "DemandProfile", "GpuSampler", "Hardware", "Layout", "Level", "Meter", "MeteredAgent", "MeteredChat", "MeteredSpeech",
           "MockCluster", "Monitor", "Plan", "Proposal", "Round", "ServiceDemand", "ServiceSpec", "SupplyProfile", "allocate", "analyze",
           "evaluate", "launch_config", "launch_script", "load_host", "loop", "measure", "place", "propose", "run_batch", "solve"]

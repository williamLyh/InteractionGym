"""The Full-Duplex-Bench material (CC BY-NC 4.0) is a separate optional component: the Apache-2.0 package imports
without it and says how to install it when a loader needs it."""

import importlib
import subprocess
import sys
import textwrap


def test_main_package_does_not_ship_the_fdbench_material():
    import interaction_gym.benchmarks.third_party as tp
    from pathlib import Path

    assert not (Path(tp.__file__).parent / "full_duplex_bench").exists()


def test_benchmarks_import_without_the_component_and_explain_when_it_is_needed():
    code = textwrap.dedent("""
        import sys
        sys.modules["interaction_gym_fdbench"] = None  # as if not installed
        import interaction_gym, interaction_gym.eval, interaction_gym.traj
        from interaction_gym.benchmarks import full_duplex_bench, fdb2, fdb3, easy_turn, humdial, clip_bank
        assert full_duplex_bench.SUBSETS and fdb3.NAME
        for f in (lambda: full_duplex_bench.take_turn, lambda: fdb2.EXAMINEE_PROMPT, lambda: fdb3.pass_at_1):
            try:
                f()
            except ImportError as e:
                assert "interaction-gym-fdbench" in str(e) and "CC BY-NC 4.0" in str(e), e
            else:
                raise AssertionError("expected ImportError")
        try:
            full_duplex_bench.no_such_name
        except AttributeError:
            pass
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_reexports_resolve_to_the_component():
    fdb = importlib.import_module("interaction_gym.benchmarks.full_duplex_bench")
    v1 = importlib.import_module("interaction_gym_fdbench.v1")
    assert fdb.take_turn is v1.take_turn and fdb._tor is v1.take_turn and fdb._merge is v1.merge
    from interaction_gym.benchmarks.fdb3 import FUNCTIONS, _strip_json_fences  # noqa: F401  (from-import works)

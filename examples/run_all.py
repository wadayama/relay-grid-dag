"""Run the paper experiment suite (EXP0-EXP7, selftest depth) and print a summary.

Each expN script also runs standalone at full (paper) depth:
    uv run --extra examples python examples/expN_*.py
This runner uses each script's --selftest path (reduced iterations) as a fast
end-to-end smoke test of the whole suite.

    uv run --extra examples python examples/run_all.py
"""
import importlib

MODS = ["exp0_setup", "exp1_engine", "exp2_precoding_gain", "exp3_selection",
        "exp4_nearfield", "exp5_handover", "exp6_multipair", "exp7_mimap"]


def main():
    results = {}
    for name in MODS:
        print("\n" + "=" * 70)
        mod = importlib.import_module(name)
        if name == "exp0_setup":
            results[name] = bool(mod.main())          # cheap; no selftest knob
        else:
            results[name] = bool(mod.main(selftest=True))
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

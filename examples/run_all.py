"""Run the whole NF-PRG smoke test (S1-S4) and print a PASS/FAIL summary.

    /Users/wadayama/Dropbox/2026-nearfield-dag/.venv/bin/python smoke/run_all.py
"""
import importlib

MODS = ["s1_position_matters", "s2_selection_value",
        "s3_cont_to_discrete", "s4_ofdm_shared"]


def main():
    results = {}
    for name in MODS:
        print("\n" + "=" * 70)
        mod = importlib.import_module(name)
        results[name] = bool(mod.main())
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

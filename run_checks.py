"""Run every project check. A phase is complete only when this exits with code 0.

    python run_checks.py                  # evaluation set, unit and API tests, real-model and API server smoke tests
    python run_checks.py --eval --phase 3 # also evaluate the models and enforce targets due by phase 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = ROOT / "evaluation" / "targets.json"


def step(name, command):
    print(f"\n=== {name} ===", flush=True)
    passed = subprocess.run(command, cwd=ROOT).returncode == 0
    print(f"--- {name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return name, passed


def lookup(result, path):
    value = result
    for key in path.split("."):
        value = value[key]
    return value


def check_targets(phase):
    """Compare saved evaluation results with targets; only targets due by this phase can fail."""
    config = json.loads(TARGETS.read_text(encoding="utf-8"))
    results = {}
    failed = False
    print(f'\n=== Targets (engine {config["engine"]}, phase {phase}) ===')
    for target in config["targets"]:
        engine = target.get("engine", config["engine"])
        if engine not in results:
            results[engine] = json.loads((ROOT / "evaluation" / "results" / f"{engine.lower()}.json").read_text(encoding="utf-8"))
        value = lookup(results[engine]["metrics"], target["metric"])
        met = value is not None and value >= target["min"]
        due = target["from_phase"] <= phase
        failed |= due and not met
        state = "PASS" if met else "FAIL" if due else f"not yet due (phase {target['from_phase']})"
        print(f'{target["metric"]:<34} {"-" if value is None else f"{value:.3f}":>6}  target >= {target["min"]:.2f}  {state}')
    return "Targets", not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", action="store_true", help="run the model evaluation (several minutes)")
    parser.add_argument("--phase", type=int, default=0, help="current build phase; targets due by it must pass")
    args = parser.parse_args()
    python = sys.executable
    results = [
        step("Evaluation set", [python, "evaluation/eval_set.py", "verify"]),
        step("Unit tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        step("Real-model smoke test", [python, "smoke_test.py"]),
        step("API server smoke test", [python, "-m", "backend.smoke_test"]),
    ]
    if args.eval:
        results.append(step("Model evaluation", [python, "evaluation/evaluate.py"]))
        if results[-1][1]:
            results.append(check_targets(args.phase))
    print("\n=== Summary ===")
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all(passed for _, passed in results) else 1)


if __name__ == "__main__":
    main()

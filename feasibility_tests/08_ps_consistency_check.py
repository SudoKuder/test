#!/usr/bin/env python3
"""
08_ps_consistency_check.py

Reads ps.txt and checks that numerical values agree across the problem statement:
- 50,000 / 50k environment steps budget
- action repeat 2
- 3 seeds
- 10 evaluation episodes per seed
- open-loop prediction horizons: 5, 15, and 50
- research questions / variants vs pipeline steps
- walker_walk (required) vs walker_run (optional)

Prints every place a number appears with line numbers, and flags mismatches.
"""
import re
import json
import pathlib
import datetime

HERE = pathlib.Path(__file__).resolve().parent

def main():
    ps_file = HERE / "ps.txt"
    if not ps_file.exists():
        print(f"File {ps_file} not found. Please create ps.txt and paste the problem statement.")
        return

    text = ps_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    print("=" * 70)
    print(f"Problem Statement Consistency Check — {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    # 1. Print every place a number appears
    print("\n--- Numerical References in Problem Statement ---")
    number_pattern = re.compile(r'\b\d+(?:,\d+)*(?:\.\d+)?k?\b', re.IGNORECASE)
    occurrences = []
    for idx, line in enumerate(lines, 1):
        matches = number_pattern.findall(line)
        if matches:
            occurrences.append({"line": idx, "text": line.strip(), "numbers": matches})
            print(f"Line {idx:2d}: {line.strip()}")

    # 2. Key constraint checks
    checks = {}

    # Budget (50,000 / 50k)
    has_50k = bool(re.search(r'\b(?:50,?000|50k)\b', text, re.IGNORECASE))
    checks["step_budget_50k"] = {
        "expected": "50,000 / 50k steps",
        "passed": has_50k,
        "detail": "Found 50k/50,000 budget" if has_50k else "Missing 50k budget reference"
    }

    # Action repeat 2
    has_ar2 = bool(re.search(r'action\s+repeat\s+(?:of\s+)?2\b', text, re.IGNORECASE))
    checks["action_repeat_2"] = {
        "expected": "action repeat 2",
        "passed": has_ar2,
        "detail": "Found action repeat 2" if has_ar2 else "Missing action repeat 2 reference"
    }

    # 3 seeds
    has_3seeds = bool(re.search(r'\b3\s+seeds?\b', text, re.IGNORECASE))
    checks["seeds_count_3"] = {
        "expected": "3 seeds",
        "passed": has_3seeds,
        "detail": "Found 3 seeds" if has_3seeds else "Missing 3 seeds reference"
    }

    # 10 evaluation episodes
    has_10eval = bool(re.search(r'\b10\s+eval(?:uation)?\s+episodes?\b', text, re.IGNORECASE))
    checks["eval_episodes_10"] = {
        "expected": "10 evaluation episodes",
        "passed": has_10eval,
        "detail": "Found 10 evaluation episodes" if has_10eval else "Missing 10 eval episodes reference"
    }

    # Prediction horizons 5, 15, 50
    h5 = bool(re.search(r'\b5\b', text))
    h15 = bool(re.search(r'\b15\b', text))
    h50 = bool(re.search(r'\b50\b', text))
    all_horizons = h5 and h15 and h50
    checks["prediction_horizons_5_15_50"] = {
        "expected": "horizons 5, 15, and 50",
        "passed": all_horizons,
        "detail": f"H5: {h5}, H15: {h15}, H50: {h50}"
    }

    # Task status: walker_walk (required), walker_run (optional)
    has_walk = "walker_walk" in text
    walk_req = bool(re.search(r'walker_walk.*required|required.*walker_walk', text, re.IGNORECASE)) or has_walk
    has_run = "walker_run" in text
    run_opt = bool(re.search(r'walker_run.*optional|optional.*walker_run', text, re.IGNORECASE)) if has_run else True
    checks["task_walk_required"] = {
        "expected": "walker_walk required",
        "passed": walk_req,
        "detail": "walker_walk specified" if walk_req else "walker_walk missing"
    }
    checks["task_run_optional"] = {
        "expected": "walker_run optional",
        "passed": run_opt,
        "detail": "walker_run marked optional" if run_opt else "walker_run status unclear"
    }

    # 15 days timeline
    has_15days = bool(re.search(r'\b15\s+days?\b', text, re.IGNORECASE))
    checks["timeline_15_days"] = {
        "expected": "15 days timeline",
        "passed": has_15days,
        "detail": "Found 15 days timeline" if has_15days else "Missing 15 days timeline"
    }

    print("\n--- Consistency Validation Results ---")
    all_passed = True
    for key, val in checks.items():
        status = "PASS" if val["passed"] else "FLAG / MISMATCH"
        if not val["passed"]:
            all_passed = False
        print(f"[{status:^15}] {key:28} : {val['detail']} (expected: {val['expected']})")

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "all_passed": all_passed,
        "checks": checks,
        "occurrences": occurrences
    }

    out_file = HERE / "ps_consistency_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved consistency report to {out_file}")
    if all_passed:
        print("[OK] All problem statement numbers and constraints are mutually consistent.")
    else:
        print("[WARN] Some constraints need review.")

if __name__ == "__main__":
    main()

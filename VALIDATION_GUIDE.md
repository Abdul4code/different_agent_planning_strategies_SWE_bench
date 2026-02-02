# Validating Mini-SWE-Agent Results

This guide explains how to validate the results submitted by the mini-swe-agent against the SWE-bench benchmark.

## Overview

After running mini-swe-agent with `run_tasks.py`, the results are saved as:
- **Predictions file**: `results/baseline/preds.json` - Contains the generated patches
- **Trajectory files**: `results/baseline/<instance_id>/<instance_id>.traj.json` - Contains agent execution logs
- **Exit status file**: `results/baseline/exit_statuses_<timestamp>.yaml` - Contains exit codes for each instance

## 1. Understanding Output Artifacts

### preds.json
Each entry contains:
```json
{
  "instance_id": {
    "model_name_or_path": "model name",
    "instance_id": "instance_id",
    "model_patch": "diff content (the proposed fix)"
  }
}
```

### Trajectory Files
JSON files containing:
- Complete message history (conversations between agent and LLM)
- All bash commands executed
- Command outputs
- Exit status and final result

## 2. Validation Methods

### Method A: Using SWE-bench's Official Evaluation Harness

The most comprehensive validation approach using the SWE-bench evaluation framework:

```bash
# Install SWE-bench if not already installed
cd SWE-bench
pip install -e .

# Run evaluation on your predictions
python -m swebench.harness.run_evaluation \
  --predictions_path results/baseline/preds.json \
  --output_dir results/evaluation \
  --max_workers 1 \
  --timeout 900
```

This will:
1. Apply each patch from `preds.json` to the actual repository
2. Run the test suite for each instance
3. Compare PASS→PASS and FAIL→PASS test results
4. Generate detailed evaluation reports

**Output:**
- `results/evaluation/<run_id>/` - Contains logs and reports
- Summary report showing which patches fixed the target tests

### Method B: Manual Patch Validation

For a quick check of specific instances:

```bash
# 1. Navigate to the instance directory
cd results/baseline/sympy__sympy-17139/

# 2. View the proposed patch
cat sympy__sympy-17139.traj.json | jq '.result'

# 3. Check what the agent attempted
cat sympy__sympy-17139.traj.json | jq '.messages'  # Full conversation history
```

### Method C: Python Script for Quick Validation

Create a validation script to inspect results:

```python
#!/usr/bin/env python3
import json
from pathlib import Path

# Load predictions
preds_path = Path("results/baseline/preds.json")
preds = json.loads(preds_path.read_text())

# Load exit statuses
exit_status_file = list(Path("results/baseline").glob("exit_statuses_*.yaml"))[0]
with open(exit_status_file) as f:
    import yaml
    exit_statuses = yaml.safe_load(f)

print(f"Total instances processed: {len(preds)}")
print(f"\nInstance Results:")

for instance_id, pred in preds.items():
    exit_status = exit_statuses.get("instances", {}).get(instance_id, {}).get("exit_status", "UNKNOWN")
    print(f"\n{instance_id}:")
    print(f"  Status: {exit_status}")
    print(f"  Model: {pred['model_name_or_path']}")
    print(f"  Patch size: {len(pred['model_patch'])} chars")
    
    # Check trajectory for more details
    traj_path = Path(f"results/baseline/{instance_id}/{instance_id}.traj.json")
    if traj_path.exists():
        traj = json.loads(traj_path.read_text())
        print(f"  Exit code: {traj.get('exit_status', 'N/A')}")
```

## 3. Key Validation Metrics

After running the evaluation harness, check these metrics:

1. **FAIL_TO_PASS**: Tests that were failing before the patch but pass after
   - This is the primary success metric for SWE-bench
   - Instance is **resolved** if ≥1 FAIL_TO_PASS test passes

2. **PASS_TO_PASS**: Tests that were passing before and still pass after
   - No regression detection
   - Should have ≥1 for a valid solution

3. **PASS_TO_FAIL**: Tests that were passing but now fail after the patch
   - Indicates a regression (bad patch)
   - Should be 0 or minimal

4. **FAIL_TO_FAIL**: Tests that were failing and still fail
   - Expected for tests outside the scope of the fix

## 4. Interpreting Results

### Success Criteria
An instance is **resolved** if:
- ✅ Patch applies cleanly without conflicts
- ✅ At least 1 test in FAIL_TO_PASS list passes
- ✅ No regressions (PASS_TO_FAIL is empty or minimal)

### Common Issues

| Issue | Cause | Check |
|-------|-------|-------|
| Patch doesn't apply | Invalid diff format | Verify patch file syntax in preds.json |
| Import errors | Missing dependencies | Check trajectory logs in .traj.json |
| Wrong repo state | Version mismatch | Verify repo hash matches instance base_commit |
| Test timeouts | Complex tests taking too long | Increase timeout in evaluation harness |

## 5. Accessing Detailed Logs

For each instance evaluation:

```bash
# Logs are stored in:
results/evaluation/<run_id>/<model_name>/<instance_id>/

# Key files:
#   - test_output.log     # Full test execution output
#   - info.txt            # Exit status and patch application result
#   - eval_report.json    # Structured evaluation results

# Example:
cat "results/evaluation/<run_id>/openai__gpt-4o-mini/django__django-13710/eval_report.json" | jq .
```

## 6. Quick Validation Checklist

```bash
# 1. Check if predictions exist
[ -f results/baseline/preds.json ] && echo "✓ Predictions found" || echo "✗ No predictions"

# 2. Count processed instances
jq '. | length' results/baseline/preds.json

# 3. View exit status summary
cat results/baseline/exit_statuses_*.yaml | head -20

# 4. Check for any failed instances
grep -E "failed|error|exception" results/baseline/exit_statuses_*.yaml

# 5. Run evaluation (if needed)
python -m swebench.harness.run_evaluation \
  --predictions_path results/baseline/preds.json \
  --output_dir results/evaluation
```

## 7. Integration with SWE-bench Leaderboard

To submit results to the official SWE-bench leaderboard:

1. Validate your evaluation using the official harness
2. Get the evaluation report: `results/evaluation/<run_id>/report.json`
3. Follow submission guidelines at https://www.swebench.com/

## References

- [mini-swe-agent Documentation](https://mini-swe-agent.com/)
- [SWE-bench Evaluation Guide](https://github.com/swe-bench/SWE-bench)
- [Harness Source Code](https://github.com/swe-bench/SWE-bench/tree/main/swebench/harness)

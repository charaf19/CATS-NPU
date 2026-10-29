CATS-NPU audit, 2026-10-23

Start with report/docs/AUDIT_2026-10-23.md and RESEARCH_CHARTER.md.
source_snapshot/ is the unchanged pre-audit 41-file source.
source_snapshot.json records its SHA-256 manifest.

From the original project root, using its existing virtual environment:
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe results/audit_2026-10-23/reproduce_blockers.py
  .\.venv\Scripts\python.exe results/audit_2026-10-23/validate_snapshot.py

Expected original suite: 105 passed, 2 failed, 1 skipped (exit 1).
Diagnostic scripts collect failures and may exit zero; read the JSON fields.
Targeted matrix: 320 cases, 143 correct, 177 exceptions or wrong results.
The failure-injection harness folder is SYNTHETIC and is NOT scientific benchmark data.
No competitive performance experiment was run.
Results are ignored by Git; preserve this bundle separately.
Re-executing diagnostics overwrites their JSON; retain this archive first.

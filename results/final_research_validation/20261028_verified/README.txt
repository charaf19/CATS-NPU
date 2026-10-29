CATS-NPU final research record, 2026-10-28

Decision: NO-GO for manuscript preparation on the investigated mechanisms.
Start with source/docs/FINAL_RESEARCH_CONTRIBUTION.md.

source/ contains the committed repository, Phase 7/8 historical results,
Phase 9 exploratory results, final audit, pinned compiler source evidence,
and the four development score arrays needed for the decisive pilot.
The bundle under source/results/final_research/20261028/ preserves source
history through the package commit, including the historical probe revisions.
Other dataset arrays can be regenerated using their provenance manifests.

Verify every payload file against manifest.json before using the evidence.
In the freshly extracted source/ folder, restore the included Git history:
git init
git fetch results/final_research/20261028/source_history.bundle HEAD
git reset --mixed FETCH_HEAD
This reset updates the fresh repository index/HEAD and preserves its files.
Then, with the repository dependencies installed, run:
python scripts/audit_final_research.py --out results/reproduced/final_audit.json
python -m pytest -q

The audit is CPU-only. It does not run a new NPU experiment. Hardware pilot
reproduction requires the documented NPU/compiler stack, optional C++ build,
and CUDA PyTorch baseline. The existing benchmark can be repeated with:
python scripts/compare_split_key_pilot.py --input results/phase5/sift1m/q0000_n1000000_neg_l2_f16.npy --k 10 --repeats 5 --out results/reproduced/paired_pilot.json

Every output path must be new. Pilot repetitions are not independent sessions
or a confirmatory protocol. No new confirmatory study was launched after the
novelty gate failed. Phase 7/8 records retain their original scope and status.

# Reviewer score check

This portable package verifies the completed F2 development comparison in Supplementary Table S17d. It reproduces accounting from four saved controller trajectories, checks paired no-yaw records and forecast hashes, and checks the frozen technical endpoints. It does not rerun LES, establish independent confirmation, or reproduce every experiment in the manuscript.

## Run

Python 3.11 or newer, NumPy 2.3.5. The verification was tested in a newly created Python 3.14 environment. No JAX, GPU, external state archives, repository checkout, credentials or network data retrieval is needed after installing NumPy.

```sh
python -m venv .venv
# Activate .venv using your platform's normal command.
python -m pip install -r requirements.txt
python scripts/verify.py
```

Expected result: `verification: pass`. A successful verification preserves the scientific failure: F2 does not meet the positive pooled energy advancement rule. The four input arms each contain 11,400 time samples for nine turbines. Every saved scoring input is bound by MANIFEST.json. Runtime is expected to be under one minute on a modern CPU, excluding environment installation; actual execution time appears in the project verification record.

`frozen_score.py` is copied byte-for-byte from the prospectively frozen B3 scorer. The wrapper verifies every packaged file and compares the scientific output with the archived expected result. Input-path strings in the result change on relocation; the original expected result retains its original paths for provenance. Scoring tolerances are unchanged: relative 1e-10 and absolute 1e-8 for recomputed accounts. NumPy uses FP64 reductions for this accounting; this is not an FP64 LES simulation.

Energy = sum(power in watts) × 0.2 / 3.6e6. Yaw electricity uses 30 kW per moving turbine and motion time reconstructed from angle travel at 0.3 degrees/s. The initial yaw is zero. Scenario energies are summed before computing pooled ratios. See configs/SCOPE.json and each frozen PROTOCOL.json.

Large state and precursor arrays are omitted. They are required for full solver replay and have externally archived identities recorded in the manuscript's other source-data manifests. No public licence or repository identifier has yet been assigned by the authors. This is a local reviewer-preparation artifact, not a claim of public availability.

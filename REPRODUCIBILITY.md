# Reproduction scope

## 1. File integrity (standard-library Python, no GPU)

Run `python3 scripts/verify_distribution.py`. This checks distributed byte
identities and Python syntax, without importing research modules. Download the
assets listed in data/ASSETS.json and pass `--assets /path/to/assets` to verify
all asset bytes and nested publication manifests. These are packaging checks,
not additional experiments.

## 2. Archived trajectory accounting (no LES)

The examples/reviewer_score_check package contains four saved F2 trajectories,
a frozen scorer, expected output and exact requirements. Its README documents
`python scripts/verify.py` from that package directory. The original package
was previously verified in a fresh environment; this publication step does not
claim a new end-to-end LES reproduction. Scientific failures remain failures.
Other source archives retain case-specific accounting scripts and records.

## 3. Full LES/controller reproduction (external inputs required)

The recorded environment is Python 3.13.9, JAX/JAXLIB 0.9.0.1, NumPy 2.3.5 and
Optax 0.2.6 on CUDA-capable NVIDIA H20-3e hardware. Requirements alone do not
reconstruct the historical GPU image. Source archives bind image, precursor,
state, solver and protocol hashes. Do not replace a frozen input with a newly
generated precursor and call it the same realization.

The original source runner is experiments/original/run_gpu.py; the original
source archive contains the complete frozen directory layout. It requires
NET_MPC_CONFIG, explicit input/output paths, matching versions and source/input
hashes, prospective mainline adjudication and experiment-specific allocation.
Historical source/input roots are retained for provenance; portability is not
claimed for those launchers. Remote-management helpers are omitted from public
copies. The src tree makes the numerical core directly inspectable, while the
source archives identify the versions actually used in each experiment.

Original decisions have a 180 s online deadline; compilation and initialization
are separate. The complete two-cell OWN-TAIL G1 campaign consumed 3159.950497
GPU-seconds; this is an observed historical cost, not a guarantee on other
hardware or for a full-paper rerun. No full-paper runtime estimate is available
without the complete external input set and equivalent hardware.

## Evidence units and status

Six original roots, twelve original scenario cells and 108 decisions are
different units. Successive windows and repeated diagnostic states are not
independent seeds. The newer causal and OWN-TAIL results remain development
evidence. Original N=22 confirmation is incomplete. Do not combine these
groups to manufacture a confirmatory sample count.

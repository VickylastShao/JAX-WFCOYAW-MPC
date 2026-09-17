# JAX-WFCOYAW-MPC

Research code and evidence for **Execution-aware wind-farm yaw control with
differentiable large-eddy simulation**, a manuscript being prepared for
Renewable Energy. This repository is a research release, not an accepted-paper
or field-deployment claim.

The original controller optimizes nine executable yaw targets directly through
a differentiable LES rollout with actuator limits, a 180 s command delay and
explicit yaw-motion electricity. B1P*(F) is an external comparator only.

## C1 representation replication and updated figures

The [C1 evidence release](https://github.com/VickylastShao/JAX-WFCOYAW-MPC/releases/tag/paper-evidence-2026-09-17-c1-v1)
adds two previously unused control-outcome sources, with two scenarios and
single-/two-segment yaw targets. Its 72 decisions meet the declared 180 s
budget. Single-segment targets reduce travel in all four paired comparisons,
while two segments deliver more net energy. Across the two source-level pools,
the single-segment energy cost is 0.315–0.362% of two-segment total net energy,
or 25.8–25.9% of its incremental benefit over no yaw.

Download `Source_Data_C1_20260917.zip` for the frozen evidence and
`Source_Data_C1_Editorial_20260917.zip` for figure source tables and a standalone
redraw script. Verify both against `SHA256SUMS_C1_20260917.txt` in that release.
The two sources remain distinct from the original six-root development pool;
this does not constitute the planned 22-root confirmation. Complete simulated
state and exact inlet preview remain the conditions of the C1 result. The
original release and its assets remain available unchanged below.

## Evidence in the original release

- Original six-root exact-preview revalidation: 108 decisions within the 180 s
  budget; pooled net-energy gains of 1.1024% over no yaw and 0.8165% over the
  specified lookup comparator, conditional on complete simulated state, exact
  inlet preview and the 30 kW motion-power assumption.
- Causal forecast experiments retain negative and incomplete outcomes; no
  completed independent 22-root confirmation is claimed.
- OWN-TAIL-200-v1 is a separate development candidate. Its completed two-cell
  G1 comparison was timely but failed the prospective pooled gain criterion;
  G2 was not launched. See results/own_tail/FINAL_REPORT.md.
- Numerical qualifications apply to the tested states and precisions. Earlier
  failed finite-difference criteria remain in the archives. Dynamic-model
  response qualification failures are retained and are not superiority tests.

## Start here

1. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the supported levels of reuse.
2. Browse [data/tables](data/tables) for source tables and original calculations.
3. Download the [versioned source-data archives](https://github.com/VickylastShao/JAX-WFCOYAW-MPC/releases/tag/paper-evidence-2026-09-16-v1).
   The filenames, sizes and SHA-256 checksums are in [data/ASSETS.json](data/ASSETS.json).
4. Verify the checkout: `python3 scripts/verify_distribution.py`.
   With downloaded assets in a local directory, append `--assets /path/to/assets`.
5. The portable [reviewer score check](examples/reviewer_score_check/README.md)
   reproduces one already archived F2 comparison from saved arrays; it is not a
   full LES rerun. The original README contains historical release-status text.
   Current public access and licence terms are those in this top-level release.

## Layout

| Directory | Contents |
|---|---|
| src | Scientific core and transitive local dependencies |
| experiments/original | Original frozen experiment entry points |
| data/tables, data/original | Source accounts and result records |
| examples/reviewer_score_check | Portable archived F2 account verification |
| results | Latest development and comparator decisions |
| provenance | Original identities and public-copy transformation records |

All scientific arrays and numerical records retained in the public archives are
byte-identical to their original files. Remote launch helpers and two VPN
records are omitted; modified archive containers have new checksums and a
PUBLICATION_MANIFEST.json. Original scientific manifests remain historical
records and can name omitted operational files. No local research history was
rewritten. The public checkout starts with a curated snapshot.

## Access, citation and licence

Code: MIT for original contributions, with preserved third-party notices.
Original data: CC BY 4.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Cite this repository and tag `paper-evidence-2026-09-16-v1`; an article DOI and archival DOI have not
been assigned. Full precursor fields, restart states and original runtime
images are external dependencies: their identifiers are recorded, but this
release does not include all of those large inputs. Their public bulk delivery
remains to be arranged; see [EXTERNAL_DATA.md](EXTERNAL_DATA.md).


## Choice valuation and joint-inflow evidence — 18 September 2026

[This separate release](https://github.com/VickylastShao/JAX-WFCOYAW-MPC/releases/tag/paper-evidence-2026-09-18-transfer-v1) provides three verified compact
archives: the fixed C1 energy–travel valuations (S31), all twelve joint-inflow
control accounts and 72 MPC decisions (S32), and the reproducible Figure S11.
The new conditions use two exposed source roots with exact preview and complete
simulated state. Full results retain every positive and negative contrast.
The portable checks reproduce saved accounts and figures without a GPU;
large-field LES reruns still require the external inputs described above.
Prior releases and their assets remain unchanged.

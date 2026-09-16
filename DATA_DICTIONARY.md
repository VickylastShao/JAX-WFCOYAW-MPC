# Data dictionary and provenance

Source CSV column names and source JSON/NPZ field names are preserved. Each
subdirectory retains its case-specific README, manifest or protocol where
available. Do not assume a common schema across development stages.

- Power fields labelled watts are instantaneous electrical generation in W.
- Yaw histories/targets are in degrees.
- The original LES time step is 0.2 s. A complete 2280 s case has 11400 steps.
- Energy accounts are labelled kWh; apply the units in each frozen scorer.
- Yaw electricity uses the declared 30 kW motion-power assumption and
  fractional movement time at the declared 0.3 degrees/s rate.
- Root IDs identify precursor realizations; T1/T2 identify imposed inflow
  scenarios. A development-state replay is not another independent root.
- Missing/incomplete cases retain their recorded failure state and must not
  be silently converted into zero-energy or successful observations.
- Ratios for pooled outcomes use summed paired energies, not unweighted means
  of rounded percentages. Preserve source precision.

Source_Data_V3 and data/original cover original accounts; strengthening and
revision archives add forecast/actuator/plant diagnostics. The focused archive
covers restored-state action replay; dynamic archives cover comparison-model
qualification. Derivative, native64, actual_target and own_tail archives
preserve the staged numerical diagnoses, original failures and latest G1.
Archived intermediate results remain labelled by their own stage and do not
supersede the final result record.

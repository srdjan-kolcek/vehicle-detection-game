# Datasets

## UA-DETRAC (demo pool)

Source: UA-DETRAC benchmark, downloaded by the owner into data/raw/ (DETRAC-Images, DETRAC-Train-Annotations-XML, DETRAC-Test-Annotations-XML).

Use: the non-commercial demo only. Owner decision (2026-10-03): the licence terms were not re-checked because the dataset is never used in production or commercially. Production and any commercial pitch will use other footage with a confirmed licence.

Licence checklist (doc 04)
- Commercial use: not needed, demo only.
- Redistribution: clips stay inside the demo; the dataset itself is never published.
- Attribution: credit UA-DETRAC in the game rules and the demo's about text.
- Privacy: 960x540 traffic footage; plates and faces are not readable at that resolution.

Split
- Train sequences (60): used for fine-tuning only, never made into clips.
- Test sequences (40): the only source of clips; 38 are mapped to cities in pipeline/angles.yaml. MVI_39501 and MVI_39511 are shorter than 30 s and unused.

Every clip row carries dataset ua-detrac and the licence note "UA-DETRAC, non-commercial demo use only".

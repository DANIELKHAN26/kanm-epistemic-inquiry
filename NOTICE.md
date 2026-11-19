# Copyright and attribution

Copyright (C) 2026 Daniel Khan.

KANM is a new implementation based on published model equations and descriptions. It does not include or translate the historical Laputa source code. The archived Laputa 1.6 source (GPLv3) was read to specify historical behaviour (scheduling, trust handling and assertion rules), which KANM reimplements independently. The historical model and software were developed by Erik J. Olsson and Staffan Angere: see Olsson (2011), *Episteme* 8(2), 127–143, doi:10.3366/epi.2011.0012; Vallinder & Olsson (2014), *Synthese* 191(9), 1991–2007, doi:10.1007/s11229-013-0375-0; and the [SourceForge project record](https://sourceforge.net/projects/epistemenet/). This notice is not a claim that the Laputa authors endorse KANM.

The optional docking harness in `docking/laputa_1_6/` is KANM's own code (stubs, a driver and a build script). When you build it, it compiles the unmodified Laputa 1.6 model files from your own copy of the 1.6 archive, verified by checksum. Those files are not distributed with KANM. The harness's `resetSimulation` restates the nine lines of per-run initialisation in the 1.6 `Simulation::reset` (Simulation.cpp, lines 126–134), because that file is not compiled.

**Data.** No third-party data are distributed with KANM. The loaders download files from their original locations on request, verify SHA-256 checksums, and parse them in memory.
- `kanm.crowd_data`: Snow, O'Connor, Jurafsky & Ng (2008), EMNLP, pp. 254–263; Welinder, Branson, Belongie & Perona (2010), NeurIPS 23; Zheng, Li, Li, Shan & Cheng (2017), PVLDB 10(5), 541–552, doi:10.14778/3055540.3055547.
- `kanm.testimony_data`: Xie & Hayes (2022), *Cognitive Science* 46(5), e13144, doi:10.1111/cogs.13144; data at [osf.io/qng93](https://osf.io/qng93/). The authors' published model predictions are quoted as test values in `tests/test_testimony.py`.
- `kanm.structure_data`: Fränken, Valentin, Lucas & Bramley (2024), *Cognition* 242, 105633, doi:10.1016/j.cognition.2023.105633; data from [github.com/janphilippfranken/spacefish](https://github.com/janphilippfranken/spacefish) (MIT licence), commit `a1681e8`.

`kanm.judgment_inversion` re-implements the Level-1 and Level-2 models of Fränken et al. (2024) from their MIT-licensed code; it is a re-implementation, not a copy. Saved inference values from that repository are quoted as test values in `tests/test_judgment_inversion.py`.

The software licence is GPL-3.0-only; see `LICENSE`.

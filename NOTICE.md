# Copyright and attribution

Copyright (C) 2026 Daniel Khan.

The KANM software is a new implementation based on published model equations and descriptions. It does not include or translate the historical Laputa source code. The archived Laputa 1.6 source (GPLv3) was read to specify historical behaviour (scheduling, trust handling and assertion rules), which KANM reimplements independently; KANM is itself GPL-3.0-only. The historical model and software were developed by Erik J. Olsson and Staffan Angere; see [research/references.md](research/references.md) and the [official SourceForge project record](https://sourceforge.net/projects/epistemenet/).

The optional docking harness in `docking/laputa_1_6/` is KANM's own code (stubs, a driver and a build script). When a user builds it, it compiles and links the unmodified Laputa 1.6 model files from the user's own copy of the 1.6 archive, which is verified by checksum. Those files are not distributed with KANM. The harness's own `resetSimulation` restates the nine lines of per-run initialisation in the 1.6 `Simulation::reset` (Simulation.cpp, lines 126–134), because that file is not compiled. Behaviour that `kanm/laputa16.py` and `Simulation(schedule="laputa_1_6")` take from the 1.6 source is restated from a reading of that code, not copied.

The public crowdsourcing data used in `research/10_public_data_benchmark.md` are not redistributed. `kanm.crowd_data` downloads them from their original or archived locations and verifies checksums. Their authors are cited in [research/references.md](research/references.md), entries 63–70.

The software license is GPL-3.0-only. This notice is not a claim that the historical Laputa authors endorse KANM.

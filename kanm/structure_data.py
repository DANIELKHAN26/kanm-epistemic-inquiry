"""Experiments 1–3 of Fränken, Valentin, Lucas & Bramley (2024), for :mod:`kanm.judgment_inversion`.

**Source:** Fränken, Valentin, Lucas & Bramley (2024), "Naïve information aggregation in human social learning", *Cognition* 242, 105633, doi:10.1016/j.cognition.2023.105633.

**Data:** github.com/janphilippfranken/spacefish, MIT licence, pinned to commit ``a1681e81a84da5934dbffb4da6722cf22eedb133``. :func:`download` fetches the three Experiment 1 files and verifies their SHA-256 checksums; :func:`load` parses them. Participants are keyed anonymously.

**Design:**
- **Evidence:** every participant (agent A) catches one fish, on trial 2, and sees the same scripted judgments from agents B and C. Judgments are stored in the frame where A's fish is red and positive means red; this was checked against the authors' processed files (research/15, section 8).
- **Conditions:** they differ only in the stated structure, ``"c_sees_b"``, ``"b_sees_c"`` or ``"none"``.

**File labels:** the condition is read from each participant's ``structureHint``, together with the file it comes from.
- ``cond_1_BC`` is "C sees B", ``cond_3_CB`` is "B sees C", and ``cond_2_no_struct`` is no structure.
- **Group sizes are 47, 49 and 50.** The article reports 50 and 47 for the first and last of these, apparently swapped.

**Experiments 2–3** (:func:`load_triads`): three real players per room. Roles follow each player's fish timing (A at index 1, B at index 3, C at index 6), and judgments are stored in the canonical frame. Only player A faces a dependence problem, so each record holds A's judgments together with the B and C judgments A saw.
- **Experiment 2:** three rooms are exact duplicates of other rooms (same player keys and records) and are removed. That leaves 21 rooms per condition, as the article reports.
- **Experiment 3:** 22 independent rooms and 21 "C sees B" rooms.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import urllib.request

COMMIT = "a1681e81a84da5934dbffb4da6722cf22eedb133"
_RAW = f"https://raw.githubusercontent.com/janphilippfranken/spacefish/{COMMIT}/data/exp_1/"

#: file name -> (condition, sha256)
FILES = {
    "cond_1_BC.json": ("c_sees_b", "9f6437490169f5f303c72d2a8225e6a963a20570bf70aa0b19dcd45d402d765a"),
    "cond_2_no_struct.json": ("none", "1144dbfb0d250153f83c3abe66ff7e658043066e2e5e076a28b977a62ff547d6"),
    "cond_3_CB.json": ("b_sees_c", "63451f15a5c8e9421e5f9a0d3706f853a2e13b31b7b099e743c3e31510dd5328"),
}
EXPECTED_PARTICIPANTS = {"c_sees_b": 47, "none": 50, "b_sees_c": 49}

_REPO = f"https://raw.githubusercontent.com/janphilippfranken/spacefish/{COMMIT}/data/"
#: experiment -> (local file name, path in the repository, sha256)
TRIAD_FILES = {
    2: ("exp_2_clean_07_12_2021.json", "exp_2/clean_07_12_2021.json",
        "31d6558d51c0e5e2e8e240654f70ab0d3820c1fdbd7f5b64dbcbbc921c90b2da"),
    3: ("exp_3_clean_07_12_2021.json", "exp_3/clean_07_12_2021.json",
        "25f871d87ec1f18b5812e25d3d36ff9514be2f98ae52b8db0435d306ef660b94"),
}
EXPECTED_ROOMS = {2: {"independent": 21, "c_sees_b": 21}, 3: {"independent": 22, "c_sees_b": 21}}
_ROLE = {"1": "A", "3": "B", "6": "C"}

#: displayed judgments of the scripted agents, trials 1–10 (negative = blue)
STRONG_BLUE = (0, -1, -1, -2, -2, -2, -2, -3, -3, -3)   # agent B
WEAK_RED = (0, 0, 0, 1, 1, 1, 1, 1, 1, 1)               # agent C
#: A's own evidence: one red fish on trial 2
OWN_EVIDENCE = (None, "red", None, None, None, None, None, None, None, None)
#: which displayed agent watches the other
WATCHER = {"c_sees_b": "C", "b_sees_c": "B", "none": None}


@dataclass(frozen=True)
class Triad:
    key: str
    experiment: int
    condition: str                   # "independent" or "c_sees_b"
    judgments: tuple[int, ...]       # focal player A, canonical frame
    autofilled: tuple[bool, ...]     # A's automatically filled responses
    b_judgments: tuple[int, ...]     # what A saw from B (auto-filled values included)
    c_judgments: tuple[int, ...]


@dataclass(frozen=True)
class Participant:
    key: str
    condition: str
    judgments: tuple[int, ...]       # canonical frame, trials 1–10
    autofilled: tuple[bool, ...]     # True where the response was filled in automatically


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(directory: str | Path) -> None:
    """Fetch and verify the Experiment 1–3 files into ``directory``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    wanted = [(filename, _RAW + filename, checksum) for filename, (_, checksum) in FILES.items()]
    wanted += [(local, _REPO + remote, checksum) for local, remote, checksum in TRIAD_FILES.values()]
    for filename, url, checksum in wanted:
        path = target / filename
        if path.is_symlink():
            raise ValueError(f"refusing to read or replace symlink at {path}")
        if path.exists() and _sha256(path.read_bytes()) == checksum:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "kanm-research-software"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed URL)
            data = response.read()
        if _sha256(data) != checksum:
            raise ValueError(f"checksum mismatch for {filename}")
        with tempfile.NamedTemporaryFile(dir=target, prefix=f".{filename}.", delete=False) as temp:
            temporary = Path(temp.name)
            temp.write(data)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def parse(text: str, condition: str, label: str) -> list[Participant]:
    """Parse one condition file and check it against the published design."""
    hint = {"none": "no"}.get(condition, "strong")
    strong_slot = "simulated_player_three" if condition == "b_sees_c" else "simulated_player_two"
    weak_slot = "simulated_player_two" if condition == "b_sees_c" else "simulated_player_three"
    out = []
    for index, room in enumerate(json.loads(text).values()):
        if tuple(room[strong_slot]["planet_selections"][:10]) != STRONG_BLUE or \
                tuple(room[weak_slot]["planet_selections"][:10]) != WEAK_RED:
            raise ValueError(f"{label} room {index}: scripted judgments differ from the design")
        humans = [(k, v) for k, v in room.items() if not k.startswith("simulated")]
        if len(humans) != 1:
            raise ValueError(f"{label} room {index}: expected one participant")
        _, record = humans[0]
        info = record["info"]
        if info.get("structureHint") != hint or str(info.get("t_fish")) != "1":
            raise ValueError(f"{label} room {index}: condition or fish timing differs from the design")
        judgments = tuple(int(j) for j in record["planet_selections"])
        autofilled = tuple(bool(s) for s in record["sim_resp"])
        if len(judgments) != 10 or len(autofilled) != 10 or any(j not in range(-3, 4) for j in judgments):
            raise ValueError(f"{label} room {index}: expected 10 judgments on the -3..3 scale")
        out.append(Participant(f"{condition}-{index:03d}", condition, judgments, autofilled))
    return out


def load(directory: str | Path) -> list[Participant]:
    """All Experiment 1 participants, verified against checksums, design and expected counts."""
    directory = Path(directory)
    participants: list[Participant] = []
    for filename, (condition, checksum) in FILES.items():
        path = directory / filename
        if path.is_symlink():
            raise ValueError(f"refusing to read symlink at {path}")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run kanm.structure_data.download first")
        data = path.read_bytes()
        if _sha256(data) != checksum:
            raise ValueError(f"checksum mismatch for {filename}")
        participants += parse(data.decode("utf-8"), condition, filename)
    for condition, expected in EXPECTED_PARTICIPANTS.items():
        found = sum(p.condition == condition for p in participants)
        if found != expected:
            raise ValueError(f"{condition}: expected {expected} participants, found {found}")
    return participants


def parse_triads(text: str, experiment: int) -> list[Triad]:
    """Parse an Experiment 2 or 3 file: one record per unique complete room."""
    out: list[Triad] = []
    seen: set[tuple] = set()
    for room in json.loads(text).values():
        if not room:
            continue  # rooms that never filled
        if len(room) != 3:
            raise ValueError(f"experiment {experiment}: room with {len(room)} players")
        players = {_ROLE.get(str(p["info"]["t_fish"])): p for p in room.values()}
        if set(players) != {"A", "B", "C"}:
            raise ValueError(f"experiment {experiment}: roles do not follow the published fish timing")
        conditions = {p["info"]["condition"] for p in room.values()}
        if len(conditions) != 1 or not conditions <= {"lr", "independent"}:
            raise ValueError(f"experiment {experiment}: inconsistent condition within a room")
        sequences = {r: tuple(int(j) for j in players[r]["planet_selections"]) for r in "ABC"}
        for r, s in sequences.items():
            if len(s) != 10 or any(j not in range(-3, 4) for j in s):
                raise ValueError(f"experiment {experiment}: player {r} lacks 10 judgments on -3..3")
        identity = (tuple(sorted(room)),) + tuple(sequences.values())
        if identity in seen:
            continue  # exact duplicate of a room already read
        seen.add(identity)
        condition = "c_sees_b" if conditions == {"lr"} else "independent"
        out.append(Triad(f"e{experiment}-{condition}-{len(out):03d}", experiment, condition,
                         sequences["A"], tuple(bool(s) for s in players["A"]["sim_resp"]),
                         sequences["B"], sequences["C"]))
    return out


def load_triads(directory: str | Path, experiment: int) -> list[Triad]:
    """Experiment 2 or 3 rooms, verified against checksum and the published room counts."""
    if experiment not in TRIAD_FILES:
        raise ValueError("experiment must be 2 or 3")
    local, _, checksum = TRIAD_FILES[experiment]
    path = Path(directory) / local
    if path.is_symlink():
        raise ValueError(f"refusing to read symlink at {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run kanm.structure_data.download first")
    data = path.read_bytes()
    if _sha256(data) != checksum:
        raise ValueError(f"checksum mismatch for {local}")
    triads = parse_triads(data.decode("utf-8"), experiment)
    for condition, expected in EXPECTED_ROOMS[experiment].items():
        found = sum(t.condition == condition for t in triads)
        if found != expected:
            raise ValueError(f"experiment {experiment} {condition}: expected {expected} rooms, found {found}")
    return triads

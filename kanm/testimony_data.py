"""Trial-level ratings from Xie & Hayes (2022), for testing :mod:`kanm.testimony`.

Source: Xie, B., & Hayes, B. K. (2022). Sensitivity to evidential dependencies in judgments under uncertainty. *Cognitive Science*, 46(5), e13144, doi:10.1111/cogs.13144. Data: https://osf.io/qng93/.

**Licence and handling:** no licence is stated for the data, so the files are **not** redistributed with KANM.
- :func:`download` fetches them into a directory of the user's choice and verifies each SHA-256 checksum.
- :func:`load` parses the files in memory without extracting anything.
- Participants are keyed anonymously. Worker codes, which appear in some file names, are never returned.

**Ratings** follow the authors' coding (their ``responseR``): 0 means "definitely the urn implied by the testimony" and 10 means "definitely the urn implied by the participant's own ball".
- **Experiments 1 and 3:** a linear 11-point scale.
- **Experiment 2:** an 11-point scale whose points are labelled with odds; see :data:`LOG_ODDS_SCALE`.

**Exclusions** are the authors' own.
- **Experiment 1:** the published files already omit attention-check failures. They contain 338 complete participants; the article reports 336, a difference that could not be resolved from the materials.
- **Experiment 2:** participants are kept only if they answered "20" to the attention check, giving 194, as in the article.
- **Experiment 3:** the published file contains the 200 participants analysed in the article.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from math import log
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

#: file name -> (url, sha256), verified 17 November 2026 through the OSF API
FILES = {
    "expt1_data_1of2.csv": (
        "https://osf.io/download/7uta6/",
        "8e0d1e6b5503c566c7feb82737fdd66f9227f1ed4d71e9f7d81b727fe5428eb7"),
    "expt1_data_2of2.csv": (
        "https://osf.io/download/9fu6h/",
        "39d6e27c18f966970b435e6c3e9269ed89d36db9082e42e137b8cbddb271b6a7"),
    "expt2.zip": (
        "https://osf.io/download/xz4qf/",
        "642d7a62de7a054ad5f1f5ddc0d3ec58f9770aab8bb2c4128e621bd6f6507b05"),
    "expt3.zip": (
        "https://osf.io/download/2c6ny/",
        "ed56b1dc4c12e113ba11095585ed2a9ef0d74d29e1cd778818b79ccb644a7b23"),
}
_EXPT3_MEMBER = ("csv2/expt3_data.csv",
                 "143f37d033d7ecdf4c398f04cd6c28dca306a1264b5240b3a19677c039c31fc2")

#: Experiment 2 scale: position -> odds for the own-ball urn (position 5 is even odds).
#: The endpoints are labelled ">40x"; their odds are a lower bound, not a value.
LOG_ODDS_SCALE = (1 / 40, 1 / 20, 1 / 10, 1 / 5, 1 / 2, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)

#: participants with complete data in the published files
EXPECTED_PARTICIPANTS = {1: 338, 2: 194, 3: 200}


@dataclass(frozen=True)
class Rating:
    experiment: int          # 1, 2 or 3
    participant: str         # anonymous key, unique across experiments
    informants: int          # 3, 5 or 7
    dependency: str          # "independent" or "sequential"
    trial: int               # 1..informants: after each statement; informants + 1: after own ball
    rating: int              # 0 = testimony urn ... 10 = own-ball urn
    scale: str               # "linear" or "log_odds"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified(data: bytes, expected: str, label: str) -> bytes:
    actual = _sha256(data)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {label}: expected {expected}, got {actual}")
    return data


def download(directory: str | Path) -> None:
    """Fetch and verify the four data files into ``directory``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for filename, (url, checksum) in FILES.items():
        path = target / filename
        if path.is_symlink():
            raise ValueError(f"refusing to read or replace symlink at {path}")
        if path.exists() and _sha256(path.read_bytes()) == checksum:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "kanm-research-software"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed URLs)
            data = _verified(response.read(), checksum, filename)
        with tempfile.NamedTemporaryFile(dir=target, prefix=f".{filename}.", delete=False) as temp:
            temporary = Path(temp.name)
            temp.write(data)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _file(directory: Path, filename: str) -> bytes:
    path = directory / filename
    if path.is_symlink():
        raise ValueError(f"refusing to read symlink at {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run kanm.testimony_data.download first")
    return _verified(path.read_bytes(), FILES[filename][1], filename)


def _dependency(code: str) -> str:
    code = code.strip().lower()
    if code in ("i", "independent"):
        return "independent"
    if code in ("sq", "sequential"):
        return "sequential"
    raise ValueError(f"unknown dependency code {code!r}")


def _rating(value: str, label: str) -> int:
    number = float(value)
    if number != int(number) or not 0 <= number <= 10:
        raise ValueError(f"{label}: rating {value!r} is not an integer in 0..10")
    return int(number)


def _complete(records: list[Rating], label: str) -> list[Rating]:
    trials = sorted(r.trial for r in records)
    expected = list(range(1, records[0].informants + 2))
    if trials != expected:
        raise ValueError(f"{label}: expected trials {expected}, found {trials}")
    return records


def parse_experiment1(texts: dict[str, str]) -> list[Rating]:
    """Parse the two Experiment 1 CSV files (``part`` -> text)."""
    ratings: list[Rating] = []
    for part, text in sorted(texts.items()):
        by_subject: dict[str, list[Rating]] = {}
        for row in csv.DictReader(io.StringIO(text)):
            if row["responseR"] == "NA":
                continue  # padding rows for trials beyond a condition's length
            informants = int(row["distribution"].split("-")[0])
            raw, coded = _rating(row["response"], "response"), _rating(row["responseR"], "responseR")
            own = row["colourPart"].strip().lower()
            if own not in ("blue", "red") or coded != (10 - raw if own == "blue" else raw):
                raise ValueError(f"experiment 1: unexpected coding for subject {row['subj_id']}")
            key = f"1{part}-{row['subj_id']}"
            by_subject.setdefault(key, []).append(Rating(
                1, key, informants, _dependency(row["dependency"]), int(row["question"]),
                coded, "linear"))
        for key, records in by_subject.items():
            ratings += _complete(records, key)
    return ratings


_LABEL = re.compile(r"<font color=white>(Q\d+|Q_final)</font>")


def parse_experiment2(files: list[str]) -> list[Rating]:
    """Parse Experiment 2 per-participant jsPsych CSV texts (in a fixed order)."""
    ratings: list[Rating] = []
    for index, text in enumerate(files):
        rows = list(csv.DictReader(io.StringIO(text)))
        answers = [json.loads(r["responses"]).get("Q0", "") for r in rows
                   if r["trial_type"] == "survey-text"]
        if len(answers) != 1 or answers[0].strip() != "20":
            continue  # failed or did not reach the attention check
        first = rows[0]
        informants = int(first["informants"])
        own = first["colourPart"].strip().lower()
        if own not in ("blue", "red") or first["scale"] != "odds":
            raise ValueError(f"experiment 2 file {index}: unexpected colour or scale")
        key = f"2-{index:03d}"
        records = []
        for r in rows:
            match = _LABEL.search(r["stimulus"]) if r["trial_type"] == "image-slider-response" else None
            if not match:
                continue
            label = match.group(1)
            trial = informants + 1 if label == "Q_final" else int(label[1:])
            raw = _rating(r["response"], key)  # 0 = >40x blue ... 10 = >40x red
            records.append(Rating(2, key, informants, _dependency(first["dependency"]), trial,
                                  10 - raw if own == "blue" else raw, "log_odds"))
        ratings += _complete(records, key)
    return ratings


def parse_experiment3(text: str) -> list[Rating]:
    """Parse the cleaned Experiment 3 CSV (dependency manipulated within participants)."""
    ratings: list[Rating] = []
    for index, row in enumerate(csv.DictReader(io.StringIO(text))):
        informants = int(row["informants"])
        key = f"3-{index:03d}"
        for code in ("i", "sq"):
            records = [Rating(3, key, informants, _dependency(code), trial,
                              _rating(row[f"{code}-{trial}"], key), "linear")
                       for trial in range(1, informants + 2)]
            ratings += _complete(records, f"{key}/{code}")
    return ratings


def load(directory: str | Path) -> list[Rating]:
    """All ratings from Experiments 1–3, verified against checksums and expected counts."""
    directory = Path(directory)
    e1 = parse_experiment1({
        "a": _file(directory, "expt1_data_1of2.csv").decode("utf-8-sig"),
        "b": _file(directory, "expt1_data_2of2.csv").decode("utf-8-sig"),
    })
    with zipfile.ZipFile(io.BytesIO(_file(directory, "expt2.zip"))) as archive:
        names = sorted(n for n in archive.namelist()
                       if n.startswith("csv-participants/") and n.endswith(".csv"))
        e2 = parse_experiment2([archive.read(n).decode("utf-8-sig") for n in names])
    with zipfile.ZipFile(io.BytesIO(_file(directory, "expt3.zip"))) as archive:
        member, checksum = _EXPT3_MEMBER
        e3 = parse_experiment3(_verified(archive.read(member), checksum, member).decode("utf-8-sig"))
    ratings = e1 + e2 + e3
    for experiment, expected in EXPECTED_PARTICIPANTS.items():
        found = len({r.participant for r in ratings if r.experiment == experiment})
        if found != expected:
            raise ValueError(f"experiment {experiment}: expected {expected} participants, found {found}")
    return ratings


def scale_position(probability_testimony: float, scale: str) -> float:
    """Model probability for the testimony urn, placed on the 0–10 rating scale.

    0 means the testimony urn and 10 the own-ball urn, as in the data.
    - **Linear scale:** ``10 * P(own-ball urn)``.
    - **Log-odds scale:** linear interpolation in log-odds between the labelled points, held at the endpoints.
    """
    if not 0.0 < probability_testimony < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    own = 1.0 - probability_testimony
    if scale == "linear":
        return 10.0 * own
    if scale != "log_odds":
        raise ValueError("scale must be 'linear' or 'log_odds'")
    x = log(own / probability_testimony)
    points = [log(o) for o in LOG_ODDS_SCALE]
    if x <= points[0]:
        return 0.0
    if x >= points[-1]:
        return 10.0
    for position in range(10):
        lo, hi = points[position], points[position + 1]
        if lo <= x <= hi:
            return position + (x - lo) / (hi - lo)
    raise AssertionError("unreachable")

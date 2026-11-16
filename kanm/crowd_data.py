"""Public crowdsourcing data sets with gold answers, for :mod:`kanm.reliability`.

The data are **not** redistributed with KANM: for most of them no licence is
stated. :func:`download` fetches the original files (or the stated archive
copy) into a directory of the user's choice and verifies every file against
the SHA-256 checksum recorded here; :func:`load` parses them. Archive members
are read in memory by exact name and never extracted, so a malformed archive
cannot write outside the target directory.

Provenance (each verified on 2026-11-15 by downloading and checksumming):

``snow2008_rte``, ``snow2008_temp``
    Snow, O'Connor, Jurafsky & Ng (2008), "Cheap and Fast - But is it Good?
    Evaluating Non-Expert Annotations for Natural Language Tasks", EMNLP,
    pp. 254-263, https://aclanthology.org/D08-1027/. The original host now
    requires a sign-in; the file is the Internet Archive copy of the original
    ``all_collected_data.tgz``. No licence stated. TEMP labels are coded 1/2
    in the file and are mapped to 0/1 here (2 -> 1); the file does not say
    which code means "before".
``welinder2010_bluebirds``
    Welinder, Branson, Perona & Belongie (2010), "The Multidimensional Wisdom
    of Crowds", NeurIPS 23. Files from github.com/welinder/cubam (repository
    licence: Clear BSD, stated for the software). Identical to "d_Duck
    Identification" in Zheng et al. (2017).
``zheng2017_product``, ``zheng2017_sentiment``
    Zheng, Li, Li, Shan & Cheng (2017), "Truth Inference in Crowdsourcing: Is
    the Problem Solved?", PVLDB 10(5):541-552, doi:10.14778/3055540.3055547.
    D_Product originates from Wang, Kraska, Franklin & Feng (2012), CrowdER,
    PVLDB 5(11), doi:10.14778/2350229.2350263. No licence stated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import re
import tarfile
import tempfile
import urllib.request
import zipfile

from .reliability import CrowdData

_SNOW_URL = (
    "https://web.archive.org/web/20160330040359id_/"
    "https://sites.google.com/site/nlpannotations/all_collected_data.tgz?attredirects=0"
)
_ZHENG_URL = "https://zhydhkcws.github.io/crowd_truth_inference/datasets.zip"
_CUBAM = "https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/"

#: file name -> (url, sha256)
FILES = {
    "all_collected_data.tgz": (
        _SNOW_URL, "158786ca0caa0b668439d3381d1bacbffd5f7cbc29bfd0eb33f575ff14d64c73"),
    "zheng2017_datasets.zip": (
        _ZHENG_URL, "c68ee01613da6dd6e2405c3252b73bb1199bc263909588bf952f57fc7d84323c"),
    "bluebirds_gt.yaml": (
        _CUBAM + "gt.yaml", "c9eb769c845ca99495a07b5c28c4290c78436fd8eac65c0747736c4ef8126340"),
    "bluebirds_labels.yaml": (
        _CUBAM + "labels.yaml", "8b1b28e4b869b19bb03beac476c7073149cef784f9524ebe17380a1292a6126c"),
}

#: archive members used, with their own checksums
MEMBERS = {
    ("all_collected_data.tgz", "rte.standardized.tsv"):
        "3187c5de59b9b9c596d490d25b5f5f108090a35cbdd10e81f21e6d1939d6ba79",
    ("all_collected_data.tgz", "temp.standardized.tsv"):
        "7f74de9f36a0d3dfdc1f51e6433526abfc788c616d0c2d04a96206780ac74860",
    ("zheng2017_datasets.zip", "datasets/d_jn-product/answer.csv"):
        "ca69e3f2cba3d6f9c3b3ecc2f8d2a81afcd747d0b91aa5ab18608a5715e26adc",
    ("zheng2017_datasets.zip", "datasets/d_jn-product/truth.csv"):
        "9a5757c726ef1a32b05f6a675924c483b24d602c61b1a2d087ab3d84f90b3766",
    ("zheng2017_datasets.zip", "datasets/d_sentiment/answer.csv"):
        "8f4006278ca288a8d73a20cc37348f71b201f1bcb471a9aacd80421d8a34be26",
    ("zheng2017_datasets.zip", "datasets/d_sentiment/truth.csv"):
        "bac9bebdfdcbc5227cee6f95e27754f1089421e39317df54b9e2680bc78d9976",
}

DATASETS = (
    "snow2008_rte",
    "snow2008_temp",
    "welinder2010_bluebirds",
    "zheng2017_product",
    "zheng2017_sentiment",
)

_REQUIRED = {
    "snow2008_rte": ("all_collected_data.tgz",),
    "snow2008_temp": ("all_collected_data.tgz",),
    "welinder2010_bluebirds": ("bluebirds_gt.yaml", "bluebirds_labels.yaml"),
    "zheng2017_product": ("zheng2017_datasets.zip",),
    "zheng2017_sentiment": ("zheng2017_datasets.zip",),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified(data: bytes, expected: str, label: str) -> bytes:
    actual = _sha256(data)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {label}: expected {expected}, got {actual}")
    return data


def download(directory: str | Path, datasets: tuple[str, ...] = DATASETS) -> None:
    """Fetch the files needed for ``datasets`` into ``directory`` (verified)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    needed = sorted({f for name in datasets for f in _REQUIRED[_check_name(name)]})
    for filename in needed:
        url, checksum = FILES[filename]
        path = target / filename
        if path.is_symlink():
            raise ValueError(f"refusing to read or replace symlink at {path}")
        if path.exists() and _sha256(path.read_bytes()) == checksum:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "kanm-research-software"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed URLs)
            data = response.read()
        data = _verified(data, checksum, filename)
        with tempfile.NamedTemporaryFile(dir=target, prefix=f".{filename}.", delete=False) as temp:
            temporary = Path(temp.name)
            temp.write(data)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _check_name(name: str) -> str:
    if name not in DATASETS:
        raise ValueError(f"unknown data set {name!r}; choose from {DATASETS}")
    return name


def _file(directory: Path, filename: str) -> bytes:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run kanm.crowd_data.download first")
    return _verified(path.read_bytes(), FILES[filename][1], filename)


def _member(directory: Path, archive: str, member: str) -> bytes:
    raw = _file(directory, archive)
    if archive.endswith(".tgz"):
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            info = tar.getmember(member)
            if not info.isfile():
                raise ValueError(f"{member} is not a regular file")
            data = tar.extractfile(info).read()
    else:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            data = zf.read(member)
    return _verified(data, MEMBERS[(archive, member)], member)


def _snow(directory: Path, member: str, name: str, mapping: dict[str, int]) -> CrowdData:
    text = _member(directory, "all_collected_data.tgz", member).decode("utf-8")
    labels, gold = [], {}
    rows = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(rows)
    if header[:5] != ["!amt_annotation_ids", "!amt_worker_ids", "orig_id", "response", "gold"]:
        raise ValueError(f"unexpected header in {member}: {header}")
    for row in rows:
        _, worker, item, response, answer = row[:5]
        labels.append((item, worker, mapping[response]))
        if item in gold and gold[item] != mapping[answer]:
            raise ValueError(f"inconsistent gold for item {item} in {member}")
        gold[item] = mapping[answer]
    return CrowdData(name, tuple(labels), gold)


def _zheng(directory: Path, folder: str, name: str) -> CrowdData:
    def table(file: str) -> list[list[str]]:
        text = _member(directory, "zheng2017_datasets.zip", f"datasets/{folder}/{file}").decode("utf-8")
        rows = list(csv.reader(io.StringIO(text)))
        return rows[1:]

    labels = [(item, worker, int(label)) for item, worker, label in table("answer.csv")]
    gold = {item: int(truth) for item, truth in table("truth.csv")}
    return CrowdData(name, tuple(labels), gold)


_YAML_ENTRY = re.compile(r"(\d+)\s*:\s*(true|false)")


def _parse_yaml_pairs(text: str, label: str) -> list[tuple[str, str]]:
    """Parse the dataset's deliberately small flow-map subset strictly."""
    body = text.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    entries = [part.strip() for part in body.split(",") if part.strip()]
    pairs = []
    for entry in entries:
        match = _YAML_ENTRY.fullmatch(entry)
        if match is None:
            raise ValueError(f"unexpected Bluebirds YAML entry in {label}: {entry!r}")
        pairs.append((match.group(1), match.group(2)))
    if not pairs:
        raise ValueError(f"no Bluebirds entries found in {label}")
    if len({item for item, _ in pairs}) != len(pairs):
        raise ValueError(f"duplicate item identifier in {label}")
    return pairs


def _bluebirds(directory: Path) -> CrowdData:
    return parse_bluebirds(
        _file(directory, "bluebirds_gt.yaml").decode("utf-8"),
        _file(directory, "bluebirds_labels.yaml").decode("utf-8"),
    )


def parse_bluebirds(gold_text: str, labels_text: str) -> CrowdData:
    """Parse the two flow-style YAML files of the Bluebirds data (no YAML library)."""
    gold = {item: int(value == "true") for item, value in _parse_yaml_pairs(gold_text, "gold")}
    labels = []
    # Top-level entries look like "<worker>: {<item>: true, ...}" over several lines.
    blocks = list(re.finditer(r"^(\S+?):\s*\{(.*?)\}", labels_text, flags=re.M | re.S))
    residual, end = [], 0
    for block in blocks:
        residual.append(labels_text[end:block.start()])
        end = block.end()
        worker = block.group(1)
        for item, value in _parse_yaml_pairs(block.group(2), f"worker {worker}"):
            labels.append((item, worker, int(value == "true")))
    residual.append(labels_text[end:])
    unparsed = "\n".join(
        line for chunk in residual for line in chunk.splitlines()
        if line.partition("#")[0].strip()
    )
    if unparsed:
        raise ValueError(f"unparsed content in Bluebirds labels: {unparsed[:80]!r}")
    if not blocks:
        raise ValueError("no Bluebirds worker records found")
    unknown_items = {item for item, _, _ in labels} - set(gold)
    if unknown_items:
        raise ValueError(f"Bluebirds labels lack gold answers for {len(unknown_items)} items")
    return CrowdData("welinder2010_bluebirds", tuple(labels), gold)


def load(name: str, directory: str | Path) -> CrowdData:
    """Load a verified data set from ``directory``."""
    directory = Path(directory)
    name = _check_name(name)
    if name == "snow2008_rte":
        return _snow(directory, "rte.standardized.tsv", name, {"0": 0, "1": 1})
    if name == "snow2008_temp":
        return _snow(directory, "temp.standardized.tsv", name, {"1": 0, "2": 1})
    if name == "welinder2010_bluebirds":
        return _bluebirds(directory)
    if name == "zheng2017_product":
        return _zheng(directory, "d_jn-product", name)
    return _zheng(directory, "d_sentiment", name)

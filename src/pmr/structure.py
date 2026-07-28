import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

TOKEN = re.compile(r"[a-z0-9']+")

VERSE = "verse"
CHORUS = "chorus"


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class Structure:
    source: str
    duration: float
    sections: list[Section]


def tokenize(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def label_units(texts: list[str], threshold: float) -> list[str]:
    tokens = [tokenize(text) for text in texts]
    labels = [VERSE] * len(texts)
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if similarity(tokens[left], tokens[right]) >= threshold:
                labels[left] = CHORUS
                labels[right] = CHORUS
    return labels


def merge_sections(
    labels: list[str], spans: list[tuple[float, float]]
) -> list[Section]:
    sections: list[Section] = []
    for label, (start, end) in zip(labels, spans):
        if sections and sections[-1].label == label:
            sections[-1] = Section(sections[-1].start, end, label)
        else:
            sections.append(Section(start, end, label))
    return sections


def write_structure(structure: Structure, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(structure), indent=2) + "\n")


def structure_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Structure]:
    dataset = config.data.dataset
    alignment_dir = paths.alignments(dataset)
    target_dir = paths.structures(dataset)

    tracks = sorted(alignment_dir.glob("*.json")) if alignment_dir.exists() else []
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [path for path in tracks if path.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no alignments found under %s", alignment_dir)
        return []

    settings = config.data.structure
    logger.info("labelling %d tracks", len(tracks))

    results: list[Structure] = []
    for index, alignment in enumerate(tracks, start=1):
        target = target_dir / alignment.name
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", target.name)
            continue
        data = json.loads(alignment.read_text())
        pairs = data["pairs"]
        spans = [(pair["melody"]["start"], pair["melody"]["end"]) for pair in pairs]
        duration = max((end for _, end in spans), default=0.0)
        if duration < settings.min_duration_s:
            sections = [Section(0.0, duration, VERSE)] if duration > 0.0 else []
            logger.info(
                "[%d/%d] %s too short (%.1fs), no chorus",
                index,
                len(tracks),
                alignment.stem,
                duration,
            )
        else:
            texts = [pair["lyric"]["text"] for pair in pairs]
            labels = label_units(texts, settings.similarity)
            sections = merge_sections(labels, spans)
            logger.info(
                "[%d/%d] %s -> %d section(s)",
                index,
                len(tracks),
                alignment.stem,
                len(sections),
            )
        structure = Structure(
            source=data["source"], duration=duration, sections=sections
        )
        write_structure(structure, target)
        results.append(structure)

    logger.info("wrote %d to %s", len(results), target_dir)
    return results

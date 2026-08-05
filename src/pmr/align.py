import json
import logging
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

SILENCE_START = re.compile(r"silence_start: (-?[0-9.]+)")
SILENCE_END = re.compile(r"silence_end: (-?[0-9.]+)")


@dataclass(frozen=True)
class Lyric:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Melody:
    start: float
    end: float


@dataclass(frozen=True)
class Pair:
    lyric: Lyric
    melody: Melody


@dataclass(frozen=True)
class Alignment:
    source: str
    pairs: list[Pair]


def detect_phrases(
    source: Path, noise_db: float, min_silence_s: float
) -> list[tuple[float, float]]:
    duration = sf.info(str(source)).duration
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    starts = [max(0.0, float(v)) for v in SILENCE_START.findall(result.stderr)]
    ends = [max(0.0, float(v)) for v in SILENCE_END.findall(result.stderr)]

    phrases: list[tuple[float, float]] = []
    cursor = 0.0
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        if start > cursor:
            phrases.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        phrases.append((cursor, duration))
    if not phrases and duration > 0:
        phrases.append((0.0, duration))
    return phrases


def align_pairs(
    lyrics: list[Lyric], phrases: list[tuple[float, float]]
) -> list[Pair]:
    pairs: list[Pair] = []
    index = 0
    for lyric in lyrics:
        midpoint = (lyric.start + lyric.end) / 2
        while index < len(phrases) - 1 and phrases[index][1] < midpoint:
            index += 1
        start, end = phrases[index]
        pairs.append(Pair(lyric=lyric, melody=Melody(start=start, end=end)))
    return pairs


def load_lyrics(source: Path) -> list[Lyric]:
    data = json.loads(source.read_text())
    return [
        Lyric(text=segment["text"], start=segment["start"], end=segment["end"])
        for segment in data["segments"]
    ]


def align_track(
    vocal: Path, transcription: Path, noise_db: float, min_silence_s: float
) -> Alignment:
    phrases = detect_phrases(vocal, noise_db, min_silence_s)
    pairs = align_pairs(load_lyrics(transcription), phrases)
    return Alignment(source=vocal.name, pairs=pairs)


def write_alignment(alignment: Alignment, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(alignment), indent=2) + "\n")


def align_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Alignment]:
    dataset = config.data.dataset
    vocal_dir = paths.vocals(dataset)
    transcription_dir = paths.transcriptions(dataset)
    target_dir = paths.alignments(dataset)

    transcriptions = (
        sorted(transcription_dir.glob("*.json"))
        if transcription_dir.exists()
        else []
    )
    if track_ids is not None:
        wanted = set(track_ids)
        transcriptions = [path for path in transcriptions if path.stem in wanted]
    if limit is not None:
        transcriptions = transcriptions[:limit]

    if not transcriptions:
        logger.warning("no transcriptions found under %s", transcription_dir)
        return []

    settings = config.data.alignment
    logger.info("aligning %d tracks", len(transcriptions))

    results: list[Alignment] = []
    skipped = 0
    for index, transcription in enumerate(transcriptions, start=1):
        vocal = vocal_dir / f"{transcription.stem}.wav"
        target = target_dir / transcription.name
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", target.name)
            continue
        try:
            alignment = align_track(
                vocal,
                transcription,
                settings.silence_noise_db,
                settings.min_silence_s,
            )
        except Exception as error:
            logger.error(
                "[%d/%d] %s failed: %s",
                index,
                len(transcriptions),
                transcription.name,
                error,
            )
            skipped += 1
            continue
        write_alignment(alignment, target)
        logger.info(
            "[%d/%d] %s -> %d pairs",
            index,
            len(transcriptions),
            transcription.name,
            len(alignment.pairs),
        )
        results.append(alignment)

    logger.info("wrote %d to %s (%d skipped)", len(results), target_dir, skipped)
    return results

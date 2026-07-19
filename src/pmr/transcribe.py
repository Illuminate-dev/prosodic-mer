import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


@dataclass(frozen=True)
class Transcription:
    source: str
    language: str
    duration: float
    segments: list[Segment]


class Transcriber:
    def __init__(self, model: str, device: str):
        logger.info("loading whisper: %s (%s)", model, device)
        self._model = WhisperModel(model, device=device)

    def transcribe(self, source: Path) -> Transcription:
        raw_segments, info = self._model.transcribe(
            str(source), word_timestamps=True, vad_filter=True
        )
        segments = [
            Segment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=[
                    Word(text=word.word.strip(), start=word.start, end=word.end)
                    for word in segment.words or []
                ],
            )
            for segment in raw_segments
        ]
        return Transcription(
            source=source.name,
            language=info.language,
            duration=info.duration,
            segments=segments,
        )


def write_transcription(transcription: Transcription, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(transcription), indent=2) + "\n")


def transcribe_dataset(
    config: ProjectConfig,
    paths: ProjectPaths,
    track_ids: Iterable[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Transcription]:
    dataset = config.data.dataset
    vocal_dir = paths.vocals(dataset)
    target_dir = paths.transcriptions(dataset)

    tracks = sorted(vocal_dir.glob("*.wav")) if vocal_dir.exists() else []
    if track_ids is not None:
        wanted = set(track_ids)
        tracks = [track for track in tracks if track.stem in wanted]
    if limit is not None:
        tracks = tracks[:limit]

    if not tracks:
        logger.warning("no vocals found under %s", vocal_dir)
        return []

    settings = config.data.transcription
    transcriber = Transcriber(settings.model, settings.device)
    logger.info("transcribing %d tracks", len(tracks))

    results: list[Transcription] = []
    for index, source in enumerate(tracks, start=1):
        target = target_dir / f"{source.stem}.json"
        if target.exists() and not overwrite:
            logger.debug("skipping existing %s", target.name)
            continue
        try:
            transcription = transcriber.transcribe(source)
        except Exception as error:
            logger.error(
                "[%d/%d] %s failed: %s", index, len(tracks), source.name, error
            )
            continue
        write_transcription(transcription, target)
        logger.info(
            "[%d/%d] %s -> %d segments",
            index,
            len(tracks),
            source.name,
            len(transcription.segments),
        )
        results.append(transcription)

    logger.info("wrote %d to %s", len(results), target_dir)
    return results

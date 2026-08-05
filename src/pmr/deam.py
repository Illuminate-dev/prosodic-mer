import logging
import urllib.request
import zipfile
from pathlib import Path

from pmr.config import ProjectConfig
from pmr.paths import ProjectPaths

logger = logging.getLogger(__name__)

BASE_URL = "http://cvml.unige.ch/databases/DEAM"
AUDIO_ARCHIVE = "DEAM_audio.zip"
ANNOTATION_ARCHIVE = "DEAM_Annotations.zip"
AUDIO_PREFIX = "MEMD_audio/"


def download(name: str, target: Path) -> Path:
    if target.exists():
        logger.info("using cached %s", target.name)
        return target
    logger.info("downloading %s", name)
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)
    return target


def extract_audio(archive: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.startswith(AUDIO_PREFIX) and member.endswith(".mp3"):
                target = target_dir / Path(member).name
                if not target.exists():
                    with zf.open(member) as source, target.open("wb") as sink:
                        sink.write(source.read())
                count += 1
    return count


def extract_annotations(archive: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)


def fetch_deam(config: ProjectConfig, paths: ProjectPaths) -> int:
    dataset = config.data.dataset
    target_dir = paths.raw_dataset(dataset)
    cache_dir = paths.raw / ".cache"
    audio = download(AUDIO_ARCHIVE, cache_dir / AUDIO_ARCHIVE)
    annotations = download(ANNOTATION_ARCHIVE, cache_dir / ANNOTATION_ARCHIVE)
    count = extract_audio(audio, target_dir)
    extract_annotations(annotations, target_dir)
    logger.info("extracted %d tracks to %s", count, target_dir)
    return count

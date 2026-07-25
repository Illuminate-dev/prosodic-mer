import os
from dataclasses import dataclass
from pathlib import Path

ENV_ROOT = "PMR_ROOT"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        override = os.environ.get(ENV_ROOT)
        if override:
            return cls(root=Path(override).expanduser().resolve())
        return cls(root=Path(__file__).resolve().parents[2])

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    def raw_dataset(self, dataset: str) -> Path:
        return self.raw / dataset

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    def vocals(self, dataset: str) -> Path:
        return self.processed / "vocals" / dataset

    def transcriptions(self, dataset: str) -> Path:
        return self.processed / "transcriptions" / dataset

    def alignments(self, dataset: str) -> Path:
        return self.processed / "alignments" / dataset

    def features(self, dataset: str, unit: str, modality: str) -> Path:
        return self.processed / "features" / dataset / unit / modality

    def ensure_outputs(self) -> None:
        for path in (self.processed, self.processed / "vocals"):
            path.mkdir(parents=True, exist_ok=True)

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeparationConfig:
    spleeter_stems: str = "2stems"


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = "small"
    device: str = "auto"


@dataclass(frozen=True)
class AlignmentConfig:
    silence_noise_db: float = -35.0
    min_silence_s: float = 0.35


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "DEAM"
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "prosodic-mer"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        if not self.data.separation.spleeter_stems:
            raise ValueError("separation.spleeter_stems must be set")
        if not self.data.transcription.model:
            raise ValueError("transcription.model must be set")

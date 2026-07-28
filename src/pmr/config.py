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
class FeatureConfig:
    unit: str = "sentence"


@dataclass(frozen=True)
class ModelConfig:
    emotion_dim: int = 128
    depth: int = 3


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "DEAM"
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "prosodic-mer"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        if not self.data.separation.spleeter_stems:
            raise ValueError("separation.spleeter_stems must be set")
        if not self.data.transcription.model:
            raise ValueError("transcription.model must be set")
        if self.data.features.unit not in ("sentence", "word"):
            raise ValueError("features.unit must be 'sentence' or 'word'")
        if self.model.emotion_dim % 2:
            raise ValueError("model.emotion_dim must be even")
        if self.model.depth < 1:
            raise ValueError("model.depth must be positive")

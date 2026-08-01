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
class StructureConfig:
    similarity: float = 0.8
    min_duration_s: float = 50.0


@dataclass(frozen=True)
class ModelConfig:
    emotion_dim: int = 128
    depth: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "DEAM"
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "prosodic-mer"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not self.data.separation.spleeter_stems:
            raise ValueError("separation.spleeter_stems must be set")
        if not self.data.transcription.model:
            raise ValueError("transcription.model must be set")
        if self.data.features.unit not in ("sentence", "word"):
            raise ValueError("features.unit must be 'sentence' or 'word'")
        if not 0.0 < self.data.structure.similarity <= 1.0:
            raise ValueError("structure.similarity must be in (0, 1]")
        if self.data.structure.min_duration_s <= 0.0:
            raise ValueError("structure.min_duration_s must be positive")
        if self.model.emotion_dim % 2:
            raise ValueError("model.emotion_dim must be even")
        if self.model.depth < 1:
            raise ValueError("model.depth must be positive")
        if self.training.batch_size < 1 or self.training.epochs < 1:
            raise ValueError("training.batch_size and epochs must be positive")
        if not 0.0 <= self.training.val_fraction < 1.0:
            raise ValueError("training.val_fraction must be in [0, 1)")
        if not 0.0 <= self.training.test_fraction < 1.0:
            raise ValueError("training.test_fraction must be in [0, 1)")
        if self.training.val_fraction + self.training.test_fraction >= 1.0:
            raise ValueError("training val + test fractions must be < 1")

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeparationConfig:
    spleeter_stems: str = "2stems"


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "DEAM"
    separation: SeparationConfig = field(default_factory=SeparationConfig)


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "prosodic-mer"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        if not self.data.separation.spleeter_stems:
            raise ValueError("separation.spleeter_stems must be set")

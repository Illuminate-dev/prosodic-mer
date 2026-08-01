import torch
from torch import nn

from pmr.config import ProjectConfig


class EmotionLSTMCell(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        gates = input_dim + dim
        self.input_gate = nn.Linear(gates, dim)
        self.forget_gate = nn.Linear(gates, dim)
        self.output_gate = nn.Linear(gates, dim)
        self.candidate = nn.Linear(gates, dim)
        self.emotion = nn.Linear(2 * dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        e_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joined = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.input_gate(joined))
        f = torch.sigmoid(self.forget_gate(joined))
        o = torch.sigmoid(self.output_gate(joined))
        candidate = torch.tanh(self.candidate(joined))
        c_candidate = f * c + i * candidate
        h_candidate = o * torch.tanh(c_candidate)
        e_candidate = self.emotion(torch.cat([h_candidate, c_candidate], dim=-1))

        retain = _cumsum_softmax(e_prev)
        refresh = _cumsum_softmax(e_candidate)
        w = retain * refresh
        c_next = (
            w * (f * c + i * candidate)
            + (retain - w) * c
            + (refresh - w) * candidate
        )
        h_next = o * torch.tanh(c_next)
        e_next = self.emotion(torch.cat([h_next, c_next], dim=-1))
        return h_next, c_next, e_next


class CrossProcessingLayer(nn.Module):
    def __init__(self, melody_dim: int, lyric_dim: int, dim: int):
        super().__init__()
        self.dim = dim
        self.melody = EmotionLSTMCell(melody_dim, dim)
        self.lyric = EmotionLSTMCell(lyric_dim, dim)
        self.fuse = nn.Linear(2 * dim, dim)
        self.emotion0 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, melody: torch.Tensor, lyric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _ = melody.shape
        device = melody.device
        h_melody = c_melody = torch.zeros(batch, self.dim, device=device)
        h_lyric = c_lyric = torch.zeros(batch, self.dim, device=device)
        emotion = self.emotion0.expand(batch, -1)

        melody_out, lyric_out, emotions = [], [], []
        for step in range(steps):
            h_melody, c_melody, e_melody = self.melody(
                melody[:, step], h_melody, c_melody, emotion
            )
            h_lyric, c_lyric, e_lyric = self.lyric(
                lyric[:, step], h_lyric, c_lyric, emotion
            )
            gate = torch.sigmoid(self.fuse(torch.cat([e_melody, e_lyric], dim=-1)))
            emotion = gate * e_melody + (1 - gate) * e_lyric
            melody_out.append(h_melody)
            lyric_out.append(h_lyric)
            emotions.append(emotion)

        return (
            torch.stack(melody_out, dim=1),
            torch.stack(lyric_out, dim=1),
            torch.stack(emotions, dim=1),
        )


class CrossProcessing(nn.Module):
    def __init__(
        self,
        melody_dim: int,
        lyric_dim: int,
        dim: int,
        depth: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossProcessingLayer(
                melody_dim if index == 0 else dim,
                lyric_dim if index == 0 else dim,
                dim,
            )
            for index in range(depth)
        )

    def forward(self, melody: torch.Tensor, lyric: torch.Tensor) -> torch.Tensor:
        emotions = None
        for layer in self.layers:
            melody, lyric, emotions = layer(melody, lyric)
        return emotions


class HierarchicalCrossProcessing(nn.Module):
    def __init__(self, melody_dim: int, lyric_dim: int, dim: int, depth: int):
        super().__init__()
        self.verse = CrossProcessing(melody_dim, lyric_dim, dim, depth)
        self.chorus = CrossProcessing(melody_dim, lyric_dim, dim, depth)

    def forward(
        self, melody: torch.Tensor, lyric: torch.Tensor, chorus: torch.Tensor
    ) -> torch.Tensor:
        verse = self.verse(melody, lyric)
        chorus_emotions = self.chorus(melody, lyric)
        return torch.where(chorus.unsqueeze(-1), chorus_emotions, verse)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class EmotionRegressor(nn.Module):
    def __init__(self, melody_dim: int, lyric_dim: int, dim: int, depth: int):
        super().__init__()
        self.cross = HierarchicalCrossProcessing(melody_dim, lyric_dim, dim, depth)
        self.head = nn.Linear(dim, 2)  # valence, arousal

    def forward(
        self,
        melody: torch.Tensor,
        lyric: torch.Tensor,
        chorus: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if chorus is None:
            chorus = torch.zeros(
                melody.shape[:2], dtype=torch.bool, device=melody.device
            )
        emotions = self.cross(melody, lyric, chorus)
        return self.head(masked_mean(emotions, mask))


def build_emotion_regressor(
    config: ProjectConfig, melody_dim: int, lyric_dim: int
) -> EmotionRegressor:
    model = config.model
    return EmotionRegressor(melody_dim, lyric_dim, model.emotion_dim, model.depth)


def _cumsum_softmax(e: torch.Tensor) -> torch.Tensor:
    half = e.shape[-1] // 2
    negative = torch.softmax(e[..., :half], dim=-1).cumsum(dim=-1)
    positive = torch.softmax(e[..., half:], dim=-1).cumsum(dim=-1)
    return torch.cat([negative, positive], dim=-1)

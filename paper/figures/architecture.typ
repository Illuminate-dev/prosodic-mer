#import "@preview/fletcher:0.5.8" as fl: diagram, edge, node

#let arch-box(pos, body, tint: gray) = fl.node(
  pos,
  align(center, body),
  width: 26mm,
  fill: tint.lighten(88%),
  stroke: 0.7pt + tint.darken(15%),
  corner-radius: 3pt,
)
#let arch-opts = (
  spacing: 6pt,
  cell-size: (13mm, 9mm),
  edge-stroke: 0.7pt,
  mark-scale: 60%,
)
#let arch-base = fl.diagram(
  ..arch-opts,
  arch-box((0, 0), [*VGGish*\ 128-d acoustic], tint: blue),
  arch-box((1, 0), [*ALBERT*\ 768-d lyric], tint: blue),
  arch-box((0, 1), [e-LSTM]),
  arch-box((1, 1), [e-LSTM]),
  arch-box((0.5, 2), align(center)[gated fusion\ (`sigmoid`)], tint: green),
  arch-box((0.5, 3), align(center)[shared\ emotion\ $bold(e)_t$], tint: red),
  arch-box((0.5, 4), align(center)[verse / chorus\ hierarchy]),
  arch-box((0.5, 5), [prediction head], tint: yellow),
  arch-box((0.5, 6), [$v, a$], tint: yellow),
  fl.edge((0, 0), (0, 1), "-|>"),
  fl.edge((1, 0), (1, 1), "-|>"),
  fl.edge((0, 1), (0.5, 2), "-|>"),
  fl.edge((1, 1), (0.5, 2), "-|>"),
  fl.edge((0.5, 2), (0.5, 3), "-|>"),
  fl.edge((0.5, 3), (0.5, 4), "-|>"),
  fl.edge((0.5, 4), (0.5, 5), "-|>"),
  fl.edge((0.5, 5), (0.5, 6), "-|>"),
  fl.edge((0.5, 3), (0, 1), "--|>", stroke: (dash: "dashed"), bend: 40deg),
  fl.edge((0.5, 3), (1, 1), "--|>", stroke: (dash: "dashed"), bend: -40deg),
)
#let arch-ext = fl.diagram(
  ..arch-opts,
  arch-box((0, 0), [*VGGish*\ 128-d acoustic], tint: blue),
  arch-box((1, 0), [*ALBERT*\ 768-d lyric], tint: blue),
  arch-box((2, 0), [*prosody-v2*\ 46-d prosodic], tint: purple),
  arch-box((0, 1), [e-LSTM]),
  arch-box((1, 1), [e-LSTM]),
  arch-box((2, 1), [e-LSTM]),
  arch-box((1, 2), align(center)[gated fusion\ (`softmax`)], tint: green),
  arch-box((1, 3), align(center)[shared\ emotion\ $bold(e)_t$], tint: red),
  arch-box((1, 4), align(center)[verse / chorus\ hierarchy]),
  arch-box((1, 5), [prediction head], tint: yellow),
  arch-box((1, 6), [$v, a$], tint: yellow),
  fl.edge((0, 0), (0, 1), "-|>"),
  fl.edge((1, 0), (1, 1), "-|>"),
  fl.edge((2, 0), (2, 1), "-|>"),
  fl.edge((0, 1), (1, 2), "-|>"),
  fl.edge((1, 1), (1, 2), "-|>"),
  fl.edge((2, 1), (1, 2), "-|>"),
  fl.edge((1, 2), (1, 3), "-|>"),
  fl.edge((1, 3), (1, 4), "-|>"),
  fl.edge((1, 4), (1, 5), "-|>"),
  fl.edge((1, 5), (1, 6), "-|>"),
  fl.edge((1, 3), (0, 1), "--|>", stroke: (dash: "dashed"), bend: 40deg),
  fl.edge((1, 3), (2, 1), "--|>", stroke: (dash: "dashed"), bend: -40deg),
  fl.edge((1, 3), (1, 1), "--|>", stroke: (dash: "dashed"), bend: 60deg),
)
#let arch-figure = figure(
  grid(
    columns: (auto, auto),
    gutter: 1.2em,
    align(center)[*(a)* two-stream baseline], align(center)[*(b)* three-stream extension],
    arch-base, arch-ext,
  ),
  caption: [Architecture of the N-stream model. (a) The two-stream baseline combines a VGGish
    acoustic stream and an ALBERT lyric stream. (b) The three-stream extension adds a prosodic
    stream (`prosody-v2` in this example). Each stream is encoded by an emotion-LSTM cell, and the
    candidate emotions are combined by a learned gate. Dashed arrows indicate the shared emotion
    vector $bold(e)_t$ fed back into each cell.],
  placement: top,
  scope: "parent",
)

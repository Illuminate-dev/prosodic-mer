#import "@preview/faithful-acmart:0.2.0": *
#import "figures/architecture.typ": arch-figure

#show: acmart.with(
  format: "sigplan",
  nonacm: true,
  title: "Utilizing Prosodic Analysis for Music Emotion Recognition",
  authors: (
    (
      name: "Henry Beveridge",
      affiliation: (institution: "AP Research", country: "USA"),
    ),
  ),
)


= Introduction

Emotions have long been centered at the core of music experience and creation, often cited as
inspiration for a creative work that later shapes the mood of the listener~@Thompson2001. The
ability to computationally classify the emotions that listeners experience from particular pieces of
music has potential to benefit fields such as music therapy, recommendation systems, and the
development of person-centered healthcare~@Kat21. However, this task is far easier for humans than
computers, motivating the field of music emotion recognition, or MER.

Despite steady progress in MER, current state-of-the-art (SOTA) systems remain well below
human-level performance, particularly when handling vocal music. Existing approaches typically treat
lyrics and audio as separate modalities that are only combined at the sentence level, neglecting the
prosodic detail of how individual words are sung. This raises the central research question of this
study: _can integrating prosodic analysis into a multimodal MER framework improve music emotion
prediction on tasks at varying levels of processing granularity?_ Building on prior evidence that
prosody is a strong cross-cultural carrier of emotion in speech, this study hypothesizes that
integrating prosodic features alongside lyrical and acoustic features within a cross-modal
architecture will yield measurable gains in regression metrics at the word granularity.

= Literature Review

Accurate MER systems depend upon the contribution of multiple components: a representation of
emotion that captures complexity, a set of features extracted from music, and an architecture that
sufficiently links the features to the output.

== Emotional Model

The task of computationally recognizing emotions relies upon the method of reliably representing
representing the complexity of human emotions within a data structure. To develop accurate methods
of MER, an understanding of the various models of emotions is necessary. Most emotion models take on
two distinct types of representation: continuous, or discrete. // maybe incorrect grammar
Those of the former type often represent a particular emotion as a vector in a multidimensional
space, where axes are determined based on theoretical factors in emotion expression. For example,
Russell's circumplex model uses valence#footnote[Valence describes whether an emotion is positive
  (pleasant) or negative (unpleasant).] and arousal#footnote[Arousal describes the energy or
  activation level of an emotion, ranging from calm to excited.] as two axes, while Thayer's Model
bases dimensions on energy-stress and calm-tired ranges #cite(<Russell1980>, <Thayer1990>). Discrete
models, on the other hand, involve the definition of a set of emotions and characteristics. In the
context of music, two discrete models are popular: the Geneva Emotional Music Scale Model, and
Hevner's Emotional model, both of which rely on descriptions of the emotional response to music
#cite(<Posner2005>, <Zentner2008>). Though the aformentioned discrete models have merit in their
application to musical terms, surveys have shown that most MER researchers choose Russell’s Model
due to dataset availability~@Liyanarachchi2025.

== Features
Equally important to the way the output of computer models for MER is represented are the inputs
that are given to the computer model, or the features of the MER model. In this aspect, MER models
vary much more with the type and level of features utilized. Historically, MER features have been
organized into a hierarchy of levels based on the complexity and degree of processing required to
extract~@Goto2024. Low-level features usually are focused on the basic acoustic properties of music,
extracted from short audio segments without context of the greater music piece. Included in this
classification are two main categories of features: spectral features, and temporal features.
Spectral features aim to capture timbre and tonal properties of music by employing methods such as
mel-frequency cepstrum coefficients#footnote[Mel-frequency cepstrum coefficients (MFCCs) are a
  compact numerical summary of an audio segment’s frequency content, designed to mirror how humans
  perceive pitch.], zero-crossing#footnote[The zero-crossing rate is the number of times an audio
  waveform changes sign per unit time, roughly indicating how percussive a sound is.], spectral
centroid#footnote[The spectral centroid is the “center of mass” of a sound’s frequency spectrum, or
  a measure of where most of the signal power in a sound lies. It is often perceived as a measure of
  brightness.], spread#footnote[Spectral spread measures how widely a sound’s energy is distributed
  around the spectral centroid.], roll-off#footnote[Spectral roll-off is the frequency below which a
  specified percentage (commonly 85%) of the total spectral energy lies.], and others #cite(
  <Jeong2016>,
  <Jitendra2020>,
  <Logan2000>,
). These features are commonly extracted using spectrograms, or analysis of the frequencies of audio
over time, and have dominated early MER research due to the efficiency and direct correlation to
audio. Temporal features, on the other hand, capture the rhythm and tempo of a song, along with
other time-related aspects of the track. Temporal features differ in comparison to spectral
features, as they typically focus on how aspects evolve over durations of time. Though this
technique has not been utilized as much as spectral features in the context of MER research,
analyses of time-variation features, such as the consistency of rhythmic patterns, demonstrate a
correlation between evolving temporal features throughout time and the respective emotions elicited
from music~@Yang2023.

In MER research, most models rarely utilize features above the low-level section, especially due to
the complexity that arises when processing mid- and high-level features. Thus, there has been
limited use of mid-level features such as pitch contour#footnote[A pitch contour is the trajectory
  of perceived pitch over time within a melody.], melodic patterns, and prosodic variations, or the
combination of rhythm, pitch, stress, and intonation to create a sense of musical
quality~@Liyanarachchi2025, despite the success in other related music information retrieval tasks,
like cover-song detection~@Salamon2012. The gap in research is particularly notable given that
mid-level features seem to bridge the divide between the low-level features of acoustic description
and high-level semantic understanding, thus potentially being able to assist in a more interpretable
and emotionally relevant representation of musical content.

Historically, lower-level descriptions were hand-crafted to maximize information extraction,
resulting in greater levels of accuraacy. However, recent usage of learned embeddings, such as the
VGGish convolutional network~@Hershey2016 pretrained on AudioSet log-mel spectrograms producing an
128-dimensional embedding per audio segment, has begun more comprehensively represent audio streams.

== Techniques

The evolution of MER systems reflects the increasing sentiment that emotional content in music
cannot simply be encapsulated through audio analysis alone. Early approaches analyzed the audio
features of timbre, tempo, and spectral characteristics separately from the lyrical content
processed through natural language processing (NLP) techniques. However, this separation, albeit
much simpler, failed to account for the relationship and interconnection between what is sung and
how it is sung. The semantic content of the lyrics does influence the emotional perception, but the
variation with which words are delivered in vocal performance adds many layers of nuance that
separation could not capture. As in speech, the way words are said matters as much as which words
that are said. Recent work has begun to address this shortcoming via the incorporation of lyrical,
auditory, and even visual data into multimodal frameworks. However, many of these approaches still
exhibit the same limited interconnection seen in prior MER systems #cite(<Wang2024>, <Yang2023>).
This is mainly because each modality is typically processed through individual feature extraction
pipelines and neural network pathways, resulting in representations that are only concatenated at
later stages of the model. Late concatenation typically prohibits models from learning how details
such as a sudden vocal intensity spike interacts with a lyrically significant word, primarily
because each modality contributes its own prediction, causing adjustments in either modality to be
lost in the final prediction head.

== Current SOTA

Current state-of-the-art models have varying degrees of cross-modal integration, or the interfacing
between each type of feature through the network pathways. The COSMIC framework by Yang et al.
introduced emotion-long-short-term-memory (emotion-LSTM)#footnote[Long short-term memory (LSTM) is a
  type of recurrent neural network designed to retain information across long sequences, making it
  well-suited for ordered data such as text or audio.] cells that allow for cross-interaction
between lyric and melody representations of music tracks, yet also included an implementation of a
structural analysis approach that processes the verse and chorus sections separately~@Yang2023. The
structural awareness included recognizes the idea that different sections of a song can serve
different functions, with sections building on each other to develop narrative or thematic content.
The emotion-LSTM cells utilized in the COSMIC framework enable the bidirectional information flow
between the lyrical and acoustic features typically seen in modern natural language processing
architectures. This allows the model to learn how the different modalities are developed throughout
the song to produce emotional impact. Expanding on this, the MMD-MII model proposed by Wang et al.
adds on a multilayered analysis approach with a dedicated cross-processing module, representing
another step in the direction of more multimodal integration~@Wang2024. The MMD-MII framework
similarly processes the chorus separately from the verses, again acknowledging the intricacies in
music structure. The dedicated cross-processing module allows greater interaction between the
lyrical content and musical content of the audio input for greater cross-modality. However, these
models are both still limited by the processing of vocal content at the coarse unit of a phrase,
with each lyrical line treated as a unified semantic and acoustic unit without examining whether the
variation within the unit carries additional emotional signal. An analysis at the word-level could
potentially be revealing regarding further data hidden between overall phrases.

This processing and analysis at the sentence level instead of a finer word-level overlooks a crucial
part of emotional expression in music: prosody. Prosody typically encompasses the
"suprasegmental"#footnote[Suprasegmental features are properties of speech that span multiple
  individual sounds.] aspects of speech and singing, including pitch variation,
intensity#footnote[Intensity refers to the energy of a sound, perceived as loudness.], duration,
rhythm, stress patterns, and intonation~@Wagner2010. These features operate at a finer granular
level than the typical sentence-level analysis that occurs in the current state-of-the-art models,
often varying significantly from word to word or syllable to syllable within a single phrase. In
spoken language, prosody is shown to be a carrier of information about emotion, attitude, emphasis,
and pragmatic meaning, as the difference between sarcastic and sincere statements lies in prosodic
delivery of even individual words rather than word choice. In music, prosody can take on even
greater importance, as singers commonly extend the constrained prosodic dictionary used for speech
to include manipulations of pitch, timing, and dynamics. Skilled vocalists may stretch words for
dramatic effect, change pitch for emotionally significant emphasis, or even apply vibrato to add a
sense of warmth or intensity to sustained notes. These prosodic choices in how notes are sung
represent intentional artistic decisions aimed at conveying certain emotions, yet are largely
unexamined by current MER systems.

Unfortunately, despite prosody's well-documented utility as a carrier of emotion, it has remained
largely absent from MER systems. Past research has demonstrated that prosodic features are
intrinsically linked with emotional responses, with this link holding even across cultural
boundaries~@Balkwill99. This consistency between cultures implies that the prosodic-emotional
correlation may reflect more universal aspects of human communication, potentially giving greater
value to an application of prosodic analysis to MER systems. MER systems including prosodic analysis
could work across diverse musical traditions and languages. Moreover, musical aspects of language,
especially the prosodic elements, have been shown to influence cognition similarly in both speech
and music, indicating shared processing mechanisms between linguistic and musical
understanding~@PastuszekLipinska2025. Given the success of emotion recognition utilizing word-level
prosodic analysis in speech information retrieval~@Vicsi2010, integrating the same mechanisms in MER
may bring greater accuracy.

== Gap

After an observation of the current state-of-the-art models, a gap clearly emerges, stemming from
three different areas of lack of research. First, the concept of cross-modal interaction is quite
novel, and thus has not been researched deeply yet. The limited research on cross-modality in
state-of-the-art MER suggests that the technique has potential to be useful, but models like MMD-MII
tend to process the cross-interaction at a unit of a lyrical line, leaving finer granularity
untested. This study therefore evaluates the framework at both a pair (sentence from lyrics paired
with associated acoustic representation) granularity and a word granularity. Second, the use of
prosodic analysis is extremely limited, as many studies focus on low-level features rather than
mid-level features such as prosody. Third, studies have found that the use of more features is
strongly linked to higher performance, further motivating the research of an extra layer of
features~@Juslin2001. These gaps elicit the following question: _can integrating prosodic analysis
into a multimodal MER framework improve music emotion prediction on tasks at varying levels of
processing granularity?_

This study aims to bridge all three gaps by introducing a cross-modal framework that integrates
prosodic feature extraction with the state-of-the-art architecture of emotion-LSTM chains, thus
adding a previously underused feature type and testing whether finer granularity improves overall
performance.

= Methodology

The goal of this study is to gauge the effect of including various prosodic features the impact of
different granularity on performance. The methodology used is a modification of the methodology and
architecture developed and enumerated in the MMD-MII paper by Wang et al. The overall methodology
involves implementing the architecture described in the aforementioned paper, adding a prosodic
input stream, and running an ablation over feature set and granularity to identify the impact of
each change. The remainder of this section reviews and defends, in chronological order, the steps
taken to train and evaluate this model.

== Dataset

One dataset was chosen for use with the training of this model: Dataset for Emotion Analysis in
Music (DEAM). This dataset contains 1,802 audio files, with a mixture of excerpts and full-length
songs. DEAM is specifically made for use with MER tasks, as it contains annotations for both
per-second valence and arousal values, as well as overall song-level valence and arousal. All
emotion annotations are in the range $[-1, 1]$, where negative values indicate negative valence or
low arousal and positive values indicate the converse. Of the 1,802 files, 801 contained lyrical
vocals. Of these, 207 were filtered out for having no annotated unit inside the lyrical annotation
window. For this study, per-second values were utilized rather than song-level values in order to
enable finer unit-level alignment with the use of LSTM modules.

However, there is a discrepancy between the defined annotations provided in DEAM and the output of
the MMD-MII architecture. While the dataset contains valence-arousal annotations, the MMD-MII
architecture outputs as classification, despite claiming to have used the DEAM dataset in their
training process. This means that there is an undocumented conversion between valence-arousal and
the classes used in the paper that this research does not have access to. As a result, a slight
modification to the architecture was made: the implemented architecture contains an output head of
valence-arousal instead of classification. This should not impact the validity of results, as
comparitive and ablation testing will be able to identify the significance of a relative change in
correlation of the trained model.

Each experiment uses a five-fold cross-validation at the song level. Within each fold, the dataset
is split with 10% of tracks reserved for validation, 10% for testing, and the remaining 80% used for
training. After potentially significant findings, the main pair-level comparison repeats the five-fold 
split under five different partition seeds, for a total of 25 paired folds for a verification of 
significance. The other versions of ablation use a single partition seed, for a total of five folds. 
Splitting is performed at the song level to prevent data leakage#footnote[Data leakage occurs when 
information from outside the training set influences the model. Splitting at the song level prevents 
excerpts from the same song from appearing in both training and test sets, which would inflate apparent
  performance.]. The validation split is used only to monitor performance between epochs and to
trigger early stopping in the case of insignificant improvement following 10 continuous epochs.

== Preprocessing Pipeline

Prior to training, preprocessing was performed on all of the data, turning the raw track audio into
data usable for machine learning. First, all track audio underwent vocal source
separation#footnote[Source separation is the process of isolating individual components (such as
  vocals) from a mixed audio recording.] via the same Spleeter model used in the MMD-MII
paper~@Hennequin2020. This is necessary for adequate accuracy when extracting prosodic features from
the vocal track, which requires the raw vocal audio.

Next, lyrics were automatically transcribed using OpenAI Whisper (a locally ran whisper-small
model). Though initially developed for non-musical vocal transcription, Whisper has been found to
achieve highly-accurate performance on automatic lyrics transcription when combined with source
separation to isolate vocals~@Syed2025. Whisper also has the advantage of providing the word-level
alignment needed for training without fine-tuning. After extracting word-level timestamps, the
lyrics underwent minimal postprocessing aimed to remove low confidence words#footnote[Words flagged
  by the transcription model as having low certainty.] and identify verse/chorus structure. In
addition, tracks with no vocals were filtered out, as previously mentioned in the dataset section,
to align with this study's (and the MMD-MII) focus on music with vocals.

After transcription, the final stage in preprocessing was to align the per-second emotion labels
from DEAM with the word-level timestamps from the Whisper transcription. This was achieved through
interpolation of the emotion labels to the word-level granularity according to the following
equation.

For each word $w_i$ with time boundaries $[t_"start"^i, t_"end"^i]$:

$ v_i = 1/(t_"end"^i - t_"start"^i) integral_(t_"start"^i)^(t_"end"^i) V(t) dif t $

$ a_i = 1/(t_"end"^i - t_"start"^i) integral_(t_"start"^i)^(t_"end"^i) A(t) dif t $

Where $V(t)$ and $A(t)$ are the continuous valence and arousal annotations respectively. As labels
fall in per-second intervals, this was implemented as the weighted average of these labels during
each word's duration. The resulting word-level labels are resampled to the model's axis when the
pair granularity is used.

== Feature extraction

After preprocessing, feature extraction was broken up into three types: acoustic features, prosodic
features, and lyrical features. Acoustic features were extracted from the full audio track (vocals
and accompaniment) using VGGish~@Hershey2016. When run through the VGGish network, each audio
segment is transformed into a 128-dimensional embedding, with each embedding overlapping a unit's
time span mean-pooled to give one 128-dimensional acoustic vector per unit.

Prosodic features were extracted using Parselmouth, a Python interface to Praat, which is the widely
used tool in both phonetic and prosodic research. Additionally, librosa, a library for interfacing
with audio, was used for extracting onset, chroma, key, and chord analysis. After continuous
development and refinement, three versioned feature sets were implemented and ablated. `prosody-v1`
emerged initially as a static descriptor set. Soon, `prosody-v2` followed as a new set of features
capturing the dynamics and tonality track. Finally, `prosody-v3` aimed to take the dynamics and
tonality features of `prosody-v2` and make them relative to the track. All are extracted per word
and resampled to the model's axis.

#figure(
  tabular(
    columns: 4,
    toprule(),
    [Variant],
    [Dims],
    [Source],
    [Purpose],
    midrule(),
    [`prosody-v1`],
    [25],
    [vocals],
    [static Praat descriptors],
    [`prosody-v2`],
    [46],
    [vocals + mix],
    [performance + tonality],
    [`prosody-v2` sidecar],
    [29],
    [track],
    [key, tempo, rubato],
    [`prosody-v3`],
    [26],
    [vocals + mix],
    [track-relative performance],
    bottomrule(),
  ),
  caption: [Dimension count and contents of the versioned prosodic feature sets.],
) <tab:prosody-versions>

Extraction used an F0#footnote[F0 (fundamental frequency) is the pitch of a sound.] floor of 75 Hz,
F0 ceiling of 600 Hz, and time step of 10 ms. `prosody-v1` comprises F0 mean, F0 standard deviation,
F0 range, F0 slope, F0 velocity, voicing ratio#footnote[The voicing ratio is the fraction of
  analysis frames in which pitched speech is detected.], jitter#footnote[Jitter is the
  cycle-to-cycle variation in F0, reflecting irregularity in vibration.] (local and absolute),
shimmer#footnote[
  Shimmer is the cycle-to-cycle variation in amplitude of the vocal signal.] (local and dB),
HNR#footnote[Harmonics-to-noise ratio (HNR) measures the proportion of periodic energy to noise in a
  voice, indicating voice quality.] mean, HNR standard deviation, F1#footnote[F1, F2, and F3 are the
  first three formants, or frequencies of the vocal tract that make up vowel identity and timbre.]
mean, F1 standard deviation, F2 mean, F2 standard deviation, F3 mean, F3 standard deviation, F1
bandwidth, intensity mean, intensity standard deviation, intensity range, intensity slope, word
duration, and syllable rate.

`prosody-v2` adds performance features on the vocal stem (F0 relative to the track median in
semitones, IQR, range, slope, rising/falling fraction, a six-bin pitch-interval histogram, interval
mean/std, up-down and down-up transition rates, vibrato rate/extent, onset rate, beat residual, and
loudness variability) and tonality features on the mix (key-relative chroma, chord root, chord mode,
chord match, tension, tension change, chord change).

`prosody-v3` adds track-relative features: F0 median, IQR, range and slope; voice direction and
mobility; F0 delta versus the previous word; speech rate; duration; gap to the previous word; beat
residual; onset rate; relative loudness; loudness range and change magnitude; loudness delta;
relative HNR; voiced fraction; jitter; chord change; chord match; tension and tension change;
phrase-context F0 and loudness means; and position in the phrase.

For lyrical features, lyrics were encoded using ALBERT (`albert-base-v2`)~@Lan2019, a lightweight
transformer model in the BERT family. ALBERT was used with task fine-tuning, and the embeddings of a
unit's tokens were mean-pooled into a single 768-dimensional contextualized lyric vector per unit.
This is consistent with the described implementation of the MMD-MII paper, thus maintaining a
comparative basis with a state-of-the-art system.

Lastly, the prosodic features underwent z-score standardization before model input, with the
following equation:

$ x_"normalized" = (x - mu_"train")/(sigma_"train" + epsilon) $

where $mu_"train"$ and $sigma_"train"$ are computed on the training set only to prevent data leakage
and $epsilon = 10^(-8)$ to account for near-zero variance features.

== Model Architecture

#arch-figure <fig:architecture_diagram>

The overall framework is an N-stream model, where N can be either two or three, referring to lyrical
and audio modalities or lyrical, audio, and prosodic modalities. Each selected modality is encoded
by its own stream, the streams interact through emotion-LSTM cells that share a single emotion
vector, and the resulting candidate emotions are combined by a learned gate. While the baseline uses
two streams (VGGish acoustic and ALBERT lyric), the experimental groups add a third prosodic stream,
with the aforementioned variable feature sets. All streams operate on the same axis (pair or word),
and each modality's native unit is resampled to that axis.

The emotion-LSTM cell extends the standard LSTM to enable cross-modal interaction through the shared
emotion vector, following the MMD-MII framework. At each time step $t$, a stream receives its own
feature vector, the previous hidden and cell state, and the shared previous emotion vector
$bold(e)_(t-1) in bb(R)^128$, and produces a candidate emotion vector $bold(e)_t^k$. The native
feature dimensions differ by modality: VGGish yields $bold(a)_t in bb(R)^128$, ALBERT yields
$bold(l)_t in bb(R)^768$, and the prosodic stream yields $bold(p)_t in bb(R)^25$ (`prosody-v1`),
$bold(p)_t in bb(R)^46$ (`prosody-v2`), or $bold(p)_t in bb(R)^26$ (`prosody-v3`).

The candidate emotions are combined by a learned gate. For two streams the gate is
$g = sigma(W [bold(e)_t^"m"; bold(e)_t^"l"])$, and the fused emotion is

$ bold(e)_t = g dot.o bold(e)_t^"m" + (1 - g) dot.o bold(e)_t^"l" $

For three streams a softmax is first performed over the streams, giving per-stream weights, and the
fused emotion is
$ bold(e)_t = sum_(k=1)^N w_k bold(e)_t^k $
with
$ w = "softmax"(W [bold(e)_t^1; ...; bold(e)_t^N]) $.

Following the MMD-MII methodology, song structure is explicitly modeled as verse and chorus
structures via automatic detection, and the verse and chorus branches are combined with a learned
selection (`torch.where`). The final prediction head outputs continuous valence and arousal
predictions in the range [-1, 1], consistent with the DEAM annotations.

== Training

Models were trained on the 594 usable DEAM tracks with mean squared error (MSE)#footnote[Mean
  squared error (MSE) is the standard loss function for regression, averaging the squared
  differences between predictions and ground-truth values.] as the loss, the AdamW
optimizer#footnote[The optimizer is the algorithm that adjusts the model's weights to minimize the
  loss; AdamW is the most frequently used optimizer.], a learning rate of $1 times 10^(-4)$, and a
batch size of 32. Training ran for at most 100 epochs#footnote[An epoch is one full pass of the
  training algorithm through the entire training dataset.] with early stopping on the validation
loss and a patience of 10 epochs.

== Evaluation

For the regression task, mean squared error (MSE) and its square root (RMSE) measure the average
squared deviation between predicted and true values, with smaller values indicating more accurate
predictions. The coefficient of determination ($R^2$) measures the proportion of variance in the
ground-truth labels that the model explains, where $1$ corresponds to perfect prediction and $0$
corresponds to predicting only the mean. The Pearson correlation coefficient (PCC) measures the
linear association between predictions and ground truth on a scale from $-1$ to $1$, where $1$
indicates perfect linear agreement. The concordance correlation coefficient (CCC) is similar to PCC
but additionally penalizes systematic offsets in scale or mean, making it a stricter measure of how
closely predictions match ground-truth values.

The ablation compares one baseline against three versioned prosodic feature sets. The main 
comparison is the baseline (VGGish + ALBERT) against the baseline plus `prosody-v2`, both at pair
granularity, evaluated over 25 paired folds (five partition seeds $times$ five folds). The remaining
points of comparison are exploratory and use a single seed over five folds: the baseline and `prosody-v2`
at word granularity, and `prosody-v3` at both pair and word granularity. A further single-fold block
ablation removes one `prosody-v2` feature block at a time. Finally, `prosody-v1` was trained under 
track supervision rather than the sentence supervision used by the headline arms, so its numbers are 
not directly comparable.

== Reproducibility Notes

All random operations were seeded. The default training seed is 42. Each of the five folds within
the main comparison uses a training seed of `partition_seed + fold`, and the five partition
seeds are 42, 1042, 2042, 3042, and 4042, yielding 25 paired folds. The alternative versions use only
partition seed 42. This maintains diversity across training environments while keeping the results
reproducible through deterministic training.

= Results

#figure(
  tabular(
    columns: 3,
    toprule(),
    [Metric],
    [Base],
    [Base + `prosody-v2`],
    midrule(),
    [RMSE],
    [0.1984],
    [0.1915],
    [$R^2$],
    [0.3018],
    [0.3465],
    [PCC],
    [0.5413],
    [0.5863],
    [CCC],
    [0.4396],
    [0.5076],
    bottomrule(),
  ),
  caption: [Pair-level regression metrics over 25 paired folds.],
) <tab:pairs-absolute>

The results indicate that a treatment of `prosody-v2` feature set improves every reported metric.
The paired differences and their 95% confidence intervals over the same 25 folds are shown in
Table~@tab:pair-delta.

#figure(
  tabular(
    columns: 3,
    toprule(),
    [Metric],
    [Delta (`prosody-v2` - base)],
    [95% CI],
    midrule(),
    [$R^2$],
    [+0.0447],
    [[+0.0213, +0.0681]],
    [CCC],
    [+0.0680],
    [[+0.0473, +0.0887]],
    [RMSE],
    [-0.0069],
    [[-0.0103, -0.0034]],
    bottomrule(),
  ),
  caption: [Pair-level treatment effect over 25 paired folds.],
) <tab:pair-delta>

A paired t-test gives $p = 0.0006$, indicating that these results are statistically significant. Per
output, valence improves by $+0.0514$ in $R^2$ (CI $[+0.0272, +0.0756]$) and arousal by $+0.0381$
(CI $[+0.0097, +0.0665]$), with both improving in 22 of 25 folds.

The same treatment at word granularity, evaluated over five exploratory folds under a single
partition seed, does not demonstrate a similar increase in performance. In fact, the results obtained
were a $+0.0213$ in $R^2$, with only 3 of 5 folds improving. These results indicate that word-level 
processing does not increase performance, at least by a statistically significant amount.

The single-fold block ablation (Figure~@fig:ablation_overview, panel c) removes one `prosody-v2`
feature block at a time at word granularity. Every removal lowers $R^2$ relative to the full set
(0.478), with timing (-0.124), f0 (-0.110), tonality (-0.073), and loudness (-0.045) contributing
most. Because this ablation uses one fold it is exploratory and thus not indicative of a guaranteed
improvement, but provides a basis for future research. As a control, `prosody-v1`
does not improve over the baseline under track supervision (0.340 vs. 0.369 for $R^2$), consistent
with the gain coming from the feature treatment rather than from adding a third stream.

#figure(
  tabular(
    columns: 2,
    toprule(),
    [Setting],
    [Value],
    midrule(),
    [Labelled tracks],
    [801],
    [Usable tracks],
    [594],
    [Folds],
    [5],
    [Validation fraction],
    [0.1],
    [Test fraction],
    [0.1],
    [Supervision],
    [sentence],
    [Partition seeds],
    [5 (main); 1 (exploratory)],
    [Paired folds per arm],
    [25 (main); 5 (exploratory)],
    bottomrule(),
  ),
  caption: [Experiment setup and dataset counts.], 
) <tab:cv>

#figure(
  image("figures/ablation_overview.png", width: 100%),
  caption: [Ablation overview. (a) $R^2$ charted at pair and word granularity respectively for each set of features over
    five folds. Each dot is an individual fold's measurement. (b) Per-fold $R^2$ difference
    between `prosody-v2` and the baseline at pair-level granularity across the 25 paired folds. The
    dashed line and band are the mean and 95% confidence interval respectively. (c) Single-fold drop-one block
    ablation of `prosody-v2` features. (d) `Prosody-v1` control under
    track supervision.],
  placement: top,
  scope: "parent",
) <fig:ablation_overview>

#figure(
  image("figures/valence_predictions.png", width: 100%),
  caption: [Valence. (a) Predicted versus ground-truth valence plotted for `prosody-v2` vs. base.
  The dashed line is the identity. (b) Prediction-error distributions for the two runs.],
  placement: top,
) <fig:valence>

#figure(
  image("figures/arousal_predictions.png", width: 100%),
  caption: [Arousal. (a) Predicted versus ground-truth arousal plotted for `prosody-v2` vs. base.
  The dashed line is the identity. (b) Prediction-error distributions for the two runs.],
  placement: top,
) <fig:arousal>

= Discussion

The central result of this study is a statistically supported improvement, but the improvement itself is generally modest.
While this does achieve state-of-the-art performance, it does so with minor improvement to accuracy and
over a different output than the MMD-MII model was designed for.
Adding `prosody-v2` at pair-level granularity specifically only raises $R^2$ by $+0.0447$ over the
two-stream baseline, supported with a 95% confidence interval that excludes zero. Associated with this are improvements to CCC
and RMSE as well. This effect is also supported by the general observation that each run generally leads to 
increased improvement in at least one of valence or arousal accuracy. The control of `prosody-v1` shows that the
gain is not explained by the mere addition of more parameters, as it uses the same three-stream architecture and similar increases in parameter count 
but does not improve over the baseline under track supervision. 

A substantial performance divergence remains between the two prediction subtasks. It is wholly evident that arousal is
predicted much more accurately than valence, as arousal $R^2$ is roughly 0.5, while valence $R^2$ is
roughly 0.13-0.21. Prosody improves both outputs, but it improves valence more ($+0.0514$ in $R^2$)
than arousal ($+0.0381$). This asymmetry is consistent with prior findings that valence is 
considerably more difficult to predict than arousal, because arousal tends to map more reliably 
onto lower-level features while valence is more reliant on the higher-level features of harmonicity, mode, 
lyrical content, and subtle timbral nuances#cite(<Liyanarachchi2025>, <Yang2012>).

The pair-level gain here is an interesting piece of evidence to note. This shows that the prosodic feature treatment can
contribute non-redundant signal to the cross-modal interaction on a pair-level. However, the lack of a significant increase in word-level analysis
indicates that moving the interaction down to individual words may not be as useful to the task of music emotion recognition. 
Taken together, the two results suggest that the benefit comes from the coarser 
`prosody-v2` feature representations rather than the finer units as initially hypothesized. This overall lends credibility to the claim that the
sentence-level unit of the existing architecture is already adequate for this task.

The prediction figures (Figures~@fig:valence and~@fig:arousal) show the same
pattern at 25-fold scale. Arousal predictions tend to line up to the line of identity closer than valence,
with errors centered near zero and a standard deviation of about 0.19, while valence predictions are
compressed toward zero. This means that tracks with a valence on the extreme ends of the spectrum (high or low) are typically under- and over-predicted
respectively, and the error distribution of valence is generally wide relative compared to that of the arousal.
However, note that adding `prosody-v2` features shifts the error distributions closer to zero, with the
larger relative improvement on valence.

== Limitations

Several limitations qualify these results. First, the study uses a single dataset (DEAM) and a
single architecture family (the N-stream emotion-LSTM), so the effect may not transfer to other
corpora or models. Second, the effect itself is not very large, as the $R^2$ increase of 0.045 is
indeed statistically supported to be significant, yet is small in absolute terms. Arousal itself still remains harder to predict than valence.
Third, only the main pair-level comparison uses the full set of 25 paired folds while the
word-level and `prosody-v3` arms use a total of five folds. Thus, there may be a statistically significant 
different between values, but the scale of performed comparison did not identify one.
Finally, the dataset is small by the modern size of typical machine learning datasets, with only 594 usable tracks, which constrains
generalization.

== Implications

The results carry implications for MER development and for applications such as music therapy and
recommendation. The pair-level gain shows that a richer prosodic feature treatment contributes a
non-redundant signal to a cross-modal framework, and that the signal can be captured at the
sentence-level unit already used by existing architectures rather than requiring word-level
alignment. At the same time, the modest effect size, despite the statistical significance, indicates 
that prosodic features do not suffice for the innovation of additional feature types.
In general, this improvement should be described as a step forward rather than a full solution; 
human performance of valence-arousal is still miles ahead of MER systems.

In the real world, the arousal-side accuracy observed here means that applications centered on
detecting the energy or activation level of a track, such as music therapy session design or
mood-aware recommendation, stand to benefit most. Applications that require high valence accuracy,
such as automated mood tagging or affect-sensitive content moderation, will require further work,
given the model's weaker valence performance.

= Conclusion

In this study, an N-stream framework for multimodal music emotion recognition was
introduced, integrating prosodic analysis of vocals alongside existing streams of lyrical and
acoustic analysis. The effectiveness of the approach was verified through valence-arousal regression
under a five-fold cross-validation. Adding the `prosody-v2` treatment at pair granularity produced a
modest but statistically supported improvement over the baseline of the MMD-MII architecture across 25 paired folds.
At the same time, treatment at word granularity produced no reliable gain, indicating that finer granularity may not necessarily lead to improvement.
Despite the limitations of dataset size, this represents a meaningful step forward in the domain
of music emotion recognition.

= Appendix

== Evaluation Metric Definitions <app:metrics>

The formal definitions of each metric referenced in Section 3.6 are provided below. In each
equation, $y_i$ denotes a ground-truth value, $hat(y)_i$ a predicted value, $bar(y)$ and
$bar(hat(y))$ the corresponding means, $sigma_y$ and $sigma_(hat(y))$ the corresponding standard
deviations, and $N$ the number of samples.

$ "MSE" = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2 $

$ "RMSE" = sqrt("MSE") $

$ R^2 = 1 - (sum_(i=1)^N (y_i - hat(y)_i)^2)/(sum_(i=1)^N (y_i - bar(y))^2) $

$
  r = (sum_(i=1)^N (y_i - bar(y))(hat(y)_i - bar(hat(y))))/(sqrt(sum_(i=1)^N (y_i - bar(y))^2) sqrt(sum_(i=1)^N (hat(y)_i - bar(hat(y)))^2))
$

$ rho_c = (2 r sigma_y sigma_(hat(y)))/(sigma_y^2 + sigma_(hat(y))^2 + (mu_y - mu_(hat(y)))^2) $

#bibliography("bibliography.bib")

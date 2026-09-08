# Utilizing Prosodic Analysis for Music Emotion Recognition

This code is based on the implementation loosely described in the MMD-MII paper.
Where there is ambiguity in the paper, decisions had to be made. The main
decisions made are as follows:

1. The paper used an output of four emotion classes, but it does not specify the
   conversion from valence/arousal to these classes, despite using the same
   dataset of DEAM. As a result, the model uses a regression head to predict
   valence and arousal directly. Because of the direct comparison between the
   MMD-MII and the proposed architecture, the results should still be valid.

2. The paper used manual annotation as to where verses and choruses lie, but due
   to time constraints this implementation detects structure automatically.

3. Optimization parameters were not specified, so the following were used:
   AdamW, learning rate 1e-4, batch size 32, early stopping with patience of 10.

This code innovates upon the MMD-MII architecture by tracking direct performance
on valence-arousal aligned data, primarily because of a lack of access to data
used in training the MMD-MII model. Additionally, the model in this code adds a
variety of prosodic features. These are separated into different versions of
prosody architecture, versioned 1 through 3. They include the following features:

**`prosody-v1` (25 dims)**

- F0: mean, standard deviation, minimum, maximum, range, slope
- Intensity: mean, standard deviation, minimum, maximum
- Formants: F1-F3 mean, F1-F2 standard deviation
- Jitter: local, RAP, PPQ5
- Shimmer: local, APQ3, APQ5
- Harmonics-to-noise ratio
- Duration: total, voiced, voiced fraction

**`prosody-v2` (46 dims, plus 29 per track)**

- *Performance group*: F0 relative to the track median in semitones, with
  statistics; pitch-interval histogram and mean/standard deviation; up-to-down
  and down-to-up transition rates; vibrato rate and extent; onset rate; beat
  residual; loudness relative to the track mean; loudness standard deviation,
  range, change count and change magnitude; attack and decay slope; gap fraction.
- *Tonality group*: key-relative chroma, chord root, chord mode, chord match,
  tonal tension and its change, chord change.
- *Extra group*: key one-hot, mode, key confidence, tempo, rubato, tempo
  variance.

**`prosody-v3` (26 dims)**

- F0: track-relative median, interquartile range, range, slope; voice direction
  and mobility; F0 delta against the previous word
- Timing: speech rate, duration, gap to the previous word, beat residual, onset
  rate
- Loudness: relative loudness, loudness range, change magnitude, loudness delta
- Voice quality: relative harmonics-to-noise ratio, voiced fraction, jitter
- Tonality: chord change, chord match, tonal tension and its change
- Context: phrase-context F0 and loudness means, position within the phrase

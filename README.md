# 🎧 Auto-DJ: Smart Playlist Flow Optimizer

Reorders any playlist for the smoothest possible listening experience by analyzing tempo, musical key, and energy — and applying the same harmonic mixing principles professional DJs use, automatically.

## The problem

User-created playlists are usually ordered randomly. A slow 70 BPM track can sit right next to a 160 BPM one, or two songs in clashing musical keys can be played back to back. DJs avoid this with **harmonic mixing** — deliberately sequencing tracks so each transition feels natural. Regular listeners have no tool to do this for their own playlists.

## What it does

Upload a set of audio files, and the system:

1. Extracts BPM, musical key, and an energy score from each track
2. Scores how well every pair of tracks would transition into one another
3. Reorders the whole playlist to maximize overall flow
4. Lets you preview any transition — or the entire reordered playlist — as a crossfaded audio clip, right in the browser

| | |
|---|---|
| **Original playlist score** | 53.0 / 100 |
| **Optimized playlist score** | 63.6 / 100 |
| **Improvement** | +10.5 |

*(scores from a real 4-track test run; results vary by playlist — see [Results](#results) below for a larger run)*

## Dataset

This project was built and tested using the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (1,000 30-second clips across 10 genres). The raw audio files aren't included in this repo due to size (~1.2GB) — download them from Kaggle and place them at `data/genres_original/` to reproduce the results above. The extracted features (`data/track_analysis_results.csv`) are included, so you can explore the results without re-running the audio analysis.

## How it works

### 1. Audio analysis

Each track is analyzed with [librosa](https://librosa.org/) to extract:

- **BPM** — tempo, via beat tracking
- **Musical key** — detected by correlating chroma features against the Krumhansl-Schmuckler major/minor key profiles
- **Camelot code** — the 12-key DJ notation (e.g. `8B`, `5A`) used to determine harmonic compatibility
- **Energy score** — a weighted combination of RMS loudness, spectral centroid, spectral rolloff, and onset strength, normalized to 0–1

### 2. Compatibility scoring

Every pair of tracks is scored 0–100 based on three weighted factors:

- **BPM compatibility (45%)** — tempo is treated as a near-baseline requirement; large jumps are penalized heavily, since a tempo mismatch is the most immediately jarring kind of bad transition
- **Camelot key compatibility (35%)** — based on the DJ Camelot Wheel: identical keys score 100, one step away on the wheel scores 75, relative major/minor scores 50, anything further scores 0
- **Energy flow (20%)** — small energy steps between consecutive tracks are rewarded; large spikes or drops are penalized

### 3. Reordering algorithm

Finding the perfect ordering of *N* songs is structurally the same problem as the Travelling Salesman Problem — NP-hard, with no fast exact solution for large playlists. This project uses a practical two-stage approach instead:

1. **Greedy nearest-neighbor with lookahead** — starting from the lowest-energy track, the algorithm repeatedly picks the next track with the best transition score, looking one step further ahead to avoid getting stuck choosing tracks that lead into dead ends
2. **2-opt local search refinement** — after the greedy pass, the algorithm tries reversing segments of the playlist to see if that improves the total flow score, repeating until no single swap helps anymore

This combination consistently outperforms greedy alone — see [Results](#results).

### 4. Transition previews

For any consecutive pair, the system extracts the last few seconds of the first track and the first few seconds of the second, and **crossfades** them together (linear fade-out on one, fade-in on the other, overlapping) so the preview sounds like an actual DJ transition rather than an abrupt cut. The same logic stitches the entire optimized playlist into one continuous mix.

## Results

On a 50-track test set spanning 5 genres (classical, disco, blues, rock, hiphop):

| Stage | Average pairwise compatibility score |
|---|---|
| Random order (as uploaded) | ~45–55 |
| Greedy reorder | 72.7 |
| Greedy + 2-opt refinement | 76.6 |

2-opt converged in 4 iterations on this dataset, completing in under 2 seconds.

## Deployed Link
[Auto-DJ] (https://huggingface.co/spaces/Alisha090706/auto-dj-playlist-optimizer)

## Known limitations

This project optimizes for the two factors DJs have traditionally relied on — tempo and harmonic key — but that isn't the whole story:

- **Timbre and genre texture aren't captured.** Two tracks can score highly compatible (matching BPM and key) while still sounding jarring back-to-back if their production styles differ a lot — for example, a vocal-heavy hip-hop track into a guitar-driven rock track. The Camelot Wheel was designed by and for DJs mixing largely instrumental, texturally similar dance music; it doesn't generalize perfectly across wildly different genres. A natural next step would be adding MFCC-based timbre similarity as a fourth scoring dimension.
- **BPM detection can double or halve on complex audio.** librosa's beat tracker occasionally misreads tempo on orchestral or rhythmically ambiguous tracks. A heuristic correction is applied for clearly unrealistic values, but it's not foolproof.
- **2-opt is O(n³) per pass**, which is fine for tens of tracks but would need optimization (e.g. restricting swaps to nearby positions) to scale to playlists of hundreds or thousands of tracks.
- **No streaming integration.** The system works on local audio files only. Spotify's API does not allow downloading playable audio, so a real integration would be limited to pulling track metadata and Spotify's own audio-feature data for reordering purposes — without in-browser playback or crossfade previews for those tracks.

## Tech stack

`Python` · `librosa` · `numpy` · `pandas` · `scikit-learn` · `Gradio` · `soundfile`

## Project structure
auto_dj/

├── app.py                      # Gradio UI

├── src/

│   └── analyzer.py              # core analysis, scoring, and reordering logic

├── notebooks/

│   ├── 01_track_analyzer.ipynb  # single-track feature extraction

│   ├── 02_multi_track.ipynb     # batch analysis + before/after charts

│   ├── 03_compatibility.ipynb   # compatibility scoring + reordering

│   └── 04_transitions.ipynb     # crossfaded transition clip generation

├── data/

│   └── track_analysis_results.csv

└── requirements.txt

## Running it locally

\`\`\`bash
pip install librosa numpy pandas scikit-learn gradio soundfile
python app.py
\`\`\`

Then open the local URL Gradio prints in your terminal, upload a few audio files, and click **Analyze & Reorder**.

## Future improvements

- MFCC-based timbre similarity as a fourth scoring dimension, to catch the genre-texture mismatches the current system misses
- Spotify Web API integration to pull playlist metadata and audio features directly, rather than requiring local file uploads
- Smarter 2-opt that restricts candidate swaps to nearby positions, to scale to much larger playlists
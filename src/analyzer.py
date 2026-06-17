import librosa
import numpy as np
import pandas as pd

# --- Constants ---
CAMELOT_MAP = {
    # Major keys (B = outer ring)
    'C Major':  '8B',
    'C# Major': '3B',
    'D Major':  '10B',
    'D# Major': '5B',
    'E Major':  '12B',
    'F Major':  '7B',
    'F# Major': '2B',
    'G Major':  '9B',
    'G# Major': '4B',
    'A Major':  '11B',
    'A# Major': '6B',
    'B Major':  '1B',

    # Minor keys (A = inner ring)
    'C Minor':  '5A',
    'C# Minor': '12A',
    'D Minor':  '7A',
    'D# Minor': '2A',
    'E Minor':  '9A',
    'F Minor':  '4A',
    'F# Minor': '11A',
    'G Minor':  '6A',
    'G# Minor': '1A',
    'A Minor':  '8A',
    'A# Minor': '3A',
    'B Minor':  '10A'
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# --- Functions ---
def detect_key(chroma_mean):
    best_score = -1
    best_key = ""

    notes = ["C","C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    for i in range(12):
        # Rotate profiles for each key
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)

        # Calculate correlation scores
        major_score = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_score = np.corrcoef(chroma_mean, minor_rotated)[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = notes[i] + " Major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = notes[i] + " Minor"

    return best_key, round(best_score, 2)

def camelot_key(key):
    return CAMELOT_MAP.get(key, "Unknown")

def extract_energy(y, sr):
    """
    Computes energy score from 4 complementary audio features.
    Uses per-feature normalization based on realistic audio ranges.
    Returns a 0-1 score.
    """
    
    # 1. RMS — average loudness
    rms = float(np.mean(librosa.feature.rms(y=y)))
    
    # 2. Spectral centroid — brightness
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    # 3. Spectral rolloff — where does high frequency energy drop off?
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    
    # 4. Onset strength — how punchy/rhythmically intense?
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset_env))
    
    # Normalize each feature using realistic observed ranges
    # These ranges are based on typical music audio — more robust than fixed divisors
    rms_norm     = np.clip((rms - 0.005) / (0.35 - 0.005), 0, 1)
    cent_norm    = np.clip((centroid - 500) / (8000 - 500), 0, 1)
    rolloff_norm = np.clip((rolloff - 1000) / (16000 - 1000), 0, 1)
    onset_norm   = np.clip((onset_mean - 0.5) / (10.0 - 0.5), 0, 1)
    
    # Weighted combination — weights reflect how much each feature
    # contributes to perceived energy
    energy = (
        0.35 * rms_norm      +   # loudness is biggest factor
        0.25 * onset_norm    +   # punchiness second
        0.25 * cent_norm     +   # brightness third
        0.15 * rolloff_norm      # rolloff adds nuance
    )
    
    return round(float(energy), 3)

def camelot_compatibility(key1, key2):
    """
    Returns compatibility score 0-100 between two Camelot codes.
    100 = perfect match
    75  = one step away (smooth transition)
    50  = relative major/minor swap
    0   = incompatible (clash)
    """
    if key1 == "Unknown" or key2 == "Unknown":
        return 0  # Can't determine compatibility

    num_1 = int(key1[:-1])
    type_1 = key1[-1]
    num_2 = int(key2[:-1])
    type_2 = key2[-1]

    #clockwise distance
    distance = min(abs(num_1 - num_2), 12 - abs(num_1 - num_2))

    if distance == 0 and type_1 == type_2:
        return 100  # Perfect match
    
    if distance == 1 and type_1 == type_2:
        return 75  # One step away
    
    if distance == 0 and type_1 != type_2:
        return 50  # Relative major/minor swap
    
    return 0  # Incompatible

def analyze_track(file_path):
    """
    Analyzes a single audio file and returns its DJ-relevant features.
    
    Returns a dictionary with:
    - filename, bpm, key, camelot code, energy score
    """

    # load the audio file
    y, sr = librosa.load(file_path, duration=30)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Key Detection
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key, confidence = detect_key(chroma_mean)

    # Camelot code
    camelot = camelot_key(key)

    # Energy score
    energy = extract_energy(y, sr)

    return {
        "filename": file_path,
        "bpm": round(float(tempo[0]), 2),
        "key": key,
        "camelot": camelot,
        "energy": energy,
        "key_confidence": confidence
    }

def compatibility_score(track1, track2):
    """
    Scores how well track_b would transition from track_a.
    Returns 0-100.
    """

    # --- BPM compatibility (the gate) ---
    diff = np.abs(track1['bpm'] - track2['bpm'])
    bpm_score = 0
    if diff <= 5:
        bpm_score = 100
    elif diff <= 15:
        bpm_score = 80
    elif diff <= 30:
        bpm_score = 50
    else:
        bpm_score = 15

    # --- Camelot key compatibility (the polish) ---
    key_score = camelot_compatibility(track1['camelot'], track2['camelot'])

    # --- Energy flow (the journey) ---
    # Small energy steps are good (gradual build/wind-down)
    # Huge energy jumps in either direction are jarring
    energy_diff = abs(track1['energy'] - track2['energy'])
    energy_score = max(0, 100 - (energy_diff * 250))  # scaled penalty

     # --- Weighted combination ---
    # BPM weighted highest since it's the listening "baseline"
    # Key second — the harmonic polish
    # Energy third — shapes the overall journey
    final_score = (
        0.45 * bpm_score   +
        0.35 * key_score   +
        0.20 * energy_score
    )
    
    return round(final_score, 1)
    
def two_opt_improve(ordered_list, max_iterations=1000):
    """
    Improves a playlist ordering by reversing segments
    whenever it increases total playlist score.
    """
    best = ordered_list.copy()
    best_score = total_playlist_score(best)
    
    n = len(best)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try every pair of positions (i, j) to reverse between them
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Create a candidate by reversing the segment [i:j+1]
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                candidate_score = total_playlist_score(candidate)
                
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True
        
    return best, best_score, iterations



def total_playlist_score(ordered_list):
    """
    Average compatibility score across all consecutive pairs.
    Higher = smoother overall playlist.
    """
    if len(ordered_list) < 2:
        return 0
    
    scores = []
    for i in range(len(ordered_list) - 1):
        s = compatibility_score(ordered_list[i], ordered_list[i+1])
        scores.append(s)
    
    return sum(scores) / len(scores)

def fix_bpm_doubling(bpm):
    """
    librosa sometimes doubles tempo for complex/orchestral tracks.
    If BPM is unrealistically high, assume it's doubled and halve it.
    """
    if bpm > 180:
        return round(bpm / 2, 1)
    return bpm

def greedy_reorder(df, lookahead=3):
    remaining = df.to_dict('records')
    remaining.sort(key=lambda x: x['energy'])
    current = remaining.pop(0)
    ordered = [current]
    
    while remaining:
        # Score all candidates against current track
        scored = [(compatibility_score(current, c), i, c) for i, c in enumerate(remaining)]
        scored.sort(key=lambda x: -x[0])
        
        # Look at top N candidates, pick the one with the best SECOND step too
        best_total = -1
        best_choice = None
        best_idx = -1
        
        for score, idx, candidate in scored[:lookahead]:
            # Simulate: if we pick this candidate, what's its best next match?
            remaining_after = [r for j, r in enumerate(remaining) if j != idx]
            if remaining_after:
                next_best = max(compatibility_score(candidate, r) for r in remaining_after)
            else:
                next_best = 0
            
            total = score + (0.5 * next_best)  # weight current step more than future
            
            if total > best_total:
                best_total = total
                best_choice = candidate
                best_idx = idx
        
        current = remaining.pop(best_idx)
        ordered.append(current)
    
    return pd.DataFrame(ordered)

def make_transition_clip(file_a, file_b, clip_seconds=8, crossfade_seconds=3, sr=22050):
    """
    Extracts the tail of file_a and head of file_b, and crossfades them
    so the transition sounds natural instead of a hard cut.
    """
    y_a, _ = librosa.load(file_a, sr=sr)
    y_b, _ = librosa.load(file_b, sr=sr)
    
    n_clip = clip_seconds * sr
    n_fade = crossfade_seconds * sr
    
    end_of_a   = y_a[-n_clip:] if len(y_a) > n_clip else y_a
    start_of_b = y_b[:n_clip] if len(y_b) > n_clip else y_b
    
    # Build fade curves — linear ramp down for A, ramp up for B
    fade_out = np.linspace(1.0, 0.0, n_fade)
    fade_in  = np.linspace(0.0, 1.0, n_fade)
    
    # Apply fade to the last n_fade samples of A and first n_fade samples of B
    end_of_a_faded = end_of_a.copy()
    end_of_a_faded[-n_fade:] *= fade_out
    
    start_of_b_faded = start_of_b.copy()
    start_of_b_faded[:n_fade] *= fade_in
    
    # The overlapping region: add the faded tail of A to the faded head of B
    overlap = end_of_a_faded[-n_fade:] + start_of_b_faded[:n_fade]
    
    # Stitch: [non-overlapping part of A] + [overlap] + [non-overlapping part of B]
    transition = np.concatenate([
        end_of_a_faded[:-n_fade],
        overlap,
        start_of_b_faded[n_fade:]
    ])
    
    return transition, sr
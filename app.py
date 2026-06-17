import os
import sys
import traceback
 
import gradio as gr
import pandas as pd
 
sys.path.append(".")
 
from src.analyzer import (
    analyze_track,
    compatibility_score,
    greedy_reorder,
    two_opt_improve,
    total_playlist_score,
    make_transition_clip,
)
 
# ---------------------------------------------------------------------------
# Visual design tokens — dark theme
# ---------------------------------------------------------------------------
# Deep charcoal background, warm rose/violet accent (matches the "DJ booth at
# night" feel), sage/amber/coral for score quality - same meaning as before,
# just recolored for contrast against dark surfaces.
 
COLOR_BG = "#15121C"          # near-black with a violet undertone
COLOR_SURFACE = "#1F1B29"     # card / panel background
COLOR_SURFACE_ALT = "#272131" # slightly lighter panel, for nested elements
COLOR_TEXT = "#F1EAF7"        # near-white, soft violet tint
COLOR_TEXT_SOFT = "#B8AEC9"   # muted secondary text
COLOR_PRIMARY = "#E08CB0"     # rose accent
COLOR_PRIMARY_HOVER = "#EC9FBE"
COLOR_GOOD = "#7FCB9B"
COLOR_OK = "#E8C26A"
COLOR_BAD = "#E08989"
COLOR_BORDER = "#352D44"
 
 
def score_color(score):
    if score >= 75:
        return COLOR_GOOD
    if score >= 50:
        return COLOR_OK
    return COLOR_BAD
 
 
def score_label(score):
    if score >= 75:
        return "smooth"
    if score >= 50:
        return "workable"
    return "clash"
 
 
CUSTOM_CSS = f"""
:root, .dark {{
    --body-background-fill: {COLOR_BG} !important;
    --background-fill-primary: {COLOR_BG} !important;
    --block-background-fill: {COLOR_SURFACE} !important;
    --block-border-color: {COLOR_BORDER} !important;
    --body-text-color: {COLOR_TEXT} !important;
    --body-text-color-subdued: {COLOR_TEXT_SOFT} !important;
    --input-background-fill: {COLOR_SURFACE_ALT} !important;
    --border-color-primary: {COLOR_BORDER} !important;
}}
body, .gradio-container {{
    background: {COLOR_BG} !important;
    color: {COLOR_TEXT} !important;
    font-family: 'Inter', -apple-system, sans-serif;
}}
#title-block h1 {{
    font-family: 'Comfortaa', 'Quicksand', sans-serif;
    color: {COLOR_TEXT} !important;
    font-weight: 700;
    letter-spacing: 0.01em;
    font-size: 2.1em;
}}
#title-block p, #title-block span {{
    color: {COLOR_TEXT_SOFT} !important;
}}
label, .label-wrap span, h3, h4 {{
    color: {COLOR_TEXT} !important;
}}
button.primary, .gr-button-primary {{
    background: {COLOR_PRIMARY} !important;
    border: none !important;
    color: {COLOR_BG} !important;
    font-weight: 700 !important;
}}
button.primary:hover, .gr-button-primary:hover {{
    background: {COLOR_PRIMARY_HOVER} !important;
}}
.score-card {{
    background: {COLOR_SURFACE_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 8px;
}}
.score-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid {COLOR_BORDER};
}}
.score-row:last-child {{
    border-bottom: none;
}}
.score-row .label {{
    color: {COLOR_TEXT_SOFT};
    font-size: 0.92em;
}}
.score-row .value {{
    color: {COLOR_TEXT};
    font-weight: 700;
    font-size: 1.05em;
}}
.delta-pill {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    background: {COLOR_GOOD};
    color: {COLOR_BG};
    font-weight: 700;
    font-size: 0.9em;
}}
.skipped-note {{
    color: {COLOR_BAD};
    font-size: 0.85em;
    margin-top: 8px;
}}
.camelot-chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 700;
    color: {COLOR_BG};
    background: {COLOR_PRIMARY};
}}
table, .table-wrap {{
    background: {COLOR_SURFACE} !important;
    color: {COLOR_TEXT} !important;
}}
thead, th {{
    background: {COLOR_SURFACE_ALT} !important;
    color: {COLOR_TEXT} !important;
}}
td {{
    color: {COLOR_TEXT} !important;
    border-color: {COLOR_BORDER} !important;
}}
footer {{
    display: none !important;
}}
"""
 
CUSTOM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill=COLOR_BG,
    body_background_fill_dark=COLOR_BG,
    block_background_fill=COLOR_SURFACE,
    block_background_fill_dark=COLOR_SURFACE,
    block_border_color=COLOR_BORDER,
    block_border_color_dark=COLOR_BORDER,
    body_text_color=COLOR_TEXT,
    body_text_color_dark=COLOR_TEXT,
    body_text_color_subdued=COLOR_TEXT_SOFT,
    body_text_color_subdued_dark=COLOR_TEXT_SOFT,
    input_background_fill=COLOR_SURFACE_ALT,
    input_background_fill_dark=COLOR_SURFACE_ALT,
    border_color_primary=COLOR_BORDER,
    border_color_primary_dark=COLOR_BORDER,
    button_primary_background_fill=COLOR_PRIMARY,
    button_primary_background_fill_dark=COLOR_PRIMARY,
    button_primary_background_fill_hover=COLOR_PRIMARY_HOVER,
    button_primary_text_color=COLOR_BG,
    button_primary_text_color_dark=COLOR_BG,
    block_radius="16px",
)
 
 
# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
 
def process_playlist(files):
    """Analyzes uploaded tracks and returns (score_html, table_df, final_list)."""
    if not files:
        return (
            "<div class='score-card'>Upload at least 2 audio files to get started.</div>",
            None,
            None,
        )
 
    results = []
    skipped = []
 
    for f in files:
        try:
            track = analyze_track(f.name)
            track["filename"] = os.path.basename(f.name)
            results.append(track)
        except Exception as e:
            skipped.append((os.path.basename(f.name), str(e)))
            print(f"Skipped {f.name}: {e}")
            traceback.print_exc()
 
    print(f"DEBUG: {len(results)} tracks analyzed successfully out of {len(files)} uploaded")
    if skipped:
        print(f"DEBUG: skipped files -> {skipped}")
 
    if len(results) < 2:
        html = "<div class='score-card'>Need at least 2 successfully analyzed audio files."
        if skipped:
            html += "<div class='skipped-note'>" + "<br>".join(
                f"Skipped {name}: {err}" for name, err in skipped
            ) + "</div>"
        html += "</div>"
        return html, None, None
 
    df = pd.DataFrame(results)
 
    original_list = df.to_dict("records")
    original_score = total_playlist_score(original_list)
 
    reordered_df = greedy_reorder(df, lookahead=3)
    reordered_list = reordered_df.to_dict("records")
 
    final_list, final_score, n_iters = two_opt_improve(reordered_list)
    improvement = final_score - original_score
 
    html = f"""
    <div class="score-card">
        <div class="score-row"><span class="label">Tracks analyzed</span>
            <span class="value">{len(results)} / {len(files)}</span></div>
        <div class="score-row"><span class="label">Original playlist score</span>
            <span class="value">{original_score:.1f}/100</span></div>
        <div class="score-row"><span class="label">Optimized playlist score</span>
            <span class="value">{final_score:.1f}/100</span></div>
        <div class="score-row"><span class="label">Improvement</span>
            <span class="delta-pill">+{improvement:.1f}</span></div>
    """
    if skipped:
        html += "<div class='skipped-note'>" + "<br>".join(
            f"Skipped {name}: {err}" for name, err in skipped
        ) + "</div>"
    html += "</div>"
 
    final_df = pd.DataFrame(final_list)[["filename", "bpm", "key", "camelot", "energy"]]
    final_df.insert(0, "order", range(1, len(final_df) + 1))
    final_df = final_df.rename(columns={
        "filename": "Track", "bpm": "BPM", "key": "Key",
        "camelot": "Camelot", "energy": "Energy", "order": "#",
    })
 
    path_lookup = {os.path.basename(f.name): f.name for f in files}
    for row in final_list:
        row["_full_path"] = path_lookup.get(row["filename"], row["filename"])
 
    return html, final_df, final_list
 
 
def get_pair_choices(final_list):
    if not final_list:
        return []
    choices = []
    for i in range(len(final_list) - 1):
        a, b = final_list[i], final_list[i + 1]
        score = compatibility_score(a, b)
        label = f"{i+1}. {a['filename']}  ->  {i+2}. {b['filename']}   ({score_label(score)} · {score:.1f})"
        choices.append(label)
    return choices
 
 
def generate_transition(final_list, pair_label):
    if not final_list or not pair_label:
        return None
 
    idx = int(pair_label.split(".")[0]) - 1
    a = final_list[idx]
    b = final_list[idx + 1]
 
    path_a = a.get("_full_path", a["filename"])
    path_b = b.get("_full_path", b["filename"])
 
    try:
        clip, sr = make_transition_clip(path_a, path_b, clip_seconds=8, crossfade_seconds=3)
    except TypeError:
        clip, sr = make_transition_clip(path_a, path_b, clip_seconds=8)
 
    out_path = "preview_transition.wav"
    import soundfile as sf
    sf.write(out_path, clip, sr)
    return out_path
 
 
def generate_full_mix(final_list, clip_seconds=8, crossfade_seconds=3):
    """
    Stitches every consecutive pair in the final playlist into one
    continuous mix, reusing the same crossfade clips end-to-end so the
    whole optimized playlist can be auditioned in a single playback.
    """
    if not final_list or len(final_list) < 2:
        return None
 
    import numpy as np
    import soundfile as sf
 
    sr_used = None
    full_audio = None
 
    for i in range(len(final_list) - 1):
        a, b = final_list[i], final_list[i + 1]
        path_a = a.get("_full_path", a["filename"])
        path_b = b.get("_full_path", b["filename"])
 
        try:
            clip, sr = make_transition_clip(
                path_a, path_b, clip_seconds=clip_seconds, crossfade_seconds=crossfade_seconds
            )
        except TypeError:
            clip, sr = make_transition_clip(path_a, path_b, clip_seconds=clip_seconds)
 
        sr_used = sr
 
        if full_audio is None:
            full_audio = clip
        else:
            # Each clip already starts with "tail of previous track" - to avoid
            # repeating that segment, only append the NEW (second-track) portion.
            n_fade = crossfade_seconds * sr
            new_portion = clip[len(clip) - (clip_seconds * sr - n_fade) - n_fade:]
            full_audio = np.concatenate([full_audio, new_portion])
 
    out_path = "full_playlist_mix.wav"
    sf.write(out_path, full_audio, sr_used)
    return out_path
 
 
def on_analyze(files):
    html, final_df, final_list = process_playlist(files)
    if final_list is None:
        return html, None, gr.update(choices=[], value=None), None, final_list, None
    choices = get_pair_choices(final_list)
    default_choice = choices[0] if choices else None
    return html, final_df, gr.update(choices=choices, value=default_choice), None, final_list, None
 
 
def on_pair_change(pair_label, final_list):
    return generate_transition(final_list, pair_label)
 
 
def on_full_mix_click(final_list):
    return generate_full_mix(final_list)
 
 
# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
 
try:
    # Older Gradio (4.x / 5.x) takes theme/css directly on Blocks()
    _blocks_ctx = gr.Blocks(title="Auto-DJ Playlist Optimizer", theme=CUSTOM_THEME, css=CUSTOM_CSS)
except TypeError:
    # Gradio 6.x moved theme/css to .launch()
    _blocks_ctx = gr.Blocks(title="Auto-DJ Playlist Optimizer")
 
with _blocks_ctx as demo:
    with gr.Column(elem_id="title-block"):
        gr.Markdown("# 🎧 Auto-DJ Playlist Flow Optimizer")
        gr.Markdown(
            "Upload your playlist tracks — get them reordered for the smoothest possible "
            "listening flow using harmonic mixing theory (Camelot Wheel), tempo, and energy analysis."
        )
 
    file_input = gr.File(file_count="multiple", label="Upload audio files (.wav / .mp3)")
    analyze_btn = gr.Button("Analyze & Reorder", variant="primary")
 
    score_output = gr.HTML()
    table_output = gr.Dataframe(label="Reordered playlist")
 
    gr.Markdown("### Preview a single transition")
    pair_dropdown = gr.Dropdown(label="Choose a transition to preview", choices=[])
    audio_output = gr.Audio(label="Transition preview")
 
    gr.Markdown("### Preview the whole optimized playlist")
    gr.Markdown(
        "Renders every track back-to-back with crossfades applied, in the order above — "
        "one continuous file so you can listen to the full flow at once."
    )
    full_mix_btn = gr.Button("Generate full playlist mix")
    full_mix_output = gr.Audio(label="Full playlist mix")
 
    final_list_state = gr.State()
 
    analyze_btn.click(
        fn=on_analyze,
        inputs=[file_input],
        outputs=[score_output, table_output, pair_dropdown, audio_output, final_list_state, full_mix_output],
    )
 
    pair_dropdown.change(
        fn=on_pair_change,
        inputs=[pair_dropdown, final_list_state],
        outputs=[audio_output],
    )
 
    full_mix_btn.click(
        fn=on_full_mix_click,
        inputs=[final_list_state],
        outputs=[full_mix_output],
    )
 
if __name__ == "__main__":
    try:
        demo.launch(show_api=False)
    except TypeError:
        try:
            demo.launch(theme=CUSTOM_THEME, css=CUSTOM_CSS, show_api=False)
        except TypeError:
            demo.launch()
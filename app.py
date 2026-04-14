"""
app.py – ClipMatch Streamlit Frontend

Professional, panel-based interface for video similarity detection.
Features:
    - Query video upload with preview
    - Configurable similarity threshold
    - Real-time progress status
    - Structured results display with matched video, timestamps, confidence
    - Execution time tracking
"""

import os
import sys
import time
import tempfile
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from main_pipeline import run_pipeline
from modules.dataset_downloader import prepare_dataset

# ──────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ClipMatch – Video Similarity Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(145deg, #0a0a0f 0%, #0f0f1a 40%, #0a0a14 100%);
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ── Header ── */
    .header-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(20px);
    }
    .glass-card-accent {
        background: rgba(96, 165, 250, 0.04);
        border: 1px solid rgba(96, 165, 250, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Section Labels ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #60a5fa;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #60a5fa;
        border-radius: 2px;
    }

    /* ── Result Panel ── */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(59, 130, 246, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .result-card-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(245, 158, 11, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }
    .result-match { color: #10b981; }
    .result-nomatch { color: #ef4444; }

    /* ── Metric Items ── */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        font-weight: 500;
    }
    .metric-value {
        font-size: 0.95rem;
        color: #e5e7eb;
        font-weight: 600;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }

    /* ── Confidence Bar ── */
    .confidence-container {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
    }
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        height: 12px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease;
    }

    /* ── Status Messages ── */
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        font-size: 0.85rem;
        color: #9ca3af;
    }
    .status-active { color: #60a5fa; }
    .status-done { color: #10b981; }

    /* ── Upload Area ── */
    .upload-area {
        border: 2px dashed rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .upload-area:hover {
        border-color: rgba(96, 165, 250, 0.4);
    }

    /* ── Stats Row ── */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-chip {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        flex: 1;
        text-align: center;
    }
    .stat-chip-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .stat-chip-label {
        font-size: 0.65rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%);
        margin: 1.5rem 0;
    }

    /* ── Fix Streamlit defaults ── */
    .stProgress > div > div > div { background: #60a5fa; }
    h1, h2, h3 { color: #e5e7eb !important; }
    .stMarkdown p { color: #d1d5db; }
    div[data-testid="stFileUploader"] label {
        color: #9ca3af !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────────────────

st.markdown("""
<div class="header-container">
    <div class="header-title">🎬 ClipMatch</div>
    <div class="header-subtitle">Video Similarity Detection & Clip Localization System</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Session State
# ──────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "dataset_ready" not in st.session_state:
    # Check if dataset exists
    ds_dir = os.path.join(PROJECT_ROOT, "Dataset")
    st.session_state.dataset_ready = os.path.exists(ds_dir) and len(
        [f for f in os.listdir(ds_dir) if f.endswith(".mp4")]
    ) > 0 if os.path.exists(ds_dir) else False


# ──────────────────────────────────────────────────────────
#  Dataset Preparation Section
# ──────────────────────────────────────────────────────────

if not st.session_state.dataset_ready:
    st.markdown("""
    <div class="glass-card-accent">
        <div class="section-label">Dataset Setup Required</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            No dataset found. Click below to generate synthetic dataset videos
            (100 dataset videos + 20 query clips). This only needs to run once.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate Dataset", use_container_width=True, type="primary"):
        with st.spinner("Generating synthetic dataset... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Creating folder structure...")
            progress_bar.progress(5)

            result = prepare_dataset(PROJECT_ROOT)

            progress_bar.progress(100)
            status_text.text("Dataset ready!")
            st.session_state.dataset_ready = True
            time.sleep(1)
            st.rerun()

    st.stop()


# ──────────────────────────────────────────────────────────
#  Main Layout: Two Columns
# ──────────────────────────────────────────────────────────

col_left, col_spacer, col_right = st.columns([5, 0.5, 5])

# ── LEFT PANEL: Upload & Controls ──
with col_left:
    # Upload Section
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">Query Input</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a query video clip (~15 seconds)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
        label_visibility="collapsed",
    )

    if uploaded_file:
        # Preview uploaded video
        st.markdown('<div class="section-label">Video Preview</div>', unsafe_allow_html=True)
        st.video(uploaded_file)

        # File info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-chip">
                <div class="stat-chip-value">{uploaded_file.name}</div>
                <div class="stat-chip-label">Filename</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-value">{file_size_mb:.1f} MB</div>
                <div class="stat-chip-label">File Size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Controls
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.3,
        max_value=0.99,
        value=0.7,
        step=0.05,
        help="Minimum NCC score required to consider a match valid.",
    )

    extraction_fps = st.select_slider(
        "Extraction FPS",
        options=[8, 12, 16, 24, 32],
        value=16,
        help="Higher FPS = more accurate but slower. 16 FPS is recommended.",
    )

    window_step = st.select_slider(
        "Window Step Size",
        options=[1, 2, 3, 4],
        value=2,
        help="Step 1 = exhaustive scan (slowest, most precise). Step 2-4 = faster but may miss exact positions.",
    )

    # Dataset stats
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    ds_dir = os.path.join(PROJECT_ROOT, "Dataset")
    q_dir = os.path.join(PROJECT_ROOT, "query")
    num_ds = len([f for f in os.listdir(ds_dir) if f.endswith(".mp4")]) if os.path.exists(ds_dir) else 0
    num_q = len([f for f in os.listdir(q_dir) if f.endswith(".mp4")]) if os.path.exists(q_dir) else 0

    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-chip">
            <div class="stat-chip-value">{num_ds}</div>
            <div class="stat-chip-label">Dataset Videos</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-value">{num_q}</div>
            <div class="stat-chip-label">Query Clips</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT PANEL: Progress & Results ──
with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">Detection Output</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        # Run Pipeline Button
        run_button = st.button(
            "▶ Run Detection",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.processing,
        )

        if run_button:
            st.session_state.processing = True
            st.session_state.result = None

            # Save uploaded video to temp file
            temp_dir = os.path.join(PROJECT_ROOT, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Progress display
            progress_bar = st.progress(0)
            status_container = st.empty()

            status_log = []

            def progress_callback(status, progress):
                status_log.append(status)
                progress_bar.progress(min(int(progress * 100), 100))
                display_lines = status_log[-6:]  # Show last 6 status lines
                status_html = ""
                for i, line in enumerate(display_lines):
                    if i == len(display_lines) - 1:
                        status_html += f'<div class="status-item status-active">⟳ {line}</div>'
                    else:
                        status_html += f'<div class="status-item status-done">✓ {line}</div>'
                status_container.markdown(status_html, unsafe_allow_html=True)

            # Run the pipeline
            result = run_pipeline(
                query_path=temp_path,
                base_dir=PROJECT_ROOT,
                target_fps=extraction_fps,
                threshold=threshold,
                step=window_step,
                progress_callback=progress_callback,
            )

            progress_bar.progress(100)
            st.session_state.result = result
            st.session_state.processing = False

            # Clean up temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

        # Display results
        if st.session_state.result:
            result = st.session_state.result

            if result["match_found"]:
                # Confidence color
                conf = result["confidence"]
                if conf >= 0.85:
                    bar_color = "#10b981"  # green
                elif conf >= 0.7:
                    bar_color = "#f59e0b"  # amber
                else:
                    bar_color = "#ef4444"  # red

                bar_width = max(5, int(conf * 100))

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title result-match">✅ Match Found</div>

                    <div class="metric-row">
                        <span class="metric-label">Matched Video</span>
                        <span class="metric-value">{result['video']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Start Time</span>
                        <span class="metric-value">{result['start_time']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">End Time</span>
                        <span class="metric-value">{result['end_time']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Execution Time</span>
                        <span class="metric-value">{result['execution_time']}s</span>
                    </div>

                    <div class="confidence-container">
                        <div class="metric-row" style="border:none; padding:0;">
                            <span class="metric-label">Confidence Score</span>
                            <span class="metric-value" style="color: {bar_color};">{conf:.4f}</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill"
                                 style="width: {bar_width}%; background: linear-gradient(90deg, {bar_color}, {bar_color}aa);">
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                best_score = result.get("details", {}).get("best_raw_score", result.get("confidence", 0))
                st.markdown(f"""
                <div class="result-card-fail">
                    <div class="result-title result-nomatch">❌ No Match Found</div>
                    <div class="metric-row">
                        <span class="metric-label">Best Score</span>
                        <span class="metric-value">{best_score:.4f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Threshold</span>
                        <span class="metric-value">{threshold}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Videos Scanned</span>
                        <span class="metric-value">{result.get('details', {}).get('videos_scanned', 'N/A')}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Execution Time</span>
                        <span class="metric-value">{result['execution_time']}s</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Details expander
            with st.expander("📊 Detailed Match Info"):
                st.json(result["details"])

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #4b5563;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
            <div style="font-size: 1.1rem; font-weight: 500; color: #6b7280;">
                Upload a query video to begin
            </div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 0.5rem;">
                Supported formats: MP4, AVI, MOV, MKV
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Quick Test Section
# ──────────────────────────────────────────────────────────

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.expander("🧪 Quick Test with Pre-generated Query"):
    q_dir = os.path.join(PROJECT_ROOT, "query")
    if os.path.exists(q_dir):
        query_files = sorted([f for f in os.listdir(q_dir) if f.endswith(".mp4")])
        if query_files:
            selected_query = st.selectbox("Select a query clip:", query_files)

            if st.button("▶ Run Quick Test", key="quick_test"):
                query_path = os.path.join(q_dir, selected_query)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def quick_callback(status, progress):
                    progress_bar.progress(min(int(progress * 100), 100))
                    status_text.text(f"⟳ {status}")

                result = run_pipeline(
                    query_path=query_path,
                    base_dir=PROJECT_ROOT,
                    target_fps=extraction_fps,
                    threshold=threshold,
                    step=window_step,
                    progress_callback=quick_callback,
                )

                progress_bar.progress(100)

                if result["match_found"]:
                    st.success(
                        f"**Match Found!** → `{result['video']}` | "
                        f"{result['start_time']} - {result['end_time']} | "
                        f"Confidence: {result['confidence']:.4f} | "
                        f"Time: {result['execution_time']}s"
                    )

                    # Check against ground truth
                    import json
                    gt_path = os.path.join(PROJECT_ROOT, "ground_truth.json")
                    if os.path.exists(gt_path):
                        with open(gt_path) as f:
                            gt = json.load(f)
                        if selected_query in gt:
                            gt_entry = gt[selected_query]
                            st.info(
                                f"**Ground Truth:** {gt_entry['source_video']} | "
                                f"{gt_entry['start_sec']:.1f}s - {gt_entry['end_sec']:.1f}s"
                            )
                else:
                    st.error(
                        f"No match above threshold ({threshold}). "
                        f"Best score: {result.get('confidence', 0):.4f}"
                    )

                with st.expander("Full Result"):
                    st.json(result)
        else:
            st.warning("No query clips found. Generate dataset first.")
    else:
        st.warning("Query directory not found.")


# ──────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0; color: #374151; font-size: 0.75rem;">
    ClipMatch v1.0 · NCC-based Video Similarity Detection · CPU-Only · Academic MVP
</div>
""", unsafe_allow_html=True)

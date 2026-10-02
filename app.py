import streamlit as st
import pysrt
import torch
import librosa
import soundfile as sf
import numpy as np
import os
import time
import asyncio
import edge_tts
import base64
from faster_whisper import WhisperModel

# Page Configuration
st.set_page_config(
    page_title="AI Movie Recap Helper Tool",
    page_icon="🎬",
    layout="wide"
)

# Setup Temporary Storage Directory
TEMP_DIR = "temp_processing"
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------------------------------
# Background Video Injection (Dynamic CSS + HTML5 Video)
# ----------------------------------------------------
video_bg_path = "bg_video.mp4"
if os.path.exists(video_bg_path):
    with open(video_bg_path, "rb") as f:
        video_bytes = f.read()
    encoded_video = base64.b64encode(video_bytes).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        #bg-video-container {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -999;
            overflow: hidden;
        }}
        #bg-video {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
        }}
        .main .block-container {{
            background: rgba(15, 23, 42, 0.85);
            padding: 2rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(15, 23, 42, 0.95);
        }}
        </style>
        
        <div id="bg-video-container">
            <video autoplay muted loop id="bg-video">
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# Multi-language Dictionary (English & Khmer UI Only)
# ----------------------------------------------------
LANG_TEXTS = {
    "English": {
        "app_title": "🎬 AI Movie Recap Helper Tool",
        "gpu_active": "⚡ Hardware Acceleration: NVIDIA GPU Active (CUDA)",
        "cpu_active": "⚠️ Hardware Acceleration: CPU Active (No NVIDIA GPU detected)",
        "step1_tab": "Step 1: Extract Subtitles",
        "step2_tab": "Step 2: Generate AI Voice",
        "step3_tab": "Step 3: Subtitle Translator",
        "step1_header": "Step 1: Ultra-Fast Subtitle Extraction",
        "upload_media_label": "Upload Media File (.mp3, .wav, .m4a, .mp4, .mkv)",
        "sub_name_label": "Subtitle File Name (Optional):",
        "whisper_model_label": "Select Whisper AI Model Quality / ជ្រើសរើសកម្រិត Whisper AI Model:",
        "start_extract_btn": "🚀 Start Ultra-Fast Subtitle Extraction",
        "extracting_msg": "⏳ Extracting subtitles smoothly... Estimated time remaining: ",
        "extract_success": "✅ Subtitle Extraction Completed Successfully!",
        "download_sub_btn": "📥 Download SRT Subtitle File",
        "step2_header": "Step 2: Generate Synced AI Voice (Turbo Speed)",
        "upload_srt_label": "Upload SRT Subtitle File:",
        "voice_label": "Select AI Voice Accent & Gender / ជ្រើសរើសប្រភេទសំឡេង AI:",
        "speech_rate_label": "⚡ Global Speech Speed Rate / តម្រូវល្បឿននិយាយសរុប:",
        "speech_rate_help": "If Khmer subtitle text duration is longer than video duration, increase rate to +8% or +10%.",
        "start_voice_btn": "🎙️ Start Generating Synced AI Voice (Turbo)",
        "generating_msg": "⏳ Generating Neural AI Voice & Syncing... Estimated time remaining: ",
        "voice_success": "🎉 Audio Generation Completed Successfully!",
        "download_audio_btn": "📥 Download Final Audio File (MP3)",
        "step3_header": "Step 3: Online Subtitle Translator (translatesubtitles.co)"
    },
    "ភាសាខ្មែរ": {
        "app_title": "🎬 ឧបករណ៍ជំនួយសម្រាយរឿង AI (Movie Recap Helper Tool)",
        "gpu_active": "⚡ Hardware Acceleration: NVIDIA GPU Active (CUDA)",
        "cpu_active": "⚠️️ Hardware Acceleration: CPU Active (No NVIDIA GPU detected)",
        "step1_tab": "ជំហានទី ១: ទាញយក Subtitle",
        "step2_tab": "ជំហានទី ២: បំប្លែង Subtitle ទៅជាសំឡេង AI",
        "step3_tab": "ជំហានទី ៣: កម្មវិធីបកប្រែ Subtitle",
        "step1_header": "ជំហានទី ១: ទាញយក Subtitle",
        "upload_media_label": "Upload File មេឌៀ (.mp3, .wav, .m4a, .mp4, .mkv)",
        "sub_name_label": "ឈ្មោះ File Subtitle (Optional):",
        "whisper_model_label": "ជ្រើសរើសកម្រិត Whisper AI Model / Select Whisper AI Model Quality:",
        "start_extract_btn": "🚀 ចាប់ផ្តើមទាញយក Subtitle",
        "extracting_msg": "⏳ កំពុងទាញយក Subtitle ដោយរលូន... រយៈពេលប៉ាន់ស្មាននៅសល់៖ ",
        "extract_success": "✅ ទាញយក Subtitle រួចរាល់ដោយជោគជ័យ!",
        "download_sub_btn": "📥 ទាញយក File Subtitle (SRT)",
        "step2_header": "ជំហានទី ២: បំប្លែង Subtitle ទៅជាសំឡេង AI ឱ្យស្មើចង្វាក់ (ល្បឿនលឿនពិសេស Turbo)",
        "upload_srt_label": "Upload File Subtitle (.srt):",
        "voice_label": "ជ្រើសរើសប្រភេទសំឡេង AI / Select AI Voice Accent & Gender:",
        "speech_rate_label": "⚡ តម្រូវល្បឿននិយាយសរុប (Voice Speed Rate):",
        "speech_rate_help": "ប្រសិនបើសាច់រឿងខ្មែរ/Subtitle វែងជាងវីដេអូ សូមដំឡើងល្បឿនទៅ +8% ឬ +10% ឱ្យស្មើចង្វាក់។",
        "start_voice_btn": "🎙️ ចាប់ផ្តើមបង្កើតសំឡេង AI ឱ្យស្មើចង្វាក់ (Turbo)",
        "generating_msg": "⏳ កំពុងបង្កើតសំឡេង AI... រយៈពេលប៉ាន់ស្មាននៅសល់៖ ",
        "voice_success": "🎉 បង្កើតសំឡេង AI ស្មើចង្វាក់រួចរាល់ដោយជោគជ័យ!",
        "download_audio_btn": "📥 ទាញយក File សំឡេងពេញលេញ (MP3)",
        "step3_header": "ជំហានទី ៣: កម្មវិធីបកប្រែ Subtitle តាមអនឡាញ (translatesubtitles.co)"
    }
}

# ----------------------------------------------------
# Sidebar Language Switcher
# ----------------------------------------------------
st.sidebar.title("🌐 Language / ភាសា")
selected_lang = st.sidebar.radio("Choose UI Language / ជ្រើសរើសភាសា UI:", ["English", "ភាសាខ្មែរ"])
T = LANG_TEXTS[selected_lang]

# ----------------------------------------------------
# AI Voice Options (Khmer & English Only)
# ----------------------------------------------------
VOICE_OPTIONS = {
    "ខ្មែរ - ស្រី (Khmer - Female)": "km-KH-SreymomNeural",
    "ខ្មែរ - ប្រុស (Khmer - Male)": "km-KH-PisethNeural",
    "អង់គ្លេស - ស្រី (English - Female)": "en-US-AvaNeural",
    "អង់គ្លេស - ប្រុស (English - Male)": "en-US-AndrewNeural",
}

async def generate_edge_tts(text, voice_code, output_file, rate="+0%"):
    try:
        communicate = edge_tts.Communicate(text, voice_code, rate=rate)
        await communicate.save(output_file)
        return True
    except Exception:
        return False

def parse_time_to_sec(sub_time):
    return sub_time.hours * 3600 + sub_time.minutes * 60 + sub_time.seconds + sub_time.milliseconds / 1000.0

def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

async def process_single_subtitle(i, sub, selected_voice, rate_str, sample_rate):
    start_sec = parse_time_to_sec(sub.start)
    end_sec = parse_time_to_sec(sub.end)
    target_duration = end_sec - start_sec
    text = sub.text.replace("\n", " ").strip()

    if not text or target_duration <= 0:
        return None

    temp_mp3 = os.path.join(TEMP_DIR, f"temp_{i}.mp3")
    success = await generate_edge_tts(text, selected_voice, temp_mp3, rate=rate_str)

    audio_clip = None
    if success and os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 0:
        try:
            audio_clip, sr = librosa.load(temp_mp3, sr=sample_rate)
            actual_duration = len(audio_clip) / sr

            if actual_duration > target_duration:
                rate = actual_duration / target_duration
                audio_clip = librosa.effects.time_stretch(audio_clip, rate=rate)

            target_samples = int(target_duration * sample_rate)
            audio_clip = audio_clip[:target_samples]
        except Exception:
            audio_clip = None

        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except:
                pass

    return (start_sec, audio_clip)

async def process_all_subtitles(subs, selected_voice, rate_str, sample_rate, progress_bar, status_placeholder):
    total_subs = len(subs)
    exact_end_sec = parse_time_to_sec(subs[-1].end)
    total_samples = int(exact_end_sec * sample_rate)
    final_audio_data = np.zeros(total_samples, dtype=np.float32)

    chunk_size = 20
    start_time_all = time.time()
    
    for chunk_idx in range(0, total_subs, chunk_size):
        chunk_subs = list(enumerate(subs))[chunk_idx:chunk_idx + chunk_size]
        
        tasks = [
            process_single_subtitle(i, sub, selected_voice, rate_str, sample_rate)
            for i, sub in chunk_subs
        ]
        
        results = await asyncio.gather(*tasks)

        for res in results:
            if res is not None:
                start_sec, audio_clip = res
                if audio_clip is not None and len(audio_clip) > 0:
                    start_sample = int(start_sec * sample_rate)
                    end_sample = start_sample + len(audio_clip)
                    if end_sample <= len(final_audio_data):
                        final_audio_data[start_sample:end_sample] += audio_clip

        completed_count = min(chunk_idx + chunk_size, total_subs)
        elapsed = time.time() - start_time_all
        avg_per_item = elapsed / completed_count
        remaining_items = total_subs - completed_count
        est_seconds = int(avg_per_item * remaining_items)
        
        mins, secs = divmod(est_seconds, 60)
        est_str = f"{mins}នាទី {secs}វិនាទី" if mins > 0 else f"{secs}វិនាទី"

        progress_bar.progress(completed_count / total_subs)
        status_placeholder.warning(f"{T['generating_msg']} {est_str} ({completed_count}/{total_subs} ប្រយោគ)")

    return final_audio_data

# ----------------------------------------------------
# App Header & GPU Detection
# ----------------------------------------------------
st.title(T["app_title"])

if torch.cuda.is_available():
    st.success(T["gpu_active"])
    device_type = "cuda"
    compute_type = "float16"
else:
    st.warning(T["cpu_active"])
    device_type = "cpu"
    compute_type = "int8"

tab1, tab2, tab3 = st.tabs([T["step1_tab"], T["step2_tab"], T["step3_tab"]])

# ====================================================
# TAB 1: Ultra-Fast Subtitle Extraction with Smooth Live Progress
# ====================================================
with tab1:
    st.header(T["step1_header"])
    
    uploaded_file = st.file_uploader(
        T["upload_media_label"], 
        type=["mp3", "wav", "m4a", "aac", "mp4", "mkv"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        custom_srt_name = st.text_input(T["sub_name_label"], value="movie_recap_subtitle")
    with col2:
        model_size = st.selectbox(
            T["whisper_model_label"],
            ["tiny", "base", "small", "medium", "large-v3"],
            index=1
        )
        
    status_placeholder_s1 = st.empty()
        
    if uploaded_file is not None:
        if st.button(T["start_extract_btn"]):
            start_time = time.time()
            input_path = os.path.join(TEMP_DIR, uploaded_file.name)
            
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            progress_bar_s1 = st.progress(0.05)
            status_placeholder_s1.info("⏳ កំពុងដំណើរការផ្ទុក Whisper AI Model...")
            
            # Load Whisper Model
            model = WhisperModel(model_size, device=device_type, compute_type=compute_type)
            
            progress_bar_s1.progress(0.15)
            status_placeholder_s1.info("🎙️ កំពុងចាប់ផ្តើមស្កេនសំឡេងវីដេអូតាមរយៈ GPU...")
            
            # Transcribe (streaming segments)
            segments, info = model.transcribe(input_path, beam_size=1)
            
            srt_path = os.path.join(TEMP_DIR, f"{custom_srt_name}.srt")
            
            collected_segments = []
            audio_duration = getattr(info, "duration", 0) # total duration in seconds if available
            
            # Smoothly iterate segments and update progress live
            for segment in segments:
                collected_segments.append(segment)
                idx = len(collected_segments)
                
                # Estimate progress based on audio duration if available, or incremental steps
                if audio_duration > 0:
                    current_sec = segment.end
                    progress_val = min(0.15 + (0.8 * (current_sec / audio_duration)), 0.98)
                    
                    elapsed = time.time() - start_time
                    if current_sec > 0:
                        total_est_time = elapsed * (audio_duration / current_sec)
                        remaining = int(max(0, total_est_time - elapsed))
                        m, s = divmod(remaining, 60)
                        est_str = f"{m}នាទី {s}វិនាទី" if m > 0 else f"{s}វិនាទី"
                    else:
                        est_str = "កំពុងគណនា..."
                else:
                    # Fallback if duration is unknown
                    progress_val = min(0.15 + (idx * 0.001), 0.98)
                    est_str = f"ប្រយោគទី {idx}"

                progress_bar_s1.progress(progress_val)
                status_placeholder_s1.warning(f"{T['extracting_msg']} {est_str} (បាន {idx} ប្រយោគ)")

            total_segs = len(collected_segments)
            if total_segs == 0:
                status_placeholder_s1.error("❌ រកមិនឃើញសំឡេងនៅក្នុងវីដេអូនេះទេ។")
            else:
                with open(srt_path, "w", encoding="utf-8") as f:
                    for idx, segment in enumerate(collected_segments, start=1):
                        start_str = format_timestamp(segment.start)
                        end_str = format_timestamp(segment.end)
                        text = segment.text.strip()
                        
                        f.write(f"{idx}\n")
                        f.write(f"{start_str} --> {end_str}\n")
                        f.write(f"{text}\n\n")

                progress_bar_s1.progress(1.0)
                status_placeholder_s1.success(f"{T['extract_success']} (ចំណាយពេលសរុប៖ {round(time.time() - start_time, 2)} វិនាទី)")
                
                with open(srt_path, "rb") as f:
                    st.download_button(
                        label=T["download_sub_btn"],
                        data=f,
                        file_name=f"{custom_srt_name}.srt",
                        mime="text/plain"
                    )

# ====================================================
# TAB 2: Synced AI Voice Generation (Turbo Parallel)
# ====================================================
with tab2:
    st.header(T["step2_header"])
    
    uploaded_srt = st.file_uploader(T["upload_srt_label"], type=["srt"])
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        selected_voice_label = st.selectbox(T["voice_label"], list(VOICE_OPTIONS.keys()))
        selected_voice = VOICE_OPTIONS[selected_voice_label]
    
    with col_v2:
        speech_rate = st.slider(
            T["speech_rate_label"],
            min_value=-10, max_value=+30, value=+8, step=1,
            help=T["speech_rate_help"]
        )
        rate_str = f"{speech_rate:+d}%"
    
    status_placeholder_s2 = st.empty()
    
    if uploaded_srt is not None:
        srt_path = os.path.join(TEMP_DIR, uploaded_srt.name)
        with open(srt_path, "wb") as f:
            f.write(uploaded_srt.getbuffer())
            
        subs = pysrt.open(srt_path, encoding="utf-8")
        
        if st.button(T["start_voice_btn"]):
            sample_rate = 22050
            progress_bar_s2 = st.progress(0.05)
            
            status_placeholder_s2.info("🎙️ កំពុងរៀបចំប្រព័ន្ធបង្កើតសំឡេង Neural AI...")
            
            final_audio_data = asyncio.run(
                process_all_subtitles(subs, selected_voice, rate_str, sample_rate, progress_bar_s2, status_placeholder_s2)
            )
            
            output_final_mp3 = os.path.join(TEMP_DIR, "final_synced_voice.mp3")
            sf.write(output_final_mp3, final_audio_data, sample_rate)
            
            status_placeholder_s2.success(T["voice_success"])
            st.audio(output_final_mp3)
            
            with open(output_final_mp3, "rb") as f:
                st.download_button(
                    label=T["download_audio_btn"],
                    data=f,
                    file_name="final_synced_voice.mp3",
                    mime="audio/mp3"
                )

# ====================================================
# TAB 3: Embedded Subtitle Translator Web
# ====================================================
with tab3:
    st.header(T["step3_header"])
    st.caption("បងអាចបកប្រែ File Subtitle (.srt) ពីអង់គ្លេសទៅខ្មែរនៅទីនេះ រួច Download File នោះមក Upload ចូល Tab ទី ២ ខាងលើបានយ៉ាងងាយស្រួល!")
    
    st.components.v1.iframe(
        "https://translatesubtitles.co/translatesub.php?srt=11.srt", 
        height=750, 
        scrolling=True
    )
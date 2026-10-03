import os
import re
import warnings

# Suppress SyntaxWarning emitted by older regex patterns in libraries on Python 3.12+
warnings.filterwarnings("ignore", category=SyntaxWarning)

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _download_with_yt_dlp(url: str, player_clients: list) -> str:
    """Download YouTube audio using specified player client(s) with FFmpeg audio extraction."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "extractor_args": {
            "youtube": {
                "player_client": player_clients
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")

        # The extracted audio file is always <id>.wav
        expected_wav = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")
        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback to checking prepared filename base
        prep = ydl.prepare_filename(info)
        base, _ = os.path.splitext(prep)
        wav_cand = base + ".wav"
        if os.path.exists(wav_cand):
            return wav_cand
        if os.path.exists(prep):
            return prep

        raise FileNotFoundError(f"Extracted audio file not found for video id '{video_id}'")

def download_youtube_audio(url: str) -> str:
    # Try different client strategies to bypass YouTube 403 Forbidden on datacenters/cloud
    client_strategies = [
        ["android", "web"],
        ["android"],
        ["mweb", "web"],
        ["ios", "web"],
    ]

    last_error = None
    for clients in client_strategies:
        try:
            return _download_with_yt_dlp(url, clients)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "unavailable" in err_str or "not a valid url" in err_str or "private" in err_str:
                raise
            continue

    if last_error:
        err = str(last_error)
        if "403" in err or "forbidden" in err.lower():
            raise RuntimeError(
                "YouTube blocked direct cloud download (HTTP 403 Forbidden). "
                "Cloud server IP addresses (like Streamlit Cloud) are frequently restricted by YouTube. "
                "💡 Please use the 'Upload Audio/Video' option in the sidebar to upload the file directly, or run the app locally."
            )
        raise last_error

    raise RuntimeError("Failed to download audio from YouTube.")



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
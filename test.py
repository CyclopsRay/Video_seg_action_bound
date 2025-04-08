import os
import subprocess

# For the video jpQBWsR3HHs specifically, try with format specification
vid_url = 'https://www.youtube.com/watch?v=jpQBWsR3HHs'
vid_prefix = 'path/to/save/jpQBWsR3HHs'

# Try explicitly requesting a specific format
cmd = ["yt-dlp", "-f", "best", "-o", f"{vid_prefix}.%(ext)s", "--verbose", vid_url]
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Failed to download {vid_url}: {result.stderr}")
    else:
        # Check if the video was downloaded
        if os.path.exists(f"{vid_prefix}.mp4") or os.path.exists(f"{vid_prefix}.mkv") or os.path.exists(f"{vid_prefix}.webm"):
            print(f'[INFO] Successfully downloaded video {vid_url}')
        else:
            print(f'[WARNING] Download command succeeded but no video file found for {vid_url}')
except Exception as e:
    print(f"[ERROR] Exception when downloading {vid_url}: {str(e)}")
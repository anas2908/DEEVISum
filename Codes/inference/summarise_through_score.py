import os
import numpy as np
from pytube import YouTube
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re

def get_video_id(youtube_url):
    """Extracts video ID from a YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    return match.group(1) if match else "video"

def download_youtube_video(youtube_url, download_path="downloads"):
    os.makedirs(download_path, exist_ok=True)
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    downloaded_file = stream.download(output_path=download_path)
    print(f"Downloaded: {downloaded_file}")
    return downloaded_file

def parse_scores(score_string):
    return list(map(int, score_string.strip().split(",")))

def get_top_segments(scores, fps=1, top_percent=15):
    scores = np.array(scores)
    n_frames = len(scores)
    top_k = int(np.ceil(n_frames * (top_percent / 100.0)))
    top_indices = np.argsort(scores)[-top_k:]
    top_indices.sort()  # Keep in chronological order

    # Merge consecutive indices into continuous segments
    segments = []
    start = top_indices[0]
    for i in range(1, len(top_indices)):
        if top_indices[i] != top_indices[i-1] + 1:
            segments.append((start, top_indices[i-1] + 1))
            start = top_indices[i]
    segments.append((start, top_indices[-1] + 1))
    return [(s, e) for s, e in segments]

def trim_video(input_path, segments, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = VideoFileClip(input_path)
    clips = [video.subclip(start, end) for start, end in segments]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Saved summary video to: {output_path}")

def main():
    youtube_link = input("Enter YouTube link: ").strip()
    score_input = input("Enter comma-separated scores (1 per second): ").strip()

    video_id = get_video_id(youtube_link)
    downloaded_video_path = download_youtube_video(youtube_link)
    scores = parse_scores(score_input)
    segments = get_top_segments(scores, fps=1, top_percent=15)

    output_dir = os.path.join(os.getcwd(), "test_set")
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    trim_video(downloaded_video_path, segments, output_path)

if __name__ == "__main__":
    main()

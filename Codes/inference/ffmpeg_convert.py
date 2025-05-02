from moviepy.editor import VideoFileClip
import os

def reencode_with_moviepy(input_path, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_reencoded.mp4"

    print(f"Re-encoding: {input_path}")
    clip = VideoFileClip(input_path)

    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4,
        preset='ultrafast'
    )

    print(f"Saved to: {output_path}")
    return output_path

def reencode_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Invalid folder: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith("without.mp4"):
            full_path = os.path.join(folder_path, file)
            reencode_with_moviepy(full_path)

# === Example Usage ===
if __name__ == "__main__":
    folder = "test_videos/without_ee"
    reencode_folder(folder)

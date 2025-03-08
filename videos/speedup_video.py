import os
from moviepy.editor import VideoFileClip, vfx
from tqdm import tqdm

def speed_up_videos_in_folder(folder_path, speed_factor=6):
    # Iterate through all files in the folder
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".mp4"):  # Process only mp4 files
            video_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(folder_path, "speed_up")
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"{file_name}")

            # Load the video
            print(f"Processing {file_name}...")
            clip = VideoFileClip(video_path)
            
            # Speed the clip up
            sped_up_clip = clip.fx(vfx.speedx, speed_factor)

            sped_up_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=int(clip.fps * speed_factor)
            )

            print(f"Saved sped-up video to {output_path}\n")

if __name__ == "__main__":
    # Set the folder path to the current script's directory
    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Call the function to speed up videos
    speed_up_videos_in_folder(folder_path)


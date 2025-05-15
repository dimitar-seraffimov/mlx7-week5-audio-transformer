import os
from pydub import AudioSegment

# I assume a folder/file structure like this:
# mlx7-week5-audio-transformer/
#   whisper-speech-to-text/
#     data/
#       m4a/
#         *.m4a -> source folder containing owner's individual audio files
#       wav/
#         *.wav -> destination folder

# source and destination folders
source_folder = "./m4a"
destination_folder = "./wav"

# ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# loop through each .m4a file in the source folder
for filename in os.listdir(source_folder):
  if filename.endswith(".m4a"):
    m4a_path = os.path.join(source_folder, filename)
    wav_filename = filename.replace(".m4a", ".wav")
    wav_path = os.path.join(destination_folder, wav_filename)

    # load and export audio
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")
    
    print(f"Converted {filename} to {wav_filename}")

print("All files have been converted and saved to /wav.")

# audio config
SAMPLE_RATE = 44100
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
CHUNK_DURATION = 0.25  # 1/4 second = 250ms

# whisper model config
WHISPER_MODEL_NAME = "tiny"

# speaker recognition config
OWNER_THRESHOLD = 0.75

# paths
OWNER_M4A_DIR = "m4a" # original owner audio files
OWNER_WAV_DIR = "wav" # converted wav files
PREPROCESSED_OUTPUT_DIR = "preprocessed" # preprocessed spectrograms

# stream config
STREAM_BUFFER_SIZE = 4096  # buffer size for real-time streaming
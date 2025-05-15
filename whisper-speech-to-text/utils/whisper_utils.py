import whisper
from config import WHISPER_MODEL_NAME

class WhisperModel:
  def __init__(self):
    self.model = whisper.load_model(WHISPER_MODEL_NAME)

  # transcribe audio chunk using whisper model
  def transcribe(self, audio_chunk):
    result = self.model.transcribe(audio_chunk, fp16=False, language="en", task="transcribe")
    return result["text"].strip()

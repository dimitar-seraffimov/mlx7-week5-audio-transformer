from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

class SpeakerRecognizer:
  def __init__(self):
    self.encoder = VoiceEncoder()
    self.owner_embeddings = None
  
  # enroll owner by averaging embeddings of multiple audio files
  def enroll_owner(self, audio_files):
    embeddings = []
    for path in audio_files:
      wav = preprocess_wav(path)
      embedding = self.encoder.embed_utterance(wav)
      embeddings.append(embedding)
    self.owner_embedding = np.mean(embeddings, axis=0)

  # predict owner by comparing embeddings with threshold
  def predict_owner(self, audio_chunk, threshold=0.75):
    if isinstance(audio_chunk, str):
      wav = preprocess_wav(audio_chunk)
    else:
      wav = audio_chunk
    embedding = self.encoder.embed_utterance(wav)
    similarity = np.dot(self.owner_embedding, embedding) / (np.linalg.norm(self.owner_embedding) * np.linalg.norm(embedding))
    return similarity > threshold


## Task 2: Fine-Tune Whisper, build a speech-controlled assistant

### Goal for Friday’s Presentation

Demonstrate a live speech-to-text system that listens during the short presentation, transcribes in real time, and watches for 2–3 custom action words coming from me. I need to pre-record

### Intended demonstration features:

    - live speech-to-text transcription of microphone input using Whisper
    - [owner] speaker detection: recognise when I am speaking based on pre-recorded samples and indicate the speaker during the transcription

### How it works? - simplified overview:

    - recorded sample data = no more than 15 minutes of me speaking, I will probably split it in 30x30sec audio files  -> DONE, but didnt have time to complete the rest - I know what needs to be done though!
    - speaker recognition model based on these samples
    - stream live audio to the model:
        - a sliding window over time (audio segmentation)
        - majority voting or threshold smoothing inside the window (need to decide)
        - speaker change detection: when classification flips, assume speaker has changed
        - Whisper itself doesn't do speaker ID - I will need to choose additional speaker model (pyannote-audio, speechbrain, resemblyze)
    - transcribe speech using Whisper
    - match live audio to Owner's voice and prepend [owner] to relevant transcriptions

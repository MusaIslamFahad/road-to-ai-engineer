# Module 24: Audio & Speech Processing

**Phase 10 — Advanced Specialized Topics** | Est. time: 1–1.5 months (full-time) · 2–3 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Process and analyze audio signals using Fourier transforms and mel features
- Build speech recognition (ASR) systems using CTC and Whisper
- Implement text-to-speech (TTS) synthesis
- Classify audio using CNNs on spectrograms
- Apply voice processing: VAD, speaker ID, noise enhancement

---

## Prerequisites

- Module 10: Deep Learning Frameworks (PyTorch)
- Basic linear algebra (Fourier transform uses it)

---

## Topics Covered

### Audio Signal Fundamentals
- Digital audio: sampling rate (8 kHz, 16 kHz, 44.1 kHz), bit depth, mono vs. stereo
- Waveform representation: amplitude over time
- **Fourier Transform**: decompose signal into frequency components
- **Short-Time Fourier Transform (STFT)**: time-frequency representation
- **Spectrogram**: magnitude of STFT; visualize frequency over time
- **Mel-Spectrogram**: frequency axis warped to mel scale (matches human perception)
- **MFCCs** (Mel-Frequency Cepstral Coefficients): compact audio fingerprint; 13–40 coefficients
- **Log-Mel filterbank features**: standard input for modern ASR
- Libraries: `librosa`, `soundfile`, `torchaudio`

### Speech Recognition (ASR)
- **CTC (Connectionist Temporal Classification)**: align output tokens to variable-length audio without forced alignment
- **Attention-based Seq2Seq**: listen-attend-spell (LAS) architecture
- **End-to-End models**: wav2vec 2.0, HuBERT (self-supervised speech representations)
- **OpenAI Whisper**:
  - Log-mel spectrogram input (80-channel, 30 seconds)
  - Transformer encoder-decoder architecture
  - Multitask: transcribe, translate, identify language, timestamps
  - Models: tiny (39M) → base → small → medium → large-v3
  - Fine-tuning Whisper on custom domains with Hugging Face
- Word Error Rate (WER) and Character Error Rate (CER) evaluation

### Text-to-Speech (TTS)
- Traditional: concatenative, parametric
- **Neural TTS pipeline**: text → phonemes → mel-spectrogram → waveform
- **Tacotron 2**: seq2seq with attention; generates mel-spectrogram
- **FastSpeech 2**: non-autoregressive; much faster inference
- **VITS**: end-to-end TTS with normalizing flows (state of the art)
- **Neural Vocoders**: convert mel-spectrograms to audio waveforms
  - WaveNet (autoregressive), WaveGlow (flow-based), HiFi-GAN (GAN-based, fast)
- **Voice Cloning**: SpeechT5, Tortoise-TTS, Bark, XTTS
  - Few-shot cloning: adapt TTS to a speaker with <10 seconds of audio

### Audio Classification
- **Approach**: compute mel-spectrogram → treat as image → CNN or ViT
- **Environmental Sound Classification**: ESC-50, UrbanSound8k datasets
- **Music Genre Classification**: GTZAN dataset
- **Emotion Recognition**: speech emotion from prosody features
- **Pre-trained models**: YAMNet (YouTube AudioSet), PANNs (Pre-trained Audio Neural Networks)
- **AudioCLIP**: audio analogue of CLIP (zero-shot audio classification)

### Music Generation
- **MusicGen** (Meta): transformer-based music generation with text conditioning
- **AudioCraft**: MusicGen + AudioGen + EnCodec in one library
- **Symbolic music**: MIDI generation with transformers (Music Transformer)

### Voice Processing
- **Voice Activity Detection (VAD)**: Silero VAD, WebRTC VAD
- **Speaker Diarization**: who speaks when; pyannote.audio
- **Speaker Identification / Verification**: d-vectors, x-vectors, resemblyzer
- **Speech Enhancement**: noise suppression, dereverb; DeepFilterNet, SEGAN
- **Keyword Spotting**: always-on wake word detection

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Project Ideas

1. **Whisper Transcription App**: fine-tune Whisper on a domain-specific dataset; build a FastAPI service
2. **Audio Classifier**: classify environmental sounds with mel-spectrograms + CNN
3. **TTS Pipeline**: build a complete TTS system with FastSpeech2 + HiFi-GAN
4. **Speaker Diarization**: "who spoke when" in a meeting recording using pyannote.audio

---

## Quick Start: Transcribe Audio with Whisper

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large
result = model.transcribe("audio.mp3")
print(result["text"])
```

```python
# With Hugging Face (supports fine-tuning)
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = asr("audio.mp3", return_timestamps=True)
print(result["text"])
```

---

## Further Reading

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Jurafsky & Martin (free PDF, Chapter 16+ for speech)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) — Free, hands-on

---

## You've Completed Phase 10!

Congratulations — you've covered all 23 modules of the AI Engineer path.

**You are now a Generalist AI Engineer.** 🎓

Review your skills with the [AI Engineer Skills Checklist](../README.md#ai-engineer-skills-checklist) and build your portfolio with the [Advanced Projects](../18-projects-advanced/README.md).

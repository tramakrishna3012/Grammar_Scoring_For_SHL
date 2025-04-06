import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import speech_recognition as sr
from inference import correct_text

def record_audio(filename="output.wav", duration=10, fs=44100):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to finish
    write(filename, fs, audio)
    print("✅ Audio recorded.")

def voice_input_to_text(filename="output.wav"):
    record_audio(filename)
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"⚠️ API error: {e}")
    return ""

def run_voice_app(mode):
    while True:
        input_text = voice_input_to_text()
        if input_text.lower() == "exit":
            break
        corrected = correct_text(input_text)
        print("✅ Corrected:", corrected)
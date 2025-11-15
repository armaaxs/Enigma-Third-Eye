#!/usr/bin/env python3
"""
AI Camera – Raspberry Pi OS edition
Live CV window + voice prompts + Gemini + TTS
FIXED: TTS now speaks every time using pyttsx3
"""

import cv2
import time
import base64
import google.generativeai as genai
import os
import threading
import queue
import json
import re
import sys
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3

# ---------- TTS FIXED ----------
class TTSManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def speak(self, txt):
        self.queue.put(txt)  # Non-blocking, just adds to queue

    def _worker(self):
        engine = pyttsx3.init()  # Engine lives in background
        while True:
            txt = self.queue.get()
            engine.say(txt)
            engine.runAndWait()  # Blocks ONLY in worker thread
tts = TTSManager()

# ---------- Gemini ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    sys.exit("Set GEMINI_API_KEY env var or create .env file")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# ---------- globals ----------
prompt_queue = queue.Queue()
latest_frame = None
frame_lock = threading.Lock()
TARGET_W, TARGET_H = 640, 480

# ---------- SR ----------
sr_energy = 1000
sr_pause  = 0.8
sr_phrase = 8
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)
recognizer.energy_threshold = sr_energy
recognizer.pause_threshold  = sr_pause
recognizer.operation_timeout = 10
voice_evt = threading.Event()

# ---------- util ----------
def format_resp(text, max_w=80):
    if not text:
        return "[No response]"
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'\\"', '"', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    lines, cur = [], []
    for w in words:
        if len(' '.join(cur + [w])) <= max_w:
            cur.append(w)
        else:
            lines.append(' '.join(cur))
            cur = [w]
    if cur:
        lines.append(' '.join(cur))
    return '\n'.join(lines)

def analyse(img_path, prompt="""Briefly describe what you see in 3 sentences or less. """):
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        rsp = model.generate_content([{"mime_type": "image/jpeg", "data": data}, prompt])
        return format_resp(rsp.text) if rsp.text else "[Empty response]"
    except Exception as e:
        return f"Analysis error: {e}"

# ---------- threads ----------
def terminal_thread():
    print("\n" + "="*50)
    print("📋 Terminal ready  –  type prompt | 'v' = voice | Ctrl-C quit")
    print("="*50 + "\n")
    while True:
        try:
            inp = input("> ").strip()
            if inp.lower() == 'v':
                voice_evt.set()
            elif inp:
                prompt_queue.put(inp)
                print("⏳ queued\n")
        except (KeyboardInterrupt, EOFError):
            print("\n🛑 terminal exit\n")
            break

def voice_thread():
    print("🎤 Calibrating mic … stay quiet 2 s")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ mic ready\n")
    except Exception as e:
        print("❌ mic failed:", e); return

    while True:
        voice_evt.wait(); voice_evt.clear()
        try:
            print("\n🎤 LISTENING …")
            with mic as source:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=sr_phrase)
            txt = recognizer.recognize_google(audio, language="en-US")
            text = txt+"""
            1. Use only numbers and alphabets
            2. Never use /,*,(,) instead replace it with words
            3. Always use speakable words instead of characters that cant be pronounced 
            4. Talk in a normal casual language
            5. MOST IMP: DO NOT USE PHRASES SUCH AS HERES WHAT I SEE ETC             
            """
            print("✅ heard:", txt, "\n")
            prompt_queue.put(txt)
        except sr.WaitTimeoutError:
            print("⚠️ timeout\n")
        except sr.UnknownValueError:
            print("❌ unclear\n")
        except Exception as e:
            print("❌ voice err:", e, "\n")

# ---------- main loop ----------
def main_loop(interval=60):
    global latest_frame
    # auto-detect camera
    for idx in range(3):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if cap.read()[0]:
            print("Using camera", idx)
            break
        cap.release()
    else:
        sys.exit("No camera found")

    cv2.namedWindow("AI-Cam", cv2.WINDOW_NORMAL)
    threading.Thread(target=terminal_thread, daemon=True).start()
    threading.Thread(target=voice_thread, daemon=True).start()

    last = time.time() - interval
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read fail"); break
        with frame_lock:
            latest_frame = frame.copy()
        cv2.imshow("AI-Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        now = time.time()

        # on-demand prompt
        try:
            pr = prompt_queue.get_nowait()
            fname = f"on_demand_{int(now)}.jpg"
            with frame_lock:
                cv2.imwrite(fname, cv2.resize(latest_frame, (TARGET_W, TARGET_H)))
            print("\n" + "="*60)
            print("🔍 ON-DEMAND:", fname)
            resp = analyse(fname, pr)
            print("🤖", resp)
            print("="*60 + "\n")
            tts.speak(resp)
        except queue.Empty:
            pass

        # auto-capture
        if now - last >= interval:
            fname = f"auto_{count:03d}.jpg"
            cv2.imwrite(fname, cv2.resize(frame, (TARGET_W, TARGET_H)))
            print("\n📸 AUTO:", fname)
            resp = analyse(fname, "No filler just pure answer in 2 sentence answer what you see no formatting nothing")
            print("🤖", resp, "\n")
            tts.speak(resp)
            last = now; count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("👋 AI-Cam stopped")

if __name__ == "__main__":
    main_loop()
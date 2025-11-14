import cv2
import time
import base64
import google.generativeai as genai
import os
import threading
import queue
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import subprocess
import sys
import json
import re

# ============= TTS SYSTEM =============
class TTSManager:
    """Manages Text-to-Speech for consistent audio output"""
    def __init__(self):
        self.use_windows_tts = sys.platform == "win32"
        if not self.use_windows_tts:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
    
    def speak(self, text):
        """Speak text using best available method"""
        if not text or len(text.strip()) == 0:
            return
        
        try:
            if self.use_windows_tts:
                # Use Windows built-in TTS (more reliable)
                subprocess.run([
                    'PowerShell', '-Command',
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$speak.Speak("{text.replace('"', '\\"')}")'
                ], timeout=30)
            else:
                # Use pyttsx3 for other systems
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")

tts_manager = TTSManager()

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Thread-safe communication
prompt_queue = queue.Queue()
latest_frame = None
frame_lock = threading.Lock()

# Image settings
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Voice recognition globals
voice_recognizer = sr.Recognizer()
voice_microphone = sr.Microphone()
voice_activation_event = threading.Event()

# Fine-tune microphone sensitivity
voice_recognizer.energy_threshold = 1000
voice_recognizer.pause_threshold = 0.8
voice_recognizer.phrase_time_limit = 8


def format_output(text, max_width=80):
    """Format API response cleanly without JSON glitches"""
    if not text:
        return "[No response]"
    
    # Try to parse as JSON if it looks like JSON
    if text.strip().startswith('{') or text.strip().startswith('['):
        try:
            parsed = json.loads(text)
            # Pretty print JSON with proper formatting
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            return formatted
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, clean the text
            pass
    
    # Clean up text: remove extra escape characters, newlines, etc
    text = text.replace('\\n', '\n').replace('\\"', '"').replace("\\", "")
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    
    # Word wrap for better display
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)



def analyze_frame(image_path, prompt="Briefly describe what you see"):
    """Analyze frame with Gemini"""
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            prompt
        ])

        if not response.text:
            return "Error: No response text generated (possibly blocked)"
        
        # Clean and format the response
        clean_text = format_output(response.text)
        return clean_text

    except Exception as e:
        return f"Error analyzing frame: {str(e)}"


def terminal_input_thread():
    """Continuously listen for terminal input or voice trigger"""
    print("\n" + "=" * 50)
    print("📋 INPUT MODE ACTIVATED")
    print("Options:")
    print("  • Type a prompt + Enter → Text input")
    print("  • Type 'v' + Enter → Voice input mode")
    print("  • Press Ctrl+C to stop input mode")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Enter prompt (or 'v' for voice): ").strip()

            if user_input.lower() == 'v':
                print("\n🎤 Activating voice recognition...")
                voice_activation_event.set()
            elif user_input:
                prompt_queue.put(user_input)
                print("⏳ Prompt queued - analyzing next available frame...\n")

        except (KeyboardInterrupt, EOFError):
            print("\n🛑 Terminal input mode stopped.")
            break


def voice_input_thread():
    """Background thread for voice recognition (on-demand)"""

    print("\n🎤 Calibrating microphone for ambient noise... (2 seconds)")
    try:
        with voice_microphone as source:
            voice_recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Microphone calibration complete. Voice input ready.\n")
    except Exception as e:
        print(f"❌ Microphone setup failed: {e}")
        print("Voice input will be unavailable.\n")
        return

    while True:
        # Wait for activation signal
        voice_activation_event.wait()
        voice_activation_event.clear()

        try:
            # Listen to microphone
            print("\n🎤 LISTENING... Speak your prompt now")
            with voice_microphone as source:
                audio = voice_recognizer.listen(source, timeout=10, phrase_time_limit=8)
            print("⏳ Processing speech...")

            # Transcribe
            prompt_text = voice_recognizer.recognize_google(audio, language="en-US")
            if prompt_text.strip():
                print(f"✅ Heard: \"{prompt_text}\"\n")
                prompt_queue.put(prompt_text)
                print("⏳ Prompt queued for analysis...\n")

        except sr.WaitTimeoutError:
            print("⚠️  No speech detected (timeout)\n")
        except sr.UnknownValueError:
            print("❌ Could not understand audio\n")
        except sr.RequestError as e:
            print(f"❌ Speech service error: {e}\n")
        except Exception as e:
            print(f"❌ Unexpected voice error: {str(e)}\n")


def show_camera_feed_with_capture(capture_interval=60):
    """Main camera loop with triple capture modes (auto, text, voice)"""
    global latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()

    # Start both input threads as daemons
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    voice_thread_instance = threading.Thread(target=voice_input_thread, daemon=True)

    input_thread.start()
    voice_thread_instance.start()

    print(f"\n🎥 Camera feed started. Press 'q' in camera window to quit.")
    print(f"📸 Auto-capture every {capture_interval} seconds")
    print(f"⌨️  Terminal input: Type prompts or 'v' for voice mode\n")

    last_capture_time = time.time() - capture_interval
    photo_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        with frame_lock:
            latest_frame = frame.copy()

        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

        current_time = time.time()

        # Process any pending user prompts
        try:
            user_prompt = prompt_queue.get_nowait()

            filename = f"on_demand_{int(current_time)}.jpg"

            # Resize and save frame
            with frame_lock:
                resized_frame = cv2.resize(latest_frame, (TARGET_WIDTH, TARGET_HEIGHT))
                cv2.imwrite(filename, resized_frame)

            print(f"\n{'='*60}")
            print(f"🔍 [ON-DEMAND] {filename}")
            print(f"{'='*60}")
            description = analyze_frame(filename, prompt=user_prompt)
            print(f"🤖 RESPONSE:\n{description}")
            print(f"{'='*60}\n")

            # TTS for on-demand (extract clean text if JSON)
            tts_text = description
            if description.strip().startswith('{') or description.strip().startswith('['):
                try:
                    parsed = json.loads(description)
                    if isinstance(parsed, dict):
                        tts_text = ' '.join(str(v) for v in parsed.values())
                    elif isinstance(parsed, list):
                        tts_text = ' '.join(str(item) for item in parsed)
                except:
                    pass
            tts_manager.speak(tts_text[:500])  # Limit TTS to 500 chars

        except queue.Empty:
            pass

        # Auto-capture if interval has passed
        if current_time - last_capture_time >= capture_interval:
            filename = f"capture_{photo_count:03d}.jpg"

            # Resize and save frame
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            cv2.imwrite(filename, resized_frame)

            print(f"\n{'='*60}")
            print(f"📸 [AUTO-CAPTURE] {filename}")
            print(f"{'='*60}")
            description = analyze_frame(filename, prompt="Describe what you see in three descriptive sentences")
            print(f"🤖 RESPONSE:\n{description}")
            print(f"{'='*60}\n")

            # TTS for auto-capture (extract clean text if JSON)
            tts_text = description
            if description.strip().startswith('{') or description.strip().startswith('['):
                try:
                    parsed = json.loads(description)
                    if isinstance(parsed, dict):
                        tts_text = ' '.join(str(v) for v in parsed.values())
                    elif isinstance(parsed, list):
                        tts_text = ' '.join(str(item) for item in parsed)
                except:
                    pass
            tts_manager.speak(tts_text[:500])  # Limit TTS to 500 chars

            last_capture_time = current_time
            photo_count += 1

        time.sleep(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🚪 Quitting camera feed...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Camera feed stopped. Total auto-captures: {photo_count}")


if __name__ == "__main__":
    show_camera_feed_with_capture()
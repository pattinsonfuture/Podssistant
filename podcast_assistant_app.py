import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
import asyncio
import threading
import queue
from audio_handler import AudioHandler
from speech_to_text_service import SpeechToTextService
from language_model_service import LanguageModelService
from wake_word_detector import WakeWordDetector
import configparser
import os


class PodcastAssistantApp:
    def __init__(self):
        self._setup_logging()
        self.logger = logging.getLogger("PodcastAssistant")
        self.logger.info("Initializing Podcast Assistant")

        self.root = tk.Tk()
        self.root.title("Podcast Assistant")
        self.root.geometry("800x600")

        self._create_ui()
        self._load_config()
        self._init_services()

        self.is_recording = False
        self.full_transcription = ""
        self.audio_queue = queue.Queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.logger.info("Application initialized")

    def _setup_logging(self):
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            'podcast_assistant.log',
            encoding='utf-8'
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler,
                logging.StreamHandler()
            ]
        )

    def _create_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Transcription section
        ttk.Label(main_frame, text="Podcast Transcription:").pack(anchor=tk.W)
        self.transcription_text = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, height=10)
        self.transcription_text.pack(fill=tk.X, pady=(0, 10))

        # Q&A section
        ttk.Label(main_frame, text="Question & Answer:").pack(anchor=tk.W)

        # Question input
        self.question_text = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, height=3)
        self.question_text.pack(fill=tk.X, pady=(0, 5))
        self.question_text.insert(tk.END, "Type your question here...")

        # Ask button
        self.ask_button = ttk.Button(
            main_frame, text="Ask AI", command=self._ask_ai)
        self.ask_button.pack(anchor=tk.E, pady=(0, 10))

        # Answer display
        self.answer_text = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, height=10, bg="#f0f0f0")
        self.answer_text.pack(fill=tk.X, pady=(0, 10))

        # Audio controls
        audio_frame = ttk.Frame(main_frame)
        audio_frame.pack(fill=tk.X)

        self.device_combobox = ttk.Combobox(audio_frame, state="readonly")
        self.device_combobox.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.record_button = ttk.Button(
            audio_frame, text="Start Recording", command=self._toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 5))

        self.status_label = ttk.Label(audio_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT)

        # Populate audio devices
        self._populate_audio_devices()

    def _load_config(self):
        self.config = configparser.ConfigParser()
        if not os.path.exists('config.ini'):
            messagebox.showerror(
                "Configuration Error",
                "config.ini not found. Please create one from config_template.ini")
            self.root.destroy()
            return

        self.config.read('config.ini')
        self.logger.info("Configuration loaded")

    def _init_services(self):
        # Initialize audio handler
        self.audio_handler = AudioHandler(
            logging.getLogger("PodcastAssistant.Audio"))

        # Initialize speech to text service
        audio_format = self.audio_handler.get_audio_format()
        self.speech_service = SpeechToTextService(
            logging.getLogger("PodcastAssistant.Speech"), audio_format)

        # Configure speech service
        if not self.speech_service.configure(
            self.config['GoogleSpeech']['CredentialsPath']
        ):
            messagebox.showerror(
                "Initialization Error",
                "Failed to configure speech service. Check logs for details.")
            self.record_button.config(state=tk.DISABLED)

        # Initialize language model service
        self.language_model = LanguageModelService(
            logging.getLogger("PodcastAssistant.Language"))

        # Configure language model
        if not self.language_model.configure(
            self.config['DeepSeek']['ApiKey']
        ):
            messagebox.showwarning(
                "Configuration Warning",
                "Azure OpenAI not configured. Q&A functionality will be disabled.")
            self.ask_button.config(state=tk.DISABLED)

        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(
            logging.getLogger("PodcastAssistant.WakeWord"))

        # Configure wake word detector
        if not self.wake_word_detector.configure(
            self.config['WakeWord']['ModelPath']
        ):
            messagebox.showwarning(
                "Configuration Warning",
                "Wake word detection not configured. Manual recording only.")

        # Start wake word detection in background
        threading.Thread(
            target=self._start_wake_word_detection,
            daemon=True
        ).start()

    def _populate_audio_devices(self):
        devices = AudioHandler.list_audio_devices()
        if not devices:
            messagebox.showerror(
                "Audio Error",
                "No audio input devices found. Recording disabled.")
            self.record_button.config(state=tk.DISABLED)
            return

        self.device_combobox['values'] = [d['name'] for d in devices]

        # 優先選擇立體聲混音設備
        stereo_mix_index = next((i for i, d in enumerate(devices)
                                 if d.get('is_stereo_mix', False)), 0)
        self.device_combobox.current(stereo_mix_index)
        self.logger.info(f"Audio devices populated: {len(devices)} found")

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        selected_device = self.device_combobox.current()
        if selected_device == -1:
            messagebox.showerror("Error", "No audio device selected")
            return

        if self.audio_handler.start_recording(selected_device):
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self._update_status("Recording...")

            # Start audio processing thread
            threading.Thread(
                target=self._process_audio,
                daemon=True
            ).start()

            # Start speech recognition
            if self.speech_service.start_recognition():
                self.logger.info("Recording and transcription started")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to start speech recognition. Check logs for details.")
                self._stop_recording()

    def _stop_recording(self):
        self.is_recording = False
        self.audio_handler.stop_recording()
        self.speech_service.stop_recognition()
        self.record_button.config(text="Start Recording")
        self._update_status("Ready")
        self.logger.info("Recording stopped")

    def _process_audio(self):
        while self.is_recording:
            try:
                audio_data = self.audio_handler.audio_queue.get(timeout=0.1)
                if audio_data is None:  # Stop signal
                    break

                self.speech_service.push_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.exception("Error processing audio:")
                self._update_status(f"Error: {str(e)}")
                break

    def _ask_ai(self):
        question = self.question_text.get("1.0", tk.END).strip()
        if not question or question == "Type your question here...":
            messagebox.showinfo("Input Needed", "Please enter a question")
            return

        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, "Thinking...")

        # Get last 2000 characters as context
        context = self.full_transcription[-2000:]
        if not context:
            messagebox.showinfo(
                "Context Needed",
                "No transcription available. Please record some audio first.")
            return

        # Run in background to avoid blocking UI
        threading.Thread(
            target=self._get_ai_response,
            args=(question, context),
            daemon=True
        ).start()

    def _get_ai_response(self, question, context):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            response = loop.run_until_complete(
                self.language_model.get_response(question, context))

            self.root.after(0, lambda: self._display_response(response))
        except Exception as e:
            self.logger.exception("Error getting AI response:")
            self.root.after(
                0, lambda: self._display_response(f"Error: {str(e)}"))

    def _display_response(self, response):
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, response)

    def _start_wake_word_detection(self):
        import time  # Add missing import
        while True:
            try:
                if not hasattr(self, 'is_recording') or not self.is_recording:
                    asyncio.run(self.wake_word_detector.start_detection())
                time.sleep(1)  # Add delay between detection attempts
            except Exception as e:
                self.logger.error(f"Wake word detection error: {str(e)}")

    def _update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def _on_close(self):
        self.logger.info("Application closing")
        if self.is_recording:
            self._stop_recording()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PodcastAssistantApp()
    app.run()

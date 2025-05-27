from google.cloud import speech_v1 as speech
import logging
import queue
import threading
from typing import Optional, Dict, Callable
import io


class SpeechToTextService:
    def __init__(self, logger, audio_format: Dict[str, int]):
        self.logger = logger.getChild(self.__class__.__name__)
        self.audio_format = audio_format
        self.speech_config = None
        self.push_stream = None
        self.audio_config = None
        self.recognizer = None
        self.is_recording = False
        self.audio_queue = queue.Queue()  # 添加音頻隊列
        self.transcription_queue = queue.Queue()
        self.callbacks = {
            'intermediate': None,
            'final': None,
            'error': None,
            'session_end': None
        }
        self.logger.info("SpeechToTextService initialized")

    def configure(self, credentials_path: str):
        try:
            print(f"Using credentials from: {credentials_path}")
            self.client = speech.SpeechClient.from_service_account_file(
                credentials_path)

            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.audio_format['samplerate'],
                language_code="en-US",
                audio_channel_count=self.audio_format['channels'],
                enable_automatic_punctuation=True,
                model="latest_long"
            )

            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=True
            )

            self.logger.info("Google Speech API configured successfully")
            return True
        except Exception as e:
            self.logger.exception("Error configuring Google Speech API:")
            return False

    def register_callback(self, event_type: str, callback: Callable):
        if event_type in self.callbacks:
            self.callbacks[event_type] = callback
            self.logger.debug(f"Registered callback for {event_type}")
        else:
            self.logger.warning(f"Invalid callback type: {event_type}")

    def _process_response(self, responses):
        try:
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                if result.is_final:
                    self.transcription_queue.put(('final', transcript))
                    if self.callbacks['final']:
                        self.callbacks['final'](transcript)
                else:
                    self.transcription_queue.put(('intermediate', transcript))
                    if self.callbacks['intermediate']:
                        self.callbacks['intermediate'](transcript)
        except Exception as e:
            error_msg = f"Error processing speech response: {str(e)}"
            self.logger.error(error_msg)
            self.transcription_queue.put(('error', error_msg))
            if self.callbacks['error']:
                self.callbacks['error'](error_msg)
            self.is_recording = False

    def start_recognition(self):
        if not self.client:
            self.logger.error("Google Speech client not initialized")
            return False

        try:
            self.audio_generator = self._audio_generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in self.audio_generator
            )

            self.responses = self.client.streaming_recognize(
                config=self.streaming_config,
                requests=requests
            )

            self.processing_thread = threading.Thread(
                target=self._process_response,
                args=(self.responses,),
                daemon=True
            )
            self.processing_thread.start()

            self.logger.info("Google Speech recognition started")
            return True
        except Exception as e:
            self.logger.exception("Error starting Google Speech recognition:")
            return False

    def stop_recognition(self):
        if not self.recognizer or not self.is_recording:
            self.logger.info("Recognition stop called but not active")
            return

        try:
            self.recognizer.stop_continuous_recognition_async()
            self.logger.info("Speech recognition stopped")
        except Exception as e:
            self.logger.exception("Error stopping speech recognition:")

    def _audio_generator(self):
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:  # Stop signal
                    break
                yield audio_data.tobytes()
            except queue.Empty:
                continue

    def push_audio(self, audio_data):
        try:
            self.audio_queue.put(audio_data)
            return True
        except Exception as e:
            self.logger.exception("Error queuing audio data:")
            return False

    def get_transcription(self, block: bool = False, timeout: Optional[float] = None):
        try:
            return self.transcription_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

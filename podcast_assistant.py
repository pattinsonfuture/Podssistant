import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import sounddevice as sd
import numpy as np
import threading
import queue
import os
import configparser # Added for configuration management
import time # For WWD loop sleep
import logging # For file logging
import logging.handlers # For potential rotating file handler later, not used yet

# Attempt to import Azure SDKs and OpenAI SDK
try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Azure Speech SDK not installed. Speech-to-Text and Wake Word will be disabled. "
          "Please install with: pip install azure-cognitiveservices-speech", flush=True)
    speechsdk = None

try:
    from openai import AzureOpenAI, APIError as OpenAI_APIError
except ImportError:
    print("OpenAI SDK (version >= 1.0.0) not installed. Q&A functionality will be disabled. "
          "Please install with: pip install openai>=1.0.0", flush=True)
    AzureOpenAI = None
    OpenAI_APIError = None

class AudioHandler:
    def __init__(self, logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.audio_queue = queue.Queue()
        self.input_stream = None
        self.is_recording = False
        self.selected_samplerate = None
        self.selected_channels = None
        self.logger.info("AudioHandler initialized.")

    @staticmethod
    def list_audio_devices():
        # Logger is not available in static method directly, so use global print or pass logger if needed
        # For this application structure, logging from static methods might be less critical
        # or would require a global logger instance.
        if sd is None: 
            print("Sounddevice SDK not available for listing devices.", flush=True)
            # logging.getLogger('PodcastAssistant.AudioHandler').error("Sounddevice SDK not available for listing devices.") # Example if global logger existed
            return []
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device.get('max_input_channels', 0) > 0:
                    input_devices.append({'index': i, 'name': f"{i}: {device['name']} (In: {device['max_input_channels']})"})
            return input_devices
        except Exception as e:
            print(f"Error listing audio devices: {e}", flush=True) # Keep print for static method
            # logging.getLogger('PodcastAssistant.AudioHandler').error(f"Error listing audio devices: {e}", exc_info=True)
            return []

    def _audio_callback(self, indata, frames, time, status):
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self, device_index):
        if self.is_recording:
            self.logger.info("Audio recording already in progress.")
            return False
        try:
            device_info = sd.query_devices(device_index)
            self.selected_samplerate = int(device_info.get('default_samplerate', 44100))
            self.selected_channels = device_info.get('max_input_channels', 1)
            if self.selected_channels == 0: self.selected_channels = 1
            
            self.logger.info(f"Selected audio device {device_index}. SR: {self.selected_samplerate}, CH: {self.selected_channels}")

            self.input_stream = sd.InputStream(
                samplerate=self.selected_samplerate,
                device=device_index,
                channels=self.selected_channels,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.input_stream.start()
            self.is_recording = True
            self.logger.info(f"Recording started on device {device_index}")
            return True
        except Exception as e:
            self.is_recording = False
            self.selected_samplerate = None
            self.selected_channels = None
            self.logger.exception(f"Error starting recording on device {device_index}:")
            return False

    def stop_recording(self):
        if not self.is_recording or self.input_stream is None:
            self.logger.info("Audio stop called but not recording or stream is None.")
            return
        try:
            self.input_stream.stop()
            self.input_stream.close()
            self.is_recording = False
            self.audio_queue.put(None) 
            self.logger.info("Recording stopped.")
        except Exception as e:
            self.logger.exception("Error stopping recording:")
        finally:
            self.input_stream = None
            
    def get_audio_format(self):
        if self.selected_samplerate and self.selected_channels:
            return {'samplerate': self.selected_samplerate, 'channels': self.selected_channels}
        
        self.logger.warning("Audio format not set from current recording. Querying default input device.")
        if sd is None: 
            self.logger.error("Sounddevice SDK not available for fallback format query.")
            return {'samplerate': 44100, 'channels': 1}
        try:
            default_device_idx = sd.default.device[0] 
            if default_device_idx != -1 :
                device_info_list = sd.query_devices() 
                if default_device_idx < len(device_info_list): 
                    device_info = device_info_list[default_device_idx]
                    if device_info and isinstance(device_info, dict):
                        sr = int(device_info.get('default_samplerate', 44100))
                        ch = int(device_info.get('max_input_channels', 1))
                        if ch == 0 : ch = 1 
                        self.logger.info(f"Using default input device '{device_info.get('name', 'Unknown')}' format: SR={sr}, CH={ch}")
                        return {'samplerate': sr, 'channels': ch}
        except Exception as e:
            self.logger.exception("Error querying default input device for fallback audio format:")
        
        self.logger.warning("Fallback audio format to 44100 Hz, 1 channel as absolute fallback.")
        return {'samplerate': 44100, 'channels': 1}


class SpeechToTextService:
    def __init__(self, speech_key, speech_region, get_audio_format_callback, stt_results_queue, stt_errors_queue, logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.get_audio_format_callback = get_audio_format_callback
        self.stt_results_queue = stt_results_queue
        self.stt_errors_queue = stt_errors_queue
        
        self.push_stream = None
        self.speech_recognizer = None
        self.audio_processor_thread = None
        self._processing_audio_active = threading.Event()
        self.is_configured = False
        self._event_connections = []


        if not speechsdk:
            self._put_error("STT Error: Azure Speech SDK not available.")
            self.logger.error("Azure Speech SDK not available.")
            return
        if not self.speech_key or self.speech_key == "YOUR_SPEECH_KEY_HERE" or \
           not self.speech_region or self.speech_region == "YOUR_SPEECH_REGION_HERE":
            self._put_error("STT Error: Azure Speech Key/Region not configured in config.ini.")
            self.logger.warning("Azure Speech Key/Region not configured in config.ini.")
            return 
        self.is_configured = True
        self.logger.info("Initialized and configured.")

    def _put_error(self, message):
        self.logger.error(f"STT_Service: {message}") # Add service context to log
        if self.stt_errors_queue: self.stt_errors_queue.put(message)


    def _initialize_recognizer(self):
        if not self.is_configured: 
            self._put_error("STT Error: Not configured, cannot initialize recognizer.")
            return False

        audio_format_info = self.get_audio_format_callback()
        if not audio_format_info or not audio_format_info.get('samplerate') or not audio_format_info.get('channels'):
            self._put_error("STT Error: Audio format (samplerate/channels) not available for STT initialization.")
            self.logger.error("Audio format not available for STT initialization.")
            return False
        
        samplerate = audio_format_info['samplerate']
        channels = audio_format_info['channels']
        self.logger.info(f"STT initializing with SR: {samplerate}, CH: {channels}")
        
        audio_format_sdk = speechsdk.audio.AudioStreamFormat(samples_per_second=samplerate, bits_per_sample=16, channels=channels)
        self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format_sdk)
        
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # Store event connections to allow for proper disconnection
        self._event_connections = [
            self.speech_recognizer.recognizing.connect(self._recognizing_handler),
            self.speech_recognizer.recognized.connect(self._recognized_handler),
            self.speech_recognizer.session_started.connect(lambda evt: self.logger.info(f'STT SESSION STARTED: {evt.session_id}')),
            self.speech_recognizer.session_stopped.connect(self._session_stopped_handler),
            self.speech_recognizer.canceled.connect(self._canceled_handler)
        ]
        self.logger.info("STT Recognizer initialized and event handlers connected.")
        return True

    def _recognizing_handler(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            self.logger.debug(f"STT Intermediate: {evt.result.text[:30]}...")
            if self.stt_results_queue: self.stt_results_queue.put({'type': 'intermediate', 'text': evt.result.text})

    def _recognized_handler(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.logger.info(f"STT Final: {evt.result.text}")
            if self.stt_results_queue: self.stt_results_queue.put({'type': 'final', 'text': evt.result.text})
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            self.logger.info("STT NoMatch.")
            if self.stt_results_queue: self.stt_results_queue.put({'type': 'no_match', 'text': ''}) 

    def _session_stopped_handler(self, evt: speechsdk.SessionEventArgs): 
        self.logger.info(f'STT SESSION STOPPED: {evt.session_id}')
        self._processing_audio_active.clear() 
        if self.push_stream: self.push_stream.close() 

    def _canceled_handler(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs): 
        self.logger.error(f'STT CANCELED: {evt.reason} (SessionId: {evt.session_id}) Details: {evt.error_details}')
        self._processing_audio_active.clear() 
        error_details = f"Reason: {evt.reason}."
        if evt.reason == speechsdk.CancellationReason.Error:
            error_details += f" ErrorCode: {evt.error_code}. Details: {evt.error_details}."
            if evt.error_code == speechsdk.CancellationErrorCode.ConnectionFailure: 
                error_details += " Check network/firewall and Azure service status."
            elif evt.error_code == speechsdk.CancellationErrorCode.ServiceUnavailable:
                 error_details += " Azure Speech Service might be temporarily unavailable."
            elif evt.error_code == speechsdk.CancellationErrorCode.RuntimeError:
                 error_details += " A runtime error occurred in the Speech SDK."
        self._put_error(f"STT Cancellation: {error_details}")
        if self.push_stream: self.push_stream.close()

    def _audio_processor_thread_target(self, audio_input_queue_ref):
        self.logger.info("STT audio processor thread started.")
        try:
            while self._processing_audio_active.is_set(): 
                try:
                    audio_chunk = audio_input_queue_ref.get(timeout=0.1) 
                    if audio_chunk is None: 
                        self.logger.info("STT: Sentinel received, stopping audio push.")
                        self._processing_audio_active.clear(); break 
                    pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
                    if self.push_stream and self._processing_audio_active.is_set(): 
                        self.push_stream.write(pcm_data)
                    else: 
                        self.logger.info("STT: Push stream closed or processing stopped, exiting audio thread.")
                        break 
                except queue.Empty: continue 
                except Exception as e:
                    self.logger.exception("Error in STT audio processor loop:")
                    self._put_error(f"STT Audio processing error: {e}")
                    self._processing_audio_active.clear(); break
        finally:
            if self.push_stream: self.push_stream.close() 
            self.logger.info("STT: Audio processor thread finished.")

    def start(self, audio_input_queue_ref):
        if not self.is_configured:
             self._put_error("STT Error: Not starting, service not configured.")
             return

        if not self._initialize_recognizer():
            self.logger.error("STT Service: Failed to initialize recognizer during start."); return
        if self.speech_recognizer is None: 
            self._put_error("STT Error: Recognizer None after init in start().")
            return

        self._processing_audio_active.set() 
        self.audio_processor_thread = threading.Thread(target=self._audio_processor_thread_target, args=(audio_input_queue_ref,), daemon=True)
        self.audio_processor_thread.start()
        self.speech_recognizer.start_continuous_recognition_async()
        self.logger.info("STT Service: Started continuous recognition.")

    def stop(self):
        self.logger.info("STT Service: Stop called.")
        self._processing_audio_active.clear() 

        if self.audio_processor_thread and self.audio_processor_thread.is_alive():
            self.logger.info("STT Service: Waiting for audio processor thread to join...")
            self.audio_processor_thread.join(timeout=1.0) 
            if self.audio_processor_thread.is_alive():
                self.logger.warning("STT Service: Audio processor thread did not join in time.")
        
        if self.speech_recognizer:
            self.logger.info("STT Service: Stopping continuous recognition...")
            try:
                stop_future = self.speech_recognizer.stop_continuous_recognition_async()
                stop_future.get(timeout=1.0) 
                self.logger.info("STT Service: Recognition stopped.")
            except Exception as e: 
                self.logger.exception("STT Service: Error or timeout during stop_continuous_recognition_async:")
                self._put_error(f"STT Stop Error: {e}")
            finally: 
                if hasattr(self, '_event_connections'): 
                    for conn in self._event_connections:
                        try: conn.disconnect() 
                        except Exception as ex_disconnect: self.logger.error(f"STT: Error disconnecting event: {ex_disconnect}", exc_info=False)
                    self._event_connections = []
                self.speech_recognizer.dispose()
                self.speech_recognizer = None
                self.logger.info("STT Service: Recognizer disposed.")
        
        self.push_stream = None 
        self.logger.info("STT Service: Stop completed.")

    def pause(self):
        if self.is_configured and self._processing_audio_active.is_set():
            self.logger.info("STT Service: Pausing audio processing.")
            self._processing_audio_active.clear()

    def resume(self):
        if self.is_configured and self.speech_recognizer and self.push_stream:
            if not self._processing_audio_active.is_set(): 
                self.logger.info("STT Service: Resuming audio processing.")
                self._processing_audio_active.set()
        else:
            msg = "STT Service: Cannot resume, not configured or recognizer not ready."
            self.logger.warning(msg)
            if self.is_configured: self._put_error(msg) 

class LanguageModelService:
    def __init__(self, api_key, azure_endpoint, api_version, deployment_name, logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.client = None
        self.is_configured = False 

        if not AzureOpenAI:
            self.logger.error("OpenAI SDK not installed. Q&A functionality will be disabled.")
            return
            
        if not (not self.api_key or self.api_key == "YOUR_AZURE_OPENAI_KEY_HERE" or
                not self.azure_endpoint or self.azure_endpoint == "YOUR_AZURE_OPENAI_ENDPOINT_HERE" or
                not self.deployment_name or self.deployment_name == "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME_HERE" or
                not self.api_version):
            try:
                self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.azure_endpoint, api_version=self.api_version)
                self.is_configured = True
                self.logger.info("Initialized and configured.")
            except Exception as e: 
                self.logger.exception("Error initializing AzureOpenAI client. Check config.ini.")
        else: 
            self.logger.warning("Azure OpenAI credentials not fully configured in config.ini.")

    def get_response(self, user_question, transcript_context):
        if not self.is_configured or not self.client: 
            self.logger.error("LLM Service get_response called but not configured.")
            return "Error: LLM Service not configured. Check config.ini."
        self.logger.info(f"LLM: Getting response for question: '{user_question[:30]}...'")
        system_message = ("You are a helpful AI assistant. Answer based *only* on the provided podcast transcript snippet. "
                          "If the answer isn't in the snippet, clearly state that.")
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": f"Podcast Snippet:\n\"\"\"\n{transcript_context}\n\"\"\"\n\nUser's Question:\n\"\"\"\n{user_question}\n\"\"\""}]
        try:
            response = self.client.chat.completions.create(model=self.deployment_name, messages=messages, max_tokens=300, temperature=0.3)
            if response.choices and response.choices[0].message:
                 resp_content = response.choices[0].message.content or "AI response empty."
                 self.logger.info(f"LLM Response: {resp_content[:50]}...")
                 return resp_content
            self.logger.warning("No response content received from AI.")
            return "No response content received from AI."
        except OpenAI_APIError as e: # type: ignore
            self.logger.exception("Azure OpenAI API Error in get_response:")
            err_msg = f"AI Error: {e.message}" + (f" (Code: {e.status_code})" if hasattr(e, 'status_code') else "")
            return err_msg
        except Exception as e: 
            self.logger.exception("Unexpected error in LLM get_response:")
            return f"Unexpected AI error: {e}"

class WakeWordDetector:
    def __init__(self, speech_key, speech_region, model_path, detection_queue, logger): 
        self.logger = logger.getChild(self.__class__.__name__)
        self.speech_key = speech_key 
        self.speech_region = speech_region 
        self.model_path = model_path
        self.detection_queue = detection_queue
        self.keyword_recognizer = None
        self.keyword_model = None
        self.audio_config_wwd = None 
        self.is_detecting = False
        self.is_configured = False 
        self._recognition_task_future = None 
        self._event_connections = []


        if not speechsdk:
            self._put_error("WWD: Azure Speech SDK missing."); return
        
        are_keys_placeholders_wwd = (not self.speech_key or self.speech_key == "YOUR_WWD_SPEECH_KEY_HERE" or self.speech_key == "YOUR_SPEECH_KEY_HERE" or
                                 not self.speech_region or self.speech_region == "YOUR_WWD_SPEECH_REGION_HERE" or self.speech_region == "YOUR_SPEECH_REGION_HERE")

        if are_keys_placeholders_wwd:
            self._put_error("WWD Error: Azure Speech Key/Region not configured in config.ini for Wake Word.")
        
        try:
            if not os.path.exists(self.model_path):
                self._put_error(f"WWD Error: Model '{self.model_path}' not found. Check path in config.ini."); return
            self.keyword_model = speechsdk.KeywordRecognitionModel(self.model_path)
            self.audio_config_wwd = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self.keyword_recognizer = speechsdk.KeywordRecognizer(self.audio_config_wwd)
            
            self.is_configured = not are_keys_placeholders_wwd 
            if not self.is_configured:
                self.logger.warning("WWD Initialized with placeholder keys. Spoken question capture will fail.")
            self.logger.info(f"Initialized (model loaded). Fully configured for spoken questions: {self.is_configured}")

        except Exception as e: 
            self._put_error(f"WWD Init Error: {e}")
            self.is_configured = False


    def _put_error(self, message):
        self.logger.error(f"WWD: {message}")
        if self.detection_queue: self.detection_queue.put({"type": "error", "message": message})

    def _wwd_recognized_handler(self, evt: speechsdk.KeywordRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedKeyword:
            self.logger.info(f"WAKE WORD DETECTED: '{evt.result.text}'")
            if self.keyword_recognizer:
                 self.keyword_recognizer.stop_recognition_async() 
            self.is_detecting = False 
            if self.keyword_recognizer: 
                try:
                    for conn in self._event_connections: conn.disconnect()
                    self._event_connections = []
                except Exception as e: self.logger.warning(f"WWD: Error deregistering handlers: {e}", exc_info=False)
            self.detection_queue.put({"type": "WAKE_WORD_DETECTED", "keyword": evt.result.text})


    def _wwd_canceled_handler(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        reason_desc = f"WWD CANCELED: Reason={evt.reason}"
        if evt.reason == speechsdk.CancellationReason.Error:
            reason_desc += f" ErrCode={evt.error_code}. Details='{evt.error_details}'"
        self._put_error(reason_desc); self.is_detecting = False
        if self.keyword_recognizer: 
            try:
                for conn in self._event_connections: conn.disconnect()
                self._event_connections = []
            except Exception: pass

    def _start_single_recognition_wwd(self): 
        if not self.keyword_recognizer or not self.keyword_model: 
            self._put_error("WWD: Recognizer or model not loaded. Cannot start recognition.")
            return
        try:
            if self.keyword_recognizer: # Re-connect handlers each time we start
                for conn in self._event_connections: conn.disconnect() # Clear old ones
                self._event_connections = [
                    self.keyword_recognizer.recognized.connect(self._wwd_recognized_handler),
                    self.keyword_recognizer.canceled.connect(self._wwd_canceled_handler)
                ]
            self.keyword_recognizer.start_keyword_recognition_async(self.keyword_model)
            self.logger.info("WakeWordDetector: start_keyword_recognition_async called.")
        except Exception as e:
            self._put_error(f"WWD Start Reco Error: {e}"); self.is_detecting = False

    def start_detection(self):
        if not self.keyword_model or not self.keyword_recognizer: 
             self._put_error("WWD: Not initialized (recognizer/model). Cannot start."); return
        if self.is_detecting: self.logger.info("WWD: Already detecting."); return
        
        self.is_detecting = True
        self.logger.info("WWD: Starting keyword recognition.")
        self._start_single_recognition_wwd() 

    def stop_detection(self):
        if not self.keyword_recognizer : 
            self.is_detecting = False; return 
        
        was_detecting = self.is_detecting 
        self.is_detecting = False 
        self.logger.info("WWD: Attempting to stop detection...")
        
        try:
            if was_detecting: 
                stop_future = self.keyword_recognizer.stop_recognition_async()
                self.logger.info("WWD: stop_recognition_async called.")
                stop_future.get(timeout=1.5) 
                self.logger.info("WWD: Keyword recognition stopped successfully.")
        except Exception as e: 
            self.logger.exception(f"WWD Stop Error or Timeout:")
        finally: 
            if self.keyword_recognizer:
                try:
                    for conn in self._event_connections: conn.disconnect()
                    self._event_connections = []
                    self.logger.debug("WWD: Event handlers disconnected during stop.")
                except Exception as e_de: self.logger.warning(f"WWD: Error deregistering handlers during stop: {e_de}")
        self.logger.info("WWD: Stop detection process completed.")


    def listen_for_single_question_async(self):
        if not speechsdk: 
            self._put_error("WWD: Azure Speech SDK missing for question listening.")
            return "Error: Speech SDK not available."
        if not self.is_configured: 
            self._put_error("WWD: Not configured for question listening (check WWD keys/region in config.ini).")
            return "Error: Speech service not configured for question."

        speech_config_q = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        audio_config_q = speechsdk.audio.AudioConfig(use_default_microphone=True) 
        
        temp_speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config_q, audio_config=audio_config_q)
        self.logger.info("Listening for spoken question...")
        try:
            question_text = None 
            result = temp_speech_recognizer.recognize_once_async().get(timeout=10) 
            if result.reason == speechsdk.ResultReason.RecognizedSpeech: question_text = result.text
            elif result.reason == speechsdk.ResultReason.NoMatch: question_text = "" 
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                question_text = f"Error: Question reco canceled ({cancellation_details.reason})"
                self.logger.error(f"Question recognition canceled: {cancellation_details.reason}, Details: {cancellation_details.error_details}")
        except Exception as e: 
            question_text = f"Exception during question capture: {e}"
            self.logger.exception("Exception during single question recognition:")
        finally: 
            if temp_speech_recognizer: 
                try: temp_speech_recognizer.stop_recognition_async().get(timeout=0.5) 
                except: pass 
                temp_speech_recognizer.dispose()
        self.logger.info(f"Spoken question result: '{question_text}'")
        return question_text

    def dispose_resources(self): 
        self.logger.info("WWD: Disposing resources...")
        if self.keyword_recognizer: 
            for conn in self._event_connections: conn.disconnect()
            self._event_connections = []
            self.keyword_recognizer.dispose(); self.keyword_recognizer = None
        if self.audio_config_wwd: self.audio_config_wwd = None 
        self.keyword_model = None; self.is_detecting = False
        self.logger.info("WWD: Resources disposed.")
    def dispose(self):
        self.logger.info("WWD: Dispose called.")
        self.stop_detection() 
        self.dispose_resources()


class PodcastAssistantApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Podcast AI Assistant")
        self.root.geometry("750x820") 

        # --- Logging Setup ---
        self.logger = logging.getLogger('PodcastAssistant')
        self.logger.setLevel(logging.DEBUG) 
        log_file_path = 'podcast_assistant.log'
        # Using 'w' to overwrite log on each start, change to 'a' to append.
        try:
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("--- Application Starting ---")
        except Exception as e:
            print(f"FATAL: Could not configure logging to file {log_file_path}: {e}", flush=True)
            # No logger available here to log this specific error, so print.

        # --- Configuration Loading ---
        self.config = configparser.ConfigParser()
        self.config_path = 'config.ini' 
        self.config_template_path = 'config_template.ini' 
        self.config_load_error = None 

        if not os.path.exists(self.config_path):
            self.config_load_error = (f"ERROR: '{self.config_path}' not found. "
                       f"Please copy '{self.config_template_path}' to '{self.config_path}' and fill in your API keys/details.")
            self.logger.error(self.config_load_error)
            self.config = None 
        elif not self.config.read(self.config_path):
            self.config_load_error = f"ERROR: Failed to parse '{self.config_path}'. Please check its format."
            self.logger.error(self.config_load_error)
            self.config = None
        else:
            self.logger.info(f"'{self.config_path}' loaded successfully.")


        self.audio_handler = AudioHandler(logger=self.logger) # Pass logger
        self.recording_thread = None 
        self.stt_results_queue = queue.Queue() 
        self.stt_errors_queue = queue.Queue()  

        # --- Service Initializations ---
        stt_speech_key_fb = "YOUR_SPEECH_KEY_HERE"
        stt_speech_region_fb = "YOUR_SPEECH_REGION_HERE"
        stt_speech_key = self.config.get('AzureSpeech', 'SubscriptionKey', fallback=stt_speech_key_fb) if self.config else stt_speech_key_fb
        stt_speech_region = self.config.get('AzureSpeech', 'Region', fallback=stt_speech_region_fb) if self.config else stt_speech_region_fb

        self.stt_service = None
        if speechsdk: 
            self.stt_service = SpeechToTextService(
                speech_key=stt_speech_key,
                speech_region=stt_speech_region,
                get_audio_format_callback=self.audio_handler.get_audio_format,
                stt_results_queue=self.stt_results_queue, 
                stt_errors_queue=self.stt_errors_queue,
                logger=self.logger 
            )
        else:
            self.logger.error("STT Service disabled: Azure Speech SDK not available.")

        self.llm_service = None 
        if AzureOpenAI: 
            llm_api_key_fb = "YOUR_AZURE_OPENAI_KEY_HERE"
            llm_endpoint_fb = "YOUR_AZURE_OPENAI_ENDPOINT_HERE"
            llm_api_version_fb = "2023-07-01-preview" 
            llm_deployment_fb = "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME_HERE"
            
            llm_api_key = self.config.get('AzureOpenAI', 'SubscriptionKey', fallback=llm_api_key_fb) if self.config else llm_api_key_fb
            llm_endpoint = self.config.get('AzureOpenAI', 'AzureEndpoint', fallback=llm_endpoint_fb) if self.config else llm_endpoint_fb
            llm_api_version = self.config.get('AzureOpenAI', 'ApiVersion', fallback=llm_api_version_fb) if self.config else llm_api_version_fb
            llm_deployment = self.config.get('AzureOpenAI', 'DeploymentName', fallback=llm_deployment_fb) if self.config else llm_deployment_fb
            
            self.llm_service = LanguageModelService(
                api_key=llm_api_key, azure_endpoint=llm_endpoint,
                api_version=llm_api_version, deployment_name=llm_deployment,
                logger=self.logger 
            )
        else:
            self.logger.error("Language Model Service disabled: OpenAI SDK not available.")
        
        self.wake_word_detector = None
        self.wake_word_event_queue = queue.Queue()
        if speechsdk:
            wwd_model_path_fb = "Hi_pod.table"
            wwd_speech_key_fb = "YOUR_WWD_SPEECH_KEY_HERE" 
            wwd_speech_region_fb = "YOUR_WWD_SPEECH_REGION_HERE" 

            wwd_model_path = self.config.get('WakeWord', 'ModelPath', fallback=wwd_model_path_fb) if self.config else wwd_model_path_fb
            cfg_wwd_key = self.config.get('WakeWord', 'WakeWordSpeechKey', fallback=wwd_speech_key_fb) if self.config else wwd_speech_key_fb
            cfg_wwd_region = self.config.get('WakeWord', 'WakeWordSpeechRegion', fallback=wwd_speech_region_fb) if self.config else wwd_speech_region_fb

            key_to_use_for_wwd = cfg_wwd_key if cfg_wwd_key not in ["YOUR_WWD_SPEECH_KEY_HERE", "", None] else \
                                 (stt_speech_key if stt_speech_key not in ["YOUR_SPEECH_KEY_HERE", "", None] else "YOUR_WWD_SPEECH_KEY_HERE")
            
            region_to_use_for_wwd = cfg_wwd_region if cfg_wwd_region not in ["YOUR_WWD_SPEECH_REGION_HERE", "", None] else \
                                    (stt_speech_region if stt_speech_region not in ["YOUR_SPEECH_REGION_HERE", "", None] else "YOUR_WWD_SPEECH_REGION_HERE")
            
            self.wake_word_detector = WakeWordDetector(
                speech_key=key_to_use_for_wwd, 
                speech_region=region_to_use_for_wwd, 
                model_path=wwd_model_path,
                detection_queue=self.wake_word_event_queue,
                logger=self.logger 
            )
            if self.wake_word_detector.is_configured: 
                self.wake_word_detector.start_detection()
            else:
                self.logger.warning("Wake Word Detector not started due to configuration/initialization issues.")
        else:
            self.logger.error("Wake Word Detector disabled: Azure Speech SDK not available.")

        self.llm_results_queue = queue.Queue() 
        self.full_transcription = "" 
        self.context_for_spoken_question = ""
        self.is_listening_for_question = False

        self._setup_ui() 
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self._check_all_queues_and_update_ui)
        self.logger.info("PodcastAssistantApp initialized.")

    def _setup_ui(self):
        # --- Style Configuration ---
        style = ttk.Style()
        style.configure("TLabel", padding=2)
        style.configure("TButton", padding=5)
        style.configure("TLabelFrame", padding=(5,5), relief="groove") 
        style.configure("Status.TLabel", font=("TkDefaultFont", 9)) 
        style.configure("Note.TLabel", foreground="dimgray", font=("TkDefaultFont", 8)) 
        style.configure("Header.TLabel", font=("TkDefaultFont", 10, "bold"))


        # --- Audio Control Area ---
        audio_frame = ttk.LabelFrame(self.root, text="Audio Capture", padding=(10, 10)) 
        audio_frame.pack(padx=10, pady=(10,5), fill="x", expand=False) 
        audio_frame.columnconfigure(1, weight=1) 

        ttk.Label(audio_frame, text="Input Device:").grid(row=0, column=0, sticky="w", padx=(0,5), pady=2)
        
        self.device_var = tk.StringVar()
        self.device_combobox = ttk.Combobox(audio_frame, textvariable=self.device_var, state="readonly", width=60)
        self.device_combobox.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2) 
        
        self.populate_device_combobox()

        button_frame_audio = ttk.Frame(audio_frame) 
        button_frame_audio.grid(row=1, column=0, columnspan=3, pady=(5,5), sticky="ew")

        self.start_listening_button = ttk.Button(button_frame_audio, text="Start Listening", command=self.handle_start_listening)
        self.start_listening_button.pack(side=tk.LEFT, padx=(0,5))
        
        self.stop_listening_button = ttk.Button(button_frame_audio, text="Stop Listening", command=self.handle_stop_listening, state=tk.DISABLED)
        self.stop_listening_button.pack(side=tk.LEFT)

        self.status_label_var = tk.StringVar(value="Status: Initializing...") 
        status_label = ttk.Label(audio_frame, textvariable=self.status_label_var, wraplength=680, justify=tk.LEFT, style="Status.TLabel") 
        status_label.grid(row=2, column=0, columnspan=3, pady=(5,0), sticky="w")
        
        loopback_note = ttk.Label(audio_frame, text="Note: For system audio (loopback), select the appropriate 'Stereo Mix', 'Monitor', or similar device.\nEnsure it's enabled in your OS audio settings.", justify=tk.LEFT, style="Note.TLabel")
        loopback_note.grid(row=3, column=0, columnspan=3, pady=(5,5), sticky="w") 

        # --- Transcription Area ---
        transcription_frame = ttk.LabelFrame(self.root, text="Podcast Transcription", padding=(10,5))
        transcription_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.transcription_text_area = scrolledtext.ScrolledText(transcription_frame, wrap=tk.WORD, height=10) 
        self.transcription_text_area.pack(fill="both", expand=True)
        self._update_text_widget(self.transcription_text_area, "Podcast transcription will appear here...", overwrite=True, scroll_to_end=True)


        # --- User Question Area ---
        question_outer_frame = ttk.Frame(self.root) 
        question_outer_frame.pack(padx=10, pady=5, fill="x", expand=False)
        
        question_frame = ttk.LabelFrame(question_outer_frame, text="Your Question", padding=(10,5))
        question_frame.pack(fill="x", expand=True)
        question_frame.columnconfigure(0, weight=1) 

        self.question_entry = scrolledtext.ScrolledText(question_frame, wrap=tk.WORD, height=4) 
        self.question_entry.grid(row=0, column=0, sticky="ew", pady=(0,5))
        self.question_entry.insert(tk.END, "Type your question here...")
        self.question_entry.config(fg="grey") 
        self.question_entry.bind("<FocusIn>", self._clear_placeholder_question)
        self.question_entry.bind("<FocusOut>", self._restore_placeholder_question)
        self.question_entry.bind("<KeyRelease>", self._on_question_change) 
        
        self.clear_question_button = ttk.Button(question_frame, text="Clear", command=self.clear_question_text, width=8)
        self.clear_question_button.grid(row=0, column=1, sticky="ne", padx=(5,0))


        # --- Ask AI Button --- 
        ask_button_frame = ttk.Frame(self.root) 
        ask_button_frame.pack(pady=5)
        self.ask_ai_button = ttk.Button(ask_button_frame, text="Ask AI", command=self.handle_ask_ai, style="Accent.TButton") 
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold")) 
        self.ask_ai_button.pack()
        
        # --- AI Answer Area ---
        answer_frame = ttk.LabelFrame(self.root, text="AI Answer", padding=(10,5))
        answer_frame.pack(padx=10, pady=5, fill="both", expand=True)
        answer_frame.columnconfigure(0, weight=1) 
        answer_frame.rowconfigure(0, weight=1) 

        self.ai_answer_text_area = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD, height=7)
        self.ai_answer_text_area.grid(row=0, column=0, sticky="nsew") 
        self._update_text_widget(self.ai_answer_text_area, "AI's answer will appear here...", overwrite=True, scroll_to_end=True)

        self.clear_answer_button = ttk.Button(answer_frame, text="Clear", command=self.clear_answer_text, width=8)
        self.clear_answer_button.grid(row=0, column=1, sticky="ne", padx=(5,0)) 
        
        # --- Setup Instructions ---
        setup_instructions_text = (
            f"SETUP: Create `{self.config_path}` from `{self.config_template_path}` and fill in your Azure Speech "
            "Key/Region (for STT & Wake Word), Azure OpenAI Key/Endpoint/Deployment Name (for Q&A), "
            f"and the correct path to your `Hi_pod.table` model file."
        )
        setup_label = ttk.Label(self.root, text=setup_instructions_text, wraplength=730, justify=tk.CENTER, foreground="#404040", font=("TkDefaultFont", 8, "italic"), style="Note.TLabel")
        setup_label.pack(pady=(10,10), padx=10, fill="x", side=tk.BOTTOM) 

        self._update_initial_status_and_button_states() 

    def _update_text_widget(self, text_widget, content, overwrite=True, scroll_to_end=True): 
        current_state = text_widget.cget('state')
        text_widget.config(state=tk.NORMAL)
        if overwrite: 
            text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
        if current_state == tk.DISABLED: 
            text_widget.config(state=tk.DISABLED)
        if scroll_to_end: text_widget.see(tk.END) 
    def populate_device_combobox(self):
        devices = self.audio_handler.list_audio_devices()
        device_names = [dev['name'] for dev in devices]
        self.device_combobox['values'] = device_names
        if device_names:
            self.device_combobox.current(0) 
            for i, name in enumerate(device_names):
                if "stereo mix" in name.lower() or "loopback" in name.lower() or "monitor" in name.lower():
                    self.device_combobox.current(i); break
        else:
            self.device_combobox.set("No input devices found")
            if hasattr(self, 'start_listening_button'): 
                self.start_listening_button.config(state=tk.DISABLED)

    def _update_initial_status_and_button_states(self): 
        initial_status_parts = ["Status:"]
        notes = []
        
        if self.config_load_error: # Check if config.ini itself had loading errors
            notes.append(self.config_load_error.replace("ERROR: ", "")) # Show specific load error
        else: # Config loaded (or self.config is None if file not found initially handled by constructor)
            sdk_missing_stt_wwd = False
            if not speechsdk: 
                notes.append("AzureSpeechSDK missing (STT & WWD)")
                sdk_missing_stt_wwd = True
            
            sdk_missing_llm = False
            if 'AzureOpenAI' not in globals() or globals()['AzureOpenAI'] is None:
                notes.append("OpenAI SDK missing (Q&A)")
                sdk_missing_llm = True

            if not sdk_missing_stt_wwd:
                if not self.stt_service or not self.stt_service.is_configured: 
                    notes.append("STT not configured (check config.ini)")
                
                if not self.wake_word_detector or not self.wake_word_detector.is_configured:
                    # Check model path directly from config if WWD object exists
                    if self.wake_word_detector: # Check if object exists first
                        wwd_model_path_check = self.wake_word_detector.model_path 
                        if not os.path.exists(wwd_model_path_check): 
                            notes.append(f"WWD Model '{wwd_model_path_check}' Missing")
                        else: # Model exists, issue is likely keys if not configured
                             notes.append("WWD Key/Region in config.ini")
                    else: # WWD service itself is None (e.g. SDK missing or init failed before self.wake_word_detector assigned)
                        notes.append("WWD service not available/init failed")
                # Check for placeholder keys even if is_configured is True (local model might allow this)
                elif self.wake_word_detector.is_configured and \
                     (self.wake_word_detector.speech_key.startswith("YOUR_") or \
                      self.wake_word_detector.speech_region.startswith("YOUR_")):
                     notes.append("WWD Key/Region in config.ini (using placeholders for spoken Q&A)") # Clarify impact
            
            if not sdk_missing_llm:
                if not self.llm_service or not self.llm_service.is_configured:
                    notes.append("OpenAI not configured (check config.ini)")
        
        if notes: 
            initial_status_parts.append("Issues: " + ", ".join(sorted(list(set(notes)))))
        else: 
            initial_status_parts.append("All services appear configured.")
            if self.wake_word_detector and self.wake_word_detector.is_configured and self.wake_word_detector.is_detecting:
                initial_status_parts.append("Listening for 'Hi pod'...")
            else:
                 initial_status_parts.append("Ready.") 
        
        final_status = " ".join(initial_status_parts)
        if not final_status.endswith("."): final_status += "."
        self.status_label_var.set(final_status)
        self.logger.info(f"Initial status: {final_status}")

        # Set button states based on configurations
        if hasattr(self, 'start_listening_button'): 
            if not self.stt_service or not self.stt_service.is_configured: 
                self.start_listening_button.config(state=tk.DISABLED)
            else: 
                 if not self.audio_handler.is_recording: 
                    self.start_listening_button.config(state=tk.NORMAL)
        
        if hasattr(self, 'ask_ai_button'): self._on_question_change() 


    def _recording_target(self):
            selected_device_name = self.device_var.get()
            try: device_index = int(selected_device_name.split(':')[0])
            except (ValueError, IndexError) as e:
                self.logger.error(f"Invalid device name format: {selected_device_name}", exc_info=True)
                self.status_label_var.set("Status: Error - Invalid device selected.")
                self.start_listening_button.config(state=tk.NORMAL); self.stop_listening_button.config(state=tk.DISABLED)
                return
            if not self.audio_handler.start_recording(device_index): 
                self.logger.error("Failed to start audio capture in _recording_target.")
                self.status_label_var.set("Status: Error - Failed to start audio capture.")
                self.start_listening_button.config(state=tk.NORMAL)
                self.stop_listening_button.config(state=tk.DISABLED)
        except Exception as e:
            self.logger.exception("Exception in _recording_target:")
            self.status_label_var.set(f"Status: Error - {e}")
            self.start_listening_button.config(state=tk.NORMAL); self.stop_listening_button.config(state=tk.DISABLED)

    def handle_start_listening(self):
        self.logger.info("Start Listening button clicked.")
        if not self.device_var.get() or self.device_var.get() == "No input devices found":
            self.status_label_var.set("Status: Please select an audio device."); return
        
        if not self.stt_service or not self.stt_service.is_configured: 
            self.status_label_var.set("Status: STT Service not configured. Check config.ini.")
            tk.messagebox.showerror("STT Error", "Speech-to-Text service is not properly configured. Please check config.ini or install Azure Speech SDK.")
            self.logger.warning("STT service not started - not configured.")
            return

        # Stop wake word detection if it's running
        if self.wake_word_detector and self.wake_word_detector.is_detecting:
            self.logger.info("Stopping wake word detection to start STT/Recording.")
            self.wake_word_detector.stop_detection() 

        self.start_listening_button.config(state=tk.DISABLED)
        self.stop_listening_button.config(state=tk.NORMAL)
        self.device_combobox.config(state=tk.DISABLED)
        self.status_label_var.set("Status: Listening for podcast...")
        
        self._update_text_widget(self.transcription_text_area, "[Transcription starting...]", overwrite=True, scroll_to_end=True)
        self.full_transcription = "" 

        while not self.audio_handler.audio_queue.empty(): 
            try: self.audio_handler.audio_queue.get_nowait()
            except queue.Empty: break
        
        self.recording_thread = threading.Thread(target=self._recording_target, daemon=True)
        self.recording_thread.start()
        if self.stt_service: self.stt_service.start(self.audio_handler.audio_queue) 

    def handle_stop_listening(self, restart_wwd=True): 
        self.logger.info(f"Stop Listening called. Restart WWD: {restart_wwd}")
        self.audio_handler.stop_recording() 
        if self.stt_service: self.stt_service.stop() 

        self.start_listening_button.config(state=tk.NORMAL)
        self.stop_listening_button.config(state=tk.DISABLED)
        self.device_combobox.config(state=tk.NORMAL)
        self.status_label_var.set("Status: Stopped. Processing transcription...") 
        
        if restart_wwd and self.wake_word_detector and self.wake_word_detector.is_configured and not self.wake_word_detector.is_detecting: 
             self.logger.info("Restarting wake word detection after podcast STT stop.")
             self.wake_word_detector.start_detection()
    
    def clear_question_text(self):
        self.logger.info("Clear Question button clicked.")
        self._update_text_widget(self.question_entry, "Type your question here...", True, False)
        self.question_entry.config(fg="grey") 
        self._on_question_change() 

    def clear_answer_text(self):
        self.logger.info("Clear Answer button clicked.")
        self._update_text_widget(self.ai_answer_text_area, "AI's answer will appear here...", True, True) 

    def _on_question_change(self, event=None): 
        question_text = self.question_entry.get("1.0", tk.END).strip()
        is_placeholder = question_text == "Type your question here..." 
        has_text = bool(question_text) and not is_placeholder

        if hasattr(self, 'ask_ai_button'): 
            if self.llm_service and self.llm_service.is_configured:
                is_processing_llm = self.ask_ai_button.cget('text') != "Ask AI" 
                if has_text and not is_processing_llm and not self.is_listening_for_question :
                    self.ask_ai_button.config(state=tk.NORMAL)
                else:
                    self.ask_ai_button.config(state=tk.DISABLED)
            else:
                self.ask_ai_button.config(state=tk.DISABLED)


    def _ask_llm_thread_target(self, user_question, transcript_context):
        self.logger.info(f"LLM Thread: Sending question: '{user_question[:30]}...' Context len: {len(transcript_context)}")
        if self.llm_service: 
            response = self.llm_service.get_response(user_question, transcript_context)
            self.llm_results_queue.put(response)
        else: 
            self.llm_results_queue.put("LLM Service not available or not configured.")


    def handle_ask_ai(self, context_override=None): 
        self.logger.info(f"handle_ask_ai called. Spoken Q mode: {self.is_listening_for_question}")
        if not self.llm_service or not self.llm_service.is_configured:
            self.status_label_var.set("Status: LLM Service not configured.")
            tk.messagebox.showwarning("Azure OpenAI Configuration", 
                                     "Azure OpenAI Service is not configured. Check config.ini.")
            self.logger.warning("handle_ask_ai: LLM service not configured.")
            return

        user_question = self.question_entry.get("1.0", tk.END).strip()
        if not user_question or user_question == "Type your question here...":
            if not self.is_listening_for_question: 
                self.status_label_var.set("Status: Please enter a question.")
                tk.messagebox.showinfo("Input Needed", "Please type your question in the text box.")
            self.logger.info("handle_ask_ai: No question entered.")
            return

        transcript_context = context_override if context_override is not None else self.full_transcription
        
        snippet_max_length = 2000 
        if len(transcript_context) > snippet_max_length and not self.is_listening_for_question: # Only snip if not from spoken Q
            self.logger.info(f"Context snippet taken from full transcript. Original length: {len(transcript_context)}, Snippet length: {snippet_max_length}")
            transcript_context = transcript_context[-snippet_max_length:]
        
        if not transcript_context.strip() and not user_question.strip(): # If both are empty, it's an issue
            self.status_label_var.set("Status: No transcript or question.") 
            tk.messagebox.showinfo("Input Needed", "No transcription context or question provided.")
            self.logger.info("handle_ask_ai: No context or question provided.")
            return

        self.ask_ai_button.config(state=tk.DISABLED, text="Asking...") 
        self._update_text_widget(self.ai_answer_text_area, "AI is thinking...", True, True)
        self.status_label_var.set("Status: Asking AI...")
        self.logger.info(f"Submitting question to LLM: '{user_question[:50]}...'")

        threading.Thread(
            target=self._ask_llm_thread_target,
            args=(user_question, transcript_context),
            daemon=True
        ).start()

    def _clear_placeholder_question(self, event=None): 
        if self.question_entry.get("1.0", tk.END).strip() == "Type your question here...":
            self._update_text_widget(self.question_entry, "", True, False) 
            self.question_entry.config(fg="black") 

    def _restore_placeholder_question(self, event=None): 
        if not self.question_entry.get("1.0", tk.END).strip():
            self._update_text_widget(self.question_entry, "Type your question here...", True, False)
            self.question_entry.config(fg="grey") 

    def _capture_and_process_spoken_question(self):
        if not self.wake_word_detector: 
            self.logger.warning("WWD not available for spoken question capture.")
            self.root.after(0, self._resume_normal_listening) 
            return
        
        self.logger.info("Thread: Attempting to listen for a single question...")
        question_text = self.wake_word_detector.listen_for_single_question_async()
        self.logger.info(f"Thread: Spoken question capture result: '{question_text}'")

        if question_text and not question_text.startswith("Error:") and not question_text.startswith("Exception:") and question_text.strip():
            self.root.after(0, self._populate_question_and_ask_llm, question_text)
        else:
            self.root.after(0, self.status_label_var.set, f"Status: Question not heard clearly. {error_msg_short(question_text or 'No input.')}. See log.")
            self.logger.info(f"Spoken question not processed: '{question_text}'")
            self.root.after(100, self._resume_normal_listening) 


    def _populate_question_and_ask_llm(self, question_text):
        self.logger.info(f"Populating spoken question to UI: {question_text[:50]}...")
        self._update_text_widget(self.question_entry, question_text, True, False) 
        self.question_entry.config(fg="black") 
        self._on_question_change() 
        self.handle_ask_ai(context_override=self.context_for_spoken_question) 

    def _resume_normal_listening(self):
        self.logger.info("Resuming normal listening (STT and Wake Word)...")
        self.is_listening_for_question = False
        
        if self.stt_service and self.stt_service.is_configured: 
            if self.stop_listening_button['state'] == tk.NORMAL or self.audio_handler.is_recording: 
                 self.stt_service.resume()
                 self.status_label_var.set("Status: Listening for podcast...")
                 self.logger.info("STT service resumed.")
            else: 
                 self._update_initial_status_and_button_states()
        
        if self.wake_word_detector and self.wake_word_detector.is_configured: 
            if not self.wake_word_detector.is_detecting: 
                self.logger.info("WWD: Calling start_detection from _resume_normal_listening")
                self.wake_word_detector.start_detection() 
        
        self._on_question_change() # Update Ask AI button state
        
        # More robust status update after resuming
        if not self.audio_handler.is_recording:
            if self.wake_word_detector and self.wake_word_detector.is_configured and self.wake_word_detector.is_detecting:
                self.status_label_var.set("Status: Listening for wake word...")
            else:
                self._update_initial_status_and_button_states() # General ready/config state
        elif self.audio_handler.is_recording: # If STT is active
             self.status_label_var.set("Status: Listening for podcast...")


    def _check_all_queues_and_update_ui(self): 
        # Check STT Queue
        if self.stt_service:
            try:
                while not self.stt_service.stt_results_queue.empty():
                    result = self.stt_service.stt_results_queue.get_nowait()
                    self.logger.debug(f"STT queue processing: {result['type']}")
                    
                    if result['type'] == 'intermediate':
                        self._update_text_widget(self.transcription_text_area, self.full_transcription + result['text'], True, True) 
                    elif result['type'] == 'final' and result['text']: 
                        self.full_transcription += result['text'] + " " 
                        self._update_text_widget(self.transcription_text_area, self.full_transcription, True, True)
                    elif result['type'] == 'no_match':
                        self.logger.info("STT NoMatch received in UI loop.")
                        pass 
                    
                while not self.stt_service.stt_errors_queue.empty():
                    error_msg = self.stt_service.stt_errors_queue.get_nowait()
                    self.status_label_var.set(f"Status: STT Error! {error_msg_short(error_msg)}. See log.")
                    # Logger already handled this in _put_error for STTService
            except Exception as e: self.logger.exception("Error processing STT queue:")

        # Check LLM Queue
        if self.llm_service: 
            try:
                while not self.llm_results_queue.empty():
                    llm_response = self.llm_results_queue.get_nowait()
                    self.logger.info(f"LLM response received: {llm_response[:50]}...")
                    self._update_text_widget(self.ai_answer_text_area, llm_response, True, True)
                    self.ask_ai_button.config(text="Ask AI") # Reset button text
                    if not self.is_listening_for_question : 
                        self._on_question_change() 
                        self.status_label_var.set("Status: AI response received. Ready.")
                    # If it was a spoken question, _resume_normal_listening will handle status and button re-enabling for Ask AI
            except Exception as e:
                self.logger.exception("Error processing LLM queue:")
                self.ask_ai_button.config(text="Ask AI") # Reset button text on error too
                if not self.is_listening_for_question: 
                    self._on_question_change()
                    self.status_label_var.set("Status: Error processing AI response. See log.")

        # Check Wake Word Queue
        if self.wake_word_detector:
            try:
                while not self.wake_word_event_queue.empty():
                    event_data = self.wake_word_event_queue.get_nowait() 
                    self.logger.info(f"Wake word queue processing: {event_data}")
                    if event_data.get("type") == "WAKE_WORD_DETECTED": 
                        if not self.is_listening_for_question: 
                            self.is_listening_for_question = True
                            keyword = event_data.get('keyword', 'Wake word') 
                            self.status_label_var.set(f"Status: '{keyword}' detected! Listening for question...") 
                            self.logger.info(f"'{keyword}' detected! Initiating spoken question capture.")
                            self.root.bell() 
                            self._update_text_widget(self.question_entry, "", True, False) 
                            self.question_entry.config(fg="black") 
                            self.question_entry.focus_set() 
                            self._on_question_change() # Disable Ask AI button
                            
                            self.context_for_spoken_question = self.full_transcription 
                            
                            if self.stt_service and self.audio_handler.is_recording:
                                self.logger.info("Pausing STT for spoken question.")
                                self.stt_service.pause()
                            
                            # WWD stops itself after one detection.
                            self.logger.info("Starting thread for _capture_and_process_spoken_question")
                            threading.Thread(target=self._capture_and_process_spoken_question, daemon=True).start()
                    elif event_data.get("type") == "error":
                         error_message = event_data.get('message','Unknown Wake Word error')
                         self.status_label_var.set(f"Status: Wake Word Error! {error_msg_short(error_message)}. See log.")
                         # Logger already handled this in _put_error for WWD
                         if not self.is_listening_for_question and self.wake_word_detector and \
                            self.wake_word_detector.is_configured and not self.wake_word_detector.is_detecting: 
                             self.logger.info("WWD Error detected, attempting to restart WWD.")
                             self.wake_word_detector.start_detection()

            except Exception as e:
                self.logger.exception("Error processing Wake Word queue:")

        self.root.after(100, self._check_all_queues_and_update_ui) 

    def on_closing(self):
        """Handles window closing event to clean up resources."""
        self.logger.info("--- Application Closing ---")
        if self.wake_word_detector: 
            self.wake_word_detector.dispose() 

        if self.audio_handler and self.audio_handler.is_recording:
            self.audio_handler.stop_recording() 

        if self.stt_service : 
            self.stt_service.stop() 
        
        self.root.destroy()
        self.logger.info("--- Application Closed ---")


def error_msg_short(error_details, max_len=60): 
    """Shortens error message for status bar display."""
    if len(error_details) > max_len:
        return error_details[:max_len-3] + "..."
    return error_details

if __name__ == "__main__":
    import os 
    # import asyncio # Not directly used by app for WWD

    app_root = tk.Tk()
    assistant_app = PodcastAssistantApp(app_root) 
    app_root.mainloop()

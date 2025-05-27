import sounddevice as sd
import numpy as np
import threading
import queue
import logging


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
        if sd is None:
            print("Sounddevice SDK not available for listing devices.", flush=True)
            return []
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device.get('max_input_channels', 0) > 0:
                    # 特別標記立體聲混音設備
                    device_name = device['name']
                    is_stereo_mix = "stereo mix" in device_name.lower() or "立體聲混音" in device_name.lower()
                    input_devices.append({
                        'index': i,
                        'name': f"{i}: {device_name} (In: {device['max_input_channels']})",
                        'is_stereo_mix': is_stereo_mix
                    })
            # 將立體聲混音設備排在前面
            input_devices.sort(key=lambda x: not x['is_stereo_mix'])
            return input_devices
        except Exception as e:
            # Keep print for static method
            print(f"Error listing audio devices: {e}", flush=True)
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
            self.selected_samplerate = int(
                device_info.get('default_samplerate', 44100))
            self.selected_channels = device_info.get('max_input_channels', 1)
            if self.selected_channels == 0:
                self.selected_channels = 1

            self.logger.info(
                f"Selected audio device {device_index}. SR: {self.selected_samplerate}, CH: {self.selected_channels}")

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
            self.logger.exception(
                f"Error starting recording on device {device_index}:")
            return False

    def stop_recording(self):
        if not self.is_recording or self.input_stream is None:
            self.logger.info(
                "Audio stop called but not recording or stream is None.")
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

        self.logger.warning(
            "Audio format not set from current recording. Querying default input device.")
        if sd is None:
            self.logger.error(
                "Sounddevice SDK not available for fallback format query.")
            return {'samplerate': 44100, 'channels': 1}
        try:
            default_device_idx = sd.default.device[0]
            if default_device_idx != -1:
                device_info_list = sd.query_devices()
                if default_device_idx < len(device_info_list):
                    device_info = device_info_list[default_device_idx]
                    if device_info and isinstance(device_info, dict):
                        sr = int(device_info.get('default_samplerate', 44100))
                        ch = int(device_info.get('max_input_channels', 1))
                        if ch == 0:
                            ch = 1
                        self.logger.info(
                            f"Using default input device '{device_info.get('name', 'Unknown')}' format: SR={sr}, CH={ch}")
                        return {'samplerate': sr, 'channels': ch}
        except Exception as e:
            self.logger.exception(
                "Error querying default input device for fallback audio format:")

        self.logger.warning(
            "Fallback audio format to 44100 Hz, 1 channel as absolute fallback.")
        return {'samplerate': 44100, 'channels': 1}

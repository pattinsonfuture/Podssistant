import logging
from speech_to_text_service import SpeechToTextService
from language_model_service import LanguageModelService
from wake_word_detector import WakeWordDetector
from audio_handler import AudioHandler

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestServices")


def test_services():
    # 初始化各服務
    audio_format = {'samplerate': 16000, 'channels': 1}

    # 語音轉文字測試
    stt = SpeechToTextService(logger, audio_format)
    stt.configure("client_secret.json")  # 替換為實際JSON路徑

    # 語言模型測試
    lm = LanguageModelService(logger)
    lm.configure("your_deepseek_api_key")  # 替換為實際API金鑰

    # 喚醒詞測試
    wwd = WakeWordDetector(logger)
    wwd.configure("Hi_pod.table")  # 替換為實際模型路徑

    # 音頻處理測試
    audio = AudioHandler(logger)
    audio.configure(audio_format)

    print("所有服務初始化成功！")
    print("請手動測試各服務功能：")
    print("1. 語音轉文字服務")
    print("2. 語言模型響應")
    print("3. 喚醒詞檢測")
    print("4. 音頻錄製與播放")


if __name__ == "__main__":
    test_services()

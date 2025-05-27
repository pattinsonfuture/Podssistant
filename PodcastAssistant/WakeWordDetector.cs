using System;
using System.Diagnostics;
using System.IO; // For Path.Combine if needed for model path
using System.Threading.Tasks;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

namespace PodcastAssistant
{
    public class WakeWordDetector : IDisposable
    {
        // IMPORTANT: Replace with your actual subscription key and region.
        // These can be the same as used in SpeechTranscriber.cs or specific to wake word detection.
        private const string SpeechKey = "YOUR_SPEECH_KEY"; 
        private const string SpeechRegion = "YOUR_SPEECH_REGION";

        // IMPORTANT: The wake word model file (e.g., "Hi_pod.table") must be:
        // 1. Generated using Azure Speech Studio for your custom wake word (e.g., "Hi pod").
        // 2. Added to the PodcastAssistant project.
        // 3. Set to "Copy to Output Directory: PreserveNewest" or "Copy if newer" in its file properties.
        private const string WakeWordModelFileName = "Hi_pod.table"; // Placeholder path

        private KeywordRecognizer? keywordRecognizer;
        private KeywordRecognitionModel? keywordModel;
        private AudioConfig? audioConfig;
        private bool isDetecting = false;

        public event Action<string>? WakeWordDetected; // Event payload can be the detected keyword text
        public event Action<string>? DetectionError;   // Event for errors during detection

        public WakeWordDetector()
        {
            if (string.IsNullOrEmpty(SpeechKey) || SpeechKey == "YOUR_SPEECH_KEY" ||
                string.IsNullOrEmpty(SpeechRegion) || SpeechRegion == "YOUR_SPEECH_REGION")
            {
                DetectionError?.Invoke("Azure Speech Key or Region is not set for WakeWordDetector.");
                // throw new InvalidOperationException("Azure Speech Key or Region not set for WakeWordDetector.");
            }
        }

        public async Task StartDetectionAsync()
        {
            if (isDetecting)
            {
                Debug.WriteLine("WakeWordDetector: Already detecting.");
                return;
            }
            
            if (SpeechKey == "YOUR_SPEECH_KEY" || SpeechRegion == "YOUR_SPEECH_REGION") {
                 var errorMsg = "WakeWordDetector: Azure Speech Key or Region is not configured.";
                 Debug.WriteLine(errorMsg);
                 DetectionError?.Invoke(errorMsg);
                 return;
            }

            try
            {
                // Path to the model file. Assumes it's in the same directory as the executable.
                string modelPath = Path.Combine(AppContext.BaseDirectory, WakeWordModelFileName);

                if (!File.Exists(modelPath))
                {
                    var errorMsg = $"WakeWordDetector: Keyword model file not found at '{modelPath}'. Please ensure '{WakeWordModelFileName}' exists and is copied to the output directory.";
                    Debug.WriteLine(errorMsg);
                    DetectionError?.Invoke(errorMsg);
                    return;
                }

                keywordModel = KeywordRecognitionModel.FromFile(modelPath);
                audioConfig = AudioConfig.FromDefaultMicrophoneInput(); // Uses the default microphone
                
                // SpeechConfig is not directly used for KeywordRecognizer initialization in the same way as SpeechRecognizer.
                // The key/region are implicitly managed if needed by the SDK for model verification or telemetry.
                // However, some SDK versions or specific setups might require it. If issues arise, consult SDK docs.
                // For now, we proceed without explicit SpeechConfig for KeywordRecognizer as per common examples.

                keywordRecognizer = new KeywordRecognizer(audioConfig);
                isDetecting = true;

                Debug.WriteLine("WakeWordDetector: Starting detection...");
                // The keyword recognition starts when RecognizeOnceAsync is called.
                // It will listen until the keyword is recognized or an error occurs.
                // For continuous wake word detection, you typically loop this call or use a session-based approach if available.
                // However, KeywordRecognizer is often used in a one-shot manner for a keyword, then you might start speech recognition.
                // For persistent listening, we'll call it in a loop (with care for resources).
                
                // Start a background task for persistent listening (simplified loop)
                _ = Task.Run(async () => {
                    while(isDetecting && keywordRecognizer != null && keywordModel != null) {
                        try {
                             Debug.WriteLine("WakeWordDetector: Listening for keyword...");
                            KeywordRecognitionResult result = await keywordRecognizer.RecognizeOnceAsync(keywordModel);
                            Debug.WriteLine($"WakeWordDetector: Recognition result: {result.Reason}");

                            if (result.Reason == ResultReason.RecognizedKeyword)
                            {
                                Debug.WriteLine($"WakeWordDetector: Detected keyword: {result.Text}");
                                WakeWordDetected?.Invoke(result.Text);
                                // After detection, you might want to stop or pause wake word detection
                                // and start another process (like speech-to-text).
                                // For this example, we'll let it continue listening unless explicitly stopped.
                                // If you want it to stop after one detection:
                                // await StopDetectionAsync(); 
                                // break; 
                            }
                            else if (result.Reason == ResultReason.Canceled)
                            {
                                var cancellation = CancellationDetails.FromResult(result);
                                Debug.WriteLine($"WakeWordDetector: CANCELED: Reason={cancellation.Reason}");
                                if (cancellation.Reason == CancellationReason.Error)
                                {
                                    Debug.WriteLine($"WakeWordDetector: CANCELED: ErrorCode={cancellation.ErrorCode}");
                                    Debug.WriteLine($"WakeWordDetector: CANCELED: ErrorDetails={cancellation.ErrorDetails}");
                                    DetectionError?.Invoke($"Detection canceled with error: {cancellation.ErrorDetails}");
                                }
                                // If canceled (e.g. by StopDetectionAsync or error), stop the loop.
                                if (isDetecting) await Task.Delay(100); // Brief pause before retrying unless stopping
                                else break;
                            }
                            else if (result.Reason == ResultReason.NoMatch)
                            {
                                // This is expected if no keyword is spoken. Continue listening.
                                // No action needed, just loop again.
                            }
                             // Add a small delay to prevent tight looping if errors occur rapidly
                            if (!isDetecting) break;
                        } catch (Exception ex) {
                            Debug.WriteLine($"WakeWordDetector: Exception in listening loop: {ex.Message}");
                            DetectionError?.Invoke($"Exception in listening loop: {ex.Message}");
                            if (isDetecting) await Task.Delay(1000); // Longer pause if exception
                            else break;
                        }
                    }
                    Debug.WriteLine("WakeWordDetector: Listening loop ended.");
                });


            }
            catch (Exception ex)
            {
                var errorMsg = $"WakeWordDetector: Error starting detection: {ex.Message}. Ensure microphone is available and model path is correct.";
                Debug.WriteLine(errorMsg);
                DetectionError?.Invoke(errorMsg);
                isDetecting = false; // Ensure state is correct
                DisposeResources();
            }
        }

        public async Task StopDetectionAsync()
        {
            if (!isDetecting || keywordRecognizer == null)
            {
                Debug.WriteLine("WakeWordDetector: Not detecting or recognizer is null.");
                return;
            }

            isDetecting = false; // Signal the loop to stop
            Debug.WriteLine("WakeWordDetector: Stopping detection...");
            try
            {
                // The RecognizeOnceAsync call in the loop will eventually complete or cancel.
                // For KeywordRecognizer, stopping is often about disposing resources or not calling RecognizeOnceAsync again.
                // If a session-based recognizer were used, it would have a StopSessionAsync type method.
                // Here, setting isDetecting = false and disposing resources is the primary way to "stop".
                // We might need to wait for the task to complete if it's stuck in RecognizeOnceAsync.
                // However, KeywordRecognizer.RecognizeOnceAsync should return if audio input stops or is disconnected.
                // Forcing a stop can be done by disposing the recognizer.
                DisposeResources(); 
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"WakeWordDetector: Error stopping detection: {ex.Message}");
                DetectionError?.Invoke($"Error stopping detection: {ex.Message}");
            }
            Debug.WriteLine("WakeWordDetector: Detection stopped.");
        }

        private void DisposeResources()
        {
            keywordRecognizer?.Dispose();
            keywordRecognizer = null;
            
            audioConfig?.Dispose(); // AudioConfig should be disposed
            audioConfig = null;
            
            // KeywordRecognitionModel does not implement IDisposable according to documentation.
            keywordModel = null; 
        }

        public void Dispose()
        {
            // Ensure StopDetectionAsync is called to clean up
            // This might need to be awaited if called from a non-async context,
            // but Dispose pattern typically doesn't await.
            if (isDetecting)
            {
                 StopDetectionAsync().GetAwaiter().GetResult(); // Synchronously wait for stop
            }
            DisposeResources(); // General cleanup
            GC.SuppressFinalize(this);
        }
    }
}

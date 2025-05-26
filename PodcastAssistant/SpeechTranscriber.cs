using System;
using System.Diagnostics; // For Debug.WriteLine
using System.Text;
using System.Threading.Tasks;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using NAudio.Wave;

namespace PodcastAssistant
{
    public class SpeechTranscriber : IDisposable
    {
        // IMPORTANT: Replace with your actual subscription key and region.
        private const string SpeechKey = "YOUR_SPEECH_KEY"; // e.g. "0123456789abcdef0123456789abcdef"
        private const string SpeechRegion = "YOUR_SPEECH_REGION"; // e.g. "westeurope"

        private SpeechRecognizer? speechRecognizer;
        private PushAudioInputStream? pushStream;
        private readonly WaveFormat naudioWaveFormat;

        public event Action<string>? IntermediateTranscription;
        public event Action<string>? FinalTranscriptionSegment;
        public event Action<string>? TranscriptionError;
        public event Action<string>? RecognitionSessionEnded; // Unified event for session stop/cancel

        public SpeechTranscriber(WaveFormat inputWaveFormat)
        {
            if (string.IsNullOrEmpty(SpeechKey) || SpeechKey == "YOUR_SPEECH_KEY")
            {
                TranscriptionError?.Invoke("Azure Speech Key is not set. Please update SpeechTranscriber.cs.");
                // throw new ArgumentException("Azure Speech Key is not set."); // Or handle more gracefully
            }
            if (string.IsNullOrEmpty(SpeechRegion) || SpeechRegion == "YOUR_SPEECH_REGION")
            {
                 TranscriptionError?.Invoke("Azure Speech Region is not set. Please update SpeechTranscriber.cs.");
                // throw new ArgumentException("Azure Speech Region is not set."); // Or handle more gracefully
            }
            this.naudioWaveFormat = inputWaveFormat;
        }

        public PushAudioInputStream? InitializePushStreamAndConfig()
        {
            if (SpeechKey == "YOUR_SPEECH_KEY" || SpeechRegion == "YOUR_SPEECH_REGION") {
                 TranscriptionError?.Invoke("Azure Speech Key or Region is not configured in SpeechTranscriber.cs.");
                 return null;
            }

            try
            {
                var speechConfig = SpeechConfig.FromSubscription(SpeechKey, SpeechRegion);
                // Optional: speechConfig.SpeechRecognitionLanguage = "en-US";
                // Optional: speechConfig.SetProfanity(ProfanityOption.Raw); // To see if profanity is filtered/masked

                var audioFormat = AudioStreamFormat.GetWaveFormatPCM(
                    (uint)naudioWaveFormat.SampleRate,
                    (byte)naudioWaveFormat.BitsPerSample,
                    (byte)naudioWaveFormat.Channels);
                
                pushStream = AudioInputStream.CreatePushStream(audioFormat);
                var audioConfig = AudioConfig.FromStreamInput(pushStream);

                speechRecognizer = new SpeechRecognizer(speechConfig, audioConfig);

                speechRecognizer.Recognizing += (s, e) =>
                {
                    if (e.Result.Reason == ResultReason.RecognizingSpeech)
                    {
                        Debug.WriteLine($"RECOGNIZING: Text={e.Result.Text}");
                        IntermediateTranscription?.Invoke(e.Result.Text);
                    }
                };

                speechRecognizer.Recognized += (s, e) =>
                {
                    if (e.Result.Reason == ResultReason.RecognizedSpeech)
                    {
                        Debug.WriteLine($"RECOGNIZED: Text={e.Result.Text}");
                        FinalTranscriptionSegment?.Invoke(e.Result.Text);
                        // FUTURE ENHANCEMENT: Investigate using e.Result.OffsetInTicks and e.Result.Duration 
                        // to get word/phrase level timestamps. This data, if stored with the transcript, 
                        // could enable more precise context windowing based on when the user asked their 
                        // question relative to the podcast audio.
                    }
                    else if (e.Result.Reason == ResultReason.NoMatch)
                    {
                        Debug.WriteLine($"NOMATCH: Speech could not be recognized.");
                        // Optionally invoke an event or log this.
                    }
                };

                speechRecognizer.Canceled += (s, e) =>
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine($"CANCELED: Reason={e.Reason}");
                    if (e.Reason == CancellationReason.Error)
                    {
                        sb.AppendLine($"CANCELED: ErrorCode={e.ErrorCode}");
                        sb.AppendLine($"CANCELED: ErrorDetails=[{e.ErrorDetails}]");
                        sb.AppendLine($"CANCELED: Did you update the Speech Key and Region in SpeechTranscriber.cs?");
                    }
                    Debug.WriteLine(sb.ToString());
                    TranscriptionError?.Invoke(sb.ToString());
                    RecognitionSessionEnded?.Invoke($"Recognition Canceled: {e.Reason}");
                };

                speechRecognizer.SessionStopped += (s, e) =>
                {
                    Debug.WriteLine($"SESSION STOPPED");
                    RecognitionSessionEnded?.Invoke("Recognition Session Stopped.");
                };
                
                speechRecognizer.SessionStarted += (s, e) => 
                {
                    Debug.WriteLine($"SESSION STARTED");
                };
                return pushStream;
            }
            catch (Exception ex)
            {
                var errorMsg = $"Error initializing SpeechRecognizer: {ex.Message}. Check Speech Key/Region.";
                Debug.WriteLine(errorMsg);
                TranscriptionError?.Invoke(errorMsg);
                return null;
            }
        }

        public async Task StartRecognitionAsync()
        {
            if (speechRecognizer == null)
            {
                TranscriptionError?.Invoke("Speech recognizer not initialized. Call InitializePushStreamAndConfig first.");
                return;
            }
            try
            {
                await speechRecognizer.StartContinuousRecognitionAsync().ConfigureAwait(false);
            }
            catch(Exception ex)
            {
                var errorMsg = $"Error starting recognition: {ex.Message}.";
                Debug.WriteLine(errorMsg);
                TranscriptionError?.Invoke(errorMsg);
            }
        }

        public async Task StopRecognitionAsync()
        {
            if (speechRecognizer != null)
            {
                try
                {
                    await speechRecognizer.StopContinuousRecognitionAsync().ConfigureAwait(false);
                }
                catch(Exception ex)
                {
                    var errorMsg = $"Error stopping recognition: {ex.Message}.";
                    Debug.WriteLine(errorMsg);
                    TranscriptionError?.Invoke(errorMsg);
                }
                // pushStream?.Close() should be called here to signal the end of audio data.
                // This tells the Speech SDK that no more audio is coming.
                pushStream?.Close(); 
            }
        }
        
        public void Dispose()
        {
            if (speechRecognizer != null)
            {
                // It's good practice to ensure StopContinuousRecognitionAsync is called and awaited
                // if a session is active, but direct Dispose is often sufficient for cleanup by SDK.
                // StopRecognitionAsync().GetAwaiter().GetResult(); // Synchronous wait for cleanup if needed
                
                speechRecognizer.SessionStarted -= (s,e) => {}; // Deregister events
                speechRecognizer.SessionStopped -= (s,e) => {};
                speechRecognizer.Recognizing -= (s,e) => {};
                speechRecognizer.Recognized -= (s,e) => {};
                speechRecognizer.Canceled -= (s,e) => {};
                speechRecognizer.Dispose();
                speechRecognizer = null;
            }
            
            // PushStream is owned and disposed by the AudioConfig, 
            // but if we created it, we should close and dispose it.
            // However, SpeechRecognizer's AudioConfig takes ownership.
            // Closing it in StopRecognitionAsync is the correct way to signal end of stream.
            // Explicitly disposing here might be redundant if SDK handles it, but can be safe.
            if (pushStream != null)
            {
                pushStream.Close(); // Ensure it's closed
                pushStream.Dispose();
                pushStream = null;
            }
            GC.SuppressFinalize(this);
        }
    }
}

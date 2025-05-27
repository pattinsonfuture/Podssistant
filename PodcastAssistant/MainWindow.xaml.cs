using System;
using System.Text;
using System.Windows;
using Microsoft.CognitiveServices.Speech.Audio;

namespace PodcastAssistant
{
    public partial class MainWindow : Window, IDisposable
    {
        private readonly AudioRecorder? audioRecorder; 
        private readonly SpeechTranscriber? speechTranscriber; 
        private readonly LanguageModelService? languageModelService; 
        private readonly WakeWordDetector? wakeWordDetector; // Added WakeWordDetector
        private StringBuilder fullTranscription;

        public MainWindow()
        {
            InitializeComponent();
            fullTranscription = new StringBuilder();

            try
            {
                audioRecorder = new AudioRecorder();
                audioRecorder.RecordingFailed += (s, errMsg) => OnAudioRecordingError(errMsg);
                audioRecorder.RecordingStoppedEvent += (s, e) => OnAudioCaptureStopped();

                if (audioRecorder != null) // Ensure audioRecorder initialized before accessing its WaveFormat
                {
                    speechTranscriber = new SpeechTranscriber(audioRecorder.RecordingWaveFormat);
                    SubscribeToTranscriberEvents();
                }
                else // Should not happen if AudioRecorder constructor throws on failure
                {
                     throw new InvalidOperationException("AudioRecorder could not be initialized.");
                }


                languageModelService = new LanguageModelService();
                if (!languageModelService.IsAvailable())
                {
                    AskAiButton.IsEnabled = false;
                    // Combining status messages carefully
                    string currentStatus = RecordingStatusTextBlock.Text;
                    UpdateRecordingStatus(currentStatus + (string.IsNullOrEmpty(currentStatus) ? "" : " | ") + "OpenAI Not Configured.");
                }
                
                // Initialize WakeWordDetector
                wakeWordDetector = new WakeWordDetector();
                SubscribeToWakeWordEvents();
                wakeWordDetector.StartDetectionAsync().ConfigureAwait(false); // Start wake word detection

                // Initial status message
                string initialStatus = "Status: Ready.";
                if (speechTranscriber?.InitializePushStreamAndConfig() == null) // Check Speech SDK config
                {
                    initialStatus += " Speech SDK (Transcription) not configured.";
                }
                if (!languageModelService.IsAvailable())
                {
                    initialStatus += " OpenAI SDK not configured.";
                }
                if (wakeWordDetector == null) // Should not happen unless constructor fails
                {
                    initialStatus += " Wake Word Detector failed to init.";
                }
                UpdateRecordingStatus(initialStatus);


            }
            catch (Exception ex)
            {
                MessageBox.Show($"Fatal error during initialization: {ex.Message}\nApplication might not function correctly.", "Initialization Error", MessageBoxButton.OK, MessageBoxImage.Error);
                UpdateRecordingStatus($"Status: Initialization Error! {ex.Message}");
                StartRecordingButton.IsEnabled = false;
                AskAiButton.IsEnabled = false;
                audioRecorder = null;
                speechTranscriber = null;
                languageModelService = null;
                wakeWordDetector = null; // Ensure it's null if init fails
                return;
            }
        }

        private void SubscribeToTranscriberEvents()
        {
            if (speechTranscriber == null) return;

            speechTranscriber.IntermediateTranscription += (transcription) =>
            {
                Dispatcher.Invoke(() =>
                {
                    TranscriptionTextBlock.Text = fullTranscription.ToString() + " " + transcription + "...";
                });
            };
            speechTranscriber.FinalTranscriptionSegment += (segment) =>
            {
                Dispatcher.Invoke(() =>
                {
                    fullTranscription.AppendLine(segment);
                    TranscriptionTextBlock.Text = fullTranscription.ToString();
                });
            };
            speechTranscriber.TranscriptionError += (error) =>
            {
                Dispatcher.Invoke(() =>
                {
                    UpdateRecordingStatus($"Status: Transcription Error! Check Azure Speech Key/Region.");
                    MessageBox.Show(error, "Transcription Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    StartRecordingButton.IsEnabled = true;
                    StopRecordingButton.IsEnabled = false;
                });
            };
            speechTranscriber.RecognitionSessionEnded += (sessionStatus) =>
            {
                Dispatcher.Invoke(() =>
                {
                    UpdateRecordingStatus($"Status: Recognition Ended. {sessionStatus}");
                    if (audioRecorder?.IsRecording ?? false) audioRecorder.StopRecording();
                    StartRecordingButton.IsEnabled = true;
                    StopRecordingButton.IsEnabled = false;
                });
            };
        }

        private void SubscribeToWakeWordEvents()
        {
            if (wakeWordDetector == null) return;

            wakeWordDetector.WakeWordDetected += (keyword) =>
            {
                Dispatcher.Invoke(() =>
                {
                    UpdateRecordingStatus($"Status: Wake word '{keyword}' detected. Ask your question or start recording.");
                    // Optionally, could auto-start recording here, but per instructions, user types question.
                    // For example: if (!StartRecordingButton.IsEnabled && !StopRecordingButton.IsEnabled) { /* Can start recording now */ }
                    // Or, flash UI, play a sound, etc.
                    System.Media.SystemSounds.Beep.Play(); // Simple feedback
                });
            };

            wakeWordDetector.DetectionError += (error) =>
            {
                 Dispatcher.Invoke(() => {
                    UpdateRecordingStatus($"Status: Wake Word Error! {error}");
                    // Optionally show a non-modal warning or log it.
                    // MessageBox.Show(error, "Wake Word Detection Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                 });
            };
        }

        private void OnAudioRecordingError(string errorMessage)
        {
            Dispatcher.Invoke(() =>
            {
                UpdateRecordingStatus($"Status: Audio Recording Error! {errorMessage}");
                MessageBox.Show(errorMessage, "Audio Recording Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StartRecordingButton.IsEnabled = true;
                StopRecordingButton.IsEnabled = false;
            });
        }

        private void OnAudioCaptureStopped()
        {
            Dispatcher.Invoke(() =>
            {
                if (StopRecordingButton.IsEnabled) // If recording was active and stopped unexpectedly
                {
                    UpdateRecordingStatus("Status: Audio capture stopped unexpectedly.");
                }
            });
        }


        private async void StartRecordingButton_Click(object sender, RoutedEventArgs e)
        {
            if (audioRecorder == null || speechTranscriber == null)
            {
                MessageBox.Show("Critical audio/speech components not initialized.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Optional: Stop wake word detection while recording to avoid conflicts or save resources
            // await wakeWordDetector?.StopDetectionAsync(); 

            fullTranscription.Clear();
            TranscriptionTextBlock.Text = "[Starting transcription...]";
            AiAnswerTextBlock.Text = "[AI Answer...]"; 
            UpdateRecordingStatus("Status: Initializing for recording...");

            PushAudioInputStream? pushStream = speechTranscriber.InitializePushStreamAndConfig();
            if (pushStream == null)
            {
                var errorMsg = "Status: Failed to init Speech SDK for transcription. Check Azure Speech Key/Region.";
                UpdateRecordingStatus(errorMsg);
                MessageBox.Show(errorMsg.Substring("Status: ".Length), "Azure Config Error", MessageBoxButton.OK, MessageBoxImage.Error);
                // await wakeWordDetector?.StartDetectionAsync(); // Restart wake word detection if recording fails to start
                return;
            }

            try
            {
                audioRecorder.StartRecording(pushStream);
                await speechTranscriber.StartRecognitionAsync();

                StartRecordingButton.IsEnabled = false;
                StopRecordingButton.IsEnabled = true;
                UpdateRecordingStatus("Status: Recording & Transcribing...");
            }
            catch (Exception ex)
            {
                UpdateRecordingStatus($"Status: Error starting recording/transcription! {ex.Message}");
                MessageBox.Show($"Error starting recording/transcription: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StartRecordingButton.IsEnabled = true;
                StopRecordingButton.IsEnabled = false;
                await speechTranscriber.StopRecognitionAsync(); // Ensure transcription stops
                // await wakeWordDetector?.StartDetectionAsync(); // Restart wake word detection
            }
        }

        private async void StopRecordingButton_Click(object sender, RoutedEventArgs e)
        {
            if (audioRecorder == null || speechTranscriber == null) return;

            UpdateRecordingStatus("Status: Stopping recording & transcription...");
            try
            {
                if (audioRecorder.IsRecording)
                {
                    audioRecorder.StopRecording();
                }
                await speechTranscriber.StopRecognitionAsync();
            }
            catch (Exception ex)
            {
                UpdateRecordingStatus($"Status: Error stopping! {ex.Message}");
                MessageBox.Show($"Error stopping: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally 
            {
                StartRecordingButton.IsEnabled = true;
                StopRecordingButton.IsEnabled = false;
                // Status will be updated by RecognitionSessionEnded or if an error occurred.
                // Restart wake word detection after recording session is fully stopped.
                // await wakeWordDetector?.StartDetectionAsync(); 
            }
        }

        private async void AskAiButton_Click(object sender, RoutedEventArgs e) 
        {
            if (languageModelService == null || !languageModelService.IsAvailable())
            {
                MessageBox.Show("Azure OpenAI Service is not configured or available. Please check credentials in LanguageModelService.cs.", "AI Service Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string userQuestion = UserQuestionTextBox.Text;
            string fullTranscript = fullTranscription.ToString(); // Use fullTranscript as base for snippet

            if (string.IsNullOrWhiteSpace(userQuestion) || userQuestion == "[Type your question here...]")
            {
                MessageBox.Show("Please type a question first.", "Input Needed", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            // Context Snippet Extraction: Get the last N characters.
            // N is hardcoded here. For configuration, it could be a private const int in this class,
            // or read from an application settings file if more dynamic configuration is needed.
            const int snippetMaxLength = 2000; 
            string transcriptContextForAI;

            if (fullTranscript.Length > snippetMaxLength)
            {
                transcriptContextForAI = fullTranscript.Substring(fullTranscript.Length - snippetMaxLength);
            }
            else
            {
                transcriptContextForAI = fullTranscript;
            }

            if (string.IsNullOrWhiteSpace(transcriptContextForAI)) // Check the snippet, not the full transcript
            {
                MessageBox.Show("No transcription context is available. Please record and transcribe audio first.", "Context Needed", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            AiAnswerTextBlock.Text = "[Thinking...]";
            AskAiButton.IsEnabled = false;
            UpdateRecordingStatus("Status: Asking AI...");

            try
            {
                // Pass the extracted snippet to the language model service
                string llmResponse = await languageModelService.GetResponseAsync(userQuestion, transcriptContextForAI);
                AiAnswerTextBlock.Text = llmResponse;
            }
            catch (Exception ex)
            {
                AiAnswerTextBlock.Text = $"Error getting response from AI: {ex.Message}";
                MessageBox.Show($"Error communicating with Azure OpenAI: {ex.Message}", "AI Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                AskAiButton.IsEnabled = true;
                UpdateRecordingStatus("Status: AI response received.");
            }
        }


        private void UpdateRecordingStatus(string status)
        {
            RecordingStatusTextBlock.Text = status;
        }

        public void Dispose()
        {
            audioRecorder?.Dispose();
            speechTranscriber?.Dispose();
            wakeWordDetector?.Dispose(); // Dispose WakeWordDetector
            // languageModelService?.Dispose(); 
            GC.SuppressFinalize(this);
        }
    }
}

using System;
using NAudio.Wave;
using Microsoft.CognitiveServices.Speech.Audio;

namespace PodcastAssistant
{
    public class AudioRecorder : IDisposable
    {
        private WasapiLoopbackCapture? captureInstance;
        private PushAudioInputStream? pushStream; 

        public bool IsRecording { get; private set; } = false;
        public WaveFormat RecordingWaveFormat { get; private set; } // Made non-nullable, set in constructor

        public event EventHandler? RecordingStoppedEvent;
        public event EventHandler<string>? RecordingFailed;
        public event EventHandler<AudioDataEventArgs>? AudioDataAvailable; // Kept for potential other uses

        public AudioRecorder()
        {
            // Initialize capture instance and WaveFormat in the constructor
            // This might throw if no audio loopback device is available.
            try
            {
                captureInstance = new WasapiLoopbackCapture();
                RecordingWaveFormat = captureInstance.WaveFormat;
            }
            catch (Exception ex)
            {
                // Handle cases where WasapiLoopbackCapture cannot be initialized (e.g. no speakers)
                // For now, rethrow or log, as the application might not be usable without it.
                // Consider a more graceful fallback or error reporting to UI.
                RecordingFailed?.Invoke(this, $"AudioRecorder: Failed to initialize audio capture device: {ex.Message}");
                // Set a default or throw to prevent usage if critical
                RecordingWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(44100, 2); // Placeholder if init fails, not ideal
                throw new InvalidOperationException($"Failed to initialize audio capture: {ex.Message}", ex);
            }
        }

        public void StartRecording(PushAudioInputStream targetPushStream)
        {
            if (IsRecording) return;
            if (captureInstance == null)
            {
                RecordingFailed?.Invoke(this, "AudioRecorder: Capture instance not initialized.");
                return;
            }

            try
            {
                this.pushStream = targetPushStream;
                IsRecording = true;

                captureInstance.DataAvailable += OnDataAvailable;
                captureInstance.RecordingStopped += OnRecordingStopped;

                captureInstance.StartRecording();
            }
            catch (Exception ex)
            {
                IsRecording = false;
                DisposeResources(); 
                RecordingFailed?.Invoke(this, $"Error starting recording: {ex.Message}");
            }
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            if (pushStream != null && e.BytesRecorded > 0)
            {
                pushStream.Write(e.Buffer, e.BytesRecorded);
                AudioDataAvailable?.Invoke(this, new AudioDataEventArgs(e.Buffer, e.BytesRecorded));
            }
        }
        
        private void OnRecordingStopped(object? sender, StoppedEventArgs e)
        {
            IsRecording = false;
            DisposeResources(); // Detach handlers, etc.
            RecordingStoppedEvent?.Invoke(this, EventArgs.Empty);
            if (e.Exception != null)
            {
                 RecordingFailed?.Invoke(this, $"Recording stopped with an error: {e.Exception.Message}");
            }
        }

        public void StopRecording()
        {
            if (!IsRecording || captureInstance == null) return;

            try
            {
                captureInstance.StopRecording(); 
            }
            catch (Exception ex)
            {
                IsRecording = false;
                DisposeResources(); // Ensure cleanup even if StopRecording throws
                RecordingFailed?.Invoke(this, $"Error stopping recording: {ex.Message}");
            }
        }
        
        private void DisposeResources()
        {
            if (captureInstance != null)
            {
                // Detach event handlers to prevent issues if StopRecording wasn't called or completed
                captureInstance.DataAvailable -= OnDataAvailable;
                captureInstance.RecordingStopped -= OnRecordingStopped;
                
                // Only dispose if it's not already disposed (NAudio can be sensitive)
                // However, for WasapiCapture, it's usually okay to call Dispose multiple times.
                captureInstance.Dispose();
                captureInstance = null;
            }
            // pushStream is externally managed by SpeechTranscriber, so it's not disposed here.
            pushStream = null; 
        }

        public void Dispose()
        {
            if (IsRecording && captureInstance != null) // Ensure it's recording and instance exists
            {
                StopRecording(); 
            }
            DisposeResources(); // General cleanup
            GC.SuppressFinalize(this);
        }
    }

    public class AudioDataEventArgs : EventArgs
    {
        public byte[] Buffer { get; }
        public int BytesRecorded { get; }

        public AudioDataEventArgs(byte[] buffer, int bytesRecorded)
        {
            // It's safer to copy the buffer if it's going to be reused or held onto,
            // as NAudio might reuse its internal buffers.
            this.Buffer = new byte[bytesRecorded];
            Array.Copy(buffer, this.Buffer, bytesRecorded);
            this.BytesRecorded = bytesRecorded;
        }
    }
}

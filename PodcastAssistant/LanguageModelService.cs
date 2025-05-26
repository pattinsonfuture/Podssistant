using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Azure; // Required for Response<T>
using Azure.AI.OpenAI; // Primary namespace for Azure OpenAI

namespace PodcastAssistant
{
    public class LanguageModelService
    {
        // IMPORTANT: Replace with your actual Azure OpenAI service details.
        private const string AzureOpenAiEndpoint = "YOUR_AZURE_OPENAI_ENDPOINT"; // e.g., "https://your-resource-name.openai.azure.com/"
        private const string AzureOpenAiKey = "YOUR_AZURE_OPENAI_KEY";
        private const string AzureOpenAiDeploymentName = "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME"; // The name of your deployment (e.g., "gpt-35-turbo")

        private readonly OpenAIClient? client;
        private bool isInitialized = false;

        public LanguageModelService()
        {
            if (string.IsNullOrEmpty(AzureOpenAiEndpoint) || AzureOpenAiEndpoint == "YOUR_AZURE_OPENAI_ENDPOINT" ||
                string.IsNullOrEmpty(AzureOpenAiKey) || AzureOpenAiKey == "YOUR_AZURE_OPENAI_KEY" ||
                string.IsNullOrEmpty(AzureOpenAiDeploymentName) || AzureOpenAiDeploymentName == "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME")
            {
                // Error should be propagated to the UI.
                // Consider throwing an exception or having an IsInitialized property that MainWindow checks.
                System.Diagnostics.Debug.WriteLine("Azure OpenAI credentials are not set in LanguageModelService.cs. Q&A functionality will be disabled.");
                isInitialized = false;
                return; 
            }

            try
            {
                client = new OpenAIClient(new Uri(AzureOpenAiEndpoint), new AzureKeyCredential(AzureOpenAiKey));
                isInitialized = true;
            }
            catch (Exception ex)
            {
                // Log or handle client creation error
                System.Diagnostics.Debug.WriteLine($"Error initializing OpenAIClient: {ex.Message}");
                isInitialized = false;
                // Propagate this error to UI if necessary
            }
        }

        public bool IsAvailable() => isInitialized && client != null;

        public async Task<string> GetResponseAsync(string userQuestion, string transcriptContext)
        {
            if (!IsAvailable())
            {
                return "Azure OpenAI Service is not configured. Please check credentials in LanguageModelService.cs.";
            }

            try
            {
                // Updated prompt engineering
                string systemPrompt = "You are a helpful AI assistant. The user is listening to a podcast and has a question about it. Your goal is to answer the user's question based *only* on the provided podcast transcript snippet. If the answer cannot be found in the snippet, clearly state that.";
                string userPrompt = $@"Podcast Snippet:
"""
{transcriptContext}
"""

User's Question:
"""
{userQuestion}
"""";

                var chatCompletionsOptions = new ChatCompletionsOptions()
                {
                    DeploymentName = AzureOpenAiDeploymentName, // Ensure this uses the correct field
                    Messages =
                    {
                        new ChatRequestSystemMessage(systemPrompt),
                        new ChatRequestUserMessage(userPrompt),
                    },
                    MaxTokens = 400, 
                    Temperature = 0.7f, 
                };

                Response<ChatCompletions> response = await client!.GetChatCompletionsAsync(chatCompletionsOptions);

                if (response.Value.Choices.Count > 0 && response.Value.Choices[0].Message != null)
                {
                    return response.Value.Choices[0].Message.Content;
                }
                return "No meaningful response received from the language model.";
            }
            catch (RequestFailedException ex)
            {
                System.Diagnostics.Debug.WriteLine($"Azure OpenAI API Error: {ex.Status} - {ex.ErrorCode} - {ex.Message}");
                return $"Error calling Azure OpenAI: {ex.Message} (Status Code: {ex.Status}). Check your Azure OpenAI service deployment and credentials.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Unexpected error in GetResponseAsync: {ex.ToString()}");
                return $"An unexpected error occurred while contacting Azure OpenAI: {ex.Message}";
            }
        }
    }
}

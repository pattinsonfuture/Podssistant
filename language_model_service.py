import requests
import logging
from typing import Optional, Dict, Generator


class LanguageModelService:
    def __init__(self, logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.api_key = None
        self.base_url = "https://api.deepseek.com/v1"
        self.is_available = False
        self.logger.info("LanguageModelService initialized")

    def configure(self, api_key: str):
        try:
            if not api_key:
                raise ValueError("API key is required")

            self.api_key = api_key
            self.is_available = True
            self.logger.info("DeepSeek API configured successfully")
            return True
        except Exception as e:
            self.logger.exception("Error configuring DeepSeek API:")
            self.is_available = False
            return False

    def is_configured(self) -> bool:
        return self.is_available

    async def get_response(self, user_question: str, context: str) -> Optional[str]:
        if not self.is_available or not self.api_key:
            self.logger.error("Service not available or not configured")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a helpful AI assistant. The user is listening to a podcast and has a question about it. 
                        Your goal is to answer the user's question based *only* on the provided podcast transcript snippet. 
                        If the answer cannot be found in the snippet, clearly state that."""
                    },
                    {
                        "role": "user",
                        "content": f"Podcast Snippet:\n{context}\n\nUser's Question:\n{user_question}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 400
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            if data.get("choices") and data["choices"][0].get("message"):
                return data["choices"][0]["message"]["content"]
            return "No response generated"
        except Exception as e:
            self.logger.exception("Error getting response from DeepSeek API:")
            return f"Error: {str(e)}"

    def get_streaming_response(self, user_question: str, context: str) -> Generator[str, None, None]:
        if not self.is_available or not self.api_key:
            self.logger.error("Service not available or not configured")
            yield "Service not available"
            return

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a helpful AI assistant. The user is listening to a podcast and has a question about it. 
                        Your goal is to answer the user's question based *only* on the provided podcast transcript snippet. 
                        If the answer cannot be found in the snippet, clearly state that."""
                    },
                    {
                        "role": "user",
                        "content": f"Podcast Snippet:\n{context}\n\nUser's Question:\n{user_question}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 400,
                "stream": True
            }

            with requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        data = line.decode('utf-8')
                        if data.startswith("data:"):
                            chunk = data[5:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(chunk)
                                if chunk_data.get("choices") and chunk_data["choices"][0].get("delta"):
                                    content = chunk_data["choices"][0]["delta"].get(
                                        "content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            self.logger.exception(
                "Error in streaming response from DeepSeek API:")
            yield f"Error: {str(e)}"

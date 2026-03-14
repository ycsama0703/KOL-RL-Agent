import requests

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    },
    json={
        "model": "google/gemini-3.1-pro-preview",
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    },
    timeout=60,
)

result = response.json()
print(result["choices"][0]["message"]["content"])






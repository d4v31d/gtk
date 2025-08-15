
### ollama_gemma3_01

# ollama list
# ollama run gemma3:latest
# ollama run gemma3:12b-it-qat

from ollama import chat
from ollama import ChatResponse



# NAME                 ID              SIZE      MODIFIED     
# gpt-oss:20b          f2b8351c629c    13 GB     2 hours ago
# gemma3:1b            8648f39daa8f    815 MB    25 hours ago
# deepseek-r1:1.5b     a42b25d8c10a    1.1 GB    2 months ago
# llama3.2:1b          baf6a787fdff    1.3 GB    2 months ago
# llava-phi3:latest    c7edd7b87593    2.9 GB    4 months ago

llm_model="gemma3:latest" #'gemma3:1b'

response: ChatResponse = chat(
    model=llm_model
    ,messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])

print(response['message']['content'])

# or access fields directly from the response object
print(response.message.content)

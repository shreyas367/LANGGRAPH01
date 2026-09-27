from dotenv import load_dotenv
import litellm
from litellm import completion
import time
from litellm.caching import Cache
litellm.cache=Cache(type="local", directory="cache")

load_dotenv()
prompt = "What is an LLM gateway in 3 bullets points only?"


start_time = time.time()
response1 = completion(
    model="groq/openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
    fallbacks=[
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.6-27b",
        "groq/qwen/qwen3.8-27b",
        "gemini-3.6-flash",
    ],
)

end_time = time.time()
print("Response:", response1.choices[0].message.content)
print("Model used:", response1.model)
print("Time taken for first call:", end_time - start_time)



print("======================================================")
print("Calling the same prompt again to check caching...")
start_time = time.time()
response2 = completion(
    model="groq/openai/gpt-oss-20b",
    messages=[{"role": "user", "content": prompt}],
    fallbacks=[
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.6-27b",
        "groq/qwen/qwen3.8-27b",
        "gemini-3.6-flash",
    ],
)
end_time = time.time()
print("Response:", response2.choices[0].message.content)
print("Model used:", response2.model)
print("Time taken for second call:", end_time - start_time)





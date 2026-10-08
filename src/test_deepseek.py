import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.llm_factory import get_llm, get_embeddings

print(f"Provider : {config.PROVIDER}")
print(f"Model    : {config.DEEPSEEK_MODEL}")
print(f"Base URL : {config.DEEPSEEK_BASE_URL}")

print("\n--- 1. Kiểm tra Embeddings (SafeFastEmbedWrapper) ---")
try:
    emb = get_embeddings()
    vec = emb.embed_query("Hello test")
    print(f"✅ Embeddings OK! Vector size: {len(vec)}, Model: {emb.model}")
except Exception as e:
    print(f"❌ Embeddings Error: {e}")

print("\n--- 2. Kiểm tra DeepSeek LLM ---")
try:
    llm = get_llm()
    res = llm.invoke("Hi! Reply with 'DeepSeek OK' in 3 words.")
    print(f"✅ DeepSeek LLM OK! Phản hồi: {res.content.strip()}")
except Exception as e:
    print(f"❌ DeepSeek Error: {e}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.llm_factory import get_llm, get_embeddings

print(f"Provider : {config.PROVIDER}")
print(f"Groq Model: {config.GROQ_MODEL}")

print("\n--- 1. Kiểm tra Embeddings (FastEmbed Local) ---")
try:
    emb = get_embeddings()
    vec = emb.embed_query("Hello test")
    print(f"✅ Embeddings OK! Vector dimensions: {len(vec)}")
except Exception as e:
    print(f"❌ Embeddings Error: {e}")

print("\n--- 2. Kiểm tra Groq LLM Inference ---")
try:
    llm = get_llm()
    res = llm.invoke("Hi! Reply in one short sentence.")
    print(f"✅ Groq LLM OK! Phản hồi: {res.content.strip()}")
except Exception as e:
    print(f"❌ Groq Error: {e}")

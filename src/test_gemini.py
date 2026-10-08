import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.llm_factory import get_llm, get_embeddings

print(f"Provider: {config.PROVIDER}")
print(f"Gemini Model: {config.GEMINI_MODEL}")
print(f"Gemini Embedding: {config.GEMINI_EMBEDDING_MODEL}")

print("\n--- 1. Kiểm tra Gemini Embeddings ---")
try:
    emb = get_embeddings()
    vec = emb.embed_query("Hello test")
    print(f"✅ Embeddings OK! Vector size: {len(vec)}")
except Exception as e:
    print(f"❌ Embeddings Error: {e}")

print("\n--- 2. Kiểm tra Gemini Chat LLM ---")
try:
    llm = get_llm()
    res = llm.invoke("Xin chào, bạn có hoạt động không?")
    print(f"✅ LLM OK! Phản hồi: {res.content[:100]}")
except Exception as e:
    print(f"❌ LLM Error: {e}")

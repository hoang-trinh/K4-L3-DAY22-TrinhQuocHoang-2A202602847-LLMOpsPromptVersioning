import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from langsmith import Client

key = config.LANGSMITH_API_KEY
masked_key = key[:10] + "..." + key[-6:] if len(key) > 16 else "(too short)"
print("LangSmith Key:", masked_key)

endpoints = [
    ("AWS Region (Chính xác của bạn)", "https://aws.api.smith.langchain.com"),
    ("US Mặc định (GCP)",              "https://api.smith.langchain.com"),
    ("EU Region",                     "https://eu.api.smith.langchain.com"),
]

for name, endpoint in endpoints:
    print(f"\nTesting endpoint: {name} ({endpoint})")
    try:
        c = Client(api_key=key, api_url=endpoint)
        projects = list(c.list_projects())
        print(f"  ✅ THÀNH CÔNG! Đọc được {len(projects)} projects:")
        for p in projects[:3]:
            print(f"     - {p.name}")
        print(f"  👉 HÃY ĐẶT TRONG .env: LANGCHAIN_ENDPOINT={endpoint}")
        break
    except Exception as e:
        print(f"  ❌ Thất bại: {e}")

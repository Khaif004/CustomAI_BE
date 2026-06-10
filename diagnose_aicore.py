"""
Run this to list all SAP AI Core deployments and check which ones are running.
Usage: python diagnose_aicore.py
"""
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

AICORE_URL   = os.getenv("SAP_AICORE_URL", "").rstrip("/")
AUTH_URL     = os.getenv("SAP_AICORE_AUTH_URL", "")
CLIENT_ID    = os.getenv("SAP_AICORE_CLIENT_ID", "")
CLIENT_SECRET= os.getenv("SAP_AICORE_CLIENT_SECRET", "")
EMB_DEPL_ID  = os.getenv("SAP_AICORE_EMBEDDING_DEPLOYMENT_ID", "")

token_url = AUTH_URL if AUTH_URL.endswith("/oauth/token") else f"{AUTH_URL}/oauth/token"

# 1. Get OAuth token
print("Fetching OAuth token...")
r = httpx.post(
    token_url,
    data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "client_credentials"},
    timeout=30,
)
r.raise_for_status()
token = r.json()["access_token"]
print("  Token OK\n")

headers = {"Authorization": f"Bearer {token}", "AI-Resource-Group": "default"}

# 2. List all deployments
print("Listing all deployments (AI-Resource-Group: default)...")
r = httpx.get(f"{AICORE_URL}/v2/lm/deployments", headers=headers, timeout=30)
print(f"  Status: {r.status_code}")
if not r.is_success:
    print(f"  Error: {r.text}")
else:
    data = r.json()
    deployments = data.get("resources", data) if isinstance(data, dict) else data
    print(f"  Found {len(deployments)} deployment(s):\n")
    for d in deployments:
        dep_id = d.get("id", "?")
        status = d.get("status", "?")
        model  = d.get("details", {}).get("resources", {}).get("backend_details", {}).get("model", {}).get("name", "")
        if not model:
            model = d.get("modelName", d.get("executableId", "?"))
        url    = d.get("deploymentUrl", "")
        print(f"  [{dep_id}]  status={status}  model={model}")
        if url:
            print(f"    deploymentUrl: {url}")

# 3. Probe the configured embedding deployment directly
print(f"\nProbing configured embedding deployment: {EMB_DEPL_ID}")
r = httpx.get(f"{AICORE_URL}/v2/lm/deployments/{EMB_DEPL_ID}", headers=headers, timeout=30)
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.text[:800]}")

# 4. Try all known URL patterns for embedding
base_url = f"{AICORE_URL}/v2/inference/deployments/{EMB_DEPL_ID}"
candidates = [
    ("plain /embeddings (string input)",          f"{base_url}/embeddings",          {"input": "hello"}),
    ("plain /embeddings (array input)",           f"{base_url}/embeddings",          {"input": ["hello"]}),
    ("/v1/embeddings (string input)",             f"{base_url}/v1/embeddings",       {"input": "hello"}),
    ("/v1/embeddings (array input)",              f"{base_url}/v1/embeddings",       {"input": ["hello"]}),
    ("Azure path ada-002 (string)",               f"{base_url}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21", {"input": "hello"}),
]

for label, url, body in candidates:
    r = httpx.post(url, headers={**headers, "Content-Type": "application/json"}, json=body, timeout=30)
    print(f"\n  [{r.status_code}] {label}")
    print(f"    URL:  {url}")
    if r.status_code not in (200, 201):
        print(f"    Body: {r.text[:300]}")
    else:
        data = r.json()
        vec = data.get("data", [{}])[0].get("embedding", [])
        print(f"    SUCCESS — embedding dim: {len(vec)}")
        break  # stop at first success

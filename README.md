# Output Bundle Viewer v1 (Extended)

Configured paths:
- Base: `/Users/joakimthomter/Desktop/1SCANS/5ANALYZERRUN`
- Output: `/Users/joakimthomter/Desktop/1SCANS/5ANALYZERRUN/2OUTPUT`

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional secrets: `.streamlit/secrets.toml`
```toml
[auth]
require_auth = false
password = ""

[flourish]
# api_token = ""
# api_url = "https://app.flourish.studio/api/datasets"

[observable]
# api_token = ""
# api_url = "https://api.observablehq.com/datasets"

[aws]
# access_key_id = ""
# secret_access_key = ""
# region = "eu-west-1"
```

# app.py — Output Bundle Viewer v1 (Cloudready, with System Health)
import io, os, json, time, zipfile, sys
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from streamlit import components
import plotly.express as px
import plotly.graph_objects as go

try:
    import networkx as nx
except Exception:
    nx = None
try:
    import boto3
except Exception:
    boto3 = None
try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="Output Bundle Viewer v1", page_icon="📦", layout="wide")

APP_VERSION = "v1 (Cloudready)"
DEFAULT_BASE = "/Users/joakimthomter/Desktop/1SCANS/5ANALYZERRUN"
DEFAULT_OUTPUT = "/Users/joakimthomter/Desktop/1SCANS/5ANALYZERRUN/2OUTPUT"
BASE_DIR = os.environ.get("OBV_BASE_DIR", DEFAULT_BASE)
OUTPUT_DIR = os.environ.get("OBV_OUTPUT_DIR", DEFAULT_OUTPUT)
CACHE_DIR = os.path.join(BASE_DIR, "CACHE")
LOGS_DIR = os.path.join(BASE_DIR, "LOGS")
EXPORTS_DIR = os.path.join(BASE_DIR, "EXPORTS")
for p in [BASE_DIR, OUTPUT_DIR, CACHE_DIR, LOGS_DIR, EXPORTS_DIR]:
    try: os.makedirs(p, exist_ok=True)
    except Exception: pass

require_auth = False
try: require_auth = bool(st.secrets.get("auth", {}).get("require_auth", False))
except Exception: pass
if require_auth:
    pw = st.text_input("Enter password", type="password")
    if pw != st.secrets.get("auth", {}).get("password", ""): st.stop()

with st.sidebar:
    st.title("📦 Output Bundle Viewer")
    st.caption("v1 — Cloudready build")
    # Health snapshot (lightweight)
    cache_files = len(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else 0
    log_files = len(os.listdir(LOGS_DIR)) if os.path.isdir(LOGS_DIR) else 0
    st.subheader("System Health")
    st.markdown(f"- **Version:** {APP_VERSION}")
    st.markdown(f"- **Python:** {sys.version.split()[0]}")
    st.markdown(f"- **Streamlit:** {st.__version__}")
    st.markdown(f"- **CWD:** `{os.getcwd()}`")
    st.markdown(f"- **Cache files:** {cache_files}  •  **Logs:** {log_files}")
    st.markdown(f"- **Output dir:** `{OUTPUT_DIR}`")
    st.divider()
    st.subheader("External Exports")
    export_target = st.selectbox("Export datasets to", ["None","Flourish","ObservableHQ"], index=0)
    st.caption("Set API keys in Streamlit Cloud secrets or streamlit_config/secrets.toml (local).")
    st.subheader("Persistence")
    mirror_s3 = st.checkbox("Mirror uploads to S3 (if creds present)", value=False)
    st.caption("Uses AWS creds in secrets or environment.")

def is_metrics_like(obj):
    if isinstance(obj, dict):
        if all(isinstance(v,(int,float)) for v in obj.values() if v is not None): return True
        for v in obj.values():
            if isinstance(v, dict) and any(isinstance(x,(int,float)) for x in v.values() if x is not None): return True
    return False

def load_from_zip(zb: bytes) -> Dict[str, bytes]:
    out = {}
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        for name in z.namelist():
            if not name.endswith("/"):
                out[name.split("/")[-1]] = z.read(name)
    return out

def normalize_name(n: str) -> str: return n.split("/")[-1]
def safe_read_text(b: bytes) -> str:
    for enc in ("utf-8","latin-1"):
        try: return b.decode(enc)
        except Exception: continue
    return b.decode("utf-8", errors="ignore")

def write_log(payload: dict):
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception: pass

def cache_save(filename: str, content: bytes):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{normalize_name(filename)}")
        with open(path, "wb") as f: f.write(content)
        return path
    except Exception: return None

def export_to_flourish(dataset: dict) -> Tuple[bool,str]:
    if export_target != "Flourish": return False, "Target not Flourish"
    if not requests: return False, "requests not installed"
    token = st.secrets.get("flourish", {}).get("api_token", None)
    url = st.secrets.get("flourish", {}).get("api_url", "https://app.flourish.studio/api/datasets")
    if not token: return False, "Flourish token missing"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"data": dataset})
        if 200 <= resp.status_code < 300:
            try: link = resp.json().get("link") or resp.json().get("id") or url
            except Exception: link = url
            return True, str(link)
        return False, f"HTTP {resp.status_code}"
    except Exception as e: return False, str(e)

def export_to_observable(dataset: dict) -> Tuple[bool,str]:
    if export_target != "ObservableHQ": return False, "Target not Observable"
    if not requests: return False, "requests not installed"
    token = st.secrets.get("observable", {}).get("api_token", None)
    url = st.secrets.get("observable", {}).get("api_url", "https://api.observablehq.com/datasets")
    if not token: return False, "Observable token missing"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=dataset)
        if 200 <= resp.status_code < 300:
            try: link = resp.json().get("url") or url
            except Exception: link = url
            return True, str(link)
        return False, f"HTTP {resp.status_code}"
    except Exception as e: return False, str(e)

st.subheader("1) Upload files or a .zip")
uploads = st.file_uploader("Upload bundle files (CSV / JSON / HTML / ZIP)", type=["csv","json","html","zip"], accept_multiple_files=True)

raw_files = {}
if uploads:
    for f in uploads:
        b = f.read()
        if f.type in ("application/zip","application/x-zip-compressed") or f.name.lower().endswith(".zip"):
            raw_files.update(load_from_zip(b))
        else:
            raw_files[normalize_name(f.name)] = b
        cache_save(f.name, b)

if not raw_files:
    st.info("No files uploaded yet. Drop your bundle to begin."); st.stop()

st.success(f"Detected **{len(raw_files)}** files in the bundle.")
write_log({"timestamp": datetime.now().isoformat(), "files": list(raw_files.keys()), "actions": ["upload"]})

st.subheader("2) File Inventory")
inv_cols = st.columns([3,1,1])
with inv_cols[0]: st.markdown("**Filename**")
with inv_cols[1]: st.markdown("**Type**")
with inv_cols[2]: st.markdown("**Size (KB)**")
inventory = [(n, n.split(".")[-1].lower(), round(len(c)/1024,1)) for n,c in sorted(raw_files.items())]
for name, ext, sz in inventory:
    row = st.columns([3,1,1]); row[0].markdown(f"`{name}`"); row[1].markdown(ext.upper()); row[2].markdown(f"{sz}")

csv_files = [n for n in raw_files if n.lower().endswith(".csv")]
json_files = [n for n in raw_files if n.lower().endswith(".json")]
html_files = [n for n in raw_files if n.lower().endswith(".html")]

if csv_files:
    st.subheader("3) CSV Tables & Quick Viz")
    csv_name = st.selectbox("Select a CSV to preview", csv_files, index=0)
    df = pd.read_csv(io.BytesIO(raw_files[csv_name]))
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True)
    with st.expander("Column Summary"):
        st.dataframe(pd.DataFrame([{"column":c,"dtype":str(df[c].dtype),"non_null":int(df[c].notna().sum()),"unique":int(df[c].nunique(dropna=True)),"sample":df[c].dropna().head(3).tolist()} for c in df.columns]), use_container_width=True)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        cat = st.selectbox("Categorical column", cat_cols, index=0)
        topn = st.slider("Top N categories", 5, 50, 15)
        vc = df[cat].value_counts().head(topn).reset_index(); vc.columns=[cat,"count"]
        st.plotly_chart(px.bar(vc, x=cat, y="count", title=f"Top {topn} values of '{cat}'"), use_container_width=True)
    if num_cols:
        num = st.selectbox("Numeric column", num_cols, index=0)
        st.plotly_chart(px.histogram(df, x=num, nbins=50, title=f"Distribution of '{num}'"), use_container_width=True)

def try_json_obj(name: str):
    try: return json.loads(safe_read_text(raw_files[name]))
    except Exception: return None

if json_files:
    st.subheader("4) JSON Preview, Metrics & Map Visuals")
    json_name = st.selectbox("Select a JSON to preview", json_files, index=0, key="json_sel")
    obj = try_json_obj(json_name)
    if obj is None: st.error("Invalid JSON")
    else:
        st.json(obj, expanded=False)
        if is_metrics_like(obj):
            rows=[]
            if all(isinstance(v,(int,float)) for v in obj.values() if v is not None):
                rows=[{"metric":str(k),"value":v,"section":"metrics"} for k,v in obj.items()]
            else:
                for s,inner in obj.items():
                    if isinstance(inner, dict):
                        for k,v in inner.items():
                            if isinstance(v,(int,float)): rows.append({"metric":str(k),"value":v,"section":str(s)})
            if rows:
                st.plotly_chart(px.bar(pd.DataFrame(rows), x="metric", y="value", color="section", title="Metrics Overview"), use_container_width=True)

        with st.expander("Artist Map — Treemap / Network", expanded=True):
            treemap_df=None
            if isinstance(obj,list) and obj and isinstance(obj[0],dict):
                keys=set(obj[0].keys())
                if "category" in keys and ("value" in keys or "size" in keys):
                    treemap_df=pd.DataFrame(obj)
            elif isinstance(obj,dict) and "treemap" in obj and isinstance(obj["treemap"],list):
                treemap_df=pd.DataFrame(obj["treemap"])
            if treemap_df is not None:
                label_col = "artist" if "artist" in treemap_df.columns else ("label" if "label" in treemap_df.columns else treemap_df.columns[0])
                value_col = "value" if "value" in treemap_df.columns else ("size" if "size" in treemap_df.columns else None)
                path = ["category", label_col] if "category" in treemap_df.columns else [label_col]
                if value_col: figt = px.treemap(treemap_df, path=path, values=value_col, title="Artist Category Treemap")
                else: treemap_df["_ones"]=1; figt=px.treemap(treemap_df, path=path, values="_ones", title="Artist Category Treemap (counts)")
                st.plotly_chart(figt, use_container_width=True)
            if nx is not None:
                nodes=None; links=None
                if isinstance(obj,dict) and "nodes" in obj and "links" in obj: nodes,links=obj.get("nodes",[]),obj.get("links",[])
                elif isinstance(obj,dict) and "graph" in obj and isinstance(obj["graph"],dict):
                    nodes=obj["graph"].get("nodes",[]); links=obj["graph"].get("edges",[])
                if isinstance(nodes,list) and isinstance(links,list) and nodes:
                    G=nx.Graph()
                    for n in nodes:
                        nid=n.get("id") or n.get("name")
                        if nid is not None: G.add_node(nid, **{k:v for k,v in n.items() if k not in ("id","name")})
                    for e in links:
                        s=e.get("source"); t=e.get("target"); w=e.get("weight",1.0)
                        if s is not None and t is not None: G.add_edge(s,t,weight=w)
                    pos=nx.spring_layout(G, seed=42, dim=2)
                    edge_x=[]; edge_y=[]
                    for s,t in G.edges():
                        edge_x += [pos[s][0], pos[t][0], None]
                        edge_y += [pos[s][1], pos[t][1], None]
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='none')
                    node_trace = go.Scatter(x=[pos[k][0] for k in G.nodes()], y=[pos[k][1] for k in G.nodes()], mode='markers+text', text=[str(k) for k in G.nodes()], textposition='top center')
                    figg=go.Figure(data=[edge_trace, node_trace]); figg.update_layout(title="Artist Relation Network", showlegend=False)
                    st.plotly_chart(figg, use_container_width=True)

if html_files:
    st.subheader("5) HTML Reports")
    html_name = st.selectbox("Select an HTML report", html_files, index=0, key="html_sel")
    components.v1.html(safe_read_text(raw_files[html_name]), height=700, scrolling=True)

st.subheader("6) Export / Download")
inv_payload = [{"filename": n, "size_bytes": len(raw_files[n])} for n in sorted(raw_files)]
st.download_button("Download bundle inventory (JSON)", data=json.dumps(inv_payload, indent=2), file_name="bundle_inventory.json", mime="application/json")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"bundle_inventory__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f: json.dump(inv_payload, f, indent=2)
    st.caption(f"Saved inventory to: `{out_path}`")
except Exception as e:
    st.caption(f"Save skipped ({e})")

st.markdown("#### External Visualization Export")
dataset_choice = st.selectbox("Choose a dataset for export", ["None"] + csv_files + json_files, index=0)

def build_dataset_payload(name: str) -> dict:
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw_files[name]))
        return {"type":"table","name":name,"data":df.to_dict(orient="records")}
    elif name.lower().endswith(".json"):
        try: data = json.loads(safe_read_text(raw_files[name]))
        except Exception: data = None
        return {"type":"json","name":name,"data":data}
    return {"type":"binary","name":name,"size":len(raw_files[name])}

def export_to_observable(dataset: dict) -> Tuple[bool,str]:
    if export_target != "ObservableHQ": return False, "Target not Observable"
    if not requests: return False, "requests not installed"
    token = st.secrets.get("observable", {}).get("api_token", None)
    url = st.secrets.get("observable", {}).get("api_url", "https://api.observablehq.com/datasets")
    if not token: return False, "Observable token missing"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=dataset)
        if 200 <= resp.status_code < 300:
            try: link = resp.json().get("url") or url
            except Exception: link = url
            return True, str(link)
        return False, f"HTTP {resp.status_code}"
    except Exception as e: return False, str(e)

def export_to_flourish(dataset: dict) -> Tuple[bool,str]:
    if export_target != "Flourish": return False, "Target not Flourish"
    if not requests: return False, "requests not installed"
    token = st.secrets.get("flourish", {}).get("api_token", None)
    url = st.secrets.get("flourish", {}).get("api_url", "https://app.flourish.studio/api/datasets")
    if not token: return False, "Flourish token missing"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"data": dataset})
        if 200 <= resp.status_code < 300:
            try: link = resp.json().get("link") or resp.json().get("id") or url
            except Exception: link = url
            return True, str(link)
        return False, f"HTTP {resp.status_code}"
    except Exception as e: return False, str(e)

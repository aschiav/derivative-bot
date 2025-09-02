import os, time, requests
from flask import Flask, request, jsonify, session

app = Flask(__name__)

# Sessions so each student keeps their own thread (works inside Canvas iframe)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

# Allow Canvas to iframe your app
@app.after_request
def add_headers(resp):
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://*.instructure.com https://*.instructuremedia.com;"
    )
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    return resp

# OpenAI constants (Assistants v2)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASSISTANT_ID_MAIN = os.environ.get("ASSISTANT_ID")           # course bot
ASSISTANT_ID_DERIV = os.environ.get("ASSISTANT_ID_DERIV")    # derivative bot

OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2",
    # Optional if your key is project/org scoped:
    **({"OpenAI-Project": os.environ["OPENAI_PROJECT"]} if os.environ.get("OPENAI_PROJECT") else {}),
    **({"OpenAI-Organization": os.environ["OPENAI_ORG"]} if os.environ.get("OPENAI_ORG") else {}),
}

def ensure_thread(key):
    """Create or reuse a thread id stored in the Flask session under `key`."""
    if key not in session:
        r = requests.post("https://api.openai.com/v1/threads", headers=OPENAI_HEADERS, timeout=30)
        if r.status_code >= 400:
            raise Exception(f"Create thread failed: {r.status_code} {r.text}")
        session[key] = r.json()["id"]
    return session[key]

@app.route("/health")
def health():
    return jsonify(ok=True), 200

# ----- API: main course bot -----
@app.route("/api/chat", methods=["POST"])
def chat_api():
    if not ASSISTANT_ID_MAIN:
        return jsonify({"error": "Missing ASSISTANT_ID (main bot)"}), 500

    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400

    try:
        thread_id = ensure_thread("thread_id_main")
        # 1) add message
        r1 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/messages",
            headers=OPENAI_HEADERS, json={"role":"user","content":msg}, timeout=30
        )
        if r1.status_code >= 400:
            return jsonify({"error":"OpenAI add message", "details": r1.text}), 502

        # 2) run (force plain text so JSON-mode settings don’t block)
        r2 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/runs",
            headers=OPENAI_HEADERS, json={"assistant_id": ASSISTANT_ID_MAIN, "response_format": {"type":"text"}}, timeout=30
        )
        if r2.status_code >= 400:
            return jsonify({"error":"OpenAI run", "details": r2.text}), 502
        run_id = r2.json()["id"]

        # 3) poll
        while True:
            rr = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                              headers=OPENAI_HEADERS, timeout=30)
            st = rr.json().get("status")
            if st in ("completed","failed","cancelled","expired"):
                break
            time.sleep(0.6)
        if st != "completed":
            return jsonify({"error": f"run status: {st}", "details": rr.text}), 502

        # 4) read last assistant message
        msgs = requests.get(
            f"https://api.openai.com/v1/threads/{thread_id}/messages?limit=1&order=desc",
            headers=OPENAI_HEADERS, timeout=30
        ).json()["data"]

        text_out = ""
        for part in msgs[0].get("content", []):
            if part.get("type") == "text":
                text_out += part["text"]["value"]
        return jsonify({"text": text_out or "[no text]"}), 200

    except Exception as e:
        return jsonify({"error":"server_exception","details":str(e)}), 500

# ----- API: derivative tutor -----
@app.route("/api/deriv_chat", methods=["POST"])
def deriv_chat():
    if not ASSISTANT_ID_DERIV:
        return jsonify({"error": "Missing ASSISTANT_ID_DERIV (derivative bot)"}), 500

    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400

    try:
        thread_id = ensure_thread("thread_id_deriv")
        r1 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/messages",
            headers=OPENAI_HEADERS, json={"role":"user","content":msg}, timeout=30
        )
        if r1.status_code >= 400:
            return jsonify({"error":"OpenAI add message", "details": r1.text}), 502

        r2 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/runs",
            headers=OPENAI_HEADERS, json={"assistant_id": ASSISTANT_ID_DERIV, "response_format": {"type":"text"}}, timeout=30
        )
        if r2.status_code >= 400:
            return jsonify({"error":"OpenAI run", "details": r2.text}), 502
        run_id = r2.json()["id"]

        while True:
            rr = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                              headers=OPENAI_HEADERS, timeout=30)
            st = rr.json().get("status")
            if st in ("completed","failed","cancelled","expired"):
                break
            time.sleep(0.6)
        if st != "completed":
            return jsonify({"error": f"run status: {st}", "details": rr.text}), 502

        msgs = requests.get(
            f"https://api.openai.com/v1/threads/{thread_id}/messages?limit=1&order=desc",
            headers=OPENAI_HEADERS, timeout=30
        ).json()["data"]

        text_out = ""
        for part in msgs[0].get("content", []):
            if part.get("type") == "text":
                text_out += part["text"]["value"]
        return jsonify({"text": text_out or "[no text]"}), 200

    except Exception as e:
        return jsonify({"error":"server_exception","details":str(e)}), 500

# ----- Minimal UIs -----
@app.route("/")
def main_page():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Course Assistant</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 860px; margin: 24px auto; }
#log { border:1px solid #ddd; padding:12px; height:460px; overflow:auto; border-radius:8px; background:#fafafa; white-space:pre-wrap }
form { display:flex; gap:8px; margin-top:12px; }
input { flex:1; padding:10px; border:1px solid #ccc; border-radius:8px; }
button { padding:10px 14px; border:0; border-radius:8px; cursor:pointer; }
a { display:inline-block; margin-top:10px; }
</style>
<h2>Course Assistant</h2>
<div id="log" aria-live="polite"></div>
<form id="f">
  <input id="m" autocomplete="off" placeholder="Ask the course assistant…" />
  <button>Send</button>
</form>
<a href="/derivative">Go to Derivative Tutor →</a>
<script>
const log=document.getElementById('log'), f=document.getElementById('f'), m=document.getElementById('m');
function add(prefix, text){ log.textContent += (prefix+text+"\\n"); log.scrollTop=log.scrollHeight; }
f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q=m.value.trim(); if(!q) return;
  add("You: ", q); m.value='';
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});
  const ct=(r.headers.get('content-type')||'').toLowerCase();
  const data = ct.includes('application/json') ? await r.json() : {error: await r.text()};
  add("Bot: ", data.text || data.error || "[no response]");
});
</script>
"""

@app.route("/derivative")
def derivative_page():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Derivative Tutor</title>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; }
#log { border:1px solid #ddd; padding:12px; min-height:160px; border-radius:8px; background:#fafafa; white-space:pre-wrap }
form { display:flex; gap:8px; margin-top:12px; }
input { flex:1; padding:10px; border:1px solid #ccc; border-radius:8px; }
button { padding:10px 14px; border:0; border-radius:8px; cursor:pointer; }
</style>
<h2>Derivative Tutor</h2>
<div id="log" aria-live="polite">Try: d/dx of x^2 e^{3x}</div>
<form id="f">
  <input id="m" autocomplete="off" placeholder="Enter a derivative problem…" />
  <button>Solve</button>
</form>
<script>
const log=document.getElementById('log'), f=document.getElementById('f'), m=document.getElementById('m');
function add(t){ log.textContent = t; if (window.MathJax) MathJax.typesetPromise(); }
f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q=m.value.trim(); if(!q) return;
  add('Solving…');
  const r=await fetch('/api/deriv_chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});
  const ct=(r.headers.get('content-type')||'').toLowerCase();
  const data = ct.includes('application/json') ? await r.json() : {error: await r.text()};
  add(data.text || data.error || '[no response]');
});
</script>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

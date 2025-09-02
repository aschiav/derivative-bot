import os, json, time, random, requests
from flask import Flask, request, jsonify, session

app = Flask(__name__)
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

# -------- OpenAI Assistants (v2) --------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASSISTANT_ID   = os.environ.get("ASSISTANT_ID")  # ← your gpt-4.1 Assistant (asst_...)
if not OPENAI_API_KEY: raise RuntimeError("Missing OPENAI_API_KEY")
if not ASSISTANT_ID:   raise RuntimeError("Missing ASSISTANT_ID")

OPENAI_ASSIST_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type":  "application/json",
    "OpenAI-Beta":   "assistants=v2",
}

def ensure_thread():
    """One thread per student session."""
    if "thread_id" not in session:
        # make a real API thread so history persists per student
        r = requests.post("https://api.openai.com/v1/threads",
                          headers=OPENAI_ASSIST_HEADERS, timeout=30)
        r.raise_for_status()
        session["thread_id"] = r.json()["id"]
    return session["thread_id"]

# -------- tiny helpers --------
def _safe_json(req):
    data = req.get_json(silent=True)
    if isinstance(data, dict): return data
    if isinstance(data, str):
        try:
            j = json.loads(data); return j if isinstance(j, dict) else {}
        except Exception: return {}
    if data is None:
        try:
            raw = req.get_data(cache=False, as_text=True)
            j = json.loads(raw); return j if isinstance(j, dict) else {}
        except Exception: return {}
    return {}

# -------- health --------
@app.route("/health")
def health():
    return jsonify(ok=True), 200

# -------- core: send images directly to your Assistant --------
@app.route("/api/assist", methods=["POST"])
def assist_api():
    """
    Body:
      {
        "f_image": "data:image/...;base64,...",
        "g_image": "data:image/...;base64,..."
      }
    Returns: { "assistant_text": "<model output>" }
    """
    data = _safe_json(request)
    f_img = data.get("f_image")
    g_img = data.get("g_image")
    if not f_img or not g_img:
        return jsonify({"error":"Both f_image and g_image are required (data URLs)."}), 400

    try:
        thread_id = ensure_thread()

        # Build message content for Assistants v2:
        # - one short text to label the images
        # - two image parts (data URLs allowed)
        content = [
            {"type": "input_text",
             "text": "Please analyze the student's derivative attempt. "
                     "The first image is the original function f(x). "
                     "The second image is the student's derivative g(x). "
                     "Respond per your tutoring instructions (LaTeX-first)."},
            {"type": "input_image", "image_url": f_img},
            {"type": "input_image", "image_url": g_img},
        ]

        # 1) Add message with images
        r1 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/messages",
            headers=OPENAI_ASSIST_HEADERS,
            json={"role": "user", "content": content},
            timeout=60
        )
        if r1.status_code >= 400:
            return jsonify({"error":"Assistant add-message error","details":r1.text}), 502

        # 2) Run the Assistant (text output)
        r2 = requests.post(
            f"https://api.openai.com/v1/threads/{thread_id}/runs",
            headers=OPENAI_ASSIST_HEADERS,
            json={"assistant_id": ASSISTANT_ID, "response_format": {"type":"text"}},
            timeout=60
        )
        if r2.status_code >= 400:
            return jsonify({"error":"Assistant run error","details":r2.text}), 502
        run_id = r2.json()["id"]

        # 3) Poll until complete
        while True:
            rr = requests.get(
                f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                headers=OPENAI_ASSIST_HEADERS, timeout=60
            )
            st = rr.json().get("status")
            if st in ("completed","failed","cancelled","expired"):
                break
            time.sleep(0.6)
        if st != "completed":
            return jsonify({"error": f"run status: {st}", "details": rr.text}), 502

        # 4) Fetch latest assistant message
        msgs = requests.get(
            f"https://api.openai.com/v1/threads/{thread_id}/messages?limit=1&order=desc",
            headers=OPENAI_ASSIST_HEADERS, timeout=60
        ).json()["data"]

        # Safely extract text
        out_text = ""
        if msgs and "content" in msgs[0]:
            for part in msgs[0]["content"]:
                if part.get("type") == "text":
                    out_text += part["text"]["value"]

        return jsonify({"assistant_text": out_text or "[no text]"}), 200

    except Exception as e:
        return jsonify({"error":"server_exception","details":str(e)}), 500

# -------- minimal UI: two image inputs -> Assistant reply --------
@app.route("/")
def ui():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Derivative Tutor — Assistant Image Test</title>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 940px; margin: 24px auto; }
  .card { border:1px solid #ddd; border-radius:10px; padding:14px; background:#fafafa; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .col { flex:1 1 320px; }
  label { font-weight:600; display:block; margin:8px 0 6px; }
  input[type="file"] { width:100%; padding:10px; border:1px solid #ccc; border-radius:8px; }
  button { padding:10px 14px; border:0; border-radius:8px; cursor:pointer; background:#0ea5e9; color:#fff; }
  #out { margin-top:16px; }
  .latex { padding:10px; background:#fff; border:1px solid #eee; border-radius:8px; white-space:pre-wrap; }
  img.preview { max-width:100%; border:1px solid #eee; border-radius:8px; }
  .bad { color:#991b1b; font-weight:700; }
  .gray { color:#666; }
</style>

<h2>Assistant Image Test</h2>
<p class="gray">Uploads go straight to your Assistant; we render its text with MathJax.</p>

<div class="card">
  <div class="row">
    <div class="col">
      <label>Function image (f)</label>
      <input id="fimg" type="file" accept="image/*" capture="environment">
      <div><img id="fprev" class="preview" /></div>
    </div>
    <div class="col">
      <label>Student derivative image (g)</label>
      <input id="gimg" type="file" accept="image/*" capture="environment">
      <div><img id="gprev" class="preview" /></div>
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="col" style="align-self:end">
      <button id="go">Ask Assistant</button>
    </div>
  </div>
</div>

<div id="out"></div>

<script>
const fimg = document.getElementById('fimg');
const gimg = document.getElementById('gimg');
const fprev = document.getElementById('fprev');
const gprev = document.getElementById('gprev');
const btn  = document.getElementById('go');
const out  = document.getElementById('out');

function toDataURL(file){
  return new Promise((resolve,reject)=>{
    const fr=new FileReader();
    fr.onload=()=>resolve(fr.result);
    fr.onerror=reject;
    fr.readAsDataURL(file); // -> data:image/...;base64,....
  });
}
fimg.addEventListener('change', async ()=>{ if(fimg.files[0]) fprev.src = await toDataURL(fimg.files[0]); });
gimg.addEventListener('change', async ()=>{ if(gimg.files[0]) gprev.src = await toDataURL(gimg.files[0]); });

btn.addEventListener('click', async ()=>{
  out.innerHTML = '<p class="gray">Sending to Assistant…</p>';
  if(!fimg.files[0] || !gimg.files[0]){ out.innerHTML='<p class="bad">Please upload both images.</p>'; return; }
  const [fdata,gdata] = await Promise.all([toDataURL(fimg.files[0]), toDataURL(gimg.files[0])]);
  try{
    const r = await fetch('/api/assist', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ f_image: fdata, g_image: gdata })
    });
    const ct=(r.headers.get('content-type')||'').toLowerCase();
    const data = ct.includes('application/json') ? await r.json() : {error: await r.text()};
    if(data.error){
      out.innerHTML = '<p class="bad">Error: '+(data.error||'unknown')+'</p><pre class="gray">'+(data.details||'')+'</pre>'; return;
    }
    out.innerHTML = '<div class="card latex">'+(data.assistant_text||'')+'</div>';
    if(window.MathJax) MathJax.typesetPromise();
  }catch(err){
    out.innerHTML = '<p class="bad">Network error: '+err+'</p>';
  }
});

// small warm-up ping
(async()=>{ try{ await fetch('/health',{cache:'no-store'}) }catch(e){} })();
</script>
"""
# -------- entry --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

import os, json, base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# (Only needed if you later add sessions/cookies inside Canvas)
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

# ---------- helpers ----------
def _safe_json_from_request(req):
    """Always return a dict (handles odd bodies or double-encoded JSON)."""
    data = req.get_json(silent=True)
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            j = json.loads(data)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    if data is None:
        try:
            raw = req.get_data(cache=False, as_text=True)
            j = json.loads(raw)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

def _data_url_info(data_url: str):
    """
    Parse a data URL like 'data:image/png;base64,AAAA...' -> (mime, is_b64, bytes, byte_len)
    Raises ValueError if not a data URL.
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    header, payload = data_url.split(",", 1)
    # header example: data:image/png;base64
    mime = "text/plain"
    is_b64 = False
    try:
        head = header[5:]  # strip 'data:'
        if ";base64" in head:
            is_b64 = True
            mime = head.replace(";base64", "") or mime
        else:
            mime = head or mime
    except Exception:
        pass

    if is_b64:
        try:
            blob = base64.b64decode(payload, validate=True)
        except Exception:
            # fall back: len of raw payload only
            blob = b""
    else:
        blob = payload.encode("utf-8", errors="ignore")
    return mime, is_b64, blob, len(blob)

# ---------- routes ----------
@app.route("/health")
def health():
    return jsonify(ok=True), 200

@app.route("/api/ping", methods=["POST"])
def ping():
    """
    JSON body:
      {
        "f_image": "data:image/...;base64,...",
        "g_image": "data:image/...;base64,..."
      }
    Returns: { "latex": "$$...$$" }
    """
    data = _safe_json_from_request(request)
    f_img = data.get("f_image")
    g_img = data.get("g_image")

    if not f_img or not g_img:
        return jsonify({"error": "Both f_image and g_image are required (data URLs)."}), 400

    try:
        f_mime, f_b64, f_bytes, f_len = _data_url_info(f_img)
        g_mime, g_b64, g_bytes, g_len = _data_url_info(g_img)

        # Build a friendly LaTeX echo proving the server received both images.
        # (MathJax on the page will render this.)
        latex = (
            "$$\\text{Upload OK: received two images.}\\\\"
            f"\\text{{f(x) mime}}={{{{ {f_mime} }}}},\\; "
            f"\\text{{base64}}={str(f_b64).lower()},\\; "
            f"\\text{{bytes}}={f_len}\\\\"
            f"\\text{{g(x) mime}}={{{{ {g_mime} }}}},\\; "
            f"\\text{{base64}}={str(g_b64).lower()},\\; "
            f"\\text{{bytes}}={g_len}$$"
        )
        return jsonify({"latex": latex}), 200

    except Exception as e:
        return jsonify({"error": "server_exception", "details": str(e)}), 500

@app.route("/")
def ui():
    # Minimal UI: two image uploads -> POST /api/ping -> render LaTeX
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Derivative Tutor — Upload Test</title>
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
  .latex { padding:10px; background:#fff; border:1px solid #eee; border-radius:8px; }
  img.preview { max-width:100%; border:1px solid #eee; border-radius:8px; }
  .bad { color:#991b1b; font-weight:700; }
  .gray { color:#666; }
</style>

<h2>Derivative Tutor — Upload Test</h2>
<p class="gray">Goal: make sure the server receives two images and returns LaTeX.</p>

<div class="card">
  <div class="row">
    <div class="col">
      <label>Upload function f(x) image</label>
      <input id="fimg" type="file" accept="image/*" capture="environment">
      <div><img id="fprev" class="preview" /></div>
    </div>
    <div class="col">
      <label>Upload your derivative g(x) image</label>
      <input id="gimg" type="file" accept="image/*" capture="environment">
      <div><img id="gprev" class="preview" /></div>
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="col" style="align-self:end">
      <button id="go">Send test</button>
    </div>
  </div>
</div>

<div id="out"></div>

<script>
const fimg = document.getElementById('fimg');
const gimg = document.getElementById('gimg');
const fprev = document.getElementById('fprev');
const gprev = document.getElementById('gprev');
const btn = document.getElementById('go');
const out = document.getElementById('out');

function toDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file); // always data:image/...;base64,....
  });
}

fimg.addEventListener('change', async () => {
  if (fimg.files[0]) fprev.src = await toDataURL(fimg.files[0]);
});
gimg.addEventListener('change', async () => {
  if (gimg.files[0]) gprev.src = await toDataURL(gimg.files[0]);
});

btn.addEventListener('click', async () => {
  out.innerHTML = '<p class="gray">Sending…</p>';
  if (!fimg.files[0] || !gimg.files[0]) {
    out.innerHTML = '<p class="bad">Please upload both images.</p>'; return;
  }
  const [fdata, gdata] = await Promise.all([toDataURL(fimg.files[0]), toDataURL(gimg.files[0])]);

  try {
    const r = await fetch('/api/ping', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ f_image: fdata, g_image: gdata })
    });
    const ct = (r.headers.get('content-type')||'').toLowerCase();
    const data = ct.includes('application/json') ? await r.json() : {error: await r.text()};

    if (data.error) {
      out.innerHTML = '<p class="bad">Error: ' + (data.error || 'unknown') + '</p><pre class="gray">' + (data.details||'') + '</pre>';
      return;
    }
    out.innerHTML = '<div class="card latex">' + (data.latex||'') + '</div>';
    if (window.MathJax) MathJax.typesetPromise();
  } catch (err) {
    out.innerHTML = '<p class="bad">Network error: ' + err + '</p>';
  }
});

// gentle warm-up ping (helps on Free tier)
(async()=>{ try{ await fetch('/health',{cache:'no-store'}) }catch(e){} })();
</script>
"""
# ---------- entry ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

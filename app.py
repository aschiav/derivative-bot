import os, time, json, math, random, re, requests
from flask import Flask, request, jsonify, session

# ── SymPy for parsing & checking ───────────────────────────────────────────────
from sympy import (
    symbols, sympify, simplify, diff, lambdify, pi, E
)
from sympy import sin, cos, tan, cot, sec, csc
from sympy import asin, acos, atan
from sympy import sinh, cosh, tanh, asinh, acosh, atanh
from sympy import log, sqrt, Abs, latex

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True  # needed in Canvas iframe over HTTPS
)

# Allow Canvas to iframe your app
@app.after_request
def add_headers(resp):
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://*.instructure.com https://*.instructuremedia.com;"
    )
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    return resp

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
ASSISTANT_ID = os.environ.get("ASSISTANT_ID")  # ← your gpt-4.1 Assistant (asst_...)
OCR_MODEL = os.environ.get("OCR_MODEL", "gpt-4o-mini")  # fast/vision model for OCR

OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}
OPENAI_ASSISTANT_HEADERS = {
    **OPENAI_HEADERS,
    "OpenAI-Beta": "assistants=v2"
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def _safe_json_from_request(req):
    """Always return a dict for incoming POST JSON (handles odd bodies)."""
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

def ensure_thread():
    """One thread per student session—used only for your Assistant reply."""
    if "thread_id" not in session:
        r = requests.post("https://api.openai.com/v1/threads",
                          headers=OPENAI_ASSISTANT_HEADERS, timeout=30)
        r.raise_for_status()
        session["thread_id"] = r.json()["id"]
    return session["thread_id"]

# ── SymPy helpers ─────────────────────────────────────────────────────────────
def allowed_locals(varname="x"):
    x = symbols(varname)
    return {
        varname: x,
        "x": x, "y": symbols("y"),
        "pi": pi, "E": E,
        "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
        "asin": asin, "acos": acos, "atan": atan,
        "sinh": sinh, "cosh": cosh, "tanh": tanh, "asinh": asinh, "acosh": acosh, "atanh": atanh,
        "ln": log, "log": log, "exp": lambda z: E**z,
        "sqrt": sqrt, "Abs": Abs
    }

def parse_sympy(expr_str, varname="x"):
    return sympify(expr_str, locals=allowed_locals(varname))

def numeric_equiv(f_expr, g_expr, varname="x"):
    """Heuristic numeric check on several points."""
    x = symbols(varname)
    f = lambdify(x, simplify(f_expr), "mpmath")
    g = lambdify(x, simplify(g_expr), "mpmath")
    pts = [-3, -2, -1, -0.5, -1/3, 0.5, 1, 2, 3]
    tested = matched = 0
    for t in pts:
        try:
            fv, gv = f(t), g(t)
            if fv is None or gv is None: continue
            if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)): continue
            if isinstance(gv, float) and (math.isnan(gv) or math.isinf(gv)): continue
            if abs(fv - gv) <= 1e-6 * (1 + abs(fv) + abs(gv)):
                matched += 1
            tested += 1
        except Exception:
            continue
    if tested >= 4 and matched >= max(3, int(0.8 * tested)):
        return True, {"tested": tested, "matched": matched}
    return False, {"tested": tested, "matched": matched}

# ── OCR (Responses API, plain text; no structured outputs) ────────────────────
OCR_PROMPT = (
    "Read the math expression in the image. Return exactly three lines, no extra text:\n"
    "SYMPY: <expression in SymPy syntax using ** for powers and log() for natural log>\n"
    "LATEX: <the same expression in LaTeX>\n"
    "VAR: <single main variable symbol, like x>\n"
    "Do not compute anything; just transcribe what is written."
)

def ocr_math(data_url, hint=""):
    """
    Send a data URL image to a vision model and parse the 3-line response.
    Returns dict {expr_sympy, expr_latex, variable}. Never returns a bare string.
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise RuntimeError("Expected data URL (data:image/...;base64,...)")

    content = [
        {"type": "input_text", "text": OCR_PROMPT + (f"\nContext: {hint}" if hint else "")},
        {"type": "input_image", "image_url": data_url}  # string URL per Responses API
    ]
    payload = {"model": OCR_MODEL, "input": [{"role": "user", "content": content}]}

    r = requests.post("https://api.openai.com/v1/responses",
                      headers=OPENAI_HEADERS, json=payload, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"OCR {r.status_code}: {r.text}")

    j = r.json()
    # Prefer output_text; fall back to digging if needed
    text = j.get("output_text") or (j.get("output") or [{}])[0].get("content", [{}])[0].get("text", {}).get("value", "")
    if not isinstance(text, str):
        text = str(text)

    # Parse lines
    sympy_s = latex_s = var_s = ""
    for line in text.strip().splitlines():
        if line.upper().startswith("SYMPY:"):
            sympy_s = line.split(":", 1)[1].strip()
        elif line.upper().startswith("LATEX:"):
            latex_s = line.split(":", 1)[1].strip()
        elif line.upper().startswith("VAR:"):
            var_s = line.split(":", 1)[1].strip()
    if not var_s: var_s = "x"

    # Minimal cleaning
    sympy_s = sympy_s.strip().strip("`")
    latex_s  = latex_s.strip().strip("`")
    var_s    = re.sub(r"[^A-Za-z]", "", var_s) or "x"

    return {
        "expr_sympy": sympy_s,
        "expr_latex": latex_s or sympy_s,
        "variable": var_s
    }

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify(ok=True), 200

# ── Core API: upload → OCR → check → ask your Assistant for the tutoring reply ─
@app.route("/api/check", methods=["POST"])
def check_derivative():
    """
    JSON body:
      {
        "f_image": "data:image/...;base64,...",
        "g_image": "data:image/...;base64,...",
        "variable": "x"   # optional
      }
    """
    data = _safe_json_from_request(request)
    f_img = data.get("f_image")
    g_img = data.get("g_image")
    varname = (data.get("variable") or "x").strip() or "x"

    if not f_img or not g_img:
        return jsonify({"error": "Both f_image and g_image are required (data URLs)."}), 400

    try:
        # 1) OCR both images
        f_ocr = ocr_math(f_img, "This is the original function f(x).")
        g_ocr = ocr_math(g_img, "This is the student's derivative g(x).")

        # 2) Parse with SymPy (robust to OCR noise if possible)
        var = (f_ocr.get("variable") or varname).strip() or varname
        f_expr = parse_sympy(f_ocr.get("expr_sympy", ""), var)
        g_expr = parse_sympy(g_ocr.get("expr_sympy", ""), var)

        # 3) Compute correct derivative
        x = symbols(var)
        fprime = simplify(diff(f_expr, x))

        # 4) Compare
        symbolic_ok = simplify(fprime - g_expr) == 0
        numeric_ok, stats = numeric_equiv(fprime, g_expr, var)
        verdict = "correct" if (symbolic_ok or numeric_ok) else "incorrect"

        # 5) Ask YOUR Assistant (gpt-4.1, response_format=text) to explain kindly in LaTeX-first
        if not ASSISTANT_ID:
            explain_text = (
                "Assistant ID is not set on the server, so I cannot generate the tutoring reply. "
                "However, here is the system check result:\n"
                f"- f(x) parsed: {latex(f_expr)}\n"
                f"- Computed d/d{var} f(x): {latex(fprime)}\n"
                f"- Student g(x): {latex(g_expr)}\n"
                f"- Match: {verdict}"
            )
        else:
            thread_id = ensure_thread()
            # Build a single user message that your Assistant will respond to in its own tone/instructions.
            user_msg = (
                "Please analyze this derivative attempt as a supportive tutor per your instructions.\n\n"
                f"Function (from image): $${latex(f_expr)}$$\n"
                f"Student's derivative (from image): $${latex(g_expr)}$$\n"
                f"Computed derivative (for reference): $${latex(fprime)}$$\n"
                f"Variable: ${var}$\n\n"
                "Important: Start with your LaTeX reasoning and derivation first. "
                "Only after that reasoning, give your gentle concluding judgment and guidance."
            )

            # Add message
            r1 = requests.post(
                f"https://api.openai.com/v1/threads/{thread_id}/messages",
                headers=OPENAI_ASSISTANT_HEADERS,
                json={"role": "user", "content": user_msg},
                timeout=30
            )
            if r1.status_code >= 400:
                return jsonify({"error": "Assistant add-message error", "details": r1.text}), 502

            # Run with text output (since your Assistant is set to text)
            r2 = requests.post(
                f"https://api.openai.com/v1/threads/{thread_id}/runs",
                headers=OPENAI_ASSISTANT_HEADERS,
                json={"assistant_id": ASSISTANT_ID, "response_format": {"type": "text"}},
                timeout=30
            )
            if r2.status_code >= 400:
                return jsonify({"error": "Assistant run error", "details": r2.text}), 502
            run_id = r2.json()["id"]

            # Poll
            while True:
                rr = requests.get(
                    f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                    headers=OPENAI_ASSISTANT_HEADERS, timeout=30
                )
                st = rr.json().get("status")
                if st in ("completed", "failed", "cancelled", "expired"):
                    break
                time.sleep(0.6)
            if st != "completed":
                return jsonify({"error": f"run status: {st}", "details": rr.text}), 502

            # Read latest assistant message
            msgs = requests.get(
                f"https://api.openai.com/v1/threads/{thread_id}/messages?limit=1&order=desc",
                headers=OPENAI_ASSISTANT_HEADERS, timeout=30
            ).json()["data"]

            explain_text = ""
            for part in msgs[0].get("content", []):
                if part.get("type") == "text":
                    explain_text += part["text"]["value"]

        # 6) Build API response for the UI
        out = {
            "latex": {
                "f_expr": f_ocr.get("expr_latex") or latex(f_expr),
                "g_expr": g_ocr.get("expr_latex") or latex(g_expr),
                "computed_fprime": latex(fprime)
            },
            "parsed": {
                "f_expr_sympy": str(f_expr),
                "g_expr_sympy": str(g_expr),
                "variable": var
            },
            "checks": {
                "symbolic_equal": bool(symbolic_ok),
                "numeric_equal": bool(numeric_ok),
                "numeric_stats": stats
            },
            "verdict": verdict,
            "assistant_text": explain_text
        }
        return jsonify(out), 200

    except Exception as e:
        return jsonify({"error": "server_exception", "details": str(e)}), 500
        
@app.route("/")
def ui():
    # Minimal UI: two image uploads → /api/check
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Derivative Tutor — Image Checker</title>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 940px; margin: 24px auto; }
  .card { border:1px solid #ddd; border-radius:10px; padding:14px; background:#fafafa; }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .col { flex:1 1 320px; }
  label { font-weight:600; display:block; margin:8px 0 6px; }
  input[type="file"], input[type="text"] { width:100%; padding:10px; border:1px solid #ccc; border-radius:8px; }
  button { padding:10px 14px; border:0; border-radius:8px; cursor:pointer; background:#0ea5e9; color:#fff; }
  #out { margin-top:16px; }
  .latex { padding:10px; background:#fff; border:1px solid #eee; border-radius:8px; }
  .ok { color:#166534; font-weight:700; }
  .bad { color:#991b1b; font-weight:700; }
  .gray { color:#666; }
  img.preview { max-width:100%; border:1px solid #eee; border-radius:8px; }
</style>

<h2>Derivative Tutor — Check your work from images</h2>
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
    <div class="col" style="max-width:220px">
      <label>Variable (default x)</label>
      <input id="var" type="text" value="x" maxlength="3">
    </div>
    <div class="col" style="align-self:end">
      <button id="go">Check derivative</button>
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
const vinput = document.getElementById('var');

function toDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

fimg.addEventListener('change', async () => {
  if (fimg.files[0]) fprev.src = await toDataURL(fimg.files[0]);
});
gimg.addEventListener('change', async () => {
  if (gimg.files[0]) gprev.src = await toDataURL(gimg.files[0]);
});

btn.addEventListener('click', async () => {
  out.innerHTML = '<p class="gray">Analyzing images…</p>';
  if (!fimg.files[0] || !gimg.files[0]) { out.innerHTML = '<p class="bad">Please upload both images.</p>'; return; }
  const [fdata, gdata] = await Promise.all([toDataURL(fimg.files[0]), toDataURL(gimg.files[0])]);
  const variable = (vinput.value || 'x').trim() || 'x';

  try {
    const r = await fetch('/api/check', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ f_image: fdata, g_image: gdata, variable })
    });
    const ct = (r.headers.get('content-type')||'').toLowerCase();
    const data = ct.includes('application/json') ? await r.json() : {error: await r.text()};

    if (data.error) {
      out.innerHTML = '<p class="bad">Error: ' + (data.error || 'unknown') + '</p><pre class="gray">' + (data.details||'') + '</pre>';
      return;
    }

    const v = data.verdict;
    const latex = data.latex || {};
    out.innerHTML = `
      <div class="card">
        <div><strong>Parsed f(x):</strong></div>
        <div class="latex">$$${latex.f_expr || ''}$$</div>
        <div style="height:8px"></div>
        <div><strong>Computed \\(\\frac{d}{d${data.parsed?.variable || 'x'}} f(x)\\):</strong></div>
        <div class="latex">$$${latex.computed_fprime || ''}$$</div>
        <div style="height:8px"></div>
        <div><strong>Student g(x):</strong></div>
        <div class="latex">$$${latex.g_expr || ''}$$</div>
        <div style="height:8px"></div>
        <div>Result: <span class="${v==='correct'?'ok':'bad'}">${v.toUpperCase()}</span></div>
        ${data.hint ? ('<div style="margin-top:8px"><em>Hint:</em> ' + data.hint.replaceAll('<','&lt;') + '</div>') : ''}
        <div style="margin-top:8px" class="gray">
          Checks — symbolic: ${data.checks?.symbolic_equal}, numeric: ${data.checks?.numeric_equal}
          ${data.checks?.numeric_stats ? '(tested: '+data.checks.numeric_stats.tested+', matched: '+data.checks.numeric_stats.matched+')' : ''}
        </div>
      </div>
    `;
    if (window.MathJax) MathJax.typesetPromise();
  } catch (err) {
    out.innerHTML = '<p class="bad">Network error: '+err+'</p>';
  }
});

// tiny warm-up so Free tier wakes sooner
(async()=>{ try{ await fetch('/health',{cache:'no-store'}) }catch(e){} })();
</script>
"""
# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

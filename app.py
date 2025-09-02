import os, time, json, math, random, requests
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
    SESSION_COOKIE_SECURE=True  # required for Canvas iframe (HTTPS)
)

# Allow Canvas to iframe your app
@app.after_request
def add_headers(resp):
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://*.instructure.com https://*.instructuremedia.com;"
    )
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    return resp

# ── OpenAI config (Responses API) ─────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")  # vision-capable

# ── Structured Outputs schema (Responses API text.format) ─────────────────────
STRUCTURED_MATH_SCHEMA = {
    "type": "object",
    "properties": {
        "expr_sympy": {"type": "string", "description": "SymPy syntax, e.g., sin(x)**2/(x+1)"},
        "expr_latex": {"type": "string", "description": "LaTeX, e.g., \\frac{\\sin^2 x}{x+1}"},
        "variable":   {"type": "string", "description": "primary variable, e.g., x"}
    },
    # REQUIRED must include every key in properties:
    "required": ["expr_sympy", "expr_latex", "variable"],
    "additionalProperties": False
}

# ── Helpers: per-student session thread id (not strictly needed) ──────────────
def ensure_thread():
    if "thread_id" not in session:
        session["thread_id"] = f"local-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    return session["thread_id"]

# ── Math parsing & comparison helpers ─────────────────────────────────────────
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
    """Heuristic numeric check on a spread of points."""
    x = symbols(varname)
    f = lambdify(x, simplify(f_expr), "mpmath")
    g = lambdify(x, simplify(g_expr), "mpmath")
    candidates = [-3, -2, -1, -0.5, -1/3, 0.5, 1, 2, 3]
    tested = matched = 0
    for t in candidates:
        try:
            fv, gv = f(t), g(t)
            if fv is None or gv is None:
                continue
            if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)):
                continue
            if isinstance(gv, float) and (math.isnan(gv) or math.isinf(gv)):
                continue
            if abs(fv - gv) <= 1e-6 * (1 + abs(fv) + abs(gv)):
                matched += 1
            tested += 1
        except Exception:
            continue
    if tested >= 4 and matched >= max(3, int(0.8 * tested)):
        return True, {"tested": tested, "matched": matched}
    return False, {"tested": tested, "matched": matched}

# ── Robust coercion (avoid .get on strings) ───────────────────────────────────
def _coerce_ocr(obj):
    if isinstance(obj, dict):
        return {
            "expr_sympy": obj.get("expr_sympy", ""),
            "expr_latex": obj.get("expr_latex", obj.get("expr_sympy", "")),
            "variable":   (obj.get("variable") or "x")
        }
    if isinstance(obj, str):
        try:
            j = json.loads(obj)
            return _coerce_ocr(j)
        except Exception:
            return {"expr_sympy": obj, "expr_latex": obj, "variable": "x"}
    return {"expr_sympy": "", "expr_latex": "", "variable": "x"}

# ── Vision OCR via Responses API (with text.format structured outputs) ────────
def extract_expr_from_image(data_url, hint_text=""):
    """
    Return dict: { expr_sympy, expr_latex, variable }
    `data_url` must be a data: URL (data:image/...;base64,...)
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise RuntimeError("Expected a data URL (data:image/...;base64,...)")

    prompt = (
        "Extract the mathematical expression from this image. "
        "Return JSON with keys: expr_sympy (SymPy syntax; use ** for powers; use log for natural log), "
        "expr_latex (best-effort LaTeX), and variable (default 'x'). "
        "If the image shows a derivative, transcribe exactly what's shown; do not compute a new derivative."
    )
    if hint_text:
        prompt += f" Hint/context: {hint_text}"

    payload = {
        "model": VISION_MODEL,  # e.g., "gpt-4o-mini" or "gpt-4o"
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url}  # string, not {"url": ...}
                ]
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "math_from_image",
                "strict": True,
                "schema": STRUCTURED_MATH_SCHEMA
            }
        }
    }

    if os.environ.get("DEBUG_SCHEMA") == "1":
        print("DEBUG text.format.schema =",
              json.dumps(payload["text"]["format"]["schema"], ensure_ascii=False))

    r = requests.post("https://api.openai.com/v1/responses",
                      headers=OPENAI_HEADERS, json=payload, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"Vision OCR error {r.status_code}: {r.text}")

    j = r.json()
    text = j.get("output_text") or \
           (j.get("output") or [{}])[0].get("content", [{}])[0].get("text", {}).get("value")

    try:
        return json.loads(text) if text else {"expr_sympy": "", "expr_latex": "", "variable": "x"}
    except Exception:
        return {"expr_sympy": text or "", "expr_latex": text or "", "variable": "x"}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify(ok=True), 200

@app.route("/api/check", methods=["POST"])
def check_derivative():
    """
    Body: { "f_image": "data:image/...;base64,...",
            "g_image": "data:image/...;base64,...",
            "variable": "x" }
    """
    data = request.get_json(silent=True) or {}
    f_img = data.get("f_image")
    g_img = data.get("g_image")
    varname = (data.get("variable") or "x").strip() or "x"

    if not f_img or not g_img:
        return jsonify({"error": "Both f_image and g_image are required (data URLs)."}), 400

    try:
        raw_f = extract_expr_from_image(f_img, "This is the original function f(x).")
        raw_g = extract_expr_from_image(g_img, "This is the student's claimed derivative g(x).")

        f_obj = _coerce_ocr(raw_f)
        g_obj = _coerce_ocr(raw_g)

        f_sym = f_obj.get("expr_sympy", "")
        g_sym = g_obj.get("expr_sympy", "")
        var = (f_obj.get("variable") or varname).strip() or varname  # prefer f's variable if present

        # Parse & compute
        f_expr = parse_sympy(f_sym, var)
        g_expr = parse_sympy(g_sym, var)
        x = symbols(var)
        fprime = simplify(diff(f_expr, x))

        # Checks
        symbolic_ok = simplify(fprime - g_expr) == 0
        numeric_ok, stats = numeric_equiv(fprime, g_expr, var)
        verdict = "correct" if (symbolic_ok or numeric_ok) else "incorrect"

        result = {
            "parsed": {
                "f_expr_sympy": str(f_expr),
                "g_expr_sympy": str(g_expr),
                "variable": var
            },
            "latex": {
                "f_expr": f_obj.get("expr_latex") or latex(f_expr),
                "g_expr": g_obj.get("expr_latex") or latex(g_expr),
                "computed_fprime": latex(fprime)
            },
            "checks": {
                "symbolic_equal": bool(symbolic_ok),
                "numeric_equal": bool(numeric_ok),
                "numeric_stats": stats
            },
            "verdict": verdict
        }

        # Optional: brief hint if incorrect
        if verdict == "incorrect":
            hint_prompt = (
                "Compare the derivative. Given f(x) and a student's g(x), "
                "explain briefly (<= 3 lines) why g(x) differs from d/dx f(x), "
                "naming any rule that seems misapplied. Use plain text."
            )
            hint_in = f"f(x) = {str(f_expr)}\nCorrect d/dx f(x) = {str(fprime)}\nStudent g(x) = {str(g_expr)}"
            hr = requests.post("https://api.openai.com/v1/responses",
                               headers=OPENAI_HEADERS,
                               json={"model": "gpt-4o-mini", "input": f"{hint_prompt}\n\n{hint_in}"},
                               timeout=60)
            if hr.status_code < 400:
                hj = hr.json()
                hint_text = hj.get("output_text") or \
                    (hj.get("output") or [{}])[0].get("content", [{}])[0].get("text", {}).get("value")
                if hint_text:
                    result["hint"] = hint_text.strip()

        return jsonify(result), 200

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

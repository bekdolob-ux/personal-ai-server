import os
import base64
import sqlite3
import logging
import threading
from pathlib import Path

import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("personal-ai")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 9 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
).strip()
API_ACCESS_KEY = os.getenv("API_ACCESS_KEY", "").strip()

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "memory.sqlite3"

db_lock = threading.Lock()
MAX_HISTORY = 20
MAX_IMAGE_BYTES = 8 * 1024 * 1024

SYSTEM_PROMPT = """
Сен акылдуу жеке AI жардамчысысың.
Кыргызча суроого кыргызча, орусча суроого орусча жооп бер.
Сүрөттөрдү талдап, көрүнгөн нерселерди түшүндүр.
Мурунку сүйлөшүүнүн контекстин колдон.
Жооптор так, сылык жана практикалык болсун.
Билбеген маалыматты ойлоп таппа.
"""

def connect_db():
    return sqlite3.connect(str(DB_PATH), timeout=30)

def init_db():
    with db_lock:
        with connect_db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON messages(session_id, id)
            """)
            db.commit()

def get_history(session_id):
    with db_lock:
        with connect_db() as db:
            rows = db.execute("""
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
            """, (session_id, MAX_HISTORY)).fetchall()

    rows.reverse()
    return [{"role": role, "content": content}
            for role, content in rows]

def save_message(session_id, role, content):
    with db_lock:
        with connect_db() as db:
            db.execute("""
                INSERT INTO messages(session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, role, str(content)[:12000]))
            db.commit()

def check_access():
    # API_ACCESS_KEY бош болсо, ачык API иштейт.
    # Коомдук серверде сөзсүз ачкыч коюу сунушталат.
    if not API_ACCESS_KEY:
        return True

    supplied = request.headers.get("X-API-Key", "")
    return supplied == API_ACCESS_KEY

def ask_ai(session_id, prompt, image=None, image_mime="image/jpeg"):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY орнотулган эмес")

    history = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    saved_prompt = prompt or "Сүрөттү талдап бер."

    if image is not None:
        encoded = base64.b64encode(image).decode("ascii")
        content = [
            {"type": "text", "text": saved_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{encoded}"
                }
            }
        ]
    else:
        content = saved_prompt

    messages.append({"role": "user", "content": content})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1200
        },
        timeout=90
    )

    if not response.ok:
        log.error("Groq API error %s: %s",
                  response.status_code, response.text[:1000])
        response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    if not isinstance(answer, str) or not answer.strip():
        answer = "Жооп түзүлбөй калды. Кайра аракет кыл."

    if image is not None:
        saved_prompt += " [Сүрөт тиркелген]"

    save_message(session_id, "user", saved_prompt)
    save_message(session_id, "assistant", answer)
    return answer.strip()

@app.get("/")
def home():
    return jsonify({
        "name": "Personal AI Server",
        "status": "running",
        "ai_configured": bool(GROQ_API_KEY),
        "image_support": True,
        "memory": "SQLite",
        "platforms": []
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/chat")
def chat():
    if not check_access():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("message", "")).strip()
    session_id = str(data.get("session_id", "")).strip()

    if not session_id or len(session_id) > 100:
        return jsonify({
            "error": "session_id керек (1–100 символ)"
        }), 400

    if not prompt:
        return jsonify({"error": "message керек"}), 400

    try:
        answer = ask_ai(session_id, prompt)
        return jsonify({"reply": answer})
    except Exception:
        log.exception("Chat request failed")
        return jsonify({
            "error": "AI жооп бере алган жок. Render Logs текшер."
        }), 502

@app.post("/analyze-image")
def analyze_image():
    if not check_access():
        return jsonify({"error": "Unauthorized"}), 401

    session_id = request.form.get("session_id", "").strip()
    prompt = request.form.get(
        "message", "Бул сүрөттө эмне бар? Түшүндүрүп бер."
    ).strip()
    uploaded = request.files.get("image")

    if not session_id or len(session_id) > 100:
        return jsonify({"error": "session_id керек"}), 400

    if not uploaded:
        return jsonify({"error": "image файлын тирке"}), 400

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    mime = (uploaded.mimetype or "").lower()

    if mime not in allowed_types:
        return jsonify({
            "error": "JPG, PNG же WEBP сүрөт гана кабыл алынат"
        }), 400

    image = uploaded.read(MAX_IMAGE_BYTES + 1)
    if not image:
        return jsonify({"error": "Сүрөт бош"}), 400

    if len(image) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Сүрөт 8 MB ашпашы керек"}), 413

    try:
        answer = ask_ai(session_id, prompt, image, mime)
        return jsonify({"reply": answer})
    except Exception:
        log.exception("Image analysis failed")
        return jsonify({
            "error": "Сүрөттү талдоо ишке ашкан жок. "
                     "Модель сүрөт колдой турганын текшер."
        }), 502

@app.post("/reset")
def reset_memory():
    if not check_access():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()

    if not session_id or len(session_id) > 100:
        return jsonify({"error": "session_id керек"}), 400

    with db_lock:
        with connect_db() as db:
            db.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            db.commit()

    return jsonify({"status": "memory_cleared"})

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Сурам өтө чоң"}), 413

init_db()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )

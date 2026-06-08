from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
load_dotenv()
import pickle
import requests
from bs4 import BeautifulSoup
import os
import re
from googlesearch import search
from pathlib import Path
import traceback

app = Flask(__name__)
app.secret_key = "fake-news-detector-secret-key-2024"

# ==============================
# GROQ API CONFIG
# ==============================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Може да смениш во: llama-3.1-70b-versatile

# ==============================
# LOAD ML MODELS
# ==============================

try:
    model = pickle.load(open("model/model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    print("ML моделите се вчитани.")
except Exception as e:
    model = None
    vectorizer = None
    print("ML модел не е пронајден:", e)

# ==============================
# SCRAPER
# ==============================

def get_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for el in soup(["script", "style"]):
            el.extract()
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])[:3000]
    except:
        return None

# ==============================
# GOOGLE SEARCH
# ==============================

def search_related_news(query):
    try:
        texts = []
        for url in search(query, num_results=5):
            txt = get_text_from_url(url)
            if txt:
                texts.append(txt[:1000])
        return "\n\n".join(texts)
    except:
        return ""

# ==============================
# GROQ HELPER (замена за Ollama)
# ==============================

def ask_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict fact-checking assistant. Always respond in the exact format requested. Never add text before LABEL:."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,  # Ниска температура = поконзистентни одговори
        "max_tokens": 1000
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# ==============================
# EXTRACT CLAIM
# ==============================

def extract_claim(text):
    try:
        return ask_groq(f"Extract the main claim from this text in one sentence. Return only the claim, nothing else:\n{text[:1200]}")
    except:
        return text[:200]

# ==============================
# MAIN AI FACT CHECK
# ==============================

def verify_with_ai(text):
    try:
        claim = extract_claim(text)
        web_sources = search_related_news(claim)

        prompt = f"""You are a strict fact checker. Your job is to determine if a claim is FAKE or REAL.

CRITICAL RULES:
- If the claim says someone is dead but they are alive = FAKE
- If the claim says someone holds a position but they do not = FAKE
- If the claim says something happened but it did not = FAKE
- If the claim is misleading, exaggerated, or false = FAKE
- If the claim is accurate and supported by evidence = REAL
- You MUST start your response with LABEL: and nothing else before it

Respond in EXACTLY this format, no extra text before LABEL:

LABEL: FAKE
CONFIDENCE: 95
REASON: Your explanation here

---

Claim to verify:
{text[:1200]}

Evidence from the internet:
{web_sources[:2000]}

Now respond starting with LABEL:"""

        output = ask_groq(prompt)

        print("=== RAW OUTPUT ===")
        print(output)
        print("==================")

        # Clean output — ако моделот напишал нешто пред LABEL:
        output_clean = output.strip()
        if "LABEL:" in output_clean.upper():
            idx = output_clean.upper().index("LABEL:")
            output_clean = output_clean[idx:]

        # Parse LABEL
        label_match = re.search(r"LABEL:\s*(FAKE|REAL)", output_clean.upper())
        if label_match:
            label = label_match.group(1)
        else:
            # Fallback: брои FAKE vs REAL keywords
            upper = output_clean.upper()
            fake_keywords = ["FAKE", "FALSE", "INCORRECT", "NOT TRUE", "MISINFORMATION", "DIED", "DEAD", "IS NOT REAL"]
            real_keywords = ["REAL", "TRUE", "ACCURATE", "CONFIRMED", "CORRECT", "VERIFIED"]
            fake_score = sum(1 for kw in fake_keywords if kw in upper)
            real_score = sum(1 for kw in real_keywords if kw in upper)
            label = "FAKE" if fake_score >= real_score else "REAL"

        # Parse CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*(\d+)", output_clean.upper())
        ai_conf = int(conf_match.group(1)) if conf_match else 75

        # Parse REASON
        reason_match = re.search(r"REASON:\s*(.+)", output_clean, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else output_clean

        # ==============================
        # SANITY CHECKS
        # ==============================

        # Sanity check 1: тврди дека некој е мртов, но причината вели дека е жив
        dead_claim = any(word in text.upper() for word in ["IS DEAD", "HAS DIED", "PASSED AWAY", "WAS KILLED"])
        alive_in_reason = any(word in reason.upper() for word in ["STILL ALIVE", "IS ALIVE", "STILL ACTIVE", "STILL LIVING"])
        if dead_claim and alive_in_reason and label == "REAL":
            label = "FAKE"
            ai_conf = max(ai_conf, 85)

        # Sanity check 2: причината содржи контрадикции — AI вели "не е точно" но label е REAL
        contradiction_phrases = [
            "NOT THE", "IS NOT", "WAS NOT", "ARE NOT",
            "COULDN'T FIND", "COULD NOT FIND", "NO EVIDENCE",
            "NOT ACCURATE", "INCORRECT", "DOES NOT HOLD",
            "RATHER THAN", "INSTEAD OF", "ACTUALLY",
            "HOWEVER", "IN FACT", "CONTRARY"
        ]
        contradiction_score = sum(1 for phrase in contradiction_phrases if phrase in reason.upper())
        if contradiction_score >= 2 and label == "REAL":
            label = "FAKE"
            ai_conf = max(ai_conf, 80)

        return label, ai_conf, reason

    except Exception as e:
        traceback.print_exc()
        return "REAL", 0, f"AI ERROR: {str(e)}"

# ==============================
# FLASK ROUTES
# ==============================

@app.route("/", methods=["GET"])
def index():
    history = session.get("history", [])
    return render_template("index.html", history=history)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Внесете текст или линк."})

    final_text = get_text_from_url(user_input) if user_input.startswith("http") else user_input

    if not final_text:
        return jsonify({"error": "Не може да се прочита линкот."})

    label, ai_conf, ai_reason = verify_with_ai(final_text)

    ml_conf = 0
    if model and vectorizer:
        try:
            vec = vectorizer.transform([final_text])
            ml_conf = max(model.predict_proba(vec)[0]) * 100
        except:
            ml_conf = 0

    status = "ЛАЖНА ВЕСТ 🔴" if label == "FAKE" else "ВЕРОЈАТНО ВИСТИНА 🟢"

    reply = {
        "status": status,
        "ai_conf": ai_conf,
        "ml_conf": round(ml_conf, 2),
        "reason": ai_reason,
        "label": label
    }

    if "history" not in session:
        session["history"] = []
    session["history"].append({"user": user_input, "bot": reply})
    session.modified = True

    return jsonify(reply)


@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return jsonify({"ok": True})


# ==============================

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import re
from googlesearch import search
from google import genai
from pathlib import Path
import traceback

# ==============================
# LOAD ENV (FIX FOR WINDOWS)
# ==============================

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY НЕ Е ПРОНАЈДЕН!")

client = genai.Client(api_key=API_KEY)

app = Flask(__name__)

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
# EXTRACT CLAIM
# ==============================

def extract_claim(text):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Extract main claim:\n{text[:1200]}"
        )
        return response.text.strip()
    except Exception:
        traceback.print_exc()
        return text[:200]

# ==============================
# MAIN AI FACT CHECK
# ==============================

def verify_with_ai(text):
    try:
        claim = extract_claim(text)
        web_sources = search_related_news(claim)

        prompt = f"""
ТИ СИ ПРОФЕСИОНАЛЕН FACT CHECKER.

ВЕСТ:
{text[:1200]}

ИНТЕРНЕТ ИЗВОРИ:
{web_sources[:3000]}

ОДГОВОР:
LABEL: REAL или FAKE
CONFIDENCE: 0-100
REASON: објаснување на македонски
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        output = response.text.upper()

        label = "FAKE" if "FAKE" in output else "REAL"
        conf_match = re.search(r"CONFIDENCE:\s*(\d+)", output)
        ai_conf = int(conf_match.group(1)) if conf_match else 80
        reason = response.text.split("REASON:")[-1].strip()

        return label, ai_conf, reason

    except Exception as e:
        traceback.print_exc()
        return "REAL", 0, f"AI ERROR: {str(e)}"

# ==============================
# FLASK
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence_display, explanation = "", "", ""

    if request.method == "POST":
        user_input = request.form.get("news", "").strip()

        if not user_input:
            return render_template("index.html", result="Внесете текст или линк.")

        final_text = get_text_from_url(user_input) if user_input.startswith("http") else user_input

        if not final_text:
            return render_template("index.html", result="Не може да се прочита линкот.")

        label, ai_conf, ai_reason = verify_with_ai(final_text)

        ml_conf = 0
        if model and vectorizer:
            try:
                vec = vectorizer.transform([final_text])
                ml_conf = max(model.predict_proba(vec)[0]) * 100
            except:
                ml_conf = 0

        final_status = "ЛАЖНА ВЕСТ 🔴" if label == "FAKE" else "ВЕРОЈАТНО ВИСТИНА 🟢"

        result = f"Резултат: {final_status}"
        confidence_display = f"AI сигурност: {ai_conf}% | ML сигурност: {round(ml_conf,2)}%"
        explanation = ai_reason

    return render_template("index.html",
                           result=result,
                           confidence=confidence_display,
                           explanation=explanation)

# ==============================

if __name__ == "__main__":
    app.run(debug=True)

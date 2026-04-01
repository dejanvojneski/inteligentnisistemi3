from flask import Flask, render_template, request
import pickle
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

app = Flask(__name__)

# 🔑 Твојот OpenAI API Клуч
client = OpenAI(api_key="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")

# Load your model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


def get_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:1500]  # Го кратиме текстот за да не трошиме премногу токени
    except Exception as e:
        return None


def get_ai_explanation(news_text, label):
    prompt = f"""
    Анализирај ја следнава вест која мојот AI модел ја класифицираше како {label}.
    Објасни накратко (2-3 реченици) на македонски јазик зошто оваа содржина делува како {label}.
    Фокусирај се на стилот на пишување, изворите или сензационализмот.

    Текст: {news_text[:500]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        return "Не можев да генерирам објаснување во моментов."


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    confidence = ""
    explanation = ""

    if request.method == "POST":
        user_input = request.form["news"]

        # Проверка за линк
        if user_input.startswith("http"):
            final_text = get_text_from_url(user_input)
            if not final_text:
                return render_template("index.html", result="Грешка: Не можев да го прочитам линкот.")
        else:
            final_text = user_input

        # ML Предикција
        vec = vectorizer.transform([final_text])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]

        # Процент и лабела
        confidence = f"{round(max(probabilities) * 100, 2)}%"
        status = "ВИСТИНА" if prediction == "REAL" else "ЛАЖНА ВЕСТ"
        result = f"Моделот вели: {status}"

        # ChatGPT Објаснување
        explanation = get_ai_explanation(final_text, status)

    return render_template("index.html", result=result, confidence=confidence, explanation=explanation)


if __name__ == "__main__":
    app.run(debug=True)
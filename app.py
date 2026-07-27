from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import webbrowser
import threading

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("✅ APP STARTING...")

app = Flask(__name__)

# ---------- STOPWORDS ----------
stop_words = set(ENGLISH_STOP_WORDS)
if "not" in stop_words:
    stop_words.remove("not")

# ---------- CLEAN ----------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)

    words = text.lower().split()
    words = [w for w in words if w not in stop_words]

    text = " ".join(words)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text.strip()

# ---------- LOAD DATA ----------
df = pd.read_csv("cyberbullying_tweets.csv")

df["cleaned"] = df["tweet_text"].apply(clean_text)
df["label"] = df["cyberbullying_type"].apply(
    lambda x: 0 if str(x).lower() == "not_cyberbullying" else 1
)

X = df["cleaned"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------- ML ----------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

ml_model = LogisticRegression(max_iter=300)
ml_model.fit(X_train_vec, y_train)

# ---------- DL ----------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=50)

dl_model = Sequential()
dl_model.add(Embedding(5000, 128))
dl_model.add(LSTM(64))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(1, activation='sigmoid'))

dl_model.compile(loss='binary_crossentropy', optimizer='adam')
dl_model.fit(X_train_pad, y_train, epochs=3, batch_size=128)

# ---------- CATEGORY ----------
def get_category(score, is_cb):
    if not is_cb:
        if score < 0.2:
            return "Positive Comment"
        else:
            return "Neutral Comment"
    else:
        if score < 0.7:
            return "Moderate Cyberbullying"
        else:
            return "Severe Cyberbullying"

# ---------- RESPONSE ----------
def get_response(score, is_cb):
    if not is_cb:
        return "✅ Comment is safe and posted."
    elif score < 0.6:
        return "⚠️ This comment may be harmful. Please edit it."
    elif score < 0.8:
        return "🗑️ This comment has been removed."
    elif score < 0.9:
        return "⏳ You are temporarily restricted."
    else:
        return "🚫 Your account is restricted."

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    user_input = request.form["text"]
    cleaned = clean_text(user_input)

    vec = vectorizer.transform([cleaned])
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=50)

    ml = ml_model.predict_proba(vec)[0][1]
    dl = float(dl_model.predict(pad)[0][0])

    score = (0.5 * ml) + (0.5 * dl)

    # 🔥 FIXED THRESHOLD
    is_cb = score >= 0.55

    category = get_category(score, is_cb)
    response = get_response(score, is_cb)

    final_label = "Cyberbullying" if is_cb else "Non-Cyberbullying"

    return jsonify({
        "user_input": user_input,
        "response": response,
        "score": round(score, 2),
        "category": category,
        "final_label": final_label
    })

# ---------- AUTO OPEN ----------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("🚀 STARTING FLASK SERVER...")
    threading.Timer(2, open_browser).start()
    app.run(debug=False)

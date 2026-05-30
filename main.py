from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import email
import imaplib
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

import pickle
tfidf = pickle.load(open("tfidf.pkl","rb"))
LR = pickle.load(open("LR_model.pkl","rb"))


app = Flask(__name__)

SPAM_THRESHOLD = 0.35
SUSPICIOUS_KEYWORDS = {
    "account",
    "act",
    "bonus",
    "cash",
    "claim",
    "click",
    "congratulations",
    "credit",
    "free",
    "limited",
    "loan",
    "lottery",
    "offer",
    "password",
    "prize",
    "reward",
    "selected",
    "suspended",
    "urgent",
    "verify",
    "winner",
    "won",
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "bad", "good"}

IMAP_PROVIDERS = {
    "gmail": {"label": "Gmail", "host": "imap.gmail.com", "port": 993, "spam_folder": "[Gmail]/Spam"},
    "outlook": {"label": "Outlook / Hotmail", "host": "outlook.office365.com", "port": 993, "spam_folder": "Junk Email"},
    "yahoo": {"label": "Yahoo Mail", "host": "imap.mail.yahoo.com", "port": 993, "spam_folder": "Bulk Mail"},
    "custom": {"label": "Custom IMAP", "host": "", "port": 993, "spam_folder": "Spam"},
}

def cleaning_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def suspicious_keyword_score(clean_text):
    words = set(clean_text.split())
    hits = len(words.intersection(SUSPICIOUS_KEYWORDS))
    if hits >= 5:
        return 0.85
    if hits >= 4:
        return 0.70
    if hits >= 3:
        return 0.55
    return 0.0

def predict_email(text):
    clean_email = cleaning_text(text)
    vectorize = tfidf.transform([clean_email])
    model_spam_probability = float(LR.predict_proba(vectorize)[0][1])
    spam_probability = max(model_spam_probability, suspicious_keyword_score(clean_email))
    prediction = 1 if spam_probability >= SPAM_THRESHOLD else 0
    return {
        "value": prediction,
        "label": "Spam" if prediction == 1 else "Ham",
        "class_name": "spam" if prediction == 1 else "ham",
        "confidence": round(spam_probability * 100, 1),
        "model_confidence": round(model_spam_probability * 100, 1),
    }

def decode_header_value(value):
    if not value:
        return ""

    decoded_parts = email.header.decode_header(value)
    decoded_value = ""
    for content, charset in decoded_parts:
        if isinstance(content, bytes):
            decoded_value += content.decode(charset or "utf-8", errors="replace")
        else:
            decoded_value += content
    return decoded_value

def normalize_email_text(text):
    text = re.sub(r"(?is)<(script|style|code|pre|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def html_to_plain_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "code", "pre", "noscript", "svg"]):
        tag.decompose()
    return normalize_email_text(soup.get_text(" "))

def get_message_body(message):
    plain_parts = []
    html_parts = []

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", "")).lower()

            if "attachment" in disposition:
                continue

            if content_type in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True)
                if not payload:
                    continue

                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if content_type == "text/plain":
                    plain_parts.append(normalize_email_text(text))
                elif content_type == "text/html":
                    html_parts.append(html_to_plain_text(text))
    else:
        payload = message.get_payload(decode=True)
        if payload:
            charset = message.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if message.get_content_type() == "text/html":
                html_parts.append(html_to_plain_text(text))
            else:
                plain_parts.append(normalize_email_text(text))

    readable_parts = plain_parts or html_parts
    return normalize_email_text(" ".join(part for part in readable_parts if part))

def safe_positive_int(value, default):
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed_value, 0)

def fetch_folder_messages(host, port, username, password, folder="INBOX", limit=15, source_label="Inbox"):
    messages = []

    if int(limit) <= 0:
        return messages

    with imaplib.IMAP4_SSL(host, int(port)) as mail:
        mail.login(username, password)
        status, _ = mail.select(folder or "INBOX", readonly=True)
        if status != "OK":
            raise ValueError(f"Could not open folder '{folder}'.")

        status, data = mail.search(None, "ALL")
        if status != "OK":
            raise ValueError("Could not search mailbox.")

        message_ids = data[0].split()
        for message_id in reversed(message_ids[-int(limit):]):
            status, message_data = mail.fetch(message_id, "(RFC822)")
            if status != "OK" or not message_data or not message_data[0]:
                continue

            raw_message = message_data[0][1]
            parsed_message = email.message_from_bytes(raw_message)
            subject = decode_header_value(parsed_message.get("Subject")) or "(No subject)"
            sender = decode_header_value(parsed_message.get("From")) or "(Unknown sender)"
            date = decode_header_value(parsed_message.get("Date"))
            body = get_message_body(parsed_message)
            text_for_prediction = f"{subject} {subject} {body}"
            prediction = predict_email(text_for_prediction)

            messages.append({
                "id": message_id.decode("utf-8", errors="ignore"),
                "subject": subject,
                "sender": sender,
                "date": date,
                "source": source_label,
                "source_folder": folder,
                "body": body[:3000] if body else "(No readable email body found.)",
                "preview": body[:180] if body else "No readable preview available.",
                "prediction": prediction,
            })

    return messages

@app.route('/')
def index():
    return render_template('index.html', providers=IMAP_PROVIDERS)

@app.route('/Prediction', methods=['POST', 'GET'])
def Prediction():
    if request.method == "POST":
        email = request.form['Email']
        if email.strip():  # Check if not empty
            prediction = predict_email(email)
            return render_template("index.html", prediction=prediction, manual_email=email, providers=IMAP_PROVIDERS)
        else:
            return render_template("index.html", prediction="empty", providers=IMAP_PROVIDERS)
    else:
        return render_template("index.html", providers=IMAP_PROVIDERS)

@app.route('/ImportEmails', methods=['POST'])
def import_emails():
    provider = request.form.get("provider", "gmail")
    provider_settings = IMAP_PROVIDERS.get(provider, IMAP_PROVIDERS["gmail"])
    host = request.form.get("host") or provider_settings["host"]
    port = request.form.get("port") or provider_settings["port"]
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    folder = request.form.get("folder", "INBOX").strip() or "INBOX"
    spam_folder = request.form.get("spam_folder", provider_settings["spam_folder"]).strip() or provider_settings["spam_folder"]
    inbox_limit = safe_positive_int(request.form.get("inbox_limit"), 8)
    spam_limit = safe_positive_int(request.form.get("spam_limit"), 7)

    if not host or not username or not password:
        return render_template(
            "index.html",
            providers=IMAP_PROVIDERS,
            email_error="Please enter your IMAP host, email address, and app password.",
        )

    try:
        messages = fetch_folder_messages(host, port, username, password, folder, inbox_limit, "Inbox")
        messages += fetch_folder_messages(host, port, username, password, spam_folder, spam_limit, "Spam")
    except Exception as exc:
        return render_template(
            "index.html",
            providers=IMAP_PROVIDERS,
            email_error=f"Could not import emails: {exc}",
        )

    return render_template(
        "index.html",
        providers=IMAP_PROVIDERS,
        imported_messages=messages,
        selected_provider=provider,
        selected_host=host,
        selected_port=port,
        selected_username=username,
        selected_folder=folder,
        selected_spam_folder=spam_folder,
        selected_inbox_limit=inbox_limit,
        selected_spam_limit=spam_limit,
    )

    
if __name__=="__main__":
    app.run(debug=True)

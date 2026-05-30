# Email Spam Detection

A Flask web app that detects whether an email is spam or ham using a trained TF-IDF vectorizer and Logistic Regression model. The app supports both manual email text checking and real inbox import through IMAP.

## Features

- Paste an email message and classify it as spam or ham.
- Import recent emails from Gmail, Outlook/Hotmail, Yahoo, or a custom IMAP server.
- Review imported emails in a dashboard-style inbox.
- Click an imported email to view its content and prediction.
- Responsive UI for desktop and mobile screens.

## Project Structure

```text
Email-Spam-Detection/
+-- main.py
+-- LR_model.pkl
+-- tfidf.pkl
+-- requirements.txt
+-- README.md
+-- static/
|   +-- style.css
+-- templates/
    +-- index.html
```

## Requirements

- Python 3.10+
- Flask
- scikit-learn
- nltk

The full dependency list is available in `requirements.txt`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python main.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000/
```

## Using Real Email Import

The inbox dashboard uses IMAP to read recent emails from your mailbox. Select your provider, enter your email address, app password, folder name, and the number of emails to import.

Common IMAP hosts:

| Provider | IMAP Host | Port |
| --- | --- | --- |
| Gmail | `imap.gmail.com` | `993` |
| Outlook / Hotmail | `outlook.office365.com` | `993` |
| Yahoo Mail | `imap.mail.yahoo.com` | `993` |

For most providers, your normal account password will not work. Create an app password from your email account security settings and use that in the app.

## Security Notes

- Email credentials are used only for the current import request.
- Credentials are not saved by the app.
- Do not commit `.env` files, local logs, virtual environments, or secrets.
- This project uses Flask's development server, so it is intended for local development and demos.

## Model Files

The app expects these trained model artifacts in the project root:

- `tfidf.pkl`
- `LR_model.pkl`

`tfidf.pkl` transforms cleaned email text into model features. `LR_model.pkl` predicts whether the email is spam or ham.

## Prediction Labels

- `Ham`: normal email
- `Spam`: unwanted or suspicious email

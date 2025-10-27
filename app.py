from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the NLLB model and tokenizer
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Detect device (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
model.eval()

# Supported languages (you can add more)
LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans"
}


def translate(text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    """Translate text using the NLLB model."""
    text = text.strip()
    if not text:
        return ""

    try:
        # Set source language
        tokenizer.src_lang = src_lang

        # Encode input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        # Get the BOS token ID for the target language
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=200
            )

        # Decode output tokens
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        return f"⚠️ Translation error: {e}"


@app.route("/", methods=["GET", "POST"])
def home():
    translated_text = ""
    input_text = ""
    src_lang = "eng_Latn"
    tgt_lang = "hin_Deva"

    if request.method == "POST":
        input_text = request.form.get("text", "")
        src_lang = request.form.get("src_lang", "eng_Latn")
        tgt_lang = request.form.get("tgt_lang", "hin_Deva")
        translated_text = translate(input_text, src_lang, tgt_lang)

    return render_template(
        "index.html",
        input_text=input_text,
        translated_text=translated_text,
        languages=LANGUAGES,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5913, host="0.0.0.0")


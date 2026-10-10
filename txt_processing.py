
import os
import re
import nltk
from nltk.corpus import stopwords

# Use a writable directory on Vercel
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Tell NLTK where to find its resources
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Download required resources
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download("omw-1.4", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download(
    "averaged_perceptron_tagger_eng",
    download_dir=NLTK_DATA_DIR,
    quiet=True
)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=True)


def preprocess(txt):
    """
    Preprocess a list of resume texts.

    :param txt: list containing texts
    :return: preprocessed list of texts
    """
    sw = set(stopwords.words("english"))
    p_txt = []

    for resume in txt:
        text = re.sub(r"\s+", " ", resume)  # Remove extra spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = text.split()
        text = [word for word in text if word.lower() not in sw]
        text = [word.lower() for word in text]
        p_txt.append(" ".join(text))

    return p_txt

import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# carregar o modelo treinado
model = load_model('game_sentiment_model.keras')

# carrega novamente o tokenizador treinado
# OBS: o tokenizador original precisa ser salvo; se não salvou, terá que refazer o fit no mesmo corpus
# para este exemplo, vamos refazer:
df = pd.read_csv('train.csv')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

df['cleaned_review'] = df['user_review'].apply(clean_text)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['cleaned_review'])

# função de demonstração
def predict_review(review_text):
    cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    prob = model.predict(padded, verbose=0)[0][0]
    sentiment = "POSITIVO" if prob >= 0.5 else "NEGATIVO"
    print(f"Frase: {review_text}")
    print(f"Probabilidade positiva: {prob:.2f}")
    print(f"Classificação: {sentiment}\n")

# exemplos de teste
predict_review("I didn't like the controls, too clunky")
predict_review("The multiplayer mode is fantastic, very engaging")

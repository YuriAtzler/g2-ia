import pandas as pd
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from sklearn.metrics import recall_score, f1_score, classification_report

# Carregar o dataset
df = pd.read_csv('train.csv')

# Função para limpar o texto
# Remove texto HTML e caracteres especiais, e converte para minúsculas
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text.lower().strip()

# Aplicar a função de limpeza
df['cleaned_review'] = df['user_review'].apply(clean_text)

# Criar tokens, limite o vocabulário a 10.000 palavras
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['cleaned_review'])

# Converter o texto em sequências de inteiros
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

# Aplicar padding
max_len = 200
X = pad_sequences(sequences, maxlen=max_len)

# Converter os rótulos para formato binário
y = df['user_suggestion'].values

# Definir o modelo com LSTM
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Definir o callback para early stopping
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Treinar o modelo
history = model.fit(X, y, epochs=10, batch_size=512, validation_split=0.2, callbacks=[es], verbose=1)

# Avaliar o modelo
loss, acc = model.evaluate(X, y)
print(f"Desempenho no teste — Loss: {loss:.4f}, Acurácia: {acc:.4f}")

# Obter probabilidades previstas
y_pred_prob = model.predict(X, verbose=0)

# Converter em classes (0 ou 1)
y_pred = (y_pred_prob >= 0.5).astype(int).reshape(-1)

# Calcular recall e F1‑score
recall = recall_score(y, y_pred)
f1    = f1_score(y, y_pred)

print(f"Recall: {recall:.4f}")
print(f"F1‑score: {f1:.4f}\n")

model.save('game_sentiment_model.keras')
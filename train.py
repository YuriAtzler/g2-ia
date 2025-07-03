import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM

# Carregar o dataset
df = pd.read_csv('train.csv')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remover tags HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remover caracteres não alfabéticos
    return text.lower().strip()

# Aplicar a função de limpeza
df['cleaned_review'] = df['user_review'].apply(clean_text)

# Inicializar o tokenizador
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

model.save('game_sentiment_model.keras')
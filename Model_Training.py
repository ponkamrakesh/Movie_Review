import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from sklearn.model_selection import train_test_split

# 🔹 Load dataset
IMDB_Df = pd.read_csv("C:\\Rakesh_DataScience\\pythonproject\\Movie_Review\\dataset\\IMDB Dataset.csv")

print(IMDB_Df.head())
print("Null Values count --> ", IMDB_Df.isnull().sum())
print(type(IMDB_Df))

# 🔹 Drop Nulls
IMDB_Df = IMDB_Df.dropna()
print("Null Values count after dropping -->", IMDB_Df.isnull().sum())
print("Dataset Shape:", IMDB_Df.shape)
print(IMDB_Df['sentiment'].value_counts())
# 🔹 Map labels: -1 → 0, 0 → 1, 1 → 2
label_map = {"negative": 0, "positive": 1}

IMDB_Df["sentiment"] = IMDB_Df["sentiment"].map(label_map)


# 🔹 Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(IMDB_Df['review'])

sequences = tokenizer.texts_to_sequences(IMDB_Df['review'])
max_len = 50  # can be tuned

X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = IMDB_Df['sentiment'].values  # ✅ Corrected

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔹 Build GRU model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),
    GRU(64),
    Dense(1, activation='sigmoid')  # ✅ ONE node, sigmoid
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

# 🔹 Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 🔹 Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.2f}")

# 🔹 Prediction function
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = model.predict(padded)[0][0]  # single sigmoid output
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, float(prob)



# 🔹 Test prediction
sentiment, confidence = predict_sentiment("This is absolutely amazing!")
print(f"🚀 Sentiment: {sentiment} → Probabilities: {confidence}")


import os

# 🔹 Define the path to save the model
model_dir = "C:\\Rakesh_DataScience\\pythonproject\\Movie_Review\\model"

# 🔹 Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# 🔹 Define the full file path
model_path = os.path.join(model_dir, "sentiment_gru_model.h5")

# 🔹 Save the model
model.save(model_path)

import pickle

with open(os.path.join(model_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved.")
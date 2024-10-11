# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Preview the dataset
print(data.head())

# Step 1: Preprocess the text data
# Convert sentiment labels from 'positive'/'negative' to 1/0
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Function to preprocess text (convert to lowercase and remove special characters)
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    return text

# Apply preprocessing to the reviews
data['review'] = data['review'].apply(preprocess_text)

# Step 2: Tokenize and pad the sequences
max_vocab_size = 10000  # Max number of words in the vocabulary
max_sequence_length = 200  # Max length of the review sequence

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(data['review'].values)
sequences = tokenizer.texts_to_sequences(data['review'].values)

# Pad sequences to make them of uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['sentiment'].values, test_size=0.2, random_state=42)

# Step 4: Build the LSTM model (or GRU model)
embedding_dim = 128  # Embedding size for each token

model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(128, return_sequences=False),  # Change to GRU(128) for GRU model
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output for sentiment classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Step 5: Train the model
epochs = 5
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Step 7: Predict and calculate accuracy using sklearn's accuracy_score
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

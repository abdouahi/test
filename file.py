import numpy as np
import tensorflow as tf
import random
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SimpleChatModel:
    def __init__(self, vocab_size=1000, max_length=20):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # Input embedding layer
            Embedding(self.vocab_size, 128, input_length=self.max_length),
            # LSTM layer to learn sequence patterns
            LSTM(256, return_sequences=False),
            # Dense layer to process LSTM outputs
            Dense(256, activation='relu'),
            # Output layer with softmax activation for word prediction
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        return model
    
    def train(self, questions, answers, epochs=100):
        """Train model on question-answer pairs"""
        # Fit tokenizer on all text
        all_text = questions + answers
        self.tokenizer.fit_on_texts(all_text)
        
        # Convert text to sequences
        X = self.tokenizer.texts_to_sequences(questions)
        X = pad_sequences(X, maxlen=self.max_length)
        
        # One-hot encode the answers for training
        y = self.tokenizer.texts_to_sequences(answers)
        y_data = []
        for seq in y:
            if seq:
                # Create one-hot vector for the first word of response
                y_one_hot = np.zeros((self.vocab_size))
                y_one_hot[seq[0]] = 1
                y_data.append(y_one_hot)
            else:
                y_data.append(np.zeros((self.vocab_size)))
                
        self.model.fit(X, np.array(y_data), epochs=epochs, verbose=1)
        
        # Save the tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_response(self, text):
        """Generate a simple response to input text"""
        # Convert input to sequence
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_length)
        
        # Predict next word probabilities
        pred = self.model.predict(padded)[0]
        
        # Get top 3 predicted words
        top_indices = pred.argsort()[-3:][::-1]
        
        # Convert indices back to words
        words = []
        for idx in top_indices:
            for word, i in self.tokenizer.word_index.items():
                if i == idx:
                    words.append(word)
                    break
        
        # Create simple response using predicted words
        if words:
            response = f"I think about {' and '.join(words)} when you say that."
        else:
            response = "I'm not sure how to respond to that."
        
        return response

def train_new_model():
    """Train a new chat model with sample data"""
    # Simple training data - question/answer pairs
    questions = [
        "hello how are you",
        "what's your name",
        "how does this work",
        "what can you do",
        "tell me about yourself",
        "do you like music",
        "what's the weather today",
        "how old are you",
        "where are you from",
        "do you have any hobbies",
        "what do you think about ai",
        "can you help me with something",
        "what time is it",
        "do you sleep",
        "are you smart",
        "who created you",
        "what's your favorite color",
        "tell me a joke",
        "do you have friends",
        "can you learn new things"
    ]

    answers = [
        "i'm doing well thanks",
        "i'm a simple chat ai",
        "i use neural networks to learn patterns",
        "i can have simple conversations",
        "i'm a basic ai model",
        "i enjoy all kinds of music",
        "i don't have access to weather data",
        "i was just created recently",
        "i exist in a computer program",
        "i like talking with people",
        "ai is fascinating and evolving quickly",
        "i'll try my best to assist you",
        "i don't have access to real-time information",
        "i don't need sleep like humans do",
        "i'm just a simple model trying my best",
        "a programmer built me with neural networks",
        "i don't really perceive colors",
        "sorry i'm not good at jokes yet",
        "you can be my friend if you want",
        "i learn from conversations like this one"
    ]

    # Create and train the model
    print("Creating chat model...")
    model = SimpleChatModel(vocab_size=500, max_length=10)
    print("Training model on sample conversations...")
    model.train(questions, answers, epochs=50)
    print("Training complete!")

    # Save the model for later use
    tf.keras.models.save_model(model.model, 'simple_chat_model')
    print("Model saved to 'simple_chat_model'")
    
    return model

def have_conversation(model):
    """Have a conversation with the trained model"""
    print("=== Conversation with Simple AI ===")
    print("(Type 'exit' to end the conversation)")
    print("AI: Hello! I'm a simple chat AI. How can I help you today?")
    
    # Some generic responses for variety
    generic_responses = [
        "That's interesting! Tell me more.",
        "I understand what you mean.",
        "I'm learning to have better conversations.",
        "I see. That makes sense.",
        "Thanks for sharing that with me!",
        "I haven't thought about it that way before.",
        "I'd like to learn more about that.",
        "That's a good point.",
        "I'm not sure I fully understand, but I'm trying.",
        "Could you explain that differently?",
        "That's something I've been thinking about too."
    ]
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("AI: Goodbye! It was nice chatting with you.")
            break
        
        # 40% chance of using the neural model, 60% chance of generic response for variety
        if random.random() < 0.4:
            response = model.get_response(user_input)
        else:
            response = random.choice(generic_responses)
            
        print(f"AI: {response}")

if __name__ == "__main__":
    # Check if model exists already
    if os.path.exists('simple_chat_model') and os.path.exists('tokenizer.pickle'):
        print("Loading existing model...")
        model = SimpleChatModel()
        model.model = tf.keras.models.load_model('simple_chat_model')
        
        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            model.tokenizer = pickle.load(handle)
        
        print("Model loaded successfully!")
    else:
        print("No existing model found. Training a new model...")
        model = train_new_model()
    
    # Start conversation
    have_conversation(model)

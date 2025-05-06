import os
import warnings
import torch
import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ModelManager:
    def __init__(self):
        self.models = None
        self.device = None
        self.initialize_models()

    @st.cache_resource(show_spinner=False)
    def initialize_models(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load models with error handling
            try:
                self.models = {
                    'lstm_model': self._load_model('./models/model_lstm.pth'),
                    'cnn_model': self._load_model('./models/model_cnn.pth'),
                    'bert_model': self._load_model('./models/model.pth'),
                }
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return None

            # Load and initialize tokenizers and encoders
            try:
                self.models['bert_tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')
                
                # Load data and initialize tokenizer
                data = pd.read_csv('./data/cyberbullying_tweets.csv')
                tokenizer = tf.keras.preprocessing.text.Tokenizer()
                tokenizer.fit_on_texts(data['tweet_text'])
                self.models['tokenizer_lstm_cnn'] = tokenizer
                
                # Initialize label encoder
                label_encoder = LabelEncoder()
                label_encoder.fit(data['cyberbullying_type'])
                self.models['label_encoder'] = label_encoder
                
                self.models['device'] = self.device
                
                return self.models
            
            except Exception as e:
                st.error(f"Error initializing tokenizers and encoders: {str(e)}")
                return None

        except Exception as e:
            st.error(f"Error in model initialization: {str(e)}")
            return None

    def _load_model(self, path):
        """Safely load a model with error handling"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            return torch.load(path, map_location=self.device)
        except Exception as e:
            raise Exception(f"Error loading model from {path}: {str(e)}")

    def preprocess_text_lstm_cnn(self, text, max_length):
        """Preprocess text for LSTM/CNN models"""
        try:
            sequence = self.models['tokenizer_lstm_cnn'].texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=max_length)
            return torch.tensor(padded_sequence, dtype=torch.int64).to(self.device)
        except Exception as e:
            raise Exception(f"Error in LSTM/CNN preprocessing: {str(e)}")

    def preprocess_text_bert(self, text, max_length=128):
        """Preprocess text for BERT model"""
        try:
            encoding = self.models['bert_tokenizer'].encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return (encoding['input_ids'].to(self.device), 
                   encoding['attention_mask'].to(self.device))
        except Exception as e:
            raise Exception(f"Error in BERT preprocessing: {str(e)}")

    def predict(self, text, max_length=128):
        """Make ensemble prediction"""
        try:
            if self.models is None:
                raise Exception("Models not properly initialized")

            # LSTM prediction
            lstm_input = self.preprocess_text_lstm_cnn(text, max_length)
            lstm_input = lstm_input.cpu().numpy()
            with tf.device('/CPU:0'):  # Force CPU usage for TF
                lstm_output = self.models['lstm_model'].predict(lstm_input, verbose=0)
                lstm_probs = tf.nn.softmax(lstm_output).numpy()

            # CNN prediction
            cnn_input = self.preprocess_text_lstm_cnn(text, max_length)
            cnn_input = cnn_input.cpu().numpy()
            with tf.device('/CPU:0'):  # Force CPU usage for TF
                cnn_output = self.models['cnn_model'].predict(cnn_input, verbose=0)
                cnn_probs = tf.nn.softmax(cnn_output).numpy()

            # BERT prediction
            bert_input_ids, bert_attention_mask = self.preprocess_text_bert(text)
            with torch.no_grad():
                outputs = self.models['bert_model'](
                    bert_input_ids, 
                    attention_mask=bert_attention_mask
                )
                logits = outputs.logits.cpu()
                bert_probs = torch.nn.functional.softmax(logits, dim=1).numpy()

            # Ensemble voting
            weights = [1/3, 1/3, 1/3]
            final_probs = (
                weights[0] * lstm_probs + 
                weights[1] * cnn_probs + 
                weights[2] * bert_probs
            )

            final_prediction = np.argmax(final_probs, axis=1)[0]
            result = self.models['label_encoder'].inverse_transform([final_prediction])[0]
            confidence = float(np.max(final_probs))

            return result, confidence, final_probs[0]

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return "not cyber bullying", 0.0, None
import streamlit as st
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.serialization import safe_globals, add_safe_globals
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import io
import csv

# Add necessary globals to the safe list
add_safe_globals(['getattr'])

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# # Set page config
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode support
def get_custom_css():
    return f"""
    <style>
    .main {{
        padding: 2rem;
    }}
    .stAlert {{
        padding: 1rem;
        margin: 1rem 0;
    }}
    .stats-card {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }}
    </style>
    """

# # Custom CSS to improve appearance
# st.markdown("""
#     <style>
#     .main {
#         padding: 2rem;
#     }
#     .stAlert {
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# Function to generate downloadable link
def get_download_link(data, filename, text):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerows(data)
    csv_string = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Add batch processing function
def process_batch_texts(texts, models_dict):
    results = []
    for text in texts:
        prediction, probability = ensemble_predict(text.strip(), models_dict)
        results.append([text, prediction, probability])
    return results

@st.cache_resource
def load_models():
    """Load all models and tokenizers"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models with updated configurations
        lstm_model = torch.load('./models/model_lstm.pth', map_location=device, weights_only=False)
        cnn_model = torch.load('./models/model_cnn.pth', map_location=device, weights_only=False)
        bert_model = torch.load('./models/model_c.pth', map_location=device, weights_only=False)
        
        # Load tokenizers
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_lstm_cnn = tf.keras.preprocessing.text.Tokenizer()
        
        # Load data and fit tokenizer
        try:
            data = pd.read_csv('./data/cyberbullying_tweets.csv')
        except FileNotFoundError:
            st.error("Could not find cyberbullying_tweets.csv. Please ensure the file is in the correct directory.")
            return None
            
        tokenizer_lstm_cnn.fit_on_texts(data['tweet_text'])
        
        # Initialize label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(data['cyberbullying_type'])
        
        # Calculate max length
        max_length = max([len(seq) for seq in tokenizer_lstm_cnn.texts_to_sequences(data['tweet_text'])])
        
        return {
            'lstm_model': lstm_model,
            'cnn_model': cnn_model,
            'bert_model': bert_model,
            'bert_tokenizer': bert_tokenizer,
            'tokenizer_lstm_cnn': tokenizer_lstm_cnn,
            'label_encoder': label_encoder,
            'max_length': max_length,
            'device': device
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("""
        Troubleshooting steps:
        1. Ensure all model files are in the correct directory
        2. Check if the model files are compatible with your PyTorch version
        3. Verify that all required packages are installed
        """)
        return None

def preprocess_text_lstm_cnn(text, tokenizer, max_length, device):
    """Preprocess text for LSTM and CNN models"""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return torch.tensor(padded_sequence, dtype=torch.int64).to(device)

def preprocess_text_bert(text, tokenizer, device, max_length=128):
    """Preprocess text for BERT model"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

def ensemble_predict(text, models_dict):
    """Make prediction using ensemble of models"""
    try:
        # Get predictions from each model
        # LSTM prediction
        lstm_input = preprocess_text_lstm_cnn(
            text, 
            models_dict['tokenizer_lstm_cnn'], 
            models_dict['max_length'],
            models_dict['device']
        )
        lstm_input = lstm_input.cpu().numpy()
        lstm_output = models_dict['lstm_model'].predict(lstm_input, verbose=0)
        lstm_probs = tf.nn.softmax(lstm_output).numpy()

        # CNN prediction
        cnn_input = preprocess_text_lstm_cnn(
            text,
            models_dict['tokenizer_lstm_cnn'],
            models_dict['max_length'],
            models_dict['device']
        )
        cnn_input = cnn_input.cpu().numpy()
        cnn_output = models_dict['cnn_model'].predict(cnn_input, verbose=0)
        cnn_probs = tf.nn.softmax(cnn_output).numpy()

        # BERT prediction
        bert_input_ids, bert_attention_mask = preprocess_text_bert(
            text,
            models_dict['bert_tokenizer'],
            models_dict['device']
        )
        with torch.no_grad():
            outputs = models_dict['bert_model'](bert_input_ids, attention_mask=bert_attention_mask)
            logits = outputs.logits.to('cpu')
            bert_probs = torch.nn.functional.softmax(logits, dim=1).numpy()

        # Ensemble voting with equal weights
        weights = [1/3, 1/3, 1/3]
        final_probs = (weights[0] * lstm_probs + weights[1] * cnn_probs + weights[2] * bert_probs)
        
        # Get prediction and probabilities
        final_prediction = np.argmax(final_probs, axis=1)[0]
        prediction_probability = float(np.max(final_probs))
        
        result = models_dict['label_encoder'].inverse_transform([final_prediction])[0]
        return result, prediction_probability

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error in prediction", 0.0
    
def visualize_prediction_distribution(history):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    fig = px.pie(
        df, 
        names='prediction', 
        title='Distribution of Predictions',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig

def visualize_confidence_trend(history):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    fig = px.line(
        df, 
        x=df.index, 
        y='confidence', 
        title='Confidence Trend Over Time',
        labels={'index': 'Prediction Number', 'confidence': 'Confidence Score'}
    )
    return fig

def main():
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # System information
    st.sidebar.markdown(f"""
    ### System Information
    - PyTorch Version: {torch.__version__}
    - TensorFlow Version: {tf.__version__}
    """)
    
    # Main content
    st.title("Cyberbullying Detection System")
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["Single Analysis", "Batch Analysis", "History & Stats", "Help"])
    
    # Load models
    with st.spinner("Loading models... This might take a minute..."):
        models_dict = load_models()
    
    if models_dict is None:
        st.error("""
        Failed to load models. Please ensure:
        1. All model files are present in the current directory
        2. The model files are compatible with your PyTorch version
        3. You have sufficient memory to load the models
        """)
        return

    # Single Analysis Tab
    with tab1:
        st.markdown("""
        ### Single Text Analysis
        Enter your text below to analyze it for cyberbullying content.
        """)
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type or paste text here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Analyze Text", type="primary")
        with col2:
            clear_button = st.button("Clear Input")
            
        if clear_button:
            text_input = ""
            
        if analyze_button and text_input:
            with st.spinner("Analyzing text..."):
                prediction, probability = ensemble_predict(text_input, models_dict)
                
                # Store in history
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': text_input,
                    'prediction': prediction,
                    'confidence': probability
                })
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Prediction")
                    if prediction == "not_cyberbullying":
                        st.success("✅ No cyberbullying detected")
                    else:
                        st.error(f"⚠️ Detected: {prediction}")
                
                with col2:
                    st.markdown("### Confidence")
                    st.progress(probability)
                    st.text(f"{probability:.2%} confidence")

    # Batch Analysis Tab
    with tab2:
        st.markdown("### Batch Text Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV or TXT file", type=['csv', 'txt'])
        
        if uploaded_file:
            try:
                if uploaded_file.type == 'text/csv':
                    df = pd.read_csv(uploaded_file)
                    texts = df.iloc[:, 0].tolist()  # Assume first column contains texts
                else:
                    texts = uploaded_file.getvalue().decode().split('\n')
                
                if st.button("Process Batch"):
                    with st.spinner("Processing batch..."):
                        results = process_batch_texts(texts, models_dict)
                        
                        # Display results
                        st.markdown("### Batch Results")
                        results_df = pd.DataFrame(results, columns=['Text', 'Prediction', 'Confidence'])
                        st.dataframe(results_df)
                        
                        # Download results
                        st.markdown(
                            get_download_link(
                                [results_df.columns.tolist()] + results_df.values.tolist(),
                                'batch_results.csv',
                                '📥 Download Results as CSV'
                            ),
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # History & Stats Tab
    with tab3:
        st.markdown("### Analysis History and Statistics")
        
        if st.session_state.history:
            # Display visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = visualize_prediction_distribution(st.session_state.history)
                if fig1:
                    st.plotly_chart(fig1)
            
            with col2:
                fig2 = visualize_confidence_trend(st.session_state.history)
                if fig2:
                    st.plotly_chart(fig2)
            
            # History table
            st.markdown("### Recent Analysis History")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            # Download history
            st.markdown(
                get_download_link(
                    [history_df.columns.tolist()] + history_df.values.tolist(),
                    'analysis_history.csv',
                    '📥 Download History as CSV'
                ),
                unsafe_allow_html=True
            )
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()
        else:
            st.info("No analysis history available yet.")

    # Help Tab
    with tab4:
        st.markdown("""
        ### How to Use This Application
        
        #### Single Analysis
        1. Go to the "Single Analysis" tab
        2. Enter or paste your text in the text area
        3. Click "Analyze Text" to get results
        
        #### Batch Analysis
        1. Go to the "Batch Analysis" tab
        2. Upload a CSV file (first column should contain texts) or TXT file (one text per line)
        3. Click "Process Batch" to analyze all texts
        4. Download results as CSV
        
        #### History & Stats
        - View analysis history and statistics in the "History & Stats" tab
        - Download history as CSV
        - Clear history if needed
        
        #### Tips
        - Use dark mode for better visibility in low-light conditions
        - Export results and history for further analysis
        - Check the confidence score to gauge prediction reliability
        
        #### About the Models
        This system uses an ensemble of three models:
        - LSTM (Long Short-Term Memory)
        - CNN (Convolutional Neural Network)
        - BERT (Bidirectional Encoder Representations from Transformers)
        
        Each model contributes equally to the final prediction.
        """)

if __name__ == "__main__":
    main()
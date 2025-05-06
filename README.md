# Deep-Learning-Driven-Cyberbullying-Detection

Step-by-Step Procedure of Project Development
1. Dataset Gathering
•	Data Sources: Collected text data from platforms like Kaggle, Github
•	Data Composition: Assembled balanced datasets containing both cyberbullying and non-bullying text samples
•	Labelling Process: Ensured proper annotation of texts as either cyberbullying or non-cyberbullying content
•	Dataset Size: Gathered sufficient data volume to train robust deep learning models
2. Data Cleaning and Preprocessing
•	Text Normalization: Converted text to lowercase, removed special characters and numbers
•	Tokenization: Split text into individual tokens or words
•	Stop Word Removal: Eliminated common words that don't contribute to classification
•	Stemming/Lemmatization: Reduced words to their root forms
•	Handling Emojis and Slang: Processed internet-specific language elements relevant to cyberbullying
•	Data Splitting: Divided dataset into training, validation, and test sets (80-10-10 split)
3. Model Development
CNN Model
•	Embedding Layer: Converted text tokens to dense vectors
•	Convolutional Layers: Applied 1D convolutions to extract local patterns from text
•	Pooling Layers: Used max pooling to reduce dimensionality
•	Dense Layers: Added fully connected layers for classification
•	Hyperparameter Tuning: Optimized filter sizes, number of filters, and activation functions
LSTM Model
•	Embedding Layer: Created word embeddings
•	LSTM Layer(s): Implemented sequential memory cells to capture contextual information
•	Dropout: Added regularization to prevent overfitting
•	Dense Layers: Connected to output layer for final classification
•	Optimization: Fine-tuned sequence length, hidden units, and learning rate
BERT Model
•	Pre-trained BERT: Utilized BERT base or other variant
•	Fine-tuning: Adapted the pre-trained model to cyberbullying detection task
•	Contextual Understanding: Leveraged bidirectional context for better semantic understanding
•	Advanced Tokenization: Implemented WordPiece tokenization
•	Optimization: Fine-tuned learning rate, batch size, and number of epochs
4. Ensemble Model Development
•	Soft Voting Strategy: Combined probabilistic outputs from all three models
•	Weighting Mechanism: Potentially assigned different weights to each model based on performance
•	Threshold Tuning: Optimized decision threshold for binary classification
5. Model Evaluation
•	Performance Metrics: Evaluated using accuracy, precision, recall, F1-score, and AUC-ROC
•	Confusion Matrix Analysis: Examined true positives, false positives, true negatives, and false negatives
•	Error Analysis: Identified patterns in misclassifications
•	Comparative Analysis: Compared individual models against the ensemble approach
6. Web Interface Development with Streamlit
•	UI Design: Created user-friendly interface with input area for text submission
•	Backend Integration: Connected the ensemble model to process submitted text
•	Results Visualization: Displayed classification results with confidence scores
•	Model Insights: Added explanatory components to highlight influential words or phrases
•	Responsive Design: Ensured compatibility across different devices
•	User Feedback Mechanism: Incorporated option for users to flag incorrect classifications
7. Testing
•	Environment Setup: Configured necessary dependencies and packages
•	Application Testing: Conducted thorough testing with various input scenarios
•	Performance Optimization: Ensured efficient response times
•	Documentation: Created user guide and technical documentation
This comprehensive development process resulted in a robust cyberbullying detection system that leverages the strengths of multiple deep learning architectures through ensemble learning, delivered through an accessible web interface.

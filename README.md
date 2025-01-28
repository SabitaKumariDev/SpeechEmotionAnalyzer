**Speech Emotion Recognition Using CNN**

**Overview**

This project implements a Speech Emotion Recognition (SER) system using Convolutional Neural Networks (CNN). The system is designed to classify speech signals into different emotion categories based on audio features. It leverages deep learning techniques to extract meaningful features from raw audio data and accurately predict emotions.

**Features**

1. Audio Feature Extraction: Utilizes Librosa for extracting Mel spectrograms and MFCCs from audio files.
   
2. Deep Learning Model: Implements a CNN architecture optimized for SER tasks.
   
3. Dataset Integration: Supports datasets such as RAVDESS (e.g., Actor_01, Actor_02) for training and evaluation.

4. Visualization: Provides tools to visualize spectrograms, model training metrics, and predictions.
   
5. Evaluation Metrics: Uses accuracy and loss to measure model performance.

**Dataset**

**Dataset Details**

Dataset Name: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

Data Files: Includes audio files of speech and song with emotional annotations.

Example Files:
    Actor_01.zip
    Actor_02.zip
    
Emotion Labels: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

**Dataset Preparation**

1. Unzip the dataset files (Actor_01.zip, Actor_02.zip, etc.) into the data/ directory.

2. Ensure the folder structure follows:

      data/
          Actor_01/
          Actor_02/
          ...

**Installation**

**Prerequisites**

  Python 3.6 or higher.

**Required Libraries**

  Install the required libraries using the requirements.txt file:

      pip install -r requirements.txt

**Dependencies:**

TensorFlow 1.9.0

Keras 2.2.2

Librosa 0.6.3

NumPy, Pandas, Matplotlib, Scipy, and others (full list in requirements.txt)​

**Usage**

1. Preprocess Data
   
Extract features (e.g., MFCCs, Mel spectrograms) from the audio files:

    from feature_extraction import extract_features
    features, labels = extract_features('data/')

2. Train the Model
   
Train the CNN on the extracted features:
    python train_model.py
    
Modify training parameters in the script (e.g., batch size, epochs).

3. Evaluate the Model

Evaluate the model on a test dataset:
    python evaluate_model.py

4. Predict Emotions
   
Predict emotions from new audio files:
    from prediction import predict_emotion
    emotion = predict_emotion('path/to/audio/file.wav')
    print(f'Predicted Emotion: {emotion}')

**Model Architecture**

The CNN model includes:

   Input Layer: Processes MFCCs or Mel spectrogram features.
   
   Convolutional Layers: Extract spatial features from spectrograms.
   
   Pooling Layers: Reduce spatial dimensions for computational efficiency.
   
   Dense Layers: Perform classification into emotion categories.
   
   Output Layer: Provides probabilities for each emotion class.

**Results**

**Performance Metrics**

    Training Accuracy: Achieved ~85% on the RAVDESS dataset.
    
    Validation Accuracy: Achieved ~80%.
    
**Confusion Matrix**

   Visualizes the model's performance across different emotion categories.

**Visualization**

   Spectrograms: Display the frequency distribution of audio signals.
   
   Training Curves: Plot accuracy and loss metrics over epochs.

**Challenges and Solutions**

   1. Imbalanced Dataset: Addressed by augmenting data using pitch shifting and time stretching.
      
   2. Noise in Audio Files: Applied preprocessing techniques like filtering and normalization.

**Future Work**

  Support real-time emotion detection via microphone input.
  
  Extend the system to include multi-modal emotion recognition (e.g., video + audio).
  
  Experiment with advanced architectures like Transformers.

**References**

   1. Librosa Documentation
      
   2. TensorFlow Documentation
      
   3. RAVDESS Dataset

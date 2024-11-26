# UrbanSound Classifier

This project uses a neural network model to classify urban sounds from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/download-urbansound8k.html). The dataset includes various urban audio samples, such as sirens, drills, and street music.

## Demo

Try out the **UrbanSound Classifier** using the interactive Streamlit app! You can upload an audio file and get instant predictions about its classification.

👉 [Launch the Streamlit App](https://devrathmukesh-urbansoundclassifiernn-bvzccdyxvazzrhfiepjrso.streamlit.app/)


## Table of Contents
- [Demo](#demo)
- [Dataset](#dataset)
- [Installation](#installation)
- [Features Extraction](#features-extraction)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)

## Dataset

Download the dataset from [UrbanSound8K](https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz) and place it in the `dataset/` directory. The dataset is organized into folders by sound type and includes metadata for each audio sample.

## Installation

Install the necessary dependencies:
```bash
pip install librosa numpy scipy scikit-learn tensorflow streamlit
```

## Features Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are used to extract features from each audio file. This process transforms each audio sample into a numerical format suitable for input into the neural network.

## Model Architecture

The neural network consists of:

- **Input Layer**: 128 neurons with ReLU activation
- **Hidden Layers**: Two dense layers with 256 and 128 neurons, each followed by dropout layers to prevent overfitting
- **Output Layer**: A softmax layer to classify the audio sample into one of the categories

## Training and Evaluation

The model is trained using categorical cross-entropy loss and the Adam optimizer. The dataset is split into training and testing sets, and the model saves the best weights based on validation accuracy. After training, the model's performance is evaluated on the test set.

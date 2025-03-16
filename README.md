# Handwriting Recognition using Deep Learning

## Introduction

In today's digital world, our reliance on online platforms has led to a decline in handwriting practice, often resulting in less legible writing. This project addresses the challenge of interpreting handwritten notes by developing a Handwritten Text Recognition (HTR) system using deep learning techniques.

The system aims to accurately recognize and transcribe handwritten text, regardless of individual handwriting styles, making it easier to digitize and understand handwritten content.

## Dataset

This project utilizes the IAM Handwriting Database, which contains various samples of handwritten English text. The dataset includes:

- PNG images with 256 gray levels
- Four main folders: ASCII, lines, sentences, and words
- Our focus was primarily on the ASCII and words folders for word-level predictions

## Methodology

We implemented a Convolutional Recurrent Neural Network (CRNN) architecture that combines:

1. **Convolutional layers** for feature extraction
2. **Recurrent layers** (LSTM units) for sequence modeling
3. **Fully connected layers** for mapping features to character probabilities
4. **Connectionist Temporal Classification (CTC)** loss function for sequence alignment

### Data Preprocessing

- Resized images to 128 x 32 pixels while preserving aspect ratio
- Applied padding to maintain dimensions
- Vectorized labels using StringLookup layer
- Split data into training (90%), validation and testing (10%) sets

### Data Augmentation

To improve model generalization, we applied various augmentation techniques:
- Rotations
- Zooming
- Padding
- Image cropping

### Model Architectures

We experimented with three different model architectures:

#### Baseline Model
- Input Layer: (128, 32, 1) grayscale image
- Conv Layer: 32 filters, (3, 3) kernel, ReLU activation
- MaxPooling: (2, 2) pool size
- Dense: 64 units, ReLU, dropout 0.2
- Output Layer: Softmax activation

#### Intermediate Model
- Additional convolutional layer (64 filters)
- Additional MaxPooling layer
- Two Bi-LSTM layers (128 and 64 units)
- Increased dropout for regularization

#### Deep Model
- Third convolutional layer (128 filters)
- Additional MaxPooling layer
- Increased dense layer units (128)
- Optional GRU layer

## Results

Performance was evaluated using mean edit distance, which measures the average number of alterations needed to transform predicted text into the actual text:

- Baseline Model: 20.7563 mean edit distance
- Intermediate Model: 19.7177 mean edit distance
- Deep Model: 19.7105 mean edit distance

The deep model showed the best performance, though the improvement from the intermediate model was minimal, suggesting a plateau in the learning capacity of the current architecture.

When tested on custom handwritten words, the model struggled to recognize text accurately, highlighting the challenges of generalizing to unseen handwriting styles.

## Limitations

- Limited computational resources restricted training time and model complexity
- Only trained for 50 epochs
- Unable to test on full-sentence images
- No implementation of transfer learning or attention mechanisms
- Limited generalization to handwriting styles not represented in the training data

## Future Improvements

- Extend training beyond 50 epochs
- Experiment with advanced architectures like attention mechanisms or transformers
- Build a front-end interface for users to upload handwritten images
- Optimize for real-time recognition on mobile devices

## Conclusion

This project demonstrates the potential of deep learning for handwriting recognition while highlighting the challenges of creating a robust system that can generalize across different handwriting styles. While we achieved promising results on the IAM dataset, further work is needed to improve performance on real-world handwritten text.

## Tech Stack

- TensorFlow/Keras
- Python
- Numpy
- OpenCV (for image preprocessing)
- Matplotlib (for visualization)

## Contributors

Aryan Singh 

## License

Apache License 2.0

## 🍅 Tomato Disease Detection

AgriPredict-AI uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify tomato leaf images into 10 categories.

### Features
- Detects 9 tomato disease categories and identifies healthy tomato leaves.
- Resizes input images to 128 × 128 pixels.
- Uses data augmentation, including random flipping, rotation, and zooming.
- Trains with the Adam optimizer and sparse categorical cross-entropy loss.
- Saves the trained model in Keras format.
- Stores class names in JSON for prediction.

### Model Architecture
1. Input layer (128 × 128 × 3)
2. Image augmentation and pixel rescaling
3. Four convolutional blocks with max pooling
4. Flatten layer
5. Dense layer with 128 units and dropout
6. Softmax output layer with 10 classes

### Training Configuration
- Image size: 128 × 128
- Batch size: 32
- Epochs: 10
- Validation split: 20%
- Random seed: 42

### Model Outputs
- `tomato_disease_model.keras`
- `tomato_class_names.json`

> **Note:** Predictions are intended as preliminary assistance, not a substitute for expert agricultural diagnosis. Actual model accuracy should be measured on independent test data.

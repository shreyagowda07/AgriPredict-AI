\# AgriPredict AI — Tomato Leaf Disease Detection



\## Project Overview

AgriPredict AI is a machine learning prototype that classifies tomato leaf images into 10 categories using a MobileNetV2-based image classification model.



\## Objectives

\- Identify possible tomato leaf disease categories from uploaded images.

\- Display the predicted category and model confidence.

\- Demonstrate integration between a machine learning model, Python API, and web interface.



\## Technologies Used

\- Python

\- TensorFlow and Keras

\- MobileNetV2 transfer learning

\- NumPy and Pillow

\- HTML, CSS, and JavaScript

\- Python HTTP server



\## Main Features

\- Upload a tomato leaf image.

\- Preview the selected image.

\- Predict one of 10 supported tomato categories.

\- Display the prediction and confidence percentage.

\- Show a caution message about automated predictions.



\## Project Structure

\- `index.html` — website interface

\- `src/disease\_api.py` — prediction API

\- `src/train\_tomato\_v2.py` — improved model training script

\- `src/test\_tomato\_v2.py` — sample prediction test

\- `models/tomato\_disease\_model\_v2.keras` — trained model

\- `models/tomato\_class\_names\_v2.json` — class-name mapping



\## How to Run



1\. Open Command Prompt in the project folder.

2\. Activate the virtual environment:



&#x20;  `venv\\Scripts\\activate`



3\. Start the API:



&#x20;  `python src\\disease\_api.py`



4\. Keep the API terminal open.

5\. Open a second terminal in the project folder.

6\. Open the website:



&#x20;  `start index.html`



7\. Upload a tomato leaf image and click \*\*Detect Disease\*\*.



\## Current Model Results

The training run reached approximately 94.80% best validation accuracy. A quick sanity check correctly classified 9 out of 10 sampled images. These are preliminary results, not independent test accuracy or a guarantee of real-world performance.



\## Limitations

\- The model is designed for the 10 supported tomato categories only.

\- Image quality, lighting, backgrounds, and leaf condition can affect predictions.

\- Confidence scores do not guarantee correct classification.

\- Predictions should not be used as the sole basis for pesticide or treatment decisions.

\- Independent evaluation on unseen images is still needed.



\## Conclusion

AgriPredict AI demonstrates a complete prototype workflow from image upload to machine learning prediction. Further testing and validation are required before real-world agricultural use.




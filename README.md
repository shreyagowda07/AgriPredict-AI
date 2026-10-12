import os

import json

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



\# ==============================

\# TOMATO DISEASE DETECTION

\# ==============================



BASE\_DIR = os.path.dirname(os.path.dirname(os.path.abspath(\_\_file\_\_)))



DATA\_DIR = os.path.join(

&#x20;   BASE\_DIR,

&#x20;   "data",

&#x20;   "plantvillage",

&#x20;   "raw",

&#x20;   "color"

)



MODEL\_DIR = os.path.join(BASE\_DIR, "models")



os.makedirs(MODEL\_DIR, exist\_ok=True)



IMG\_SIZE = (128, 128)

BATCH\_SIZE = 32

SEED = 42

EPOCHS = 10



\# Tomato disease classes

CLASS\_NAMES = \[

&#x20;   "Tomato\_\_\_Bacterial\_spot",

&#x20;   "Tomato\_\_\_Early\_blight",

&#x20;   "Tomato\_\_\_healthy",

&#x20;   "Tomato\_\_\_Late\_blight",

&#x20;   "Tomato\_\_\_Leaf\_Mold",

&#x20;   "Tomato\_\_\_Septoria\_leaf\_spot",

&#x20;   "Tomato\_\_\_Spider\_mites\_Two-spotted\_spider\_mite",

&#x20;   "Tomato\_\_\_Target\_Spot",

&#x20;   "Tomato\_\_\_Tomato\_mosaic\_virus",

&#x20;   "Tomato\_\_\_Tomato\_Yellow\_Leaf\_Curl\_Virus"

]



print("=" \* 60)

print("       AGRIPREDICT AI")

print("   TOMATO DISEASE DETECTION")

print("=" \* 60)



print("\\nDataset location:")

print(DATA\_DIR)



print("\\nClasses:")

for i, name in enumerate(CLASS\_NAMES):

&#x20;   print(f"{i}: {name}")



\# --------------------------------

\# Load dataset

\# --------------------------------



print("\\nLoading training dataset...")



train\_ds = tf.keras.utils.image\_dataset\_from\_directory(

&#x20;   DATA\_DIR,

&#x20;   labels="inferred",

&#x20;   label\_mode="int",

&#x20;   class\_names=CLASS\_NAMES,

&#x20;   image\_size=IMG\_SIZE,

&#x20;   batch\_size=BATCH\_SIZE,

&#x20;   validation\_split=0.2,

&#x20;   subset="training",

&#x20;   seed=SEED

)



print("\\nLoading validation dataset...")



val\_ds = tf.keras.utils.image\_dataset\_from\_directory(

&#x20;   DATA\_DIR,

&#x20;   labels="inferred",

&#x20;   label\_mode="int",

&#x20;   class\_names=CLASS\_NAMES,

&#x20;   image\_size=IMG\_SIZE,

&#x20;   batch\_size=BATCH\_SIZE,

&#x20;   validation\_split=0.2,

&#x20;   subset="validation",

&#x20;   seed=SEED

)



\# Improve performance

AUTOTUNE = tf.data.AUTOTUNE



train\_ds = train\_ds.prefetch(buffer\_size=AUTOTUNE)

val\_ds = val\_ds.prefetch(buffer\_size=AUTOTUNE)



\# --------------------------------

\# Data augmentation

\# --------------------------------



data\_augmentation = keras.Sequential(\[

&#x20;   layers.RandomFlip("horizontal"),

&#x20;   layers.RandomRotation(0.1),

&#x20;   layers.RandomZoom(0.1)

])



\# --------------------------------

\# CNN model

\# --------------------------------



model = keras.Sequential(\[

&#x20;   layers.Input(shape=(128, 128, 3)),



&#x20;   data\_augmentation,



&#x20;   layers.Rescaling(1.0 / 255),



&#x20;   layers.Conv2D(32, 3, activation="relu"),

&#x20;   layers.MaxPooling2D(),



&#x20;   layers.Conv2D(64, 3, activation="relu"),

&#x20;   layers.MaxPooling2D(),



&#x20;   layers.Conv2D(128, 3, activation="relu"),

&#x20;   layers.MaxPooling2D(),



&#x20;   layers.Conv2D(256, 3, activation="relu"),

&#x20;   layers.MaxPooling2D(),



&#x20;   layers.Flatten(),



&#x20;   layers.Dense(128, activation="relu"),

&#x20;   layers.Dropout(0.4),



&#x20;   layers.Dense(len(CLASS\_NAMES), activation="softmax")

])



model.compile(

&#x20;   optimizer="adam",

&#x20;   loss="sparse\_categorical\_crossentropy",

&#x20;   metrics=\["accuracy"]

)



print("\\nCNN model created.")



model.summary()



\# --------------------------------

\# Train

\# --------------------------------



print("\\nStarting training...")

print("This may take some time.")



history = model.fit(

&#x20;   train\_ds,

&#x20;   validation\_data=val\_ds,

&#x20;   epochs=EPOCHS

)



\# --------------------------------

\# Save model

\# --------------------------------



model\_path = os.path.join(

&#x20;   MODEL\_DIR,

&#x20;   "tomato\_disease\_model.keras"

)



model.save(model\_path)



\# Save class names

class\_file = os.path.join(

&#x20;   MODEL\_DIR,

&#x20;   "tomato\_class\_names.json"

)



with open(class\_file, "w") as f:

&#x20;   json.dump(CLASS\_NAMES, f, indent=4)



print("\\n" + "=" \* 60)

print("TRAINING COMPLETED")

print("=" \* 60)



print("\\nModel saved:")

print(model\_path)



print("\\nClass names saved:")

print(class\_file)



print("\\nFinal training accuracy:")

print(history.history\["accuracy"]\[-1])



print("\\nFinal validation accuracy:")

print(history.history\["val\_accuracy"]\[-1])


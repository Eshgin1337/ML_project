# Machine Learning and AI Fundamentals Project

## Description

This project focuses on implementing machine learning and AI techniques for classifying images from the UCMerced LandUse dataset. The goal is to prepare the dataset, perform data augmentation, and set up the project environment for subsequent model training.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Training Visualization](#training-visualization)
- [Model Saving](#model-saving)
## Installation

To run this project, you need to install the required dependencies. Execute the following command to install necessary packages:

```bash
pip install tensorflow matplotlib
```

## Data Preparation


1. Install necessary packages by running the provided code:
```bash
    !apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6
``` 
2. Import required libraries and modules:
```python
    import os
    import glob
    import shutil
    import zipfile
    import numpy as np
    # ...
```
3. Mount Google Drive to access your dataset:
```python
    from google.colab import drive
    drive.mount('/content/drive')
```
 4. Extract the dataset:

```python
    local_zip = '/content/drive/MyDrive/Copy of UCMerced_LandUse.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/content/drive/MyDrive/training')
```
5. Define the base directory for your dataset:

```python
    base_dir = os.path.join('/content/drive/MyDrive/training', 'UCMerced_LandUse', 'Images')
```

6. Create class labels:

```python
    classes = ['agricultural', 'airplane', 'baseballdiamond', ...]
```

7. Split images into train, validation, and test sets and organize the directory structure:

```python
    for cl in classes:
        # Split images
        train, val, test = ...
        # Move images to appropriate directories
        for t in train:
            shutil.move(t, os.path.join(base_dir, 'train', cl))
        # ...
```

8. Set up directory paths:

```python
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
```

## Data Augmentation
1. Set batch size and image shape:
```python
    batch_size = 32
    IMG_SHAPE = 256
```
2. Define data generators for training, validation, and testing:
```python
    train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   zoom_range=0.2,
                                   rotation_range=40,
                                   shear_range=0.2)
# ...
```
3. Visualize augmented images:
```python
    def plotImages(images_arr):
    # ...
    plotImages(augmented_images)
```

## Model Architecture
1. Define the CNN model architecture:
```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        # ...
        tf.keras.layers.Dense(21)
    ])
```
2. Compile the model:
```python
    model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## Model Training
1. Train the model
```python
    epochs = 5

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
```

## Training Visualization
1. Visualize training progress:
```python
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # ...

```

## Model Saving
1. Save the trained model:
```python
    mymodel = model.save('model_v1.h5')
```
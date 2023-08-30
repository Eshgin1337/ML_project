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
- [Transfer Learning with Inception Model](#transfer-learning-with-inception-model)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [Image Prediction](#image-prediction)
- [F1 Score Calculation for Inception Model](#f1-score-calculation-for-inception-model)
- [Transfer Learning with MobileNet V2 Model](#transfer-learning-with-mobilenet-v2-model)
- [F1 Score Calculation for MobileNet V2 Model](#f1-score-calculation-for-mobilenet-v2-model)
- [Model Training Visualization](#model-training-visualization)
- [Model Saving](#model-saving)
- [Image Prediction](#image-prediction)

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

## Transfer Learning with Inception Model
1. Define augmented data generators for training and validation:

```python
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   zoom_range=0.2,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2)
# ...
```
2. Import the necessary library:
```python
    import tensorflow_hub as hub
```
3. Create the Inception model for transfer learning:
```python
    URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(150, 150, 3))
    feature_extractor.trainable = False
    model_inception = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(21)
    ])

```
4. Define a callback to stop training when accuracy reaches 99.9%:
```python
    class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.999):
        print("\nReached 99.9% accuracy, cancelling training!")
        self.model.stop_training = True

```
5. Compile and train the Inception model:
```python
    model_inception.compile(
        optimizer='sgd',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    EPOCHS = 4
    callbacks = myCallback()
    history = model_inception.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=callbacks)
```
6. Plot training and validation accuracies:
```python
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
```

## Model Evaluation
1. Evaluate the model on the test dataset:
```python
    loss, accuracy = model_inception.evaluate_generator(test_generator)
```

## Saving and Loading Models
1. Save the trained Inception model:
```python
    model_inception = model_inception.save('model_inception.h5', include_optimizer=True)
```
2. Load the saved model:
```python
    model_inception = tf.keras.models.load_model('model_inception.h5', custom_objects={'KerasLayer': hub.KerasLayer})
```

## Image Prediction
1. Define a function to format input images:
```python
    def format_image(image, IMAGE_RES):
        image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))
        image = image/255.0
        return image
```
2. Load and predict images:
```python
    from google.colab import files
    uploaded = files.upload()
    for fn in uploaded.keys():
        path = fn
        img = tf.keras.utils.load_img(path, target_size=(150, 150))
        x = tf.keras.utils.img_to_array(img)
        x = format_image(x, 150)
        plt.imshow(x)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        logits = model_inception.predict(images)
        probs = tf.nn.softmax(logits)

        # Get the predicted class with the highest probability
        pred_index = np.argmax(probs)
        #As index starts from zero, add 1 to index to identify class
        pred_class = pred_index+1
        # Print the predicted class and its probability
        print(fn)
        print(f"Predicted class: {classes[pred_index]}, Predicted index: {pred_index}")
        print(f"Probability: {probs[0][pred_index] * 100:.2f}%")
```

## F1 Score Calculation for Inception Model
1. Calculate and display the F1 score:
```python
    from sklearn.metrics import f1_score
    y_pred = model_inception.predict(test_generator)

    # Calculate the F1 score
    y_true = test_generator.classes
    y_pred = np.argmax(y_pred, axis=1)
    f1score = f1_score(y_true, y_pred, average='weighted')
    print('F1 score:', f1score)
```

## Transfer Learning with MobileNet V2 Model
1. Define augmented data generators for training and validation for MobileNet V2 model:
```python
    train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   zoom_range=0.2,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2)
# ...
```
2. Import MobileNet V2 feature extractor:
```python
    import tensorflow_hub as hub
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3))
```
3. Set the feature extractor as non-trainable:
```python
    feature_extractor.trainable = False
```
4. Build the MobileNet V2 model:
```python
    model_mobile = tf.keras.Sequential([feature_extractor,
                            layers.Dense(21)])
```
5. Compile and train the MobileNet V2 model:
```python
    !pip install tensorflow_addons
    import tensorflow_addons as tfa
    model_mobile.compile(optimizer='SGD',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    EPOCHS = 10

    history = model_mobile.fit(train_generator,
                        validation_data = validation_generator,
                        epochs = EPOCHS)
```
6. Evaluate the MobileNet V2 model:
```python
    loss, accuracy = model_mobile.evaluate_generator(test_generator)
    loss, accuracy
```

## F1 Score Calculation for MobileNet V2 Model
1. Calculate and display the F1 score for the MobileNet V2 model:
```python
    from sklearn.metrics import f1_score
    # Get predictions for the test set
    y_pred = model_mobile.predict(test_generator)

    # Calculate the F1 score
    y_true = test_generator.classes
    y_pred = np.argmax(y_pred, axis=1)
    f1score = f1_score(y_true, y_pred, average='weighted')
    print('F1 score:', f1score)
```

## Model Training Visualization
1. Plot training and validation graphs for both models:
```python
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
```

## Model Saving
1. Save the MobileNet V2 model:
```python
    model_mobile = model_mobile.save('model_mobilenet.h5', include_optimizer=True)
```

## Image Prediction
1. Load and predict images using the MobileNet V2 model:
```python
    from google.colab import files
    uploaded = files.upload()
    for fn in uploaded.keys():
        path = fn
        img = tf.keras.utils.load_img(path, target_size=(224,224))
        x = tf.keras.utils.img_to_array(img)
        x = format_image(x, 224)
        plt.imshow(x)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        logits = model_mobile.predict(images)
        probs = tf.nn.softmax(logits)

        # Get the predicted class with the highest probability
        pred_index = np.argmax(probs)
        #As index starts from zero, add 1 to index to identify class
        pred_class = pred_index+1
        # Print the predicted class and its probability
        print(fn)
        print(f"Predicted class: {classes[pred_index]}, Predicted index: {pred_index}")
        print(f"Probability: {probs[0][pred_index] * 100:.2f}%")
```
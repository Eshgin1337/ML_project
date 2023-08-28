# Machine Learning and AI Fundamentals Project

## Description

This project focuses on implementing machine learning and AI techniques for classifying images from the UCMerced LandUse dataset. The goal is to prepare the dataset, perform data augmentation, and set up the project environment for subsequent model training.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Data Augmentation](#data-augmentation)
- [Project Structure](#project-structure)
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
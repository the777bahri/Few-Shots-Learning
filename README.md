Project Description

This project implements a few-shot learning approach using Prototypical Networks for image classification. It provides the ability to classify images from a limited dataset and dynamically set up different layers and filters for training.

Features

Prototypical Networks for few-shot learning.
Multiple configuration options (layers, filters, dropout rate, batch normalization).
Support for training with custom datasets.
Evaluation of model accuracy with minimal training data.

How to set up the code environment.

1. Install Anaconda
Anaconda makes it easy to manage packages and environments for Python projects.

3. Create a New Virtual Environment

        conda create -n fewshot python=3.11

5. Activate the Environment
Activate the virtual environment using the following command:

        conda activate fewshot

7. Install PyTorch
https://pytorch.org/get-started/locally/

        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

9. Install Additional Packages

        pip install numpy matplotlib seaborn scikit-learn pillow collections
   
11. Set Up the Project Folder, each folder name is the class name. 

few_shot_project/

├── train.py               # Script to train the model

├── test.py                # Script to test the model

├── datasets/

│   ├── train/             # Training dataset

│   │   ├── class1/

│   │   │   ├── image1.png # Images for class1

│   │   ├── class2/

│   │   │   ├── image1.png # Images for class2

│   ├── test/              # Test dataset

│   │   ├── image1.png

│   ├── support set/       # Support set for few-shot learning

│       ├── class1/

│       │   ├── image1.png # Support images for class1

│       ├── class2/

│           ├── image1.png # Support images for class2

└── saved_models/          # Directory to store trained models

Model Accuracy

The model accuracy was 60% in classifying bird, butterfly, and uknown class. 

the reason for the unkown class is reduce overvitting from the other two classes




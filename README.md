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

├── train.py

├── test.py

├── datasets/

│   ├── train/

│     ├── class1/

│       ├── image1.png

│     ├── class2/

│       ├── image1.png

│   ├── test/

│     ├── image1.png

│   ├── support set/

│     ├── class1/

│       ├── image1.png

│     ├── class2/

│       ├── image1.png

└── saved_models/




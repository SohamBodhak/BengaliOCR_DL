# Bengali OCR Project

## Overview
This project implements an Optical Character Recognition (OCR) system for recognizing Bengali digits and characters using deep learning models, specifically LeNet5 and ResNet18 architectures. The models are built using PyTorch and trained on a dataset of grayscale images (32x32 pixels) from the BanglaLekha-Isolated dataset. The system achieves high accuracy in classifying 84 unique Bengali characters, with the ResNet18 model reaching a final validation accuracy of 94.33%.

## Features
- **Dataset**: Processes images from the BanglaLekha-Isolated dataset, organized by character labels.
- **Preprocessing**: Images are resized to 32x32 pixels, converted to grayscale, and normalized.
- **Data Augmentation**: Random rotation and affine transformations are applied to the training set to improve model robustness.
- **Models**:
  - **LeNet5**: A convolutional neural network with two convolutional layers and three fully connected layers.
  - **ResNet18**: A deeper network with residual blocks for better performance on complex patterns.
- **Training**: Both models are trained for 20 epochs using the Adam optimizer and CrossEntropyLoss.
- **Evaluation**: Training and validation metrics (loss and accuracy) are tracked and visualized.
- **Output**: Models are saved when validation accuracy improves, and training history is plotted as PNG files.

## Project Structure
- `BengaliOCR.ipynb`: Jupyter notebook containing the implementation of LeNet5 and ResNet18 models, including data loading, preprocessing, training, and evaluation.
- `Images/`: Directory containing the dataset (not included in the repository; see Dataset section below).
- `models/`: Directory where trained models are saved (`lenet5.pth` and `resnet18.pth`).
- `training_history_lenet.png`: Plot of training and validation metrics for LeNet5.
- `training_history_resnet.png`: Plot of training and validation metrics for ResNet18.
- `sample_digit_lenet.png`: Visualization of a sample digit predicted by LeNet5.
- `sample_digit_resnet.png`: Visualization of a sample digit predicted by ResNet18.
- `requirements.txt`: List of Python dependencies required to run the project.

## Requirements
To run this project, ensure you have Python 3.10 or later installed. The required packages are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the BanglaLekha-Isolated dataset, which contains images of isolated Bengali characters. The dataset should be organized in a directory named `Images`, with subdirectories for each character class containing PNG images. For example:
```
Images/
├── 1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 2/
│   ├── image1.png
│   └── ...
└── ...
```
Download the dataset from a reliable source (e.g., Kaggle) and place it in the `Images` directory, or modify the `data_path` variable in the notebook to point to your dataset location.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset**:
   - Place the BanglaLekha-Isolated dataset in the `Images` directory or update the `data_path` variable in the notebook.
4. **Run the Notebook**:
   - Open `BengaliOCR.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to load data, train the models, and visualize results.
   - Alternatively, convert the notebook to a Python script using `jupyter nbconvert --to script BengaliOCR.ipynb` and run it with `python BengaliOCR.py`.
5. **Outputs**:
   - Trained models will be saved in the `models/` directory.
   - Training history plots and sample digit visualizations will be saved as PNG files in the project root.

## Model Performance
- **LeNet5**: Achieves a final validation accuracy of 88.37% after 20 epochs.
- **ResNet18**: Achieves a final validation accuracy of 94.33% after 20 epochs, demonstrating superior performance due to its deeper architecture and residual connections.

## Notes
- The notebook assumes a CUDA-enabled GPU is available for faster training. If not, it defaults to CPU.
- The commented-out `read_images_from_zip` function can be used if the dataset is provided as a ZIP file (e.g., `BanglaLekha-Isolated.zip`).
- Adjust hyperparameters (e.g., `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`) in the notebook for experimentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The BanglaLekha-Isolated dataset for providing a comprehensive set of Bengali character images.
- PyTorch for the flexible deep learning framework.

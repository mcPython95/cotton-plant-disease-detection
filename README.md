# Cotton Plant Disease Detection Using ResNet152

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture and Optimization](#model-architecture-and-optimization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Saving the Model](#saving-the-model)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project focuses on detecting diseases in cotton plants using machine learning techniques. Early detection of diseases in cotton plants can help farmers take preventive measures and ensure better crop yields. The project uses a Convolutional Neural Network (CNN) based on ResNet152 architecture for image classification.

## Dataset

The dataset used for this project consists of images of cotton plants with different diseases. It includes labels for various diseases such as leaf blight, bacterial blight, and healthy plants.

You can download the dataset from [this link](https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset).

## Dependencies

The project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- OpenCV
- torch
- torchvision
- PIL
- jax

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python torch torchvision pillow jax
```

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mcPython95/cotton-plant-disease-detection.git
   cd cotton-plant-disease-detection
   ```

## Usage

1. Open the Google Colab notebook:

   [Open in Colab](https://colab.research.google.com/github/mcPython95/cotton-plant-disease-detection/blob/main/cotton_plant_disease_detection.ipynb)

2. Set the data directory in the notebook:
   ```python
   DataDir = '/content/drive/MyDrive/cotton plant disease detection/dataset/Cotton Disease/'
   ```

3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Model Architecture and Optimization

### ResNet152 Architecture

- **ResNet152**: ResNet, short for Residual Network, is a type of deep neural network that uses residual learning to ease the training of networks that are substantially deeper than those used previously. ResNet152 is a variant with 152 layers, known for its ability to handle the vanishing gradient problem through the use of skip connections or shortcuts, allowing gradients to pass through layers directly, bypassing certain blocks. This architecture improves the training of very deep networks and has proven to be highly effective in image recognition tasks.

### Optimization Techniques

- **Loss Function**: CrossEntropyLoss was used as the loss function to measure the performance of the classification model.
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9 was employed to update the weights of the network.
- **Learning Rate Scheduler**: ReduceLROnPlateau was used to adjust the learning rate when the validation loss plateaued, ensuring efficient training.

## Model Training and Evaluation

The model is built using a ResNet152 architecture. The training and evaluation process includes data loading and preprocessing, image augmentation, building and compiling the CNN model, training the model, evaluating the model's performance, and visualizing the results.

## Results

The model performed exceptionally well in detecting diseases in cotton plants. It achieved an overall accuracy of approximately 97%. Precision and recall were both high, indicating that the model is effective at identifying both diseased and healthy plants accurately. The F1-Score, which balances precision and recall, was also high. The model demonstrated a low error rate and high specificity, meaning it correctly identified the majority of healthy plants without falsely marking them as diseased. The ROC-AUC score was also high, indicating excellent performance in distinguishing between different classes of plant health.

### Classification Report Summary

- **Diseased Cotton Leaf**: The model perfectly identified diseased cotton leaves with high precision, recall, and F1-score.
- **Diseased Cotton Plant**: The model showed high precision and recall for detecting diseased cotton plants, ensuring minimal misclassification.
- **Fresh Cotton Leaf**: Fresh cotton leaves were accurately identified, maintaining a high F1-score.
- **Fresh Cotton Plant**: The model also accurately detected fresh cotton plants, with high precision and recall.

### Overall Performance Metrics

- **Accuracy**: 96.84%
- **Precision**: 97.10%
- **Recall**: 96.74%
- **F1-Score**: 96.92%
- **Specificity**: 97.10%
- **ROC-AUC**: 97.83%
- **Error Rate**: 3.16%

## Saving the Model

After training, the model can be saved to your Google Drive for future use:

```python
torch.save(model, '/content/drive/MyDrive/cotton plant disease detection/cotton_plant_dis_det_model.pt')
```

## Contributing

We welcome contributions to improve this project. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Author's Name](https://github.com/mcPython95) - Project creator and maintainer.
- Thanks to all the contributors who helped in improving this project.

---

Feel free to customize this README file based on your specific project details and requirements. If you have any additional content or sections to add, let me know!

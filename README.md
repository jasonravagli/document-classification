# Study of Feature Fusion Methods for Document Image Classification

State-of-the-art methods for document image classification rely on visual features extracted by deep convolutional neural networks (CNNs). These methods do not use rich semantic information present in the text of the document, which can be extracted using Optical Character Recognition (OCR). In this work we study a feature fusion technique original proposed by R. Jain and C. Wigington [Multimodal Document Image Classification](https://ieeexplore.ieee.org/abstract/document/8977998). It allows to build a model that considers both visual and textual features of a document image to perform the classification. We compare this method to other approaches for the document image classification task using the RVL-CDIP dataset [RVL-CDIP  dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/).

Further details about this work can be found in our [paper](https://github.com/jasonravagli/document-classification/blob/master/paper.pdf).

The following image show the main idea behind the feature fusion technique.

<img src="https://github.com/jasonravagli/document-classification/blob/master/img/overview.png" width="400" height="400"/>

The project can be run almost entirely on Google Colab and it is composed of 2 Jupyter Notebook and 8 Colab notebooks.
Since the dataset is too large (about 120GB) to be processed easily on Google Colab, we realized the two Jupyter notebooks that allows to create on a local machine a mini-dataset starting from the original one (see below for further details).

To be able to use the Colab notebooks without any configuration you have to create inside your Google Drive root folder a directory tree with the following structure:

```
document-classification
├── datasets
│   └── rvl-cdip
│       └── mini-dataset-1488-288-192
└── models
    ├── feature-fusion
    ├── final-models
    ├── image-classifier
    ├── output-late-fusion
    └── text-classificator
```

mini-dataset-1488-192 will containsthe mini-dataset generated for the experiments.

In the following sections we give a brief description of each notebook and its functionalities.

We used the [fast.ai](https://www.fast.ai/) library, a high-level machine learning library based on PyTorch. The main advantage of this library is that it abstracts a lot of implementation details to the user, providing even to inexperienced users a well functioning environment already set up.

### 1-Mini-dataset-generation.ipynb (not used)
Jupyter notebook that generates the mini-dataset sampling images from the original RVL-CDIP dataset. It applies a preprocessing step to the selected images, scaling them to a target shape and adding a border to preserve their aspect ratio. The mini-dataset is saved as an HDF5 file. This notebook was used in the early phases of the development and it was replaced by the notebook 1b. See the next section for further details about how to run it.

### 1b-Mini-dataset-generation-no-preprocess.ipynb
Jupyter notebook that generates the mini-dataset sampling images from the original RVL-CDIP dataset. It has to be executed on a PC where the entire RVL-CDIP is available. In the section "Global parameters" you can configure the path to the original dataset and some other settings. The final mini-dataset will be saved inside the folder resources/output. As opposed to the notebook 1, this notebook does not perform any preprocessing to the images, saving them as they are inside simple folders.

### 2-VGG16-FastAI-Image-Classification.ipynb
Colab notebook to train an image classifier using the VGG16 network. 
The output model (whose name is configurable in section "Setup") is placed inside the document-classification/models/image-classifier Google Drive folder.

### 3-Text-Classification.ipynb
Colab notebook to train a text classifier using ULMFiT.
The output model (whose name is configurable in section "Setup") is placed inside the document-classification/models/text-classificator Google Drive folder.

### 4-Feature-Fusion.ipynb
Colab notebook to train a classifier using the feature fusion technique. The classifier is built starting from the two models trained in notebooks 2 and 3. The paths where these two models can be found are configured in section "Setup".
The output model (whose name is configurable in section "Setup") is placed inside the document-classification/models/feature-fusion Google Drive folder.

### 5-Late-Fusion.ipynb
Colab notebook to train a classifier using the late fusion technique. As for notebook 4, it requires models trained in notebooks 2 and 3.
The output model (whose name is configurable in section "Setup") is placed inside the document-classification/models/output-late-fusion Google Drive folder.

### Results notebooks
> **2R-Results-VGG16-Image-Classification.ipynb**

> **3R-Results-Text-Classification.ipynb**

> **4R-Results-Feature-Fusion.ipynb**

> **5R-Results-Late-Fusion.ipynb**

Colab notebooks to evaluate the performance of their respective classifiers. In the initial section can be configured some parameters to choose the model to evaluate.
Each notebook calculates the accuracy on the test set and plots the confusion matrix.

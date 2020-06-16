# Document Classification

State-of-the-art methods for document image classi- fication rely on visual features extracted by deep convolutional neural networks (CNNs). These methods do not use rich semantic information present in the text of the document, which can be extracted using Optical Character Recognition (OCR). In this work we study a feature fusion technique original proposed by R. Jain and C. Wigington [Multimodal Document Image Classification](https://ieeexplore.ieee.org/abstract/document/8977998). It allows to build a model that considers both visual and textual features of a document image to perform the classification. We compare this method to other approaches for the document image classification task using the RVL-CDIP dataset [RVL-CDIP  dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/).

The following image show the main idea behind the feature fusion technique.

<img src="https://github.com/jasonravagli/document-classification/blob/master/img/overview.png" width="400" height="400"/>

We realized 5 Google Colab notebooks using the [Fast.ai library](https://www.fast.ai/).

1. **Mini dataset generation:** Here we have two notebooks 1 and  1b, this notebooks allow us to extract a mini dataset from original RVL-CDIP. We have done this for upload faster a small dataset on Google Drive for use it in the next 4 notebooks. (The rvl dataset is approximately 120 Gb)
2. **VGG Image Document Classifier:** Here we create a vgg model using fastai, its possible to modify the hyperparameters of the networks.
3. **Text Document Classifier:** Like the previus notebooks this create a text classifier model.
4. **Features Fusion Document Classifier:** This notebooks allow us to load the two previus model created, manage the concatenation of the two representation and create a concat model for do experimets and fine tuning. 
5. **Late Fusion Document Classifier:** This notebooks allow us to load the two previus model created, extract the probability vector of the frist two models and create the late fusion classifier.


To replicate our experiment the only things you have to do is download [RVL-CDIP  dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/) and use locally the notebook 1, when you have extracted a mini dataset and you have uploaded on google drive you can just run other notebooks for replicate the experiments and change the hyperparameters.

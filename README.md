# Document Classification

State-of-the-art methods for document image classi-fication  rely  on  visual  features  extracted  by  deep  convolutionalneural networks (CNNs). These methods do not use rich semanticinformation  present  in  the  text  of  the  document,  which  can  beextracted  using  Optical  Character  Recognition  (OCR),  this  is the main purpose of the article of R. Jain and C. Wigington [Multimodal Document Image Classification](https://ieeexplore.ieee.org/abstract/document/8977998).

We studied their approach for the fusion of semantic and visualinformation  of  an  image,  then  we  carried  out  an  experiment  to try to replicate their techniques on classification using the [RVL-CDIP  dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/).

Here we can see the main idea of our project, is to combine two classificator, one extract features from visual image and the other one extract information from text.

<img src="https://github.com/jasonravagli/document-classification/blob/master/img/overview.png" width="400" height="400" />

We have realized 5 Google Colab noteboks, written in Python 3.6 using the [Fast.Ai libriray](https://www.fast.ai/)

1. Mini dataset generation: 
2. VGG Image Document Classifier:
3. Text Document Classifier:
4. Features Fusion Document Classifier:
5. Late Fusion Document Classifier:

# Glaucoma-Detection

![glaucoma_image](https://github.com/kesaroid/Glaucoma-Detection/blob/master/data/repository-open-graph-template.jpg "glaucoma_image")

Glaucoma is a chronic and irreversible eye disease, which leads to deterioration in vision and quality of life. In this paper, we develop a Deep Learning (DL) with convolutional neural network and implement it using a Raspberry Pi module for automated glaucoma diagnosis. Deep learning systems, such as convolutional neural networks (CNNs), can infer a hierarchical representation of images to discriminate between glaucoma and non-glaucoma patterns for diagnostic decisions. The model is trained with the ROI of RIGA, DRISHTI-GS1 dataset. The Network architecture used gives great accuracy. A graphical user interface is used to diagnose the condition of test images and give a graphical analysis of the patients. The entire program is run on a Raspberry Pi 3B with a 5” LCD touch screen as a stand-alone device with the power input.

Packages Required:
1. Keras
2. Tensorflow
3. Numpy
4. Pandas
5. Matplotlib
6. OpenCV
7. h5py
8. imgaug

Which—with the exception of OpenCV—can be installed with:
```
pip install -r requirements.txt
```

Citation: "Kesar T. N, T. C Manjunath, 'Diagnosis & detection of eye diseases using Deep Convolutional Neural Networks & Raspberry Pi', Second IEEE International Conference on Green Computing & Internet of Things (IOT), ICGCIoT, IEEE ISBN: 978-1-5386-5657-0"

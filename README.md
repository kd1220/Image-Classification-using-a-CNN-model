Image Classification with CIFAR-10 Dataset


This project demonstrates image classification using the CIFAR-10 dataset and a Convolutional Neural Network (CNN) built with TensorFlow and Keras.




[Install Dependencies: Ensure you have TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and PIL installed.]
1. Data Preparation
Load CIFAR-10 Dataset: The CIFAR-10 dataset is loaded using `keras.datasets.cifar10.load_data()`.
Preprocess Data:
- Reshape labels to 1D arrays.
- Normalize pixel values to the range [0, 1] by dividing by 255.
2. Model Building
CNN Architecture:A CNN model is created using `keras.models.Sequential` with the following layers:
- Convolutional layers with ReLU activation.
- Batch normalization layers.
- Max pooling layers.
- Spatial dropout layers.
- Residual block.
- Global average pooling layer.
- Dense layer with ReLU activation.
- Output dense layer with softmax activation for 10 classes.
Compilation: The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
3. Model Training
Training: The model is trained using `model.fit()` with early stopping based on validation accuracy.
Callbacks: EarlyStopping is used to prevent overfitting.
4. Evaluation
Prediction: Predictions are made on the test set using `model.predict()`.
Classification Report: A classification report is generated using `sklearn.metrics.classification_report()`.
Accuracy and Loss Plots: Training and validation accuracy and loss are plotted.
Confusion Matrix: A confusion matrix is plotted using `seaborn` and `matplotlib`.
5. Prediction on New Images
Preprocessing: A function `preprocess_image()` is defined to resize and normalize new images.
Prediction Function: A function `predict_class()` takes the model and preprocessed image as input and returns the predicted class and confidence.
6. Saving the Model
Saving: The trained model is saved as 'ciphar10_imageclass.h5' using `model.save()` for future use and deployment.

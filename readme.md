# Hand Written Digit Classifier Numpy Only
The Handwritten Digit Classifier is a project developed to classify digits drawn by the user in an interface. The training and calculations of the algorithm were implemented using only the Numpy library. The main objective of this project is to gain an in-depth understanding of how a neural network functions. It aims to provide insights into the underlying mechanics of neural networks without relying on high-level libraries, focusing instead on core concepts like forward propagation, backpropagation, and gradient descent.

Richard Feynman: What I cannot create, I do not understand.

https://github.com/user-attachments/assets/39132ce1-bf7e-4020-86cf-b6afc05fa541

## How to Play
To start the canvas, execute the `app.py` file.<br>
Draw a number between 0 and 9 and press the "Tell the Number" button.<br>
If you want to start over, press the "Erase" button.<br>


Make sure you have all the necessary libraries installed by running:`pip install -r requirements.txt`<br>

## How it works
### Algorithm
The algorithm is a `neural network` with dense layers, following this configuration:

- **Input Size**: 784   
- **Hidden Layer 1**: 50 neurons, activation function **ReLU**  
- **Hidden Layer 2**: 20 neurons, activation function **ReLU**  
- **Output Layer**: 10 neurons, activation function **Softmax**

**Backpropagation**: The `batch gradient descent` method was used with a batch size of 40 and 100 epochs, with a learning rate of 0.005.

### Training
The neural network training was done using the well-known MNIST dataset, which contains 60,000 digits (0 through 9) for training and 10,000 digits for testing the model.
![mnist](https://github.com/user-attachments/assets/58a83c54-9620-422d-a2e1-e35cf3498e50)

### Performance
Accuracy on the training set: 100%
Accuracy on the test set: 97.74%

## Files 
- **dense_neural_class.py**: Structure of the `Class Dense_Neural_Diy`<br>
Each neural network is treated as an object of the Dense_Neural_Diy class, organizing them in this way makes it easier to create multiple models and compare their performances.
Important Methods:<br>
`fit`: Trains the model.<br>
`predict`: Makes predictions with the trained model.<br>
`improve_train`: Allows the training of the model to continue from where it left off. In other words, it takes the current set of weights and biases and continues the convergence process. This can be useful for gradually training the model when there isn't enough time to wait for full convergence. More importantly, it's possible to add new images to the training set without needing to restart the training process from the beginning. The algorithm can update itself incrementally. In the future, an implementation will be created where the user can judge whether the program's predictions are correct or incorrect, and the results will serve as reinforcement for updating the model's weights.

- **model.pkl**:It is the pre-trained model that will be used within the app. The `.pkl` file is a Python format for saving various types of data; in this case, it is saved as an already created and pre-trained object. Within any application, this model can be called using the `load_model()` function in the `utils.py` file.

- **utils.py**: File containing various functions necessary for understanding the `Dense_Neural_Diy class`, as well as other functions for the correct processing of forward and backpropagation.

- **training_example.ipynb**: `Notebook` containing the complete `step-by-step` process of how to create, train, test the performance, and save a neural network using the Dense_Neural_Diy class.
- **app.py**: It is the interface between the drawing on the canvas and the pre-trained model; the file serves only for this purpose.







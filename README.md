# MLB Project 2 - Neural Network Project: Handwritten Digit Recognition with TensorFlow

---

## 1) What this project is about

You'll build a **neural network** using **TensorFlow/Keras** to recognize handwritten digits (0-9) from the famous **MNIST dataset**. This is a classic introduction to deep learning—teaching a computer to "see" and classify images.

You'll use a provided notebook (`project_2_mnist_nn_template.ipynb`) that walks you through:
- Loading and preprocessing the MNIST dataset
- Building a neural network architecture with Keras
- Training the model with backpropagation
- Evaluating accuracy and visualizing results
- Making predictions on test images
- Visualizing what the model learned

By the end, you'll understand how neural networks process images, how to build models with TensorFlow, and how to evaluate classification performance.

---

## 2) What is a Neural Network (in plain English)

A **neural network** is a machine learning model inspired by how the brain works. It consists of layers of interconnected "neurons" that learn patterns from data.

### How it works
1. **Input Layer:** Takes in your data (in this case, 784 pixels from a 28×28 image)
2. **Hidden Layer(s):** Processes the data through mathematical transformations
   - Each neuron applies: `output = activation(weights * input + bias)`
   - **Weights** are learned parameters that get adjusted during training
   - **Activation functions** (like ReLU) add non-linearity so the network can learn complex patterns
3. **Output Layer:** Produces predictions (10 probabilities for digits 0-9)
   - Uses **softmax** to convert outputs into probabilities that sum to 1

### The training process
- **Forward pass:** Data flows through the network to make predictions
- **Loss calculation:** Compare predictions to actual labels (using cross-entropy loss)
- **Backward pass (backpropagation):** Calculate how to adjust weights to reduce error
- **Optimization:** Update weights using an optimizer (like Adam)
- Repeat for many iterations (epochs) until the model learns to classify accurately

### Why use TensorFlow/Keras?
- **TensorFlow** is one of the most popular deep learning frameworks
- **Keras** is TensorFlow's high-level API—makes building neural networks simple
- With just a few lines of code, you can create powerful models
- Handles all the complex math (gradients, optimizations) automatically

### Key concepts
- **Epochs:** One complete pass through the training data
- **Batch size:** Number of samples processed before updating weights
- **Accuracy:** Percentage of correct predictions
- **Loss:** How "wrong" the model's predictions are (lower is better)

---

## 3) The dataset (what you'll see)

This project uses the **MNIST dataset**, a benchmark in machine learning:

- **70,000 images** of handwritten digits (0-9)
  - 60,000 for training
  - 10,000 for testing
- **28×28 pixels** per image (784 total pixels)
- **Grayscale** values from 0 (black) to 255 (white)
- **Labels:** The actual digit each image represents

Example tasks:
- Normalize pixel values from [0, 255] to [0, 1]
- Flatten 28×28 images into 784-dimensional vectors
- One-hot encode labels (digit 3 → `[0,0,0,1,0,0,0,0,0,0]`)
- Build a network that learns to classify these images with ~97-98% accuracy

---

## 4) Project files

- **`project_2_mnist_nn_template.ipynb`** (the notebook you'll complete)
- **This README (`README.md`)**: instructions + context

---

## 5) How to open and run in **Google Colab** (recommended)

### Upload the notebook file directly
1. Download the notebook from Canvas
2. Go to <https://colab.research.google.com/>
3. Click **Upload** → select `project_2_mnist_nn_template.ipynb`
4. TensorFlow and Keras come pre-installed in Colab—no extra setup needed!

### Dataset loading
- MNIST is built into Keras, so it downloads automatically!
- We've already imported the dataset in the notebook for you. 

---

## 6) How we'll grade (submission criteria)

To receive credit:

1. **Complete the notebook**  
   - Fill in all the `# TODO:` cells
   - The notebook should **run end-to-end** without errors
   - It should produce:
     - A trained neural network model
     - Training history plots (loss and accuracy curves) - this is given by us anyway!
     - **Test accuracy of 95% or higher**
  - There are a lot of other code snippets on there that are just for you to explore! Our main criteria is that the code runs end to end and that it produces an accuracy of
    95% or higher. 

2. **Push your work to your GitHub Classroom repo**
   - If working in Colab:
     - **File → Save a copy in GitHub** → choose your **Classroom repo** → commit message like "Completed neural network project". We know that there was some errors with this, so manually
       uploading the file works as well, no need to submit using the command line! 
     - Or download the `.ipynb` and use Git locally:
       ```bash
       git clone <your-classroom-repo-url>
       cd <repo>
       git add mnist_neural_network_tensorflow.ipynb
       git commit -m "Completed neural network project"
       git push origin main
       ```
   - Verify your notebook is visible in your repo on GitHub
   - If the above fails, manually uploading to GitHub totally works!

3. **Submit on Canvas**
   - Post your **GitHub username** (e.g., `octocat`)
   - Post your **commit SHA** (a long hex string like `3f5c2a9...`)
     - How to find it: open your repo on GitHub → **Commits** → copy the **full SHA** of your final submission commit

> Your Canvas submission must include **both** your GitHub username and the **exact commit SHA** to get credit.

---

## 7) Tips & common pitfalls

- **Run All:** Use **Runtime → Run all** after edits to ensure the notebook still works start-to-finish
- **Training time:** Training should take 1-3 minutes on Colab. If it's taking much longer, check your architecture
- **Shape errors:** Common mistake—make sure to flatten images from (28, 28) to (784,) for the input layer
- **Activation functions:** Use ReLU for hidden layers and softmax for the output layer
- **Loss not decreasing:** Make sure you compiled the model with an optimizer and loss function before training
- **Colab timeouts:** If a session times out, reopen the notebook and re-run cells (training might need to restart)

### Expected results
- **Training accuracy:** ~98-99%
- **Test accuracy:** ~97-98%
- If you're getting much lower accuracy:
  - Check that images are normalized (divided by 255.0)
  - Verify you're using the right loss function (sparse_categorical_crossentropy or categorical_crossentropy)
  - Make sure the output layer has 10 neurons with softmax activation
  - Try training for more epochs (10-20 is typical)

---

## 8) Group Work

You all are free to work in groups, around 2-3 people! Just please submit a copy of your code anyway, it just helps us keep track of who submitted what work.

---

## 9) Need help?

- Re-read the comments in the notebook cells
- Check the TensorFlow/Keras documentation: <https://www.tensorflow.org/api_docs/python/tf/keras>
- Look at the console output for error messages—they often tell you exactly what's wrong
- Common issues:
  - Shape mismatches: Print shapes with `.shape` to debug
  - Import errors: Make sure you're importing from `tensorflow.keras`, not just `keras`
  - Model not training: Verify you called `model.compile()` before `model.fit()`
- Reach out to us!

---

## 10) What you'll learn

By completing this project, you will:
- ✅ Understand how neural networks classify images
- ✅ Build models using TensorFlow and Keras
- ✅ Preprocess image data for machine learning
- ✅ Train models with backpropagation and gradient descent
- ✅ Evaluate classification performance with accuracy and confusion matrices
- ✅ Visualize training progress and model predictions
- ✅ Debug and improve neural network architectures

This is your foundation for more advanced deep learning topics like CNNs, RNNs, and transfer learning!

Good luck—and have fun teaching a computer to recognize handwritten digits! 🧠✨

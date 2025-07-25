# digit classifier model
Handwritten Digit Classifier using MLP & Gradio A lightweight AI-powered web application that classifies handwritten digits (0–9) using a Multi-layer Perceptron (MLP) neural network. Built with Python, trained on scikit-learn’s digit dataset, and deployed using Gradio for interactive user input through a simple drawing canvas.  Ideal for beginners exploring machine learning, GUI integration, and real-time digit recognition with high accuracy.


 Handwritten Digit Classifier using MLP & Gradio

This project is a Handwritten Digit Classifier built using Python, scikit-learn, and Gradio. It uses a Multi-layer Perceptron (MLP) Classifier trained on the MNIST dataset to recognize handwritten digits from 0 to 9.

___________________________________________________
___________________________________________________

 Demo

The app provides a simple Gradio GUI where you can draw a digit and get a prediction instantly!

![Digit GUI](./screenshots/demo.png)

___________________________________________________
___________________________________________________

Features

- 🧠 MLPClassifier trained with `sklearn`
- 🎨 Gradio-based Web GUI for digit input
- 📊 Training Accuracy: ~98.6%
- 📈 Testing Accuracy: ~97.0%
- ⚠ Warning if convergence isn't reached
- 🌐 Option to run locally or via public Gradio link
___________________________________________________
___________________________________________________

GUI Preview

![Gradio App Screenshot](./screenshots/gui-preview.png)
___________________________________________________
___________________________________________________
Tech Stack

| Component         | Tool               |
|------------------|--------------------|
| Programming Lang | Python             |
| ML Library       | Scikit-learn       |
| GUI Framework    | Gradio             |
| Data Used        | `load_digits()` from sklearn |
| IDE              | Visual Studio Code |
| Deployment       | Localhost & Gradio Share URL |

___________________________________________________
___________________________________________________

 Installation

1. Clone the Repository
bash
git clone https://github.com/your-username/handwritten-digit-classifier.git
cd handwritten-digit-classifier

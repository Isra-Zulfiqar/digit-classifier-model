import gradio as gr
from sklearn.neural_network import MLPClassifier
import torchvision.datasets as datasets
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt

# Dark mode seaborn
sns.set_style("darkgrid")

# Load MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True)

# Extract data and labels
X_train = mnist_trainset.data.numpy()
Y_train = mnist_trainset.targets.numpy()
X_test = mnist_testset.data.numpy()
Y_test = mnist_testset.targets.numpy()

'''# Optional: Plot some visuals
sns.histplot(data=X_train[0].reshape(784, 1), bins=30)
plt.show()

sns.heatmap(X_train[0], cmap="gray")
plt.show()'''

# Normalize and reshape
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0

# Train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=20)
mlp.fit(X_train, Y_train)

print("Training Accuracy:", mlp.score(X_train, Y_train))
print("Testing Accuracy:", mlp.score(X_test, Y_test))

# Prediction function for sketchpad
from PIL import Image
import numpy as np

def predict(img_dict):
    # Gradio's sketchpad returns a dict with image data
    img_data = img_dict["image"]  # extract the actual image (as a 2D list)
    
    # Convert to a NumPy array, PIL image, then reshape for prediction
    img_array = np.array(img_data).astype("uint8")
    img = Image.fromarray(img_array).convert("L").resize((28, 28))
    
    img = np.array(img).reshape(1, 784) / 255.0
    prediction = mlp.predict(img)[0]
    return int(prediction)


gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(canvas_size=(280, 280), image_mode="L"),
    outputs="number",
    title="Digit Classifier"
).launch(share=True)

# The code above sets up a Gradio interface for a digit classifier using a sketchpad input.
# The model is a simple MLP trained on the MNIST dataset, and the prediction function processes the input image to make a prediction.
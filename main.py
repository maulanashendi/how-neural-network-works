import torch
from model import simple_neural_network
from data_generator import generate_dummy_data
from train import train_model
from visualizer import generate_gif

# Generate dummy data
input_data, target_data = generate_dummy_data()

# Initialize the model
model = simple_neural_network()

# Train the model
epochs = 600
losses, frames_data, frames_loss = train_model(model, input_data, target_data, epochs=epochs, lr=0.01)

# Generate GIFs
generate_gif(frames_data, "data_visualization.gif")
generate_gif(frames_loss, "loss_visualization.gif")

print("GIFs created: data_visualization.gif and loss_visualization.gif")

# Plot final loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Loss Over Time')
plt.show()
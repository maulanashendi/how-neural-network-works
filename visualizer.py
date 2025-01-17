import os
import matplotlib.pyplot as plt
import imageio

# Create directories for frames
os.makedirs("frames", exist_ok=True)

def save_data_visualization(epoch, input_data, target_data, predictions, frames_data, loss_value):
    plt.figure(figsize=(8, 6))
    plt.scatter(input_data.detach().numpy(), target_data.detach().numpy(), label="True Data", color="orange")
    plt.plot(input_data.detach().numpy(), predictions.detach().numpy(), label="Model Predictions", color="blue")
    plt.xlabel("Input")
    plt.ylabel("Target")
    plt.title(f"Epoch {epoch}, Loss: {loss_value:.4f}")  # Add loss value to the title
    plt.legend()
    filename = f"frames/frame_data_{epoch}.png"
    plt.savefig(filename)
    frames_data.append(filename)
    plt.close()

def save_loss_visualization(epoch, losses, frames_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve at Epoch {epoch}")
    plt.legend()
    filename = f"frames/frame_loss_{epoch}.png"
    plt.savefig(filename)
    frames_loss.append(filename)
    plt.close()


def generate_gif(frames, output_file):
    with imageio.get_writer(output_file, mode="I", duration=0.2) as writer:
        for frame in frames:
            image = imageio.v2.imread(frame)
            writer.append_data(image)

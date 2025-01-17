import torch
from visualizer import save_data_visualization, save_loss_visualization

def train_model(model, input_data, target_data, epochs=600, lr=0.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    frames_data = []
    frames_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(input_data)
        loss = criterion(predictions, target_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Save frames every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                predictions = model(input_data)
            save_data_visualization(epoch + 1, input_data, target_data, predictions, frames_data, loss.item())
            save_loss_visualization(epoch + 1, losses, frames_loss)

    return losses, frames_data, frames_loss

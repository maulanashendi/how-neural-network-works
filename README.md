# **Neural Network Project: Learning the Basics**

This project demonstrates how a neural network processes data, learns from it, and predicts outputs. It includes:
1. A fully connected neural network model (`simple_neural_network`).
2. Detailed visualization of predictions and loss during training.
3. An explanation of each component, including calculations, key functions, and uploaded files.

---

## **Project Purpose**

This project serves as a tool to:
1. **Understand Neural Networks**: How they work and how data propagates through layers.
2. **Explore Techniques**: Including dropout, ReLU activation, and Adam optimizer.
3. **Visualize Learning**: Track predictions and loss evolution over epochs using GIFs.

---

## **1. Step-by-Step Calculations**

### **a. Input Data**
- Input data is generated as:
```math
y = \sin(x) \cdot 10 + \text{noise}, \quad \text{noise} \sim \mathcal{N}(0, 0.2)
```
- **Normalization**:
```math
x_{\text{normalized}} = \frac{x - \mu}{\sigma}
```
Where:
- $$x$$: Original input.
- $$\mu$$: Mean of \(x\).
- $$\sigma$$: Standard deviation of \(x\).

**Why Normalize?**
- Normalizing input data ensures that features have a mean of 0 and standard deviation of 1, improving the convergence speed and stability of training.

---

### **b. First Hidden Layer**
- Transformation:
```math
Z_1 = X \cdot W_1^T + b_1
```
- Activation:
```math
A_1 = \text{ReLU}(Z_1)
```
- Dropout:
```math
A_1^{\text{dropout}} = A_1 \cdot M, \quad M \sim \text{Bernoulli}(1-p)
```
Where:
- $$X \in \mathbb{R}^{n_{\text{batch}} \times n_{\text{input}}}$$: Input data.
- $$W_1 \in \mathbb{R}^{n_{\text{hidden1}} \times n_{\text{input}}}$$: Weights.
- $$b_1 \in \mathbb{R}^{n_{\text{hidden1}}}$$: Biases.
- $$M$$: Dropout mask, with \(p = 0.2\).

**Why ReLU?**
- **Prevents vanishing gradients**: ReLU outputs gradients of 1 for positive values, ensuring they do not diminish as in sigmoid or tanh.
- **Non-linearity**: Allows the model to learn complex patterns.

**Why Dropout?**
- **Prevents overfitting**: By randomly deactivating neurons during training, dropout encourages the network to learn robust and generalizable features.

---

### **c. Second Hidden Layer**
- Transformation:
```math
Z_2 = A_1^{\text{dropout}} \cdot W_2^T + b_2
```
- Activation:
```math
A_2 = \text{ReLU}(Z_2)
```
Where:
- $$W_2 \in \mathbb{R}^{n_{\text{hidden2}} \times n_{\text{hidden1}}}$$: Weights.
- $$b_2 \in \mathbb{R}^{n_{\text{hidden2}}}$$: Biases.

---

### **d. Output Layer**
- Transformation:
```math
Z_{\text{output}} = A_2 \cdot W_{\text{output}}^T + b_{\text{output}}
```
Where:
- $$W_{\text{output}} \in \mathbb{R}^{n_{\text{output}} \times n_{\text{hidden2}}}$$: Weights.
- $$b_{\text{output}} \in \mathbb{R}^{n_{\text{output}}}$$: Biases.

---

### **e. Loss Calculation**
Loss is calculated using **Mean Squared Error (MSE)**:
```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2
```
Where:
- $$\hat{y}_i$$: Predicted output.
- $$y_i$$: True target.
- $$n$$: Number of samples.

**Why MSE?**
- Suitable for regression tasks as it penalizes larger errors more significantly.
- Provides smooth gradients for optimization.

---

### **f. Backward Pass and Parameter Updates**
- Gradients are calculated using backpropagation.
- Parameters (\(W\) and \(b\)) are updated using the Adam optimizer:
```math
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
```
Where:
- $$\eta$$: Learning rate.
- $$m_t$$: Moving average of gradients.
- $$v_t$$: Moving average of squared gradients.

**Why Adam Optimizer?**
- Combines momentum and adaptive learning rates for faster convergence.
- Handles sparse gradients effectively.

---

## **2. Uploaded Files Explanation**

### **1. `main.py`**
- Main script to coordinate the entire process:
  - Generates dummy data.
  - Trains the model using `train.py`.
  - Visualizes results using `visualizer.py`.

### **2. `model.py`**
- Defines the `simple_neural_network` architecture:
  - Input layer.
  - Two hidden layers with ReLU and dropout.
  - Output layer for regression tasks.

### **3. `train.py`**
- Implements the training process:
  - Performs forward pass, loss calculation, and backpropagation.
  - Updates parameters using Adam optimizer.
  - Tracks and stores loss over epochs.

### **4. `data_generator.py`**
- Generates synthetic dummy data for training:
  - Sinusoidal pattern with added noise.
  - Normalizes the input for better convergence.

### **5. `visualizer.py`**
- Handles visualization tasks:
  - Creates scatter plots for predictions vs. true data.
  - Tracks loss over epochs and generates line plots.
  - Generates GIFs from saved frames (`data_visualization.gif` and `loss_visualization.gif`).

### **6. Generated GIFs**
- **`data_visualization.gif`**:
  - Shows how model predictions improve over epochs.
- **`loss_visualization.gif`**:
  - Tracks the decline in training loss over time.

---

## **3. How to Use the Project**

### **a. Install Dependencies**
Install the required Python packages:
```bash
pip install torch matplotlib imageio
```

### **b. Run the Project**
Execute the main script:
```bash
python main.py
```

---

## **4. Visual Outputs**

### **a. Data Visualization**
- **File**: `data_visualization.gif`.
- **Purpose**: Shows how the model's predictions approach the true data over epochs.

### **b. Loss Visualization**
- **File**: `loss_visualization.gif`.
- **Purpose**: Displays the loss curve to monitor training progress.

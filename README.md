# 3D Neural Network Visualizer with Interactive Training

- A powerful, interactive 3D visualization tool for neural networks built with Python and Three.js.
-  Watch your neural network learn in real-time with beautiful 3D graphics showing neurons, connections, and training dynamics.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)
![Three.js](https://img.shields.io/badge/Three.js-r128-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features
### 🧠 **Pure Python Neural Network**
- **NumPy-based implementation** - No external ML libraries required
- **Xavier weight initialization** for optimal convergence
- **Sigmoid activation** with numerical stability
- **Backpropagation algorithm** with proper gradient computation
- **Configurable architecture** (2-5 layers, customizable hidden sizes)

### 🎮 **Interactive 3D Visualization**
- **Real-time network rendering** in beautiful 3D space
- **Dynamic neuron activation** - Watch neurons light up during training
- **Weight visualization** - Green/red connections showing positive/negative weights
- **Interactive camera controls** - Mouse rotation and zoom
- **Live training statistics** - Epoch, loss, and accuracy updates

### 📊 **Multiple Dataset Types**
- **Circular Classification** - Points inside/outside a circle
- **Linear Separation** - Basic linear classification boundary
- **XOR Problem** - Classic non-linearly separable dataset
- **Spiral Classification** - Complex spiral pattern recognition

### 🔧 **Training Controls**
- **Adjustable learning rate** (0.01 - 1.0)
- **Start/Stop/Reset** training functionality
- **Real-time parameter modification**
- **Threaded training** - Non-blocking UI during training

## 🚀 Quick Start

### Prerequisites
```bash
# Required Python packages
pip install numpy flask flask-cors
```

### Installation & Running
1. **Clone or download the project files**
2. **Run the Python server:**
   ```bash
   python neural_network_server.py
   ```
3. **Open your browser to:**
   ```
   http://localhost:5000
   ```

That's it! The server will automatically create the HTML interface and start the neural network backend.

## 📁 Project Structure
```
neural-network-visualizer/
├── neural_network_server.py    # Main Python backend server
├── templates/
│   └── index.html              # Auto-generated HTML interface
└── README.md                   # This file
```

## 🎯 How to Use

### 1. **Select a Dataset**
Choose from four different classification problems:
- **Circular**: Learn to classify points inside vs outside a circle
- **Linear**: Basic linear classification boundary
- **XOR**: The classic XOR problem (non-linearly separable)
- **Spiral**: Complex spiral pattern classification

### 2. **Configure Network Architecture**
- **Layers**: Choose 2-5 layers total (input + hidden + output)
- **Hidden Size**: Set neurons per hidden layer (2-8 neurons)
- The network automatically uses 2 input neurons (for 2D data) and 1 output neuron

### 3. **Adjust Training Parameters**
- **Learning Rate**: Use the slider to set learning rate (0.01 - 1.0)
- Higher values = faster learning but less stable
- Lower values = slower but more stable convergence

### 4. **Start Training**
- Click "Start Training" to begin the learning process
- Watch neurons activate and connections change in real-time
- Monitor epoch, loss, and accuracy in the stats panel

### 5. **Interactive Controls**
- **Mouse drag**: Rotate the 3D view
- **Mouse wheel**: Zoom in/out
- **Stop/Reset**: Control training process
- **Architecture changes**: Automatically reset the network

## 🔬 Technical Details
### Neural Network Implementation
```python
class NeuralNetwork:
    - Xavier weight initialization
    - Sigmoid activation function
    - Backpropagation with gradient descent
    - Mean squared error loss
    - Batch training (32 samples per batch)
```

### API Endpoints
- `POST /api/create_network` - Create new network architecture
- `POST /api/generate_data` - Generate training datasets
- `POST /api/start_training` - Begin training process
- `POST /api/stop_training` - Stop training
- `GET /api/get_network_state` - Get current network state
- `POST /api/predict` - Make predictions

### Visualization Features
- **Neurons**: Colored spheres (blue=input, green=hidden, red=output)
- **Connections**: Lines between neurons (green=positive weights, red=negative)
- **Activation**: Neuron brightness and size based on activation level
- **Weight Strength**: Connection opacity based on weight magnitude

## 🎨 Visualization Guide

### Color Coding
- **🔵 Blue Neurons**: Input layer
- **🟢 Green Neurons**: Hidden layers  
- **🔴 Red Neurons**: Output layer
- **🟢 Green Connections**: Positive weights
- **🔴 Red Connections**: Negative weights

### Visual Cues
- **Brighter/Larger Neurons**: Higher activation values
- **Thicker Connections**: Stronger weights (higher absolute value)
- **Fading Connections**: Weaker weights (closer to zero)

## 🛠️ Customization
### Adding New Datasets
```python
@staticmethod
def generate_custom_data(n_samples=200):
    """Add your custom dataset here"""
    X = # Your 2D input data (2, n_samples)
    Y = # Your binary labels (1, n_samples)
    return X, Y
```

### Modifying Network Architecture
```python
# In the frontend JavaScript
const architecture = [
    2,        # Input layer (fixed for 2D data)
    8, 6, 4,  # Hidden layers (customizable)
    1         # Output layer (fixed for binary classification)
];
```

### Changing Activation Functions
```python
def relu(self, z):
    return np.maximum(0, z)

def tanh(self, z):
    return np.tanh(z)
```

## 📈 Training Tips
### Getting Good Results
1. **Start with simpler datasets** (Linear, Circular)
2. **Use moderate learning rates** (0.1 - 0.3)
3. **Try different architectures** for complex problems (XOR, Spiral)
4. **Watch the loss curve** - should generally decrease over time
5. **Reset and try again** if training gets stuck

### Troubleshooting
- **High loss, low accuracy**: Try lower learning rate or more hidden neurons
- **Training too slow**: Increase learning rate or reduce network size
- **Network not learning**: Check if dataset is appropriate for architecture
- **Visualization not updating**: Ensure Python server is running

## 🔧 System Requirements

### Python Dependencies
- **Python 3.7+**
- **NumPy**: For matrix operations and numerical computations
- **Flask**: Web server framework
- **Flask-CORS**: Cross-origin resource sharing

### Browser Requirements
- **Modern web browser** with WebGL support
- **JavaScript enabled**
- **Stable internet connection** (for Three.js CDN)

## 🚨 Known Limitations
- **Binary classification only** (single output neuron)
- **2D input data** (visualized datasets are 2-dimensional)  
- **Limited to sigmoid activation** (easily extendable)
- **No GPU acceleration** (pure NumPy implementation)
- **Single training instance** (one network at a time)

## 🤝 Contributing
- We welcome contributions! Here's how you can help:
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add feature-name'`
5. **Push to branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 for Python code
- Add comments for complex algorithms
- Test new features thoroughly
- Update documentation as needed

## 📄 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- **Three.js** - Amazing 3D graphics library
- **Flask** - Lightweight web framework
- **NumPy** - Fundamental package for scientific computing
- **Neural Network concepts** - Based on classic backpropagation algorithm

## 📞 Support
- Having issues? Here are some resources:
1. **Check the console** for error messages (F12 in browser)
2. **Verify Python dependencies** are installed correctly
3. **Ensure port 5000** is not blocked by firewall
4. **Try different browsers** if visualization issues occur
5. **Restart the Python server** if connectivity problems persist

## 📊 Performance Notes
- **Training speed**: ~20-50 epochs per second (depending on architecture)
- **Memory usage**: Minimal (pure NumPy arrays)
- **Browser compatibility**: Modern browsers with WebGL 1.0+
- **Network size limits**: Tested up to 5 layers × 8 neurons each

---

**Made with ❤️ using Python, NumPy, Flask, and Three.js**

*Visualize the magic of neural networks learning in real-time!*

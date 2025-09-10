import numpy as np
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import threading
import time
import os

app = Flask(__name__)
CORS(app)

class NeuralNetwork:
    def __init__(self, architecture):
        """
        Initialize neural network with given architecture
        architecture: list of integers representing neurons in each layer
        """
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        
        # Initialize weights using Xavier initialization
        for i in range(len(architecture) - 1):
            input_size = architecture[i]
            output_size = architecture[i + 1]
            
            # Xavier initialization
            limit = np.sqrt(6.0 / (input_size + output_size))
            weight_matrix = np.random.uniform(-limit, limit, (output_size, input_size))
            self.weights.append(weight_matrix)
            
            # Initialize biases to small random values
            bias_vector = np.random.uniform(-0.1, 0.1, (output_size, 1))
            self.biases.append(bias_vector)
    
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        X: input data (features, samples)
        """
        self.activations = [X]
        self.z_values = []
        
        activation = X
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)
        
        return activation
    
    def backward_propagation(self, X, Y, learning_rate):
        """
        Backward propagation to update weights and biases
        X: input data
        Y: target output
        learning_rate: learning rate for weight updates
        """
        m = X.shape[1]  # number of samples
        
        # Forward propagation
        output = self.forward_propagation(X)
        
        # Initialize gradients
        dW = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        # Calculate output layer error
        dA = output - Y
        
        # Backward propagation
        for i in range(len(self.weights) - 1, -1, -1):
            dZ = dA * self.sigmoid_derivative(self.z_values[i])
            dW[i] = (1/m) * np.dot(dZ, self.activations[i].T)
            db[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
        
        return output
    
    def predict(self, X):
        """Make predictions on input data"""
        return self.forward_propagation(X)
    
    def compute_loss(self, Y_pred, Y_true):
        """Compute mean squared error loss"""
        return np.mean((Y_pred - Y_true) ** 2)
    
    def compute_accuracy(self, Y_pred, Y_true):
        """Compute classification accuracy"""
        predictions = (Y_pred > 0.5).astype(int)
        return np.mean(predictions == Y_true) * 100
    
    def get_network_state(self):
        """Get current network state for visualization"""
        # For activations, use the first sample's values to get scalars per neuron
        activations_for_viz = [a[:, 0].tolist() if a.ndim == 2 and a.shape[1] > 0 else [] for a in self.activations] if self.activations else []
        return {
            'architecture': self.architecture,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'activations': activations_for_viz
        }

class DataGenerator:
    @staticmethod
    def generate_circular_data(n_samples=200):
        """Generate circular classification data"""
        X = np.random.uniform(-2, 2, (2, n_samples))
        distances = np.sqrt(X[0]**2 + X[1]**2)
        Y = (distances < 1.2).astype(int).reshape(1, -1)
        return X, Y
    
    @staticmethod
    def generate_linear_data(n_samples=200):
        """Generate linearly separable data"""
        X = np.random.uniform(-2, 2, (2, n_samples))
        Y = (X[0] + X[1] > 0).astype(int).reshape(1, -1)
        return X, Y
    
    @staticmethod
    def generate_xor_data(n_samples=200):
        """Generate XOR data"""
        X = np.random.uniform(-2, 2, (2, n_samples))
        Y = ((X[0] > 0) != (X[1] > 0)).astype(int).reshape(1, -1)
        return X, Y
    
    @staticmethod
    def generate_spiral_data(n_samples=200):
        """Generate spiral classification data"""
        t = np.random.uniform(0, 4*np.pi, n_samples)
        r = np.random.uniform(0, 2, n_samples)
        
        X = np.zeros((2, n_samples))
        X[0] = r * np.cos(t) * 0.5
        X[1] = r * np.sin(t) * 0.5
        
        Y = (np.floor(t / (2 * np.pi)) % 2).astype(int).reshape(1, -1)
        return X, Y

class NetworkTrainer:
    def __init__(self):
        self.neural_network = None
        self.training_data = None
        self.training_labels = None
        self.is_training = False
        self.training_thread = None
        self.epoch = 0
        self.training_stats = {
            'epoch': 0,
            'loss': 0,
            'accuracy': 0
        }
    
    def create_network(self, architecture):
        """Create a new neural network with given architecture"""
        self.neural_network = NeuralNetwork(architecture)
        self.epoch = 0
        self.training_stats = {'epoch': 0, 'loss': 0, 'accuracy': 0}
    
    def set_training_data(self, dataset_type):
        """Generate and set training data"""
        if dataset_type == 'circular':
            self.training_data, self.training_labels = DataGenerator.generate_circular_data()
        elif dataset_type == 'linear':
            self.training_data, self.training_labels = DataGenerator.generate_linear_data()
        elif dataset_type == 'xor':
            self.training_data, self.training_labels = DataGenerator.generate_xor_data()
        elif dataset_type == 'spiral':
            self.training_data, self.training_labels = DataGenerator.generate_spiral_data()
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_type))
    
    def start_training(self, learning_rate=0.1, max_epochs=1000):
        """Start training in a separate thread"""
        if self.is_training:
            return False
        
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_loop, 
            args=(learning_rate, max_epochs)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        return True
    
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1.0)
    
    def _training_loop(self, learning_rate, max_epochs):
        """Main training loop"""
        while self.is_training and self.epoch < max_epochs:
            try:
                # Shuffle data
                indices = np.random.permutation(self.training_data.shape[1])
                X_shuffled = self.training_data[:, indices]
                Y_shuffled = self.training_labels[:, indices]
                
                # Train on batch
                batch_size = min(32, X_shuffled.shape[1])
                X_batch = X_shuffled[:, :batch_size]
                Y_batch = Y_shuffled[:, :batch_size]
                
                # Backward propagation
                Y_pred = self.neural_network.backward_propagation(X_batch, Y_batch, learning_rate)
                
                # Calculate statistics
                loss = self.neural_network.compute_loss(Y_pred, Y_batch)
                accuracy = self.neural_network.compute_accuracy(Y_pred, Y_batch)
                
                self.epoch += 1
                self.training_stats = {
                    'epoch': self.epoch,
                    'loss': float(loss),
                    'accuracy': float(accuracy)
                }
                
                # Small delay to make training visible
                time.sleep(0.05)
            except Exception as e:
                print("Training error:", str(e))
                break

# Global trainer instance
trainer = NetworkTrainer()

# Flask routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/create_network', methods=['POST'])
def create_network():
    """Create a new neural network"""
    try:
        data = request.get_json()
        architecture = data.get('architecture', [2, 4, 4, 1])
        
        trainer.create_network(architecture)
        
        return jsonify({
            'status': 'success',
            'architecture': architecture
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    """Generate training data"""
    try:
        data = request.get_json()
        dataset_type = data.get('type', 'circular')
        
        trainer.set_training_data(dataset_type)
        
        # Return data for visualization
        data_points = []
        for i in range(min(100, trainer.training_data.shape[1])):
            data_points.append({
                'input': trainer.training_data[:, i].tolist(),
                'output': int(trainer.training_labels[0, i])
            })
        
        return jsonify({
            'status': 'success',
            'data_points': data_points,
            'total_samples': trainer.training_data.shape[1]
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start neural network training"""
    try:
        data = request.get_json()
        learning_rate = data.get('learning_rate', 0.1)
        
        if trainer.neural_network is None:
            return jsonify({'status': 'error', 'message': 'No network created'})
        
        if trainer.training_data is None:
            return jsonify({'status': 'error', 'message': 'No training data available'})
        
        success = trainer.start_training(learning_rate)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Training started' if success else 'Training already in progress'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop neural network training"""
    trainer.stop_training()
    return jsonify({'status': 'success', 'message': 'Training stopped'})

@app.route('/api/get_network_state')
def get_network_state():
    """Get current network state for visualization"""
    try:
        if trainer.neural_network is None:
            return jsonify({'status': 'error', 'message': 'No network available'})
        
        # Get network state
        network_state = trainer.neural_network.get_network_state()
        
        # Add training statistics
        network_state['training_stats'] = trainer.training_stats
        network_state['is_training'] = trainer.is_training
        
        return jsonify({
            'status': 'success',
            'network_state': network_state
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    try:
        data = request.get_json()
        input_data = np.array(data.get('input')).reshape(-1, 1)
        
        if trainer.neural_network is None:
            return jsonify({'status': 'error', 'message': 'No network available'})
        
        prediction = trainer.neural_network.predict(input_data)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Create templates directory and HTML file
def create_html_template():
    """Create the HTML template file"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python 3D Neural Network Visualizer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }
        
        #container {
            position: relative;
            width: 100vw;
            height: 100vh;
        }
        
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            color: white;
            z-index: 100;
            min-width: 280px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        #stats {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            color: white;
            z-index: 100;
            min-width: 200px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .control-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #e0e0e0;
        }
        
        input[type="range"] {
            width: 100%;
            margin-bottom: 5px;
        }
        
        input[type="number"] {
            width: 60px;
            padding: 5px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            text-align: center;
        }
        
        button {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .dataset-btn {
            background: linear-gradient(135deg, #74b9ff, #0984e3);
        }
        
        .dataset-btn:hover {
            box-shadow: 0 5px 15px rgba(116, 185, 255, 0.4);
        }
        
        .python-badge {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #3776ab, #ffde57);
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            z-index: 200;
            font-size: 14px;
            box-shadow: 0 4px 15px rgba(55, 118, 171, 0.3);
        }
    </style>
</head>
<body>
    <div class="python-badge">Python-Powered Neural Network</div>
    
    <div id="container">
        <div id="controls">
            <h3 style="margin-top: 0; color: #fff;">Python Neural Network Controls</h3>
            
            <div class="control-group">
                <label>Dataset:</label>
                <button class="dataset-btn" onclick="generateData('circular')">Circular</button>
                <button class="dataset-btn" onclick="generateData('linear')">Linear</button>
                <button class="dataset-btn" onclick="generateData('xor')">XOR</button>
                <button class="dataset-btn" onclick="generateData('spiral')">Spiral</button>
            </div>
            
            <div class="control-group">
                <label>Network Architecture:</label>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span>Layers:</span>
                    <input type="number" id="numLayers" value="3" min="2" max="5" onchange="updateArchitecture()">
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>Hidden Size:</span>
                    <input type="number" id="hiddenSize" value="4" min="2" max="8" onchange="updateArchitecture()">
                </div>
            </div>
            
            <div class="control-group">
                <label>Learning Rate: <span id="lrValue">0.1</span></label>
                <input type="range" id="learningRate" min="0.01" max="1" step="0.01" value="0.1" oninput="updateLearningRate()">
            </div>
            
            <div class="control-group">
                <button onclick="startTraining()" id="trainBtn">Start Training</button>
                <button onclick="stopTraining()" id="stopBtn" disabled>Stop</button>
                <button onclick="resetNetwork()">Reset</button>
            </div>
        </div>
        
        <div id="stats">
            <h3 style="margin-top: 0; color: #fff;">Training Stats</h3>
            <div style="margin-bottom: 10px; font-size: 14px;">Epoch: <span style="font-weight: bold; color: #74b9ff;" id="epoch">0</span></div>
            <div style="margin-bottom: 10px; font-size: 14px;">Loss: <span style="font-weight: bold; color: #74b9ff;" id="loss">0.000</span></div>
            <div style="margin-bottom: 10px; font-size: 14px;">Accuracy: <span style="font-weight: bold; color: #74b9ff;" id="accuracy">0%</span></div>
            <div style="margin-bottom: 10px; font-size: 14px;">Backend: <span style="font-weight: bold; color: #00b894;">Python Flask</span></div>
        </div>
    </div>

    <script>
        // Global variables
        let scene, camera, renderer;
        let networkMeshes = { nodes: [], connections: [] };
        let isTraining = false;
        let updateInterval;

        // Initialize Three.js
        function initThreeJS() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 10);
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);
            
            setupMouseControls();
            animate();
            
            // Start polling for network state
            startStatePolling();
        }

        function setupMouseControls() {
            let mouseDown = false;
            let mouseX = 0;
            let mouseY = 0;
            
            renderer.domElement.addEventListener('mousedown', (e) => {
                mouseDown = true;
                mouseX = e.clientX;
                mouseY = e.clientY;
            });
            
            renderer.domElement.addEventListener('mouseup', () => {
                mouseDown = false;
            });
            
            renderer.domElement.addEventListener('mousemove', (e) => {
                if (!mouseDown) return;
                
                const deltaX = e.clientX - mouseX;
                const deltaY = e.clientY - mouseY;
                
                scene.rotation.y += deltaX * 0.01;
                scene.rotation.x += deltaY * 0.01;
                
                mouseX = e.clientX;
                mouseY = e.clientY;
            });
            
            renderer.domElement.addEventListener('wheel', (e) => {
                camera.position.z += e.deltaY * 0.01;
                camera.position.z = Math.max(5, Math.min(20, camera.position.z));
            });
        }

        // API functions
        async function generateData(type) {
            try {
                // Store current training state
                const wasTraining = isTraining;
                const learningRate = parseFloat(document.getElementById('learningRate').value);
                
                const response = await fetch('/api/generate_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type: type })
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    console.log('Generated ' + result.total_samples + ' ' + type + ' data points');
                    await resetNetwork(false); // Pass false to avoid stopping training
                    // Restart training if it was active
                    if (wasTraining) {
                        await startTraining(learningRate);
                    }
                } else {
                    alert('Error generating data: ' + result.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error generating data');
            }
        }

        async function startTraining(learningRate = null) {
            const lr = learningRate || parseFloat(document.getElementById('learningRate').value);
            
            try {
                const response = await fetch('/api/start_training', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ learning_rate: lr })
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    isTraining = true;
                    document.getElementById('trainBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                } else {
                    alert('Error starting training: ' + result.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error starting training');
            }
        }

        async function stopTraining() {
            try {
                const response = await fetch('/api/stop_training', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    isTraining = false;
                    document.getElementById('trainBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        async function resetNetwork(stop = true) {
            if (stop) {
                await stopTraining();
            }
            
            const numLayers = parseInt(document.getElementById('numLayers').value);
            const hiddenSize = parseInt(document.getElementById('hiddenSize').value);
            
            const architecture = [2];
            for (let i = 0; i < numLayers - 2; i++) {
                architecture.push(hiddenSize);
            }
            architecture.push(1);
            
            try {
                const response = await fetch('/api/create_network', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ architecture: architecture })
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    console.log('Network reset with architecture:', architecture);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        function updateArchitecture() {
            resetNetwork(true);
        }

        function updateLearningRate() {
            const lr = document.getElementById('learningRate').value;
            document.getElementById('lrValue').textContent = lr;
        }

        // Polling for network state
        function startStatePolling() {
            updateInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/get_network_state');
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        updateVisualization(result.network_state);
                        updateStats(result.network_state.training_stats);
                        
                        if (!result.network_state.is_training && isTraining) {
                            isTraining = false;
                            document.getElementById('trainBtn').disabled = false;
                            document.getElementById('stopBtn').disabled = true;
                        }
                    }
                } catch (error) {
                    // Silently handle polling errors
                }
            }, 100);
        }

        function updateVisualization(networkState) {
            if (!networkState.architecture || !networkState.weights) return;
            
            // Clear existing meshes
            networkMeshes.nodes.forEach(layerNodes => {
                layerNodes.forEach(node => scene.remove(node));
            });
            networkMeshes.connections.forEach(conn => scene.remove(conn));
            
            networkMeshes.nodes = [];
            networkMeshes.connections = [];
            
            const architecture = networkState.architecture;
            const weights = networkState.weights;
            const activations = networkState.activations;
            
            const layerSpacing = 4;
            const nodeSpacing = 1.5;
            
            // Create nodes
            architecture.forEach((layerSize, layerIndex) => {
                const x = (layerIndex - (architecture.length - 1) / 2) * layerSpacing;
                const layerNodes = [];
                
                for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
                    const y = (nodeIndex - (layerSize - 1) / 2) * nodeSpacing;
                    
                    const geometry = new THREE.SphereGeometry(0.2, 16, 16);
                    const material = new THREE.MeshPhongMaterial({
                        color: layerIndex === 0 ? 0x74b9ff : 
                               layerIndex === architecture.length - 1 ? 0xff6b6b : 0x00b894
                    });
                    
                    const node = new THREE.Mesh(geometry, material);
                    node.position.set(x, y, 0);
                    
                    // Update based on activation
                    if (activations && activations[layerIndex] && activations[layerIndex][nodeIndex] !== undefined) {
                        const activation = activations[layerIndex][nodeIndex];
                        const intensity = Math.max(0.2, activation);
                        node.material.emissive.setRGB(intensity * 0.5, intensity * 0.5, intensity * 0.5);
                        node.scale.setScalar(0.8 + activation * 0.4);
                    }
                    
                    scene.add(node);
                    layerNodes.push(node);
                }
                
                networkMeshes.nodes.push(layerNodes);
            });
            
            // Create connections
            for (let i = 0; i < architecture.length - 1; i++) {
                const currentLayer = networkMeshes.nodes[i];
                const nextLayer = networkMeshes.nodes[i + 1];
                
                currentLayer.forEach((currentNode, currentIndex) => {
                    nextLayer.forEach((nextNode, nextIndex) => {
                        const geometry = new THREE.BufferGeometry();
                        const positions = new Float32Array([
                            currentNode.position.x, currentNode.position.y, currentNode.position.z,
                            nextNode.position.x, nextNode.position.y, nextNode.position.z
                        ]);
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        
                        const weight = weights[i][nextIndex][currentIndex];
                        const opacity = Math.min(Math.abs(weight) * 2, 1);
                        const color = weight > 0 ? 0x00ff00 : 0xff0000;
                        
                        const material = new THREE.LineBasicMaterial({
                            color: color,
                            opacity: opacity,
                            transparent: true
                        });
                        
                        const line = new THREE.Line(geometry, material);
                        scene.add(line);
                        networkMeshes.connections.push(line);
                    });
                });
            }
        }

        function updateStats(stats) {
            if (!stats) return;
            
            document.getElementById('epoch').textContent = stats.epoch || 0;
            document.getElementById('loss').textContent = (stats.loss || 0).toFixed(4);
            document.getElementById('accuracy').textContent = (stats.accuracy || 0).toFixed(1) + '%';
        }

        function animate() {
            requestAnimationFrame(animate);
            
            if (!isTraining) {
                scene.rotation.y += 0.002;
            }
            
            renderer.render(scene, camera);
        }

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Initialize
        window.addEventListener('load', () => {
            initThreeJS();
            resetNetwork();
            generateData('circular');
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    # Create HTML template
    create_html_template()
    
    # Initialize with default network
    trainer.create_network([2, 4, 4, 1])
    
    print("Python Neural Network Visualizer Server")
    print("=" * 50)
    print("Features:")
    print("- NumPy-based neural network implementation")
    print("- Flask REST API backend")
    print("- Real-time 3D visualization")
    print("- Multiple dataset generators")
    print("- Configurable network architecture")
    print("- Live training statistics")
    print("")
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        trainer.stop_training()
    except Exception as e:
        print(f"Server error: {e}")
        trainer.stop_training()
# Energy Prediction System

<div align="center">

![Energy Prediction System](images/main.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.9](https://img.shields.io/badge/tensorflow-2.9-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-1.6-red.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/yourusername/Electrical-energy-prediction/actions)

</div>

## 🔋 Overview

An advanced machine learning system for accurately predicting energy consumption in electric vehicles. This system provides highly precise energy usage and range estimations based on driving conditions, weather, route characteristics, and vehicle parameters.

**Real-world applications:**

-  Improving range estimation accuracy in Electrical vehicles
-  Optimizing route planning to minimize energy consumption
-  Supporting battery management systems for improved longevity
-  Providing drivers with more confidence in their vehicle's capabilities

## ✨ Key Features

-  **High prediction accuracy** with mean error below 4% across diverse conditions
-  **Multi-model approach** combining gradient boosting and deep learning techniques
-  **Physics-informed features** based on EV energy dynamics
-  **Production-ready API** with monitoring and scalability
-  **End-to-end pipeline** from data processing to real-time predictions
-  **Comprehensive evaluation** framework with real-world testing scenarios

## 🚀 Quick Start

### Prerequisites

-  Python 3.8+
-  Docker and Docker Compose (for containerized deployment)
-  NVIDIA GPU (recommended for training, not required for inference)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/e-forecast.git
cd e-forecast
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv/Scripts/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

### Running the API

The fastest way to start the prediction service:

```bash
# Deploy with Docker
docker-compose -f docker/docker-compose.yml up -d

# Or run directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then access:

-  API endpoints: http://localhost:8000/api/v1/
-  Interactive documentation: http://localhost:8000/docs
-  Monitoring dashboards: http://localhost:3000 (if using Docker)

### Making Predictions

```python
import requests
import json

# Example prediction request
payload = {
  "vehicle": {
    "vehicle_id": "model_y_2022",
    "weight": 2100.0,
    "drag_coefficient": 0.23,
    "frontal_area": 2.7,
    "battery_capacity": 75.0
  },
  "driving_conditions": [
    {
      "timestamp": "2023-05-15T14:30:00Z",
      "speed": 105.0,
      "altitude": 150.0,
      "temperature": 22.5,
      "wind_speed": 15.0,
      "precipitation": 0.0,
      "road_type": "highway"
    },
    # Additional data points...
  ]
}

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(json.dumps(response.json(), indent=2))
```

### XGBoost Model

Gradient boosting model specialized for tabular data with complex feature interactions:

-  Efficiently captures non-linear relationships between features
-  Handles mixed data types and missing values robustly
-  Provides native feature importance for interpretability

### LSTM Model

Deep learning model capturing temporal patterns in driving sequences:

-  Learns from sequences of driving conditions
-  Recognizes long-term dependencies in energy consumption
-  Adapts to different driving styles and patterns

### Ensemble Integration

Combines predictions from multiple models to maximize accuracy:

-  Weighted averaging based on model confidence
-  Optimized weights through validation performance
-  Provides confidence intervals for uncertainty estimation

## 📊 Performance Results

The system achieves high accuracy across various driving conditions:

| Model    | MAPE | RMSE (kWh) | Inference Time (ms) |
| -------- | ---- | ---------- | ------------------- |
| XGBoost  | 3.8% | 0.92       | 5.2                 |
| LSTM     | 3.5% | 0.85       | 18.7                |
| Ensemble | 3.2% | 0.78       | 24.1                |

### Key Findings

Our analysis identified the most important factors affecting energy consumption:

1. **Speed profiles** (42% contribution)
2. **Elevation changes** (23% contribution)
3. **Temperature** (18% contribution, especially in extreme conditions)
4. **Vehicle parameters** (12% contribution)
5. **Traffic and driving style** (5% contribution)

## 🛠️ Technology Stack

-  **Machine Learning**: TensorFlow, XGBoost, Scikit-learn
-  **Data Processing**: Pandas, NumPy
-  **API Development**: FastAPI, Pydantic, Uvicorn
-  **Containerization**: Docker, Docker Compose
-  **Testing**: Pytest, Hypothesis
-  **Monitoring**: Prometheus, Grafana
-  **Development Tools**: Black, Flake8, Pre-commit, MyPy

## 📁 Project Structure

```
e-forecast/
├── data/                    # Data directory
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── api/                 # API service
│   ├── data/                # Data processing
│   ├── features/            # Feature engineering
│   ├── models/              # Model implementation
│   └── visualization/       # Visualization utilities
├── tests/                   # Test suite
├── configs/                 # Configuration files
├── docker/                  # Docker configuration
└── scripts/                 # Utility scripts
```

## 📈 Future Work

-  Integration with real-time traffic prediction systems
-  Driver behavior profiling for personalized predictions
-  Federated learning approach for privacy-preserving model improvements
-  Battery degradation modeling for long-term accuracy
-  Edge deployment for on-device inference

## 🧪 Testing

Run the test suite to verify all components:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/
```

## 🔍 Documentation

Detailed documentation is available in the `docs/` directory. To build the documentation:

```bash
cd docs
make html
```

Then open `docs/_build/html/index.html` in your browser.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Muhammed Kartal - [kartal.dev](https://kartal.dev)

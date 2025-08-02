#  MLOps Major Assignment - California Housing Linear Regression Pipeline

This repository contains a complete MLOps pipeline for Linear Regression using the California Housing dataset from `sklearn`.  
The pipeline includes training, testing, quantization, Dockerization, and CI/CD automation.

---

##  Project Structure
```

project-root/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD workflow
├── src/
│   ├── **init**.py
│   ├── train.py                   # Model training script
│   ├── quantize.py                # Model quantization script
│   ├── predict.py                 # Prediction script for Docker
│   └── utils.py                   # Utility functions
├── Dockerfile                     # Docker configuration
├── model.joblib
├── quant\_params.joblib
├── unquant\_params.joblib
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # This file

````

---

##  Features
**Model Training** – Linear Regression on California Housing dataset  
**Model Testing** – Comprehensive unit tests with `pytest`  
**Model Quantization** – Manual 8-bit quantization of model parameters  
**Containerization** – Docker setup for model deployment  
**CI/CD Pipeline** – Automated testing, training, and deployment with GitHub Actions  

---

##  Requirements
- Python 3.8+
- scikit-learn
- numpy
- joblib
- pytest

---

##  Quick Start

###  Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

###  Training the Model

```bash
python src/train.py
```
This will:

* Load the California Housing dataset
* Train a Linear Regression model
* Print **R² score** and loss metrics
* Save the trained model as `model.joblib`

---

###  Quantizing the Model

```bash
python src/quantize.py
```
 This will:

* Load the trained model
* Extract coefficients and intercept
* Save raw parameters as `unquant_params.joblib`
* Quantize parameters to **8-bit unsigned integers**
* Save quantized parameters as `quant_params.joblib`
* Perform inference with de-quantized weights

---

###  Running Tests

```bash
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Build Docker image
docker build -t mlops-pipeline.

# Run Docker container
docker run mlops-pipeline
```

### **DockerHub Image**

You can pull the latest Docker image from [DockerHub](https://hub.docker.com/repository/docker/toteabhinav/mlops-pipeline/tags):

```bash
docker pull toteabhinav/mlops-pipeline
docker run --rm toteabhinav/mlops-pipeline
```

---

##  CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes three jobs:

**test\_suite** – Runs pytest tests
**train\_and\_quantize** – Trains model and runs quantization
**build\_and\_test\_container** – Builds Docker image and tests container

The pipeline is **triggered on every push to the `main` branch**.

---

##  Model Performance

| Metric         | Value                        |
| -------------- | ---------------------------- |
| **Model Type** | Linear Regression            |
| **Dataset**    | California Housing (sklearn) |
| **R² Score**   | \~0.60 (typical)             |
| **Features**   | 8 numerical features         |
| **Target**     | Median house value           |

---

## Quantization Details

* **Precision**: 8-bit unsigned integers (0-255)
* **Parameters Quantized**: Model coefficients and intercept
* **Format**: Manual quantization with scale and zero-point
* **Storage**: Separate files for original and quantized parameters

---

## Docker Configuration

The Docker container:

* Uses Python 3.9 slim base image
* Installs all required dependencies
* Includes the trained model and prediction script
* Runs `predict.py` on container start

---

## Testing Strategy

Unit tests cover:
Dataset loading functionality
Model creation and validation
Training process verification
R² score threshold validation
Parameter extraction for quantization




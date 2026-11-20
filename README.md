# AI Model Health Monitor

An end-to-end, local-first MLOps project for customer churn prediction.

The system progressively implements:

1. Data ingestion, validation and preprocessing
2. Model training and evaluation
3. MLflow experiment tracking and model versioning
4. FastAPI prediction service
5. Docker packaging
6. Data and model monitoring with Evidently
7. Streamlit monitoring dashboard
8. GitHub Actions CI
9. Controlled retraining, model promotion and rollback

> Current status: Phase 5 — Dockerization. Phases 0–4 are implemented, and the Docker image, container, model loading and prediction endpoints have been successfully tested.

## Technology stack

Python 3.12, pandas, NumPy, scikit-learn, MLflow, SQLite, FastAPI, Docker, Streamlit, Evidently, Pytest and GitHub Actions.

## Project structure

```text
ai-model-health-monitor/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── docs/
├── models/
├── reports/
├── scripts/
│   └── register_docker_model.py
├── src/
│   ├── data/
│   ├── training/
│   ├── tracking/
│   ├── monitoring/
│   ├── serving/
│   └── retraining/
├── dashboard/
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

For local development:

- Python 3.12
- Git
- Docker Desktop (for containerized deployment)
- VS Code (recommended)

## Local setup

```bash
git clone https://github.com/Harshini20-20/ai-model-health-monitor.git
cd ai-model-health-monitor

python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and update values only when needed.

Never commit credentials, secrets or private customer data.

## Phase 1 — Data engineering

The project uses the IBM Telco Customer Churn dataset.

Place the dataset at:

```text
data/raw/telco_customer_churn.csv
```

Prepare the processed training and testing datasets:

```bash
python -m src.training.prepare_data
```

The pipeline performs data validation, preprocessing and a stratified train-test split.

## Phase 2 — Model training

Train and compare the following algorithms:

- Logistic Regression
- Random Forest
- Gradient Boosting

Run:

```bash
python -m src.training.train
```

Training uses:

- Stratified 5-fold cross-validation
- GridSearchCV for hyperparameter tuning
- Balanced class weights
- ROC-AUC for model selection

The selected model is determined by mean cross-validation ROC-AUC.

The current trained model is Gradient Boosting.

Run tests:

```bash
python -m pytest
```

### Model artifacts

- `models/churn_pipeline_v1.joblib` — fitted preprocessing and prediction pipeline.
- `models/churn_pipeline_v1.metadata.json` — model metadata, input columns, hyperparameters and evaluation metrics.
- `reports/model_comparison.csv` — model comparison.
- `reports/evaluation_metrics.json` — evaluation metrics and confusion matrices.

The model was trained using scikit-learn 1.8.0. Use a compatible version when loading the serialized model.

## Phase 3 — MLflow experiment tracking and model registry

MLflow tracks training experiments, parameters, metrics and model artifacts.

The project uses:

- SQLite as the tracking backend.
- Local artifact storage.
- MLflow experiments and nested runs.
- Registered model versions.
- The `candidate` model alias.

Configuration is managed in `config/config.yaml`.

Start the MLflow UI locally:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://localhost:5000
```

The registered model is named `churn_pipeline`.

## Phase 4 — FastAPI model serving

The project exposes a REST API using FastAPI.

The application loads the registered MLflow model using its configured alias.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model health |
| GET | `/model-info` | Returns registered model information |
| POST | `/predict` | Predicts customer churn |

Start the API locally:

```bash
python -m uvicorn src.serving.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Example prediction response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8274415700265743,
  "model_name": "churn_pipeline",
  "model_version": "1"
}
```

## Phase 5 — Dockerization

Docker packages the application, dependencies and trained model into a portable image.

The Docker build:

1. Uses Python 3.12.10.
2. Installs the project dependencies.
3. Copies the application source code and trained model artifacts.
4. Creates a separate Linux-compatible MLflow registry using the existing trained model.
5. Registers the model with the `candidate` alias.
6. Starts the FastAPI application on port 8000.

The Docker registration process reuses the existing fitted model. It does not retrain the model.

### Build the Docker image

From the project root:

```bash
docker build -t ai-model-health-monitor .
```

### Start the container

```bash
docker run -d --name ai-model-health-monitor -p 8000:8000 ai-model-health-monitor
```

### Verify the container

```bash
docker ps
```

### Test the API

Health check:

```text
http://localhost:8000/health
```

Model information:

```text
http://localhost:8000/model-info
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Container logs

```bash
docker logs ai-model-health-monitor
```

### Stop and restart

```bash
docker stop ai-model-health-monitor
docker start ai-model-health-monitor
```

### Remove the container

```bash
docker stop ai-model-health-monitor
docker rm ai-model-health-monitor
```

To recreate it later, use the Docker run command above.

### Docker implementation notes

- `.dockerignore` excludes development environments, raw and processed datasets, Git history and the original Windows MLflow database.
- The trained model and metadata are included in the Docker image.
- A separate MLflow registry is generated inside the Linux container environment.
- The original Windows MLflow database remains independent of the Docker registry.
- The container exposes port 8000.

The Docker image and container have been tested successfully, including model loading, health checks, model information, prediction and container restart.

## Development approach

Build one phase at a time. Run relevant tests, inspect outputs and commit only working milestones.

## Current limitations

- **Monitoring:** Data drift and model performance monitoring are not yet implemented.
- **Dashboard:** The Streamlit monitoring dashboard is planned for Phase 7.
- **CI:** Automated GitHub Actions workflows are planned for Phase 8.
- **Retraining:** Automated retraining, model promotion and rollback are planned for Phase 9.
- **Model registry:** MLflow currently uses a local SQLite tracking database and artifact storage. It is intended for local, single-user development and is not a shared multi-user tracking service.
- **Classification threshold:** The model uses the default 0.5 classification threshold. It has not been optimized against business-specific false-positive and false-negative costs.

## Future development

| Phase | Objective |
|---|---|
| Phase 6 | Data and model monitoring using Evidently |
| Phase 7 | Streamlit monitoring dashboard |
| Phase 8 | GitHub Actions continuous integration |
| Phase 9 | Controlled retraining, model promotion and rollback |

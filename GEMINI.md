# Tutor.IA ML Project Context

## Learning Directive

Given that the user is new to Data Science & ML, all interactions will not only focus on problem resolution but also on ensuring a comprehensive learning experience. This includes explaining concepts, justifying decisions, providing context, offering best practices, and encouraging questions related to Data Science and Machine Learning.

This document provides instructional context for the Tutor.IA ML project, which focuses on the creation and validation of Machine Learning and Deep Learning models for the broader Tutor.IA intelligent tutoring system.

## Project Overview

This project has two main purposes:

*   **Primary Purpose (User-Centric):** To serve as a practical learning environment for the user to gain hands-on experience and deepen their understanding of Machine Learning and Deep Learning concepts, tools, and best practices.
*   **Secondary Purpose (Technical):** To develop, train, and deploy Machine Learning and Deep Learning models within the broader Tutor.IA intelligent tutoring system. Specifically, it aims to enhance the personalized learning experience by providing intelligent capabilities such as learning profile inference and content recommendation.

The `tutor-ia-ml` project is a core component of the Tutor.IA system, specializing in the development, training, and deployment of machine learning models.

## Architecture and Technologies

The project leverages a robust stack of data science and machine learning tools:

*   **Data Analysis & Manipulation:**
    *   **Pandas:** Used for data loading, cleaning, and exploratory data analysis (EDA) of tabular data.
    *   **NumPy:** Provides the foundation for scientific computing, enabling efficient mathematical operations on arrays.
*   **Data Visualization:**
    *   **Matplotlib:** The primary tool for creating static graphs and visualizations.
*   **Machine Learning Modeling:**
    *   **Scikit-learn:** A versatile library for classic machine learning tasks, including data preprocessing, training various models (regression, classification, clustering), and performance evaluation.
    *   **PyTorch:** A deep learning framework used for building and training neural networks.
*   **Serving & Deployment:**
    *   **FastAPI:** Utilized for building high-performance APIs to serve trained models in production, allowing other systems to consume predictions.
    *   **Uvicorn:** A high-performance ASGI server used to run FastAPI applications.
    *   **Streamlit:** Employed for creating interactive web applications and dashboards quickly, ideal for prototyping and demonstrating models.
*   **Ecosystem & Production Tools:**
    *   **Virtual Environments (venv, uv):** Used for isolating and managing project dependencies.
    *   **Docker:** For containerizing applications, ensuring consistent execution across different environments.

## Project Structure

```
/Users/dionatasantosdasilva/Repos/personal/tcc/tutor-ia-project/tutor-ia-ml/
├───.gitignore
├───.python-version
├───main.py             # Main application entry point (likely FastAPI)
├───pyproject.toml      # Python project definition and dependencies
├───README.md           # Project README
├───uv.lock             # Dependency lock file
├───.git/...            # Git repository
└───.venv/...           # Python virtual environment
```

## Building and Running

The `README.md` mentions `uv` for dependency management and `FastAPI` with `Uvicorn` for serving. It also mentions Docker for containerization.

### Dependency Management

The project uses `uv` for dependency management, as indicated by `pyproject.toml` and `uv.lock`.

*   **Install dependencies:**
    ```bash
    uv sync
    ```

### Running the Application

The `main.py` file is likely the entry point for the FastAPI application.

*   **Locally (assuming `main.py` contains the FastAPI app):**
    ```bash
    source .venv/bin/activate
    uvicorn main:app --reload
    ```
    *(Note: The exact command might vary depending on the variable name of the FastAPI app in `main.py`)*

*   **With Docker:**
    The `README.md` mentions Docker for containerization, implying a `Dockerfile` would be present for building and running. (A `Dockerfile` is not currently visible in the provided directory structure, so this section is a placeholder.)
    ```bash
    # TODO: Add Docker build and run commands once Dockerfile is available
    # docker build -t tutor-ia-ml .
    # docker run -p 8000:8000 tutor-ia-ml
    ```

## Development Conventions

*   Nunca utilizar a tool `readManyFiles`, a não ser que seja explicitamente requisitado pelo usuário.
*   Na geração de código, nunca exiba o código gerado na resposta. Em vez disso, detalhe o que será implementado e, após isso, peça ao usuário para validar o código gerado no(s) arquivo(s) específico(s).
*   Nunca utilize uma estratégia de reescrita completa de arquivos. Em vez disso, prefira modificações cirúrgicas e seguras, como a ferramenta `replace` ou outras abordagens que minimizem o risco de corrupção de dados.
*   **Dependency Management:** `uv` is used for managing Python dependencies.
*   **API Development:** FastAPI is the chosen framework for building APIs.
*   **Containerization:** Docker is intended for packaging applications for consistent environments.
*   **Model Development:** Scikit-learn and PyTorch are the primary libraries for ML and DL model development, respectively.
*   **Prototyping/Demonstration:** Streamlit is used for quick interactive web applications.

## Future Enhancements & MLOps Radar

While the current stack is excellent for core ML development and learning, for robust, scalable, and production-grade ML systems, consider exploring the following areas as you advance:

*   **Experiment Tracking & Model Versioning:** Tools like MLflow, Weights & Biases, or DVC to track experiments, hyperparameters, metrics, and manage different versions of models and their artifacts.
*   **Data Versioning:** Tools like DVC (Data Version Control) or LakeFS to version datasets alongside code, ensuring reproducibility and traceability of data changes.
*   **Orchestration of ML Pipelines:** Tools such as Apache Airflow, Prefect, or Kubeflow Pipelines to automate, schedule, and manage complex end-to-end ML workflows (data ingestion, preprocessing, training, evaluation, deployment).
*   **Monitoring of Models in Production:** Solutions like Prometheus/Grafana or specialized ML monitoring tools (e.g., Evidently AI) to continuously track model performance, detect data drift, concept drift, and ensure model health in a production environment.
*   **Data Validation:** Libraries like Great Expectations or Pandera to define and enforce data quality expectations throughout the ML pipeline, preventing "garbage in, garbage out" scenarios.
*   **Feature Store:** For projects with many models or complex features, a Feature Store (e.g., Feast, Tecton) can centralize feature engineering, serving, and ensure consistency between training and inference. (Likely an advanced topic for later stages).
*   **Cloud Platform Integration:** Leveraging cloud-specific ML services (e.g., Google Cloud Vertex AI, AWS SageMaker, Azure Machine Learning) for managed services, scaling, and integrated MLOps workflows, especially for large-scale deployments.

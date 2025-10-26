import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- 1. App & Model Loading ---

# Define os caminhos para os artefatos do modelo
MODEL_PATH = os.path.join('src', 'models', 'lp_model.joblib')
LE_PATH = os.path.join('src', 'models', 'label_encoder.joblib')


# Cria a aplicação FastAPI
app = FastAPI(
    title="Tutor.IA - Modelo de Diagnóstico de Perfil de Aprendizagem",
    description="Uma API para inferir o perfil de aprendizagem de um usuário com base em seus hábitos de estudo.",
    version="0.2.0"
)

# Carrega os artefatos (pipeline e encoder) na inicialização
try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    print("Pipeline de modelo e LabelEncoder carregados com sucesso.")
except FileNotFoundError as e:
    print(f"ERRO: Arquivo de modelo não encontrado. Detalhes: {e}")
    model = None
    le = None

# --- 2. Pydantic Schemas (Data Validation) ---

# Define a estrutura de dados que a API espera receber (dados brutos).
class RawInput(BaseModel):
    time_spent_on_video: float
    time_spent_on_audio: float
    time_spent_on_reading: float
    time_spent_on_writing: float
    time_spent_on_quizz: float
    time_spent_on_flashcards: float
    time_spent_on_projects: float
    completed_exercices: int
    completed_quizzes: int
    completed_flashcards: int
    text_quizzes_accuracy: int
    visual_quizzes_accuracy: int
    most_preferred_resource_type: str # ex: 'video', 'audio', 'text', 'practical'

    class Config:
        # Exemplo de como a API espera os dados
        schema_extra = {
            "example": {
                "time_spent_on_video": 5.0,
                "time_spent_on_audio": 1.0,
                "time_spent_on_reading": 3.0,
                "time_spent_on_writing": 2.0,
                "time_spent_on_quizz": 2.0,
                "time_spent_on_flashcards": 1.0,
                "time_spent_on_projects": 4.0,
                "completed_exercices": 10,
                "completed_quizzes": 5,
                "completed_flashcards": 3,
                "text_quizzes_accuracy": 75,
                "visual_quizzes_accuracy": 85,
                "most_preferred_resource_type": "video"
            }
        }

class InferenceResponse(BaseModel):
    predicted_profile: str


# --- 3. API Endpoints ---

@app.get("/", summary="Endpoint raiz da API", description="Verifica se a API está online e se os artefatos do modelo foram carregados.")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "label_encoder_loaded": le is not None
    }

@app.post("/predict", response_model=InferenceResponse, summary="Realiza o diagnóstico do perfil de aprendizagem")
def predict_learning_profile(data: RawInput):
    """
    Recebe os hábitos de estudo de um usuário e retorna o perfil de aprendizagem previsto.

    - **data**: Um objeto JSON com as features do usuário em seu formato bruto.
    \f
    :param data: Dados de entrada validados pelo Pydantic.
    :return: Um JSON com o perfil de aprendizagem previsto.
    """
    if model is None or le is None:
        return {"error": "Modelo ou LabelEncoder não foram carregados. A inferência não pode ser realizada."}

    # Converte os dados de entrada Pydantic para um dicionário e depois para um DataFrame
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])

    # Realiza a previsão (o pipeline cuida de todo o pré-processamento)
    prediction_numeric = model.predict(input_df)
    
    # Usa o LabelEncoder para obter o nome do perfil
    predicted_profile_name = le.inverse_transform(prediction_numeric)

    return {"predicted_profile": predicted_profile_name[0]}
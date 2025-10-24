import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- 1. App & Model Loading ---

# Define o caminho para o modelo
MODEL_PATH = os.path.join('src', 'models', 'lp_inference_model.joblib')

# Cria a aplicação FastAPI
app = FastAPI(
    title="Tutor.IA - Modelo de Inferência de Perfil de Aprendizagem",
    description="Uma API para inferir o perfil de aprendizagem de um usuário com base em suas interações.",
    version="0.1.0"
)

# Carrega o modelo na inicialização da aplicação
# O modelo ficará em memória para ser reutilizado em todas as requisições
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo de inferência carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo do modelo não encontrado em {MODEL_PATH}")
    model = None

# Mapeamento do resultado numérico para o nome do perfil
LEARNING_PROFILE_MAP = {
    0: 'Auditivo',
    1: 'Cinestésico',
    2: 'Leitura/Escrita',
    3: 'Visual'
}

# --- 2. Pydantic Schemas (Data Validation) ---

# Define a estrutura de dados que a API espera receber para uma previsão.
# Isso garante que os dados de entrada sejam válidos.
class InferenceInput(BaseModel):
    time_spent_on_video: float
    time_spent_on_audio: float
    time_spent_reading: float
    time_spent_writing: float
    time_spent_on_quizz: float
    time_spent_on_flashcards: float
    completed_exercices: float
    completed_quizzes: float
    completed_flashcards: float
    text_quizzes_accuracy: float
    visual_quizzes_accuracy: float
    most_preferred_resource_type_audio: float
    most_preferred_resource_type_practical: float
    most_preferred_resource_type_text: float
    most_preferred_resource_type_video: float

    class Config:
        # Exemplo de como a API espera os dados
        schema_extra = {
            "example": {
                "time_spent_on_video": 1.8,
                "time_spent_on_audio": -0.5,
                "time_spent_reading": -0.8,
                "time_spent_writing": -0.9,
                "time_spent_on_quizz": 0.5,
                "time_spent_on_flashcards": 1.5,
                "completed_exercices": -1.0,
                "completed_quizzes": 1.2,
                "completed_flashcards": 1.8,
                "text_quizzes_accuracy": -1.5,
                "visual_quizzes_accuracy": 1.9,
                "most_preferred_resource_type_audio": 0.0,
                "most_preferred_resource_type_practical": 0.0,
                "most_preferred_resource_type_text": 0.0,
                "most_preferred_resource_type_video": 1.0
            }
        }

class InferenceResponse(BaseModel):
    predicted_profile: str


# --- 3. API Endpoints ---

@app.get("/", summary="Endpoint raiz da API", description="Verifica se a API está online e se o modelo foi carregado.")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=InferenceResponse, summary="Realiza a inferência do perfil de aprendizagem")
def predict_learning_profile(data: InferenceInput):
    """
    Recebe os dados de interação de um usuário e retorna o perfil de aprendizagem previsto.

    - **data**: Um objeto JSON com as features do usuário.
    \f
    :param data: Dados de entrada validados pelo Pydantic.
    :return: Um JSON com o perfil de aprendizagem previsto.
    """
    if model is None:
        return {"error": "Modelo não foi carregado. A inferência não pode ser realizada."}

    # Converte os dados de entrada Pydantic para um dicionário e depois para um DataFrame
    input_data = data.dict()
    # O modelo espera um array 2D, então colocamos o dicionário dentro de uma lista
    input_df = pd.DataFrame([input_data])

    # Realiza a previsão
    prediction_numeric = model.predict(input_df)
    
    # Mapeia o resultado numérico para o nome do perfil
    predicted_profile_name = LEARNING_PROFILE_MAP.get(prediction_numeric[0], "Desconhecido")

    return {"predicted_profile": predicted_profile_name}
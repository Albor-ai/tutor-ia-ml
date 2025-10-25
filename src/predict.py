import pandas as pd
import joblib
import os

# --- 1. Definição de Caminhos ---
MODEL_DIR = os.path.join('src', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lp_model.joblib')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# --- 2. Carregamento dos Artefatos ---
print("Carregando o pipeline e o LabelEncoder...")
try:
    pipeline = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    print("Artefatos carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Detalhes: {e}")
    print("Por favor, execute o script 'src/save_model.py' primeiro.")
    exit()

# --- 3. Preparação de Novos Dados (Exemplo com DADOS BRUTOS) ---
print("\nPreparando dados de exemplo para inferência (formato bruto)...")

# Exemplo: Usuário com forte inclinação para vídeos e recursos visuais
sample_data_visual_raw = {
    'time_spent_on_video': [120],
    'time_spent_on_audio': [10],
    'time_spent_reading': [20],
    'time_spent_writing': [5],
    'time_spent_on_quizz': [30],
    'time_spent_on_flashcards': [40],
    'completed_exercices': [5],
    'completed_quizzes': [25],
    'completed_flashcards': [80],
    'text_quizzes_accuracy': [60],
    'visual_quizzes_accuracy': [95],
    'most_preferred_resource_type': ['video']
}

sample_df_raw = pd.DataFrame(sample_data_visual_raw)
print("Dados de exemplo (brutos):")
print(sample_df_raw)

# --- 4. Execução da Inferência ---
print("\nExecutando inferência através do pipeline...")
# O pipeline aplica o pré-processamento e retorna a previsão numérica
prediction_numeric = pipeline.predict(sample_df_raw)
print(f"Previsão numérica do modelo: {prediction_numeric}")

# Usa o LabelEncoder carregado para decodificar a previsão numérica para o nome do perfil
predicted_profile_name = le.inverse_transform(prediction_numeric)

# --- 5. Exibição do Resultado ---
print("\n--- Resultado da Inferência ---")
print(f"O perfil de aprendizagem previsto para o usuário de exemplo é: {predicted_profile_name[0]}")
print("-----------------------------")

import pandas as pd
import joblib
import os

# --- 1. Definição de Caminhos e Constantes ---
MODEL_PATH = os.path.join('src', 'models', 'lp_inference_model.joblib')
# Mapeamento inverso do target (deve ser o mesmo usado no pré-processamento)
# Em um sistema real, isso também seria salvo ou gerenciado de forma centralizada.
LEARNING_PROFILE_MAP = {
    0: 'Auditivo',
    1: 'Cinestésico',
    2: 'Leitura/Escrita',
    3: 'Visual'
}

# --- 2. Carregamento do Modelo ---
print("Carregando o modelo treinado...")
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo do modelo não encontrado em '{MODEL_PATH}'.")
    print("Por favor, execute o script 'src/save_model.py' primeiro.")
    exit()

# --- 3. Preparação de Novos Dados (Exemplo) ---
# Em um cenário real, esses dados viriam de uma requisição de API, um banco de dados, etc.
# Os dados devem ter a mesma estrutura e pré-processamento (scaling, one-hot encoding)
# que os dados de treino. Para este exemplo, vamos simular dados já pré-processados.
print("\nPreparando dados de exemplo para inferência...")

# Exemplo 1: Usuário com forte inclinação para vídeos e recursos visuais
sample_data_visual = {
    'time_spent_on_video': [1.8],
    'time_spent_on_audio': [-0.5],
    'time_spent_reading': [-0.8],
    'time_spent_writing': [-0.9],
    'time_spent_on_quizz': [0.5],
    'time_spent_on_flashcards': [1.5],
    'completed_exercices': [-1.0],
    'completed_quizzes': [1.2],
    'completed_flashcards': [1.8],
    'text_quizzes_accuracy': [-1.5],
    'visual_quizzes_accuracy': [1.9],
    'most_preferred_resource_type_audio': [0],
    'most_preferred_resource_type_practical': [0],
    'most_preferred_resource_type_text': [0],
    'most_preferred_resource_type_video': [1]
}

# Converte o dicionário para um DataFrame do Pandas
# É crucial que as colunas estejam na mesma ordem do treino.
sample_df = pd.DataFrame(sample_data_visual)
print("Dados de exemplo:")
print(sample_df)


# --- 4. Execução da Inferência ---
print("\nExecutando inferência...")
prediction_numeric = model.predict(sample_df)
print(f"Previsão numérica do modelo: {prediction_numeric}")

# Mapeia o resultado numérico para o nome do perfil
predicted_profile_name = LEARNING_PROFILE_MAP.get(prediction_numeric[0], "Desconhecido")


# --- 5. Exibição do Resultado ---
print("\n--- Resultado da Inferência ---")
print(f"O perfil de aprendizagem previsto para o usuário de exemplo é: {predicted_profile_name}")
print("-----------------------------")

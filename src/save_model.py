import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Definição de Caminhos ---
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_DIR = os.path.join('src', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lp_inference_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# --- 2. Carregamento dos Dados de Treino ---
print("Carregando dados de treino...")
try:
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivos de dados de treino não encontrados.")
    print("Por favor, execute o notebook de pré-processamento (preprocessing_lp_data.ipynb) primeiro.")
    exit()

# --- 3. Treinamento e Salvamento do Scaler ---
print("\nTreinando o StandardScaler...")
# Identifica as colunas numéricas (todas exceto as que são one-hot encoded)
numeric_features = [col for col in X_train.columns if 'most_preferred_resource_type' not in col]

scaler = StandardScaler()
# Treina o scaler APENAS nos dados numéricos de treino
scaler.fit(X_train[numeric_features])

# Garante que o diretório de modelos exista
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Salvando o scaler em: {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)
print("Scaler salvo com sucesso.")

# --- 4. Treinamento do Modelo ---
print("\nTreinando o modelo RandomForestClassifier...")
# O modelo é treinado nos dados já escalados do arquivo, como antes
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

# --- 5. Salvando o Modelo ---
print(f"\nSalvando o modelo em: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print("\nModelo e Scaler salvos com sucesso!")
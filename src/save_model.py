import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- 1. Definição de Caminhos ---
RAW_DATA_PATH = os.path.join('data', 'raw', 'simulated_lp_data.csv')
MODEL_DIR = os.path.join('src', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lp_model.joblib')

# --- 2. Carregamento dos Dados Brutos ---
print("Carregando dados brutos...")
try:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo de dados brutos não encontrado em '{RAW_DATA_PATH}'.")
    print("Por favor, execute o script 'src/utils/generate_simulated_data.py' primeiro.")
    exit()

# --- 3. Divisão dos Dados ---
# Separamos as features (X) do target (y) ANTES de qualquer outra coisa
X = df_raw.drop('perfil_aprendizagem', axis=1)
y = df_raw['perfil_aprendizagem']

# Dividimos em treino e teste para treinar o pipeline
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 4. Definição do Pipeline de Pré-processamento ---
# Definimos explicitamente quais colunas são numéricas e quais são categóricas
numeric_features = [
    'time_spent_on_video', 'time_spent_on_audio', 'time_spent_reading',
    'time_spent_writing', 'time_spent_on_quizz', 'time_spent_on_flashcards',
    'completed_exercices', 'completed_quizzes', 'completed_flashcards',
    'text_quizzes_accuracy', 'visual_quizzes_accuracy'
]
categorical_features = ['most_preferred_resource_type']

# Cria o transformador que aplica StandardScaler nas colunas numéricas
# e OneHotEncoder na categórica.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 5. Definição do Modelo Final (XGBoost com melhores parâmetros) ---
# O LabelEncoder do XGBoost pode ser problemático, então codificamos o target antes.
le = LabelEncoder()
_ = le.fit(y_train) # Treina o LabelEncoder
y_train_encoded = le.transform(y_train)

best_params_xgb = {
    'subsample': 0.8,
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.01,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

final_model = XGBClassifier(**best_params_xgb)

# --- 6. Criação e Treinamento do Pipeline Completo ---
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', final_model)
])

print("\nTreinando o pipeline completo (pré-processador + modelo XGBoost)...")
# O pipeline é treinado nos dados de treino brutos (X_train) e no target codificado.
full_pipeline.fit(X_train, y_train_encoded)

print("Pipeline treinado com sucesso.")

# --- 7. Salvando o Pipeline e o LabelEncoder ---
# Agora precisamos salvar tanto o pipeline quanto o LabelEncoder para a inferência.
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Salvando o pipeline completo em: {MODEL_PATH}")
joblib.dump(full_pipeline, MODEL_PATH)

# Salva também o LabelEncoder
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
joblib.dump(le, LE_PATH)
print(f"Salvando o LabelEncoder em: {LE_PATH}")

print("Pipeline e LabelEncoder salvos com sucesso!")
import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Tutor.IA - Diagnóstico de Perfil",
    page_icon="🧠",
    layout="wide"
)

# --- 1. Título e Descrição ---
st.title("🧠 Diagnóstico Interativo de Perfil de Aprendizagem")
st.markdown("""
Responda ao formulário abaixo com base nos seus hábitos de estudo para descobrir o seu perfil de aprendizagem predominante.
""")

# --- 2. Carregamento do Modelo e do Scaler ---
MODEL_PATH = os.path.join('src', 'models', 'lp_inference_model.joblib')
SCALER_PATH = os.path.join('src', 'models', 'scaler.joblib')
LEARNING_PROFILE_MAP = {0: 'Auditivo', 1: 'Cinestésico', 2: 'Leitura/Escrita', 3: 'Visual'}

@st.cache_resource
def load_artifacts():
    """Carrega o modelo e o scaler, com cache para performance."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("ERRO: Arquivo do modelo ou do scaler não encontrado. Por favor, execute 'src/save_model.py' primeiro.")
else:
    # --- 3. Definição do Formulário ---
    st.subheader("Formulário de Hábitos de Estudo")

    with st.form("profile_form"):
        st.write("Responda com valores aproximados sobre sua rotina de estudos semanal.")
        
        # Usar colunas para organizar o layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Tempo Gasto em Recursos (horas/semana)")
            time_spent_on_video = st.number_input("Em vídeos (aulas, tutoriais)", min_value=0.0, value=5.0, step=0.5)
            time_spent_on_audio = st.number_input("Em áudios (podcasts, audiobooks)", min_value=0.0, value=1.0, step=0.5)
            time_spent_reading = st.number_input("Lendo (livros, artigos)", min_value=0.0, value=3.0, step=0.5)
            time_spent_writing = st.number_input("Escrevendo (resumos, anotações)", min_value=0.0, value=2.0, step=0.5)
            time_spent_on_quizz = st.number_input("Em quizzes e testes", min_value=0.0, value=2.0, step=0.5)
            time_spent_on_flashcards = st.number_input("Usando flashcards", min_value=0.0, value=1.0, step=0.5)

        with col2:
            st.markdown("##### Conclusão de Atividades (unidades/semana)")
            completed_exercices = st.number_input("Exercícios práticos concluídos", min_value=0, value=10)
            completed_quizzes = st.number_input("Quizzes concluídos", min_value=0, value=5)
            completed_flashcards = st.number_input("Conjuntos de flashcards concluídos", min_value=0, value=3)
            
            st.markdown("##### Desempenho (0-100)")
            text_quizzes_accuracy = st.slider("Nota média em provas de texto", 0, 100, 75)
            visual_quizzes_accuracy = st.slider("Nota média em provas com imagens/gráficos", 0, 100, 85)

            st.markdown("##### Preferência Principal")
            most_preferred_resource = st.selectbox(
                "Qual tipo de recurso você mais prefere?",
                ('video', 'audio', 'text', 'practical')
            )

        # Botão de envio do formulário
        submitted = st.form_submit_button("Descobrir meu Perfil de Aprendizagem")

        if submitted:
            with st.spinner('Analisando seu perfil...'):
                # --- 4. Pré-processamento dos Dados do Formulário ---
                
                # a. Criar DataFrame com dados crus
                raw_data = {
                    'time_spent_on_video': [time_spent_on_video], 'time_spent_on_audio': [time_spent_on_audio],
                    'time_spent_reading': [time_spent_reading], 'time_spent_writing': [time_spent_writing],
                    'time_spent_on_quizz': [time_spent_on_quizz], 'time_spent_on_flashcards': [time_spent_on_flashcards],
                    'completed_exercices': [completed_exercices], 'completed_quizzes': [completed_quizzes],
                    'completed_flashcards': [completed_flashcards], 'text_quizzes_accuracy': [text_quizzes_accuracy],
                    'visual_quizzes_accuracy': [visual_quizzes_accuracy]
                }
                input_df_numeric = pd.DataFrame(raw_data)

                # b. Aplicar o StandardScaler
                scaled_numeric_data = scaler.transform(input_df_numeric)
                input_df_scaled = pd.DataFrame(scaled_numeric_data, columns=input_df_numeric.columns)

                # c. Aplicar o One-Hot Encoding para a preferência
                input_df_scaled['most_preferred_resource_type_audio'] = 1 if most_preferred_resource == 'audio' else 0
                input_df_scaled['most_preferred_resource_type_practical'] = 1 if most_preferred_resource == 'practical' else 0
                input_df_scaled['most_preferred_resource_type_text'] = 1 if most_preferred_resource == 'text' else 0
                input_df_scaled['most_preferred_resource_type_video'] = 1 if most_preferred_resource == 'video' else 0

                # d. Garantir a ordem correta das colunas
                final_columns_order = [
                    'time_spent_on_video', 'time_spent_on_audio', 'time_spent_reading', 'time_spent_writing',
                    'time_spent_on_quizz', 'time_spent_on_flashcards', 'completed_exercices', 'completed_quizzes',
                    'completed_flashcards', 'text_quizzes_accuracy', 'visual_quizzes_accuracy',
                    'most_preferred_resource_type_audio', 'most_preferred_resource_type_practical',
                    'most_preferred_resource_type_text', 'most_preferred_resource_type_video'
                ]
                final_df = input_df_scaled[final_columns_order]

                # --- 5. Realizar a Previsão ---
                prediction_numeric = model.predict(final_df)
                predicted_profile_name = LEARNING_PROFILE_MAP.get(prediction_numeric[0], "Desconhecido")

                # --- 6. Exibir Resultado ---
                st.success(f"**Diagnóstico Concluído!**")
                st.markdown(f"### Com base nas suas respostas, seu perfil de aprendizagem predominante é: **{predicted_profile_name}**")
                st.info(f"Isso sugere que você pode aprender de forma mais eficaz através de recursos e atividades associadas ao estilo **{predicted_profile_name}**.", icon="💡")
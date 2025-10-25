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

# --- 2. Carregamento dos Artefatos (Pipeline e LabelEncoder) ---
MODEL_PATH = os.path.join('src', 'models', 'lp_model.joblib')
LE_PATH = os.path.join('src', 'models', 'label_encoder.joblib')

@st.cache_resource
def load_artifacts():
    """Carrega o pipeline completo e o LabelEncoder, com cache para performance."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        return pipeline, le
    except FileNotFoundError:
        return None, None

pipeline, le = load_artifacts()

if pipeline is None or le is None:
    st.error("ERRO: Arquivos de modelo não encontrados. Por favor, execute 'src/save_model.py' primeiro.")
else:
    # --- 3. Definição do Formulário ---
    st.subheader("Formulário de Hábitos de Estudo")

    with st.form("profile_form"):
        st.write("Responda com valores aproximados sobre sua rotina de estudos semanal.")
        
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
            most_preferred_resource_type = st.selectbox(
                "Qual tipo de recurso você mais prefere?",
                ('video', 'audio', 'text', 'practical')
            )

        submitted = st.form_submit_button("Descobrir meu Perfil de Aprendizagem")

        if submitted:
            with st.spinner('Analisando seu perfil...'):
                # --- 4. Preparação dos Dados de Entrada ---
                # Criamos um DataFrame com os dados brutos do formulário.
                # A ordem das colunas não importa mais, pois o ColumnTransformer do pipeline usa os nomes.
                input_data = {
                    'time_spent_on_video': [time_spent_on_video],
                    'time_spent_on_audio': [time_spent_on_audio],
                    'time_spent_reading': [time_spent_reading],
                    'time_spent_writing': [time_spent_writing],
                    'time_spent_on_quizz': [time_spent_on_quizz],
                    'time_spent_on_flashcards': [time_spent_on_flashcards],
                    'completed_exercices': [completed_exercices],
                    'completed_quizzes': [completed_quizzes],
                    'completed_flashcards': [completed_flashcards],
                    'text_quizzes_accuracy': [text_quizzes_accuracy],
                    'visual_quizzes_accuracy': [visual_quizzes_accuracy],
                    'most_preferred_resource_type': [most_preferred_resource_type]
                }
                input_df = pd.DataFrame(input_data)

                # --- 5. Realizar a Previsão ---
                # O pipeline cuida de todo o pré-processamento e retorna a previsão numérica.
                prediction_numeric = pipeline.predict(input_df)
                
                # Usamos o LabelEncoder carregado para obter o nome do perfil.
                predicted_profile_name = le.inverse_transform(prediction_numeric)

                # --- 6. Exibir Resultado ---
                st.success(f"**Diagnóstico Concluído!**")
                st.markdown(f"### Com base nas suas respostas, seu perfil de aprendizagem predominante é: **{predicted_profile_name[0]}**")
                st.info(f"Isso sugere que você pode aprender de forma mais eficaz através de recursos e atividades associadas ao estilo **{predicted_profile_name[0]}**.", icon="💡")

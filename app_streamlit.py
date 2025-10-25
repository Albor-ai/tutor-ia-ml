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

# --- 2. Carregamento dos Artefatos ---
MODEL_PATH = os.path.join('src', 'models', 'lp_model.joblib')
LE_PATH = os.path.join('src', 'models', 'label_encoder.joblib')

@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        # Extrai as classes do LabelEncoder para usar no mapeamento
        profile_names = le.classes_.tolist()
        return pipeline, le, profile_names
    except FileNotFoundError:
        return None, None, None

pipeline, le, profile_names = load_artifacts()

# --- 3. Conteúdo das Descrições e Dicas ---
# Usando os nomes dos perfis carregados do LabelEncoder para garantir consistência
if profile_names:
    PROFILE_INFO = {
        profile_names[3]: { # Visual
            "icon": "👁️",
            "description": "Alunos com perfil visual absorvem melhor a informação através de estímulos visuais. Gráficos, diagramas, vídeos e anotações coloridas são suas ferramentas mais fortes.",
            "tips": [
                "Crie mapas mentais para conectar conceitos.",
                "Use cores e símbolos diferentes em suas anotações.",
                "Prefira assistir a documentários ou videoaulas.",
                "Transforme informações de texto em tabelas ou fluxogramas."
            ]
        },
        profile_names[0]: { # Auditivo
            "icon": "🎧",
            "description": "Alunos com perfil auditivo aprendem melhor ouvindo. Debates, audiobooks, podcasts e explicar a matéria em voz alta são métodos muito eficazes para eles.",
            "tips": [
                "Grave suas aulas e ouça novamente mais tarde.",
                "Participe de grupos de estudo para debater os temas.",
                "Crie rimas ou músicas para memorizar fórmulas e conceitos.",
                "Leia seus resumos em voz alta."
            ]
        },
        profile_names[1]: { # Cinestésico
            "icon": "✋",
            "description": "Alunos com perfil cinestésico (ou prático) precisam \'colocar a mão na massa\'. Eles aprendem fazendo, através de experimentos, projetos práticos e aplicação direta do conhecimento.",
            "tips": [
                "Busque por laboratórios práticos ou simuladores online.",
                "Construa modelos físicos ou digitais do que está aprendendo.",
                "Faça pausas curtas e movimente-se durante os estudos.",
                "Aplique o conhecimento em projetos pessoais."
            ]
        },
        profile_names[2]: { # Leitura/Escrita
            "icon": "✍️",
            "description": "Alunos com perfil de leitura e escrita preferem interagir com o texto. Ler, escrever, fazer anotações detalhadas e criar listas são suas principais formas de aprendizado.",
            "tips": [
                "Reescreva os conceitos com suas próprias palavras.",
                "Crie resumos bem estruturados e listas de tópicos.",
                "Leia livros e artigos complementares sobre o assunto.",
                "Use flashcards com perguntas e respostas escritas."
            ]
        }
    }
else:
    PROFILE_INFO = {}


# --- 4. Barra Lateral (Sidebar) e Seleção de Modelo ---

st.sidebar.title("🤖 Painel de Modelos")
model_selection = st.sidebar.selectbox(
    "Escolha a ferramenta que deseja usar:",
    ("Diagnóstico de Perfil de Aprendizagem", "Recomendação de Conteúdo (em desenvolvimento)")
)

# --- LÓGICA PARA O MODELO DE DIAGNÓSTICO DE PERFIL ---
if model_selection == "Diagnóstico de Perfil de Aprendizagem":
    # Título e Descrição ---
    st.title("🧠 Diagnóstico Interativo de Perfil de Aprendizagem")
    
    st.sidebar.header("📝 Formulário de Diagnóstico")

    if pipeline is None or le is None:
        st.sidebar.error("ERRO: Arquivos do modelo de diagnóstico não encontrados. Por favor, execute 'src/save_model.py' primeiro.")
    else:
        with st.sidebar.form("profile_form"):
            st.write("Responda com valores aproximados sobre sua rotina de estudos semanal.")
            
            with st.expander("Tempo Gasto em Recursos (horas/semana)", expanded=True):
                time_spent_on_video = st.number_input("Em vídeos (aulas, tutoriais)", min_value=0.0, max_value=168.0, value=0.0, step=0.5)
                time_spent_on_audio = st.number_input("Em áudios (podcasts, audiobooks)", min_value=0.0, max_value=168.0, value=0.0, step=0.5)
                time_spent_reading = st.number_input("Lendo (livros, artigos)", min_value=0.0, max_value=168.0, value=0.0, step=0.5)
                time_spent_writing = st.number_input("Escrevendo (resumos, anotações)", min_value=0.0, max_value=168.0, value=0.0, step=0.5)
                time_spent_on_quizz = st.number_input("Em quizzes e testes", min_value=0.0, max_value=168.0, value=0.0, step=0.5)
                time_spent_on_flashcards = st.number_input("Usando flashcards", min_value=0.0, max_value=168.0, value=0.0, step=0.5)

            with st.expander("Atividades e Desempenho", expanded=True):
                completed_exercices = st.number_input("Exercícios práticos concluídos", min_value=0, value=0)
                completed_quizzes = st.number_input("Quizzes concluídos", min_value=0, value=0)
                completed_flashcards = st.number_input("Conjuntos de flashcards concluídos", min_value=0, value=0)
                st.markdown("---_Desempenho (%)_---")
                text_quizzes_accuracy = st.slider("Percentual de acertos em avaliações de texto (%)", 0, 100, 0)
                visual_quizzes_accuracy = st.slider("Percentual de acertos em avaliações com imagens/gráficos (%)", 0, 100, 0)

            with st.expander("Preferências", expanded=True):
                most_preferred_resource_type = st.selectbox(
                    "Qual tipo de recurso você mais prefere?",
                    ('video', 'audio', 'text', 'practical')
                )

            submitted = st.form_submit_button("✨ Descobrir meu Perfil!")

        with st.sidebar.expander("ℹ️ Sobre o Diagnóstico"):
            st.info("""
            Este diagnóstico é gerado por um modelo de Machine Learning (XGBoost) treinado com dados simulados de hábitos de estudo.
            
            O resultado é uma **sugestão** para te ajudar a encontrar os melhores métodos de estudo, e não um rótulo definitivo.
            """)

        # --- Área Principal (Resultados ou Boas-Vindas) ---
        if submitted:
            # --- TELA DE RESULTADOS ---
            with st.spinner('Analisando seu perfil...'):
                input_data = {
                    'time_spent_on_video': [time_spent_on_video], 'time_spent_on_audio': [time_spent_on_audio],
                    'time_spent_reading': [time_spent_reading], 'time_spent_writing': [time_spent_writing],
                    'time_spent_on_quizz': [time_spent_on_quizz], 'time_spent_on_flashcards': [time_spent_on_flashcards],
                    'completed_exercices': [completed_exercices], 'completed_quizzes': [completed_quizzes],
                    'completed_flashcards': [completed_flashcards], 'text_quizzes_accuracy': [text_quizzes_accuracy],
                    'visual_quizzes_accuracy': [visual_quizzes_accuracy], 'most_preferred_resource_type': [most_preferred_resource_type]
                }
                input_df = pd.DataFrame(input_data)
                prediction_numeric = pipeline.predict(input_df)
                predicted_profile_name = le.inverse_transform(prediction_numeric)[0]
                
                st.balloons()
                st.success(f"**Diagnóstico Concluído!**")
                profile = PROFILE_INFO.get(predicted_profile_name)
                if profile:
                    st.title(f"{profile['icon']} {predicted_profile_name}")
                    st.markdown("---")
                    st.metric(label="Seu Perfil Predominante", value=predicted_profile_name)
                    st.markdown(f"**_{profile['description']}_**")
                    st.subheader("💡 Dicas de Estudo para Você")
                    for tip in profile['tips']:
                        st.markdown(f"- {tip}")
                else:
                    st.error("Não foi possível encontrar informações para o perfil previsto.")
        else:
            # --- TELA DE BOAS-VINDAS ---
            st.markdown("""
            Bem-vindo ao Diagnóstico Interativo de Perfil de Aprendizagem do Tutor.IA! 

            Preencha o formulário na barra lateral à esquerda para descobrir seu perfil predominante e receber dicas personalizadas.
            """)
            st.subheader("Como funciona?")
            st.markdown("""
            Esta ferramenta utiliza um modelo de **Machine Learning** para analisar seus hábitos de estudo e identificar seu perfil de aprendizagem predominante. 
            
            O modelo, chamado **XGBoost**, foi treinado para reconhecer padrões em informações como:
            - O tempo que você dedica a diferentes tipos de materiais (vídeos, textos, etc.).
            - Suas preferências de recursos.
            - Seu desempenho em diferentes tipos de avaliação.

            O resultado não é um rótulo fixo, mas uma **sugestão poderosa** para te ajudar a entender seus pontos fortes e a escolher as melhores técnicas de estudo para você.
            """)
            st.info("👈 **Preencha o formulário na barra lateral para começar!**")


# --- LÓGICA PARA O MODELO DE RECOMENDAÇÃO DE CONTEÚDO ---
elif model_selection == "Recomendação de Conteúdo (em desenvolvimento)":
    st.title("📚 Recomendação de Conteúdo")
    st.sidebar.warning("Modelo em desenvolvimento!.")

    # --- TELA DE BOAS-VINDAS (PLACEHOLDER) ---
    st.subheader("O que esperar deste modelo?")
    st.markdown("""
        O objetivo deste futuro modelo é sugerir materiais de estudos mais aderentes ao seu perfil de aprendizagem.
        
        **Como vai funcionar?**
        1.  Você informará:
            - seu objetivo de estudo (ex: "aprender sobre Redes Neurais");
            - seu perfil de aprendizagem (ex: "Visual") obtido pelo modelo de diagnóstico de perfil de aprendizagem;
            - seu nível atual de conhecimento do assunto (ex: "Iniciante", "Intermediário", "Avançado").
            - sua preferência de formato de conteúdo (ex: "vídeos", "artigos", "cursos interativos").
        2.  Com base nessas informações, ele irá buscar e recomendar os melhores recursos (vídeos, artigos, cursos, exercícios práticos) alinhados ao seu estilo de aprender.
    """)
    st.info("Em breve o modelo estará disponível para uso.")

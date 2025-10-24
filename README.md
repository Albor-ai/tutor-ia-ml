# Tutor.IA - Repositório de Modelos de Machine Learning

Este repositório contém os projetos de Machine Learning para o sistema Tutor.IA. O desenvolvimento é guiado por um duplo propósito:

*   **Aprendizado Prático:** Servir como um ambiente para aplicar e aprofundar conhecimentos em conceitos e ferramentas de Machine Learning, do protótipo ao deploy.
*   **Solução Técnica:** Desenvolver modelos de ML funcionais e implantá-los como serviços, simulando um ciclo de vida de produto de dados real.

---

## Modelos neste Projeto

### 1. Inferência de Perfil de Aprendizagem (LP)

*   **Status:** ✅ Concluído
*   **Objetivo:** Classificar usuários em um de quatro perfis de aprendizagem (Visual, Auditivo, Leitura/Escrita, Cinestésico) com base em suas interações na plataforma.
*   **Plano de Desenvolvimento:** [`docs/LP_PLAN.md`](docs/LP_PLAN.md)

### 2. Recomendação de Conteúdo (CR)

*   **Status:** ⏳ Em Desenvolvimento
*   **Objetivo:** Recomendar conteúdos (vídeos, textos, quizzes) de forma personalizada, utilizando o perfil de aprendizagem do usuário e seu histórico de interações.
*   **Plano de Desenvolvimento:** [`CR_PLAN.md`](CR_PLAN.md)

---

## Protótipos Disponíveis e Como Executar

### Protótipo do Modelo de Perfil de Aprendizagem (LP)

Este protótipo consiste em uma API REST que serve o modelo treinado e um dashboard interativo para diagnóstico.

#### 1. Pré-requisitos
- Python 3.8+
- `uv` instalado (`pip install uv`)

#### 2. Instalação de Dependências
Na raiz do projeto, execute:
```bash
uv sync
```

#### 3. Preparação dos Artefatos de ML
Para treinar o modelo LP e o scaler, execute o script:
```bash
python src/save_model.py
```
Este comando criará os arquivos `lp_inference_model.joblib` e `scaler.joblib` no diretório `src/models/`.

#### 4. Executando as Aplicações

**Opção A: API REST com FastAPI**

Para iniciar o servidor da API:
```bash
uvicorn main:app --reload
```
- A API estará disponível em `http://127.0.0.1:8000`.
- A **documentação interativa (Swagger UI)** para testes estará em `http://127.0.0.1:8000/docs`.

**Opção B: Dashboard Interativo com Streamlit**

Para iniciar a aplicação web interativa:
```bash
streamlit run app_streamlit.py
```
- A aplicação abrirá automaticamente no seu navegador, geralmente em `http://localhost:8501`.

---

## Stack de Tecnologias

*   **Análise e Manipulação de Dados:** Pandas, NumPy
*   **Visualização de Dados:** Matplotlib
*   **Modelagem de Machine Learning:** Scikit-learn
*   **Serviço de API:** FastAPI, Uvicorn
*   **Dashboard Interativo:** Streamlit
*   **Gerenciamento de Dependências:** uv
*   **Persistência de Modelos:** Joblib

---

## Visão Futura e MLOps

Para uma análise detalhada sobre os próximos passos para levar os protótipos para um ambiente de produção com dados reais, consulte o documento: [Visão Futura e Radar de MLOps](docs/VISAO_FUTURA.md).

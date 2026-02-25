# Previsão de Canal Aduaneiro

Este repositório contém a solução técnica para o desafio de Cientista de Dados da Logcomex. O desafio é desenvolver um modelo de Machine Learning capaz de prever o canal aduaneiro que será atribuído a uma DI (Declaração de Importação).

## Estrutura do Projeto

    data/ -  Local onde o dataset sample_data.parquet deve ser inserido.

    models/ - Contém os artefatos salvos (.pkl) gerados pelo pipeline (Modelos treinados, LabelEncoder, Lookup Tables de agregação histórica).

    eval_plots/ - Gráficos gerados pelo script de avaliação (Matrizes de Confusão, Gini Importance, Permutation Importance).

    EDA.ipynb - Jupyter Notebook contendo toda a Análise Exploratória de Dados (EDA) e análises estatísticas.

    EDA_figs/ - Gráficos gerados durante a EDA.

    main_pipeline.py - Script Python modular e focado em produção. Executa todo o fluxo: Engenharia de Features, Split Temporal, Features de Agregação Histórica (lag features), Treinamento (Oversampling via imblearn) e salvamento de artefatos.

    evaluation_metrics.py - Script de avaliação e explicabilidade. Gera Matrizes de Confusão comparativas, Gini Importance e Permutation Importance a partir dos artefatos salvos.

    test_pipeline.py - Testes unitários com dados sintéticos para validar as funções críticas do pipeline (feature_engineering, temporal_split, build_lag_features, build_preprocessor).

    model_config.json - Arquivo de configuração contendo os hiperparâmetros dos modelos finais.

    documentacao_tecnica.md - Respostas às questões técnicas, avaliação de modelos e estratégia de monitoramento em produção.

    relatorio_executivo.md - Relatório executivo com os principais insights.

## Execute o projeto:

### 0. Clone o repositório e acesse a pasta:

    git clone https://github.com/alvarofrancomartins/previsao-canal-aduaneiro.git
    cd previsao-canal-aduaneiro

### 1. Inserindo o dataset

Crie a pasta data

     mkdir data

e insira o dataset sample_data.parquet 

### 2. Crie um ambiente virtual e instale as dependências:

Requer Python >= 3.11

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

### 3. Execute o Pipeline Principal (Treinamento e Avaliação):

    python main_pipeline.py

Isso irá treinar os 3 modelos (Random Forest, HistGradientBoosting, XGBoost) e salvar os artefatos .pkl na pasta models/.

### 4. Gere os Gráficos de Avaliação:

    python evaluation_metrics.py

Isso irá gerar as Matrizes de Confusão, Gini Importance e Permutation Importance na pasta eval_plots/.

### 5. Rode os Testes Unitários:

    pytest test_pipeline.py -v  

### 🛠️ Stack Tecnológica

    Python
    Pandas, Numpy, Scikit-Learn
    Imbalanced-learn (imblearn)
    XGBoost
    Matplotlib, Seaborn
    Pytest
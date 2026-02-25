import os
import json
import joblib
import logging
import warnings

import numpy  as np
import pandas as pd

from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.preprocessing  import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble       import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy          import DummyClassifier
from sklearn.metrics        import classification_report
from imblearn.pipeline      import Pipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost                import XGBClassifier

warnings.filterwarnings('ignore')

# Configuração de Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_PATH = 'data/sample_data.parquet'

def load_data(file_path):
    """Carrega o dataset real no formato parquet."""
    return pd.read_parquet(file_path)

def feature_engineering(df):
    """Aplica engenharia de atributos e limpeza de dados."""
    df_feat = df.copy()
    
    # Tratamento de datas nulas
    df_feat['registry_date'] = df_feat['registry_date'].fillna(df_feat['yearmonth'])
    df_feat['registry_date'] = pd.to_datetime(df_feat['registry_date'])
    
    # Extração de componentes temporais (Sazonalidade)
    df_feat['ano']        = df_feat['registry_date'].dt.year
    df_feat['mes']        = df_feat['registry_date'].dt.month
    df_feat['dia_semana'] = df_feat['registry_date'].dt.dayofweek 
    df_feat['trimestre']  = df_feat['registry_date'].dt.quarter
    
    # Padronização de códigos de país
    df_feat['country_origin_code'] = df_feat['country_origin_code'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # NaN Strategy (sem imputação global para evitar leakage pré-split):
    #   - ncm_code NaN         → 'DESCONHECIDO' (fallback explícito); ncm_grau_elaboracao recebe -1
    #   - shipper_name NaN     → shipper_tipo recebe 'OUTROS' (via fillna no regex)
    #   - clearance_place* NaN → _tipo recebe 'OUTROS' (via fillna no regex)
    #   - Demais categóricas   → SimpleImputer(most_frequent) no sklearn pipeline, fitted APENAS no treino (dentro do ColumnTransformer)
    
    # Capítulo NCM (Grau de Elaboração) — NaN vira -1 (categoria "desconhecido")
    df_feat['ncm_grau_elaboracao'] = pd.to_numeric(df_feat['ncm_code'].astype(str).str[:2], errors='coerce').fillna(-1)
    
    # NCM Code completo (28 valores únicos — cardinalidade baixa, usar como categórica)
    # NaN recebe fallback explícito para não vazar informação via SimpleImputer
    df_feat['ncm_code'] = df_feat['ncm_code'].fillna('DESCONHECIDO').astype(str)
    
    # Segmentação de empresas por palavras-chave
    keywords = ['INTERNATIONAL', 'TRADING', 'IMPORTADORA', 'INDUSTRIAL', 'COMERCIAL', 'LOGÍSTICA', 'DISTRIBUIDORA']
    pattern  = '(?i)(' + '|'.join(keywords) + ')'
    df_feat['consignee_tipo'] = df_feat['consignee_name'].str.extract(pattern, expand=False).str.upper().fillna('OUTROS')
    df_feat['shipper_tipo']   = df_feat['shipper_name'].str.extract(pattern, expand=False).str.upper().fillna('OUTROS')
    
    # Extração de Tipo de Recinto (Porto vs Aeroporto)
    pattern_recinto = r'(?i)(PORTO|PORTUÁRI[OA]|AEROPORTO)'
    for col in ['clearance_place_entry', 'clearance_place_dispatch', 'clearance_place']:
        df_feat[f'{col}_tipo'] = (
            df_feat[col].str.extract(pattern_recinto, expand=False)
            .str.upper()
            .replace({'PORTUÁRIO': 'PORTO', 'PORTUÁRIA': 'PORTO'})
            .fillna('OUTROS')
        )

    # Remoção de identificadores únicos para evitar Data Leakage
    cols_to_drop = ['document_number', 'yearmonth', 'consignee_name', 'shipper_name', 'consignee_code']
    return df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns])

def temporal_split(df):
    """Realiza o split temporal baseado no cutoff de Novembro/2024."""
    CUTOFF_TEST = pd.Timestamp('2024-11-01')
    
    df_train = df[df['registry_date'] < CUTOFF_TEST]
    df_test  = df[df['registry_date'] >= CUTOFF_TEST]

    logging.info(f"Treino: {df_train['registry_date'].min().date()} → {df_train['registry_date'].max().date()} ({df_train['registry_date'].dt.to_period('M').nunique()} meses)")
    logging.info(f"Teste:  {df_test['registry_date'].min().date()} → {df_test['registry_date'].max().date()} ({df_test['registry_date'].dt.to_period('M').nunique()} meses)")
    
    X_train = df_train.drop(columns=['channel', 'registry_date'])
    y_train = df_train['channel']
    X_test  = df_test.drop(columns=['channel', 'registry_date'])
    y_test  = df_test['channel']
    
    return X_train, X_test, y_train, y_test

def build_lag_features(X_train, y_train_str, X_test):
    """
    Constrói features de agregação histórica usando APENAS dados de treino.
    
    Estratégia anti-leakage:
      - Todas as estatísticas são calculadas exclusivamente no conjunto de treino.
      - Os lookup tables resultantes são aplicados (map) em treino e teste.
      - Valores não vistos no treino recebem fallback = taxa global de risco.
      - Os lookup tables são retornados para persistência (uso em produção).
    
    Features criadas:
      - taxa_risco_pais:               % de DIs com canal Vermelho, Amarelo ou Cinza por país de origem.
      - taxa_risco_ncm:                % de DIs com canal Vermelho, Amarelo ou Cinza por capítulo NCM.
      - taxa_risco_ncm_code:           % de DIs com canal Vermelho, Amarelo ou Cinza por NCM completo.
      - volume_dis_pais:               Quantidade histórica de DIs por país (proxy de familiaridade).
      - taxa_risco_clearance_entry:    % de risco por local de entrada.
      - taxa_risco_clearance_dispatch: % de risco por local de despacho.
      - taxa_risco_clearance_place:    % de risco por local de desembaraço.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    # Flag binária: 1 se canal de risco (Vermelho, Amarelo ou Cinza), 0 caso contrário
    is_risk = y_train_str.isin(['VERMELHO', 'AMARELO', 'CINZA']).astype(int)
    fallback_rate = is_risk.mean()

    # Helper: cria taxa de risco para qualquer coluna categórica
    def _build_risk_rate(col_name):
        risk = (
            pd.DataFrame({col_name: X_train[col_name], 'risk': is_risk})
            .groupby(col_name)['risk']
            .mean()
        )
        return risk

    # Taxa de risco por país de origem 
    risk_by_country = _build_risk_rate('country_origin_code')
    X_train['taxa_risco_pais'] = X_train['country_origin_code'].map(risk_by_country).fillna(fallback_rate)
    X_test['taxa_risco_pais']  = X_test['country_origin_code'].map(risk_by_country).fillna(fallback_rate)

    # Taxa de risco por capítulo de NCM 
    risk_by_ncm = _build_risk_rate('ncm_grau_elaboracao')
    X_train['taxa_risco_ncm'] = X_train['ncm_grau_elaboracao'].map(risk_by_ncm).fillna(fallback_rate)
    X_test['taxa_risco_ncm']  = X_test['ncm_grau_elaboracao'].map(risk_by_ncm).fillna(fallback_rate)

    # Taxa de risco por NCM completo (28 valores — granularidade fina)
    risk_by_ncm_code = _build_risk_rate('ncm_code')
    X_train['taxa_risco_ncm_code'] = X_train['ncm_code'].map(risk_by_ncm_code).fillna(fallback_rate)
    X_test['taxa_risco_ncm_code']  = X_test['ncm_code'].map(risk_by_ncm_code).fillna(fallback_rate)

    # Volume histórico de DIs por país (familiaridade/exposição)
    volume_by_country = X_train['country_origin_code'].value_counts()
    X_train['volume_dis_pais'] = X_train['country_origin_code'].map(volume_by_country).fillna(0)
    X_test['volume_dis_pais']  = X_test['country_origin_code'].map(volume_by_country).fillna(0)

    # Taxa de risco por local de entrada (clearance_place_entry)
    risk_by_entry = _build_risk_rate('clearance_place_entry')
    X_train['taxa_risco_clearance_entry'] = X_train['clearance_place_entry'].map(risk_by_entry).fillna(fallback_rate)
    X_test['taxa_risco_clearance_entry']  = X_test['clearance_place_entry'].map(risk_by_entry).fillna(fallback_rate)

    # Taxa de risco por local de despacho (clearance_place_dispatch)
    risk_by_dispatch = _build_risk_rate('clearance_place_dispatch')
    X_train['taxa_risco_clearance_dispatch'] = X_train['clearance_place_dispatch'].map(risk_by_dispatch).fillna(fallback_rate)
    X_test['taxa_risco_clearance_dispatch']  = X_test['clearance_place_dispatch'].map(risk_by_dispatch).fillna(fallback_rate)

    # Taxa de risco por local de desembaraço (clearance_place)
    risk_by_place = _build_risk_rate('clearance_place')
    X_train['taxa_risco_clearance_place'] = X_train['clearance_place'].map(risk_by_place).fillna(fallback_rate)
    X_test['taxa_risco_clearance_place']  = X_test['clearance_place'].map(risk_by_place).fillna(fallback_rate)

    # Lookup tables para persistência (inferência em produção)
    lookups = {
        'risk_by_country':   risk_by_country,
        'risk_by_ncm':       risk_by_ncm,
        'risk_by_ncm_code':  risk_by_ncm_code,
        'volume_by_country': volume_by_country,
        'risk_by_entry':     risk_by_entry,
        'risk_by_dispatch':  risk_by_dispatch,
        'risk_by_place':     risk_by_place,
        'fallback_rate':     fallback_rate,
    }

    logging.info(
        f"Lag features criadas — taxa_risco global (fallback): {fallback_rate:.4f}, "
        f"países únicos: {len(risk_by_country)}, NCMs únicos: {len(risk_by_ncm_code)}, "
        f"capítulos NCM: {len(risk_by_ncm)}, "
        f"recintos entrada: {len(risk_by_entry)}, despacho: {len(risk_by_dispatch)}, desembaraço: {len(risk_by_place)}"
    )

    return X_train, X_test, lookups


def build_preprocessor():
    """Constrói o ColumnTransformer para variáveis numéricas e categóricas."""
    num_features = [
        'ano', 'mes', 'dia_semana', 'trimestre', 'ncm_grau_elaboracao',
        # Features de agregação histórica (lag features)
        'taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code', 'volume_dis_pais',
        'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place',
    ]
    cat_features = [
        'ncm_code',  # 28 valores únicos 
        'country_origin_code', 'consignee_company_size', 'transport_mode_pt',
        'consignee_tipo', 'shipper_tipo',
        # Recintos: nome completo (cardinalidade moderada, carrega info de jurisdição)
        'clearance_place_entry', 'clearance_place_dispatch', 'clearance_place',
        # Recintos: tipo derivado (Porto vs Aeroporto)
        'clearance_place_entry_tipo', 'clearance_place_dispatch_tipo', 'clearance_place_tipo',
    ]

    return ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imp', SimpleImputer(strategy='median')),
            ('std', StandardScaler()),
        ]), num_features),
        ('cat', Pipeline(steps=[
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(min_frequency=0.01, handle_unknown='ignore', sparse_output=False)),
        ]), cat_features),
    ])

if __name__ == "__main__":
    logging.info("Iniciando Pipeline de Produção...")
    
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    df_raw = load_data(DATA_PATH)
    df_features = feature_engineering(df_raw)
    X_train, X_test, y_train_str, y_test_str = temporal_split(df_features)

    logging.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    logging.info(f"Distribuição treino: {y_train_str.value_counts(normalize=True).round(4).to_dict()}")
    logging.info(f"Distribuição teste:  {y_test_str.value_counts(normalize=True).round(4).to_dict()}")
    
    # FEATURES DE AGREGAÇÃO HISTÓRICA
    # Calculadas APÓS o split para garantir zero leakage temporal.
    # Usam apenas y_train_str (labels de texto) do conjunto de treino.
    X_train, X_test, lag_lookups = build_lag_features(X_train, y_train_str, X_test)
    
    le          = LabelEncoder()
    y_train     = le.fit_transform(y_train_str)
    y_test      = le.transform(y_test_str)
    class_names = le.classes_
    
    preprocessor = build_preprocessor()
    
    # Treinando os modelos
    # 0. Baseline (sempre prevê a classe majoritária: Verde)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    print("\n--- Avaliação Baseline (sempre Verde) ---")
    print(classification_report(y_test, baseline.predict(X_test), target_names=class_names, zero_division=0))

    # 1. Random Forest 
    rf_pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('sam', RandomOverSampler(random_state=config['oversampling']['random_state'])),
        ('clf', RandomForestClassifier(**config['random_forest']))
    ])
    
    logging.info("Treinando Modelo 1: Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    print("\n--- Avaliação Random Forest ---")
    print(classification_report(y_test, rf_pipeline.predict(X_test), target_names=class_names))

    # 2. HistGradientBoosting 
    hgb_pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('sam', RandomOverSampler(random_state=config['oversampling']['random_state'])),
        ('clf', HistGradientBoostingClassifier(**config['hist_gradient_boosting']))
    ])
    logging.info("Treinando Modelo 2: HistGradientBoosting...")
    hgb_pipeline.fit(X_train, y_train)
    print("\n--- Avaliação HistGradientBoosting ---")
    print(classification_report(y_test, hgb_pipeline.predict(X_test), target_names=class_names))

    # 3. XGBoost
    xgb_pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('sam', RandomOverSampler(random_state=config['oversampling']['random_state'])),
        ('clf', XGBClassifier(**config['xgboost']))
    ])
    logging.info("Treinando Modelo 3: XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    print("\n--- Avaliação XGBoost ---")
    print(classification_report(y_test, xgb_pipeline.predict(X_test), target_names=class_names))

    # Salvando os artefatos
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_pipeline,  'models/rf_pipeline.pkl')
    joblib.dump(hgb_pipeline, 'models/hgb_pipeline.pkl')
    joblib.dump(xgb_pipeline, 'models/xgb_pipeline.pkl')
    joblib.dump(le,           'models/label_encoder.pkl')
    joblib.dump(lag_lookups,  'models/lag_lookups.pkl')
    
    # Salva dados de teste para uso no script de avaliação
    joblib.dump((X_test, y_test, class_names), 'models/test_data.pkl')
    
    logging.info("Todos os modelos e dados de teste foram salvos com sucesso.")
    logging.info("Execute 'python evaluation_metrics.py' para gerar gráficos de explicabilidade e matrizes de confusão.")
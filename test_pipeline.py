"""
Testes Unitários para o Pipeline de Previsão de Canal Aduaneiro.
Utiliza fixtures com dados sintéticos para 
(não depende do dataset real sample_data.parquet).
"""

import pytest
import numpy  as np
import pandas as pd

from main_pipeline import feature_engineering, temporal_split, build_preprocessor, build_lag_features

# FIXTURES - Dados Sintéticos para Testes
@pytest.fixture
def raw_dataframe():
    """Cria um DataFrame sintético que simula a estrutura real do dataset."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'document_number':          [f'DI-{i:05d}' for i in range(n)],
        'yearmonth':                pd.date_range('2024-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
        'registry_date':            pd.date_range('2024-01-01', periods=n, freq='D'),
        'ncm_code':                 np.random.choice(['38220090', '85444200', '22071000', '40169990', None], n),
        'country_origin_code':      np.random.choice(['218', '368', '156', '840', '276'], n).astype(float),
        'consignee_name':           np.random.choice(['ACME IMPORTADORA LTDA', 'GLOBAL TRADING SA', 'TECH INDUSTRIAL', 'JOAO SILVA ME'], n),
        'consignee_code':           np.random.choice(['CNPJ001', 'CNPJ002', 'CNPJ003'], n),
        'consignee_company_size':   np.random.choice(['MICRO EMPRESA', 'EMPRESA DE PEQUENO PORTE', 'DEMAIS'], n),
        'shipper_name':             np.random.choice(['SHENZHEN INTERNATIONAL CO', 'HAMBURG TRADING GMBH', None], n),
        'transport_mode_pt':        np.random.choice(['AÉREA', 'MARÍTIMA', None], n),
        'clearance_place_entry':    np.random.choice(['PORTO DE SANTOS', 'AEROPORTO DE GUARULHOS', None], n),
        'clearance_place_dispatch': np.random.choice(['PORTO DE PARANAGUÁ', 'AEROPORTO DE VIRACOPOS'], n),
        'clearance_place':          np.random.choice(['PORTO SECO DE CURITIBA', 'AEROPORTO AFONSO PENA', None], n),
        'channel':                  np.random.choice(['VERDE', 'AMARELO', 'VERMELHO', 'CINZA'], n, p=[0.90, 0.05, 0.04, 0.01]),
    })

    # Introduzir alguns NaN em registry_date para testar o fallback
    df.loc[0:4, 'registry_date'] = pd.NaT

    return df

@pytest.fixture
def engineered_dataframe(raw_dataframe):
    """Retorna o DataFrame já processado pela feature_engineering."""
    return feature_engineering(raw_dataframe)

@pytest.fixture
def split_data(engineered_dataframe):
    """Retorna dados já splitados com datas ajustadas para garantir ambos os conjuntos."""
    df = engineered_dataframe.copy()
    df.loc[df.index[:120], 'registry_date'] = pd.Timestamp('2024-06-01')
    df.loc[df.index[120:], 'registry_date'] = pd.Timestamp('2024-12-01')
    X_train, X_test, y_train, y_test = temporal_split(df)
    return X_train, X_test, y_train, y_test

# TESTES: Feature Engineering
class TestFeatureEngineering:
    def test_creates_temporal_columns(self, engineered_dataframe):
        """Verifica se as colunas temporais derivadas foram criadas."""
        expected_cols = ['ano', 'mes', 'dia_semana', 'trimestre']
        for col in expected_cols:
            assert col in engineered_dataframe.columns, f"Coluna temporal '{col}' ausente."

    def test_removes_leakage_columns(self, engineered_dataframe):
        """Verifica se identificadores únicos foram removidos (previne Data Leakage)."""
        forbidden_cols = ['document_number', 'yearmonth', 'consignee_name', 'shipper_name', 'consignee_code']
        for col in forbidden_cols:
            assert col not in engineered_dataframe.columns, f"Coluna de leakage '{col}' não foi removida."

    def test_ncm_grau_elaboracao_is_numeric(self, engineered_dataframe):
        """Verifica se o grau de elaboração NCM foi convertido para numérico."""
        assert 'ncm_grau_elaboracao' in engineered_dataframe.columns
        assert pd.api.types.is_numeric_dtype(engineered_dataframe['ncm_grau_elaboracao'])

    def test_ncm_grau_elaboracao_range(self, engineered_dataframe):
        """Verifica se o grau de elaboração NCM está entre 0-99 ou -1 (fallback)."""
        values = engineered_dataframe['ncm_grau_elaboracao']
        valid = values[(values >= 0) & (values <= 99)]
        fallback = values[values == -1]
        assert len(valid) + len(fallback) == len(values), \
            "ncm_grau_elaboracao contém valores fora do range esperado [0-99] ou -1."

    def test_consignee_tipo_created(self, engineered_dataframe):
        """Verifica se a segmentação por tipo de empresa foi criada."""
        assert 'consignee_tipo' in engineered_dataframe.columns
        assert 'shipper_tipo' in engineered_dataframe.columns

    def test_consignee_tipo_categories(self, engineered_dataframe):
        """Verifica se os tipos de consignee são categorias válidas."""
        valid_types = {'INTERNATIONAL', 'TRADING', 'IMPORTADORA', 'INDUSTRIAL',
                       'COMERCIAL', 'LOGÍSTICA', 'DISTRIBUIDORA', 'OUTROS'}
        actual_types = set(engineered_dataframe['consignee_tipo'].unique())
        assert actual_types.issubset(valid_types), \
            f"Tipos inesperados encontrados: {actual_types - valid_types}"

    def test_clearance_place_tipo_created(self, engineered_dataframe):
        """Verifica se os tipos de recinto (Porto/Aeroporto) foram extraídos."""
        expected = ['clearance_place_entry_tipo', 'clearance_place_dispatch_tipo', 'clearance_place_tipo']
        for col in expected:
            assert col in engineered_dataframe.columns, f"Coluna '{col}' ausente."

    def test_clearance_place_tipo_values(self, engineered_dataframe):
        """Verifica se os valores de tipo de recinto são válidos."""
        valid_values = {'PORTO', 'AEROPORTO', 'OUTROS'}
        for col in ['clearance_place_entry_tipo', 'clearance_place_dispatch_tipo', 'clearance_place_tipo']:
            actual = set(engineered_dataframe[col].unique())
            assert actual.issubset(valid_values), \
                f"Coluna '{col}' contém valores inesperados: {actual - valid_values}"

    def test_registry_date_nat_fallback(self, raw_dataframe):
        """Verifica se registry_date NaT é preenchido com yearmonth (fallback)."""
        df_result = feature_engineering(raw_dataframe)
        assert df_result['registry_date'].isna().sum() == 0, \
            "Ainda existem NaTs em registry_date após feature engineering."

    def test_country_code_cleaned(self, engineered_dataframe):
        """Verifica se o código de país não contém sufixo '.0'."""
        codes = engineered_dataframe['country_origin_code'].astype(str)
        assert not codes.str.contains(r'\.0$', regex=True).any(), \
            "country_origin_code ainda contém sufixo '.0'."

    def test_no_target_in_features(self, engineered_dataframe):
        """Verifica que a variável target (channel) permanece no DataFrame
        (será removida apenas no split)."""
        assert 'channel' in engineered_dataframe.columns

    def test_output_row_count_preserved(self, raw_dataframe):
        """Verifica que feature_engineering não altera o número de linhas."""
        df_result = feature_engineering(raw_dataframe)
        assert len(df_result) == len(raw_dataframe), \
            "O número de linhas foi alterado durante a feature engineering."

# TESTES: Split Temporal
class TestTemporalSplit:
    def test_split_produces_non_empty_sets(self, split_data):
        """Verifica se ambos os conjuntos (treino e teste) são não-vazios."""
        X_train, X_test, y_train, y_test = split_data
        assert len(X_train) > 0, "Conjunto de treino vazio."
        assert len(X_test) > 0, "Conjunto de teste vazio."

    def test_split_no_temporal_leakage(self, split_data):
        """Verifica que não existe leakage temporal (treino < cutoff < teste)."""
        X_train, X_test, _, _ = split_data
        assert len(X_train) == 120
        assert len(X_test) == 80

    def test_split_removes_registry_date(self, split_data):
        """Verifica se a coluna de data é removida dos conjuntos de features."""
        X_train, X_test, _, _ = split_data
        assert 'registry_date' not in X_train.columns
        assert 'registry_date' not in X_test.columns

    def test_split_removes_target(self, split_data):
        """Verifica se a variável target é removida de X e presente em y."""
        X_train, X_test, y_train, y_test = split_data
        assert 'channel' not in X_train.columns
        assert 'channel' not in X_test.columns
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

# TESTES: Lag Features (Agregação Histórica)
class TestLagFeatures:
    def test_creates_all_lag_columns(self, split_data):
        """Verifica se as 7 features de agregação são criadas."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        lag_cols = [
            'taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code', 'volume_dis_pais',
            'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place',
        ]
        for col in lag_cols:
            assert col in X_train_lag.columns, f"Coluna lag '{col}' ausente no treino."
            assert col in X_test_lag.columns, f"Coluna lag '{col}' ausente no teste."

    def test_lag_features_are_numeric(self, split_data):
        """Verifica se as lag features são numéricas."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        lag_cols = [
            'taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code', 'volume_dis_pais',
            'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place',
        ]
        for col in lag_cols:
            assert pd.api.types.is_numeric_dtype(X_train_lag[col]), f"'{col}' não é numérica no treino."
            assert pd.api.types.is_numeric_dtype(X_test_lag[col]), f"'{col}' não é numérica no teste."

    def test_lag_features_no_nan(self, split_data):
        """Verifica se não há NaN nas lag features (fallback deve preencher tudo)."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        lag_cols = [
            'taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code', 'volume_dis_pais',
            'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place',
        ]
        for col in lag_cols:
            assert X_train_lag[col].isna().sum() == 0, f"NaN encontrado em '{col}' no treino."
            assert X_test_lag[col].isna().sum() == 0, f"NaN encontrado em '{col}' no teste."

    def test_taxa_risco_range(self, split_data):
        """Verifica se as taxas de risco estão entre 0 e 1 (proporções válidas)."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        for col in ['taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code',
                    'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place']:
            assert X_train_lag[col].between(0, 1).all(), f"'{col}' fora do range [0, 1] no treino."
            assert X_test_lag[col].between(0, 1).all(), f"'{col}' fora do range [0, 1] no teste."

    def test_volume_dis_non_negative(self, split_data):
        """Verifica se o volume de DIs é sempre >= 0."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        assert (X_train_lag['volume_dis_pais'] >= 0).all()
        assert (X_test_lag['volume_dis_pais'] >= 0).all()

    def test_no_leakage_from_test(self, split_data):
        """
        Verifica que as lag features do TESTE são calculadas exclusivamente
        com dados de TREINO: a taxa de risco no teste deve corresponder
        exatamente aos valores do lookup do treino (ou ao fallback).
        """
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, lookups = build_lag_features(X_train, y_train, X_test)

        # Para cada país no teste, o valor deve ser o do lookup ou o fallback
        for _, row in X_test_lag.iterrows():
            country = row['country_origin_code']
            expected = lookups['risk_by_country'].get(country, lookups['fallback_rate'])
            assert row['taxa_risco_pais'] == pytest.approx(expected), \
                f"Leakage detectado: país {country} com taxa {row['taxa_risco_pais']} != {expected}"

            entry = row['clearance_place_entry']
            expected_entry = lookups['risk_by_entry'].get(entry, lookups['fallback_rate'])
            assert row['taxa_risco_clearance_entry'] == pytest.approx(expected_entry), \
                f"Leakage detectado: recinto {entry} com taxa {row['taxa_risco_clearance_entry']} != {expected_entry}"

            ncm = row['ncm_code']
            expected_ncm = lookups['risk_by_ncm_code'].get(ncm, lookups['fallback_rate'])
            assert row['taxa_risco_ncm_code'] == pytest.approx(expected_ncm), \
                f"Leakage detectado: NCM {ncm} com taxa {row['taxa_risco_ncm_code']} != {expected_ncm}"

    def test_does_not_mutate_original(self, split_data):
        """Verifica que build_lag_features não altera os DataFrames originais."""
        X_train, X_test, y_train, _ = split_data
        original_train_cols = set(X_train.columns)
        original_test_cols = set(X_test.columns)

        build_lag_features(X_train, y_train, X_test)

        assert set(X_train.columns) == original_train_cols, "X_train original foi mutado."
        assert set(X_test.columns) == original_test_cols, "X_test original foi mutado."

    def test_row_count_preserved(self, split_data):
        """Verifica que build_lag_features não altera o número de linhas."""
        X_train, X_test, y_train, _ = split_data
        n_train, n_test = len(X_train), len(X_test)

        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        assert len(X_train_lag) == n_train
        assert len(X_test_lag) == n_test

    def test_lookups_returned(self, split_data):
        """Verifica se os lookup tables são retornados para persistência."""
        X_train, X_test, y_train, _ = split_data
        _, _, lookups = build_lag_features(X_train, y_train, X_test)

        assert 'risk_by_country' in lookups
        assert 'risk_by_ncm' in lookups
        assert 'risk_by_ncm_code' in lookups
        assert 'volume_by_country' in lookups
        assert 'risk_by_entry' in lookups
        assert 'risk_by_dispatch' in lookups
        assert 'risk_by_place' in lookups
        assert 'fallback_rate' in lookups
        assert 0 <= lookups['fallback_rate'] <= 1

    def test_unseen_country_gets_fallback(self):
        """
        Verifica que um país de origem NUNCA visto no treino
        recebe a taxa de risco global (fallback) em vez de NaN.
        """
        X_train = pd.DataFrame({
            'country_origin_code':      ['156', '156', '218', '218'],
            'ncm_grau_elaboracao':      [38, 38, 85, 85],
            'ncm_code':                 ['38220090', '38220090', '85444200', '85444200'],
            'clearance_place_entry':    ['PORTO DE SANTOS', 'PORTO DE SANTOS', 'AEROPORTO GRU', 'AEROPORTO GRU'],
            'clearance_place_dispatch': ['PORTO DE PARANAGUÁ', 'PORTO DE PARANAGUÁ', 'PORTO SECO', 'PORTO SECO'],
            'clearance_place':          ['PORTO SECO CURITIBA', 'PORTO SECO CURITIBA', 'AEROPORTO AFONSO PENA', 'AEROPORTO AFONSO PENA'],
        })
        y_train = pd.Series(['VERDE', 'VERMELHO', 'VERDE', 'VERDE'])

        X_test = pd.DataFrame({
            'country_origin_code':      ['999'],  # País nunca visto
            'ncm_grau_elaboracao':      [38],
            'ncm_code':                 ['99999999'],  # NCM nunca visto
            'clearance_place_entry':    ['RECINTO NOVO'],  # Recinto nunca visto
            'clearance_place_dispatch': ['PORTO DE PARANAGUÁ'],
            'clearance_place':          ['PORTO SECO CURITIBA'],
        })

        _, X_test_lag, lookups = build_lag_features(X_train, y_train, X_test)

        assert X_test_lag['taxa_risco_pais'].iloc[0]            == pytest.approx(lookups['fallback_rate'])
        assert X_test_lag['taxa_risco_ncm_code'].iloc[0]        == pytest.approx(lookups['fallback_rate'])
        assert X_test_lag['taxa_risco_clearance_entry'].iloc[0] == pytest.approx(lookups['fallback_rate'])
        assert X_test_lag['volume_dis_pais'].iloc[0]            == 0


# TESTES: Preprocessor
class TestPreprocessor:

    def test_preprocessor_builds_without_error(self):
        """Verifica se o ColumnTransformer é construído sem erros."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2  # num + cat

    def test_preprocessor_num_pipeline(self):
        """Verifica se o pipeline numérico contém Imputer + Scaler."""
        preprocessor = build_preprocessor()
        num_pipeline = preprocessor.transformers[0][1]
        step_names = [name for name, _ in num_pipeline.steps]
        assert 'imp' in step_names, "Imputer ausente no pipeline numérico."
        assert 'std' in step_names, "Scaler ausente no pipeline numérico."

    def test_preprocessor_cat_pipeline(self):
        """Verifica se o pipeline categórico contém Imputer + OneHotEncoder."""
        preprocessor = build_preprocessor()
        cat_pipeline = preprocessor.transformers[1][1]
        step_names = [name for name, _ in cat_pipeline.steps]
        assert 'imp' in step_names, "Imputer ausente no pipeline categórico."
        assert 'ohe' in step_names, "OneHotEncoder ausente no pipeline categórico."

    def test_preprocessor_includes_lag_features(self):
        """Verifica se as lag features estão na lista de features numéricas."""
        preprocessor = build_preprocessor()
        num_features = preprocessor.transformers[0][2]
        for col in ['taxa_risco_pais', 'taxa_risco_ncm', 'taxa_risco_ncm_code', 'volume_dis_pais',
                    'taxa_risco_clearance_entry', 'taxa_risco_clearance_dispatch', 'taxa_risco_clearance_place']:
            assert col in num_features, f"Lag feature '{col}' ausente no preprocessor numérico."

    def test_preprocessor_includes_clearance_places(self):
        """Verifica se as colunas de recinto completas estão no preprocessor."""
        preprocessor = build_preprocessor()
        cat_features = preprocessor.transformers[1][2]
        for col in ['clearance_place_entry', 'clearance_place_dispatch', 'clearance_place']:
            assert col in cat_features, f"Coluna '{col}' ausente no preprocessor categórico."

    def test_preprocessor_includes_ncm_code(self):
        """Verifica se ncm_code está incluído como feature categórica."""
        preprocessor = build_preprocessor()
        cat_features = preprocessor.transformers[1][2]
        assert 'ncm_code' in cat_features, "ncm_code ausente no preprocessor categórico."

    def test_preprocessor_fit_transform(self, split_data):
        """Verifica se o preprocessor executa fit_transform sem erro nos dados com lag features."""
        X_train, X_test, y_train, _ = split_data
        X_train_lag, X_test_lag, _ = build_lag_features(X_train, y_train, X_test)

        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_train_lag)

        assert X_transformed.shape[0] == len(X_train_lag), "Número de linhas alterado."
        assert X_transformed.shape[1] > 0, "Nenhuma feature gerada."
        assert not np.isnan(X_transformed).any(), "NaN encontrado após transformação."


# TESTES: Edge Cases
class TestEdgeCases:
    def test_all_null_registry_date(self):
        """Testa feature_engineering quando todas as registry_date são nulas."""
        df = pd.DataFrame({
            'document_number':          ['DI-001', 'DI-002'],
            'yearmonth':                ['2024-06-01', '2024-07-01'],
            'registry_date':            [pd.NaT, pd.NaT],
            'ncm_code':                 ['38220090', '85444200'],
            'country_origin_code':      [218.0, 368.0],
            'consignee_name':           ['ACME IMPORTADORA', 'GLOBAL TRADING'],
            'consignee_code':           ['C001', 'C002'],
            'consignee_company_size':   ['MICRO EMPRESA', 'MICRO EMPRESA'],
            'shipper_name':             ['SHIPPER A', 'SHIPPER B'],
            'transport_mode_pt':        ['AÉREA', 'MARÍTIMA'],
            'clearance_place_entry':    ['PORTO DE SANTOS', 'AEROPORTO GRU'],
            'clearance_place_dispatch': ['PORTO DE PARANAGUÁ', 'PORTO SECO'],
            'clearance_place':          ['PORTO SECO CURITIBA', 'AEROPORTO AFONSO PENA'],
            'channel':                  ['VERDE', 'VERMELHO'],
        })
        result = feature_engineering(df)
        assert result['registry_date'].isna().sum() == 0

    def test_unknown_ncm_code(self):
        """Testa se NCM inválido/desconhecido gera fallback -1."""
        df = pd.DataFrame({
            'document_number':          ['DI-001'],
            'yearmonth':                ['2024-06-01'],
            'registry_date':            [pd.Timestamp('2024-06-15')],
        'ncm_code':                     ['INVALIDO'],
            'country_origin_code':      ['218'],
            'consignee_name':           ['EMPRESA TESTE'],
            'consignee_code':           ['C001'],
            'consignee_company_size':   ['MICRO EMPRESA'],
            'shipper_name':             ['SHIPPER TESTE'],
            'transport_mode_pt':        ['AÉREA'],
            'clearance_place_entry':    ['PORTO DE SANTOS'],
            'clearance_place_dispatch': ['PORTO DE PARANAGUÁ'],
            'clearance_place':          ['PORTO SECO'],
            'channel':                  ['VERDE'],
        })
        result = feature_engineering(df)
        assert result['ncm_grau_elaboracao'].iloc[0] == -1, \
            "NCM inválido deveria gerar fallback -1."

    def test_shipper_name_null_handling(self):
        """Testa se shipper_name nulo é tratado antes da extração de tipo."""
        df = pd.DataFrame({
            'document_number':          ['DI-001'],
            'yearmonth':                ['2024-06-01'],
            'registry_date':            [pd.Timestamp('2024-06-15')],
            'ncm_code':                 ['38220090'],
            'country_origin_code':      ['218'],
            'consignee_name':           ['EMPRESA TESTE'],
            'consignee_code':           ['C001'],
            'consignee_company_size':   ['MICRO EMPRESA'],
            'shipper_name':             [None],
            'transport_mode_pt':        ['AÉREA'],
            'clearance_place_entry':    ['PORTO DE SANTOS'],
            'clearance_place_dispatch': ['PORTO DE PARANAGUÁ'],
            'clearance_place':          ['PORTO SECO'],
            'channel':                  ['VERDE'],
        })
        result = feature_engineering(df)
        assert result['shipper_tipo'].notna().all()
"""
Avaliação e Explicabilidade dos Modelos.

Gera:
  1. Matrizes de Confusão (RF, HGB, XGBoost)
  2. Gini Importance (Top 20 features - Random Forest)
  3. Permutation Importance (Impacto no Recall-Macro no Teste)

Pré-requisito: executar main_pipeline.py primeiro para gerar os artefatos em models/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

OUTPUT_DIR = 'eval_plots'


def load_artifacts():
    """Carrega todos os artefatos salvos pelo main_pipeline.py."""
    rf_pipeline = joblib.load('models/rf_pipeline.pkl')
    hgb_pipeline = joblib.load('models/hgb_pipeline.pkl')
    xgb_pipeline = joblib.load('models/xgb_pipeline.pkl')
    le = joblib.load('models/label_encoder.pkl')
    X_test, y_test, class_names = joblib.load('models/test_data.pkl')
    return rf_pipeline, hgb_pipeline, xgb_pipeline, le, X_test, y_test, class_names


def plot_confusion_matrices(pipelines, X_test, y_test, class_names):
    """Gera matrizes de confusão lado a lado para os 3 modelos."""
    model_names = ['Random Forest', 'HistGradientBoosting', 'XGBoost']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    for ax, pipeline, name in zip(axes, pipelines, model_names):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f'Matriz de Confusão — {name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
    
    plt.suptitle('Comparativo de Matrizes de Confusão', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logging.info("Matrizes de confusão salvas em eval_plots/confusion_matrices.png")


def plot_gini_importance(rf_pipeline, top_n=20):
    """Gera gráfico de Gini Importance (nativa da Random Forest)."""
    feature_names = rf_pipeline.named_steps['pre'].get_feature_names_out()
    clean_names = [n.split('__')[1] if '__' in n else n for n in feature_names]
    importances = rf_pipeline.named_steps['clf'].feature_importances_
    
    feat_imp = pd.Series(importances, index=clean_names).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feat_imp.head(top_n).values, y=feat_imp.head(top_n).index, palette='viridis')
    plt.title(f'Top {top_n} Features (Gini Importance) — Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importância (Gini)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_gini.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logging.info("Gini Importance salvo em eval_plots/feature_importance_gini.png")


def plot_permutation_importance(rf_pipeline, X_test, y_test, n_repeats=5):
    """
    Gera gráfico de Permutation Importance usando recall_macro como scoring.
    
    Nota: Usamos recall_macro em vez de f1_macro porque o objetivo de negócio é
    maximizar a detecção de canais de risco (Vermelho/Amarelo). O F1-Macro pode
    mascarar o impacto real das features quando a classe majoritária (Verde) domina.
    """
    logging.info("Calculando Permutation Importance (Recall-Macro)... Pode levar alguns minutos.")
    
    perm = permutation_importance(
        rf_pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='recall_macro'
    )
    
    perm_series = pd.Series(perm.importances_mean, index=X_test.columns)
    perm_std = pd.Series(perm.importances_std, index=X_test.columns)
    
    # Ordenar por importância decrescente e pegar top 15
    sorted_idx = perm_series.sort_values(ascending=True).index
    top_features = sorted_idx[-15:]
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(top_features)),
        perm_series[top_features].values,
        xerr=perm_std[top_features].values,
        color='teal',
        capsize=3,
        alpha=0.85,
    )
    plt.yticks(range(len(top_features)), top_features)
    plt.title('Permutation Feature Importance (Impacto no Recall-Macro)', fontsize=14, fontweight='bold')
    plt.xlabel('Queda no Recall-Macro ao permutar a feature')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logging.info("Permutation Importance salvo em eval_plots/permutation_importance.png")


if __name__ == "__main__":
    logging.info("Carregando artefatos do pipeline...")
    rf_pipeline, hgb_pipeline, xgb_pipeline, le, X_test, y_test, class_names = load_artifacts()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Matrizes de Confusão
    plot_confusion_matrices(
        pipelines=[rf_pipeline, hgb_pipeline, xgb_pipeline],
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )
    
    # 2. Gini Importance (Random Forest)
    plot_gini_importance(rf_pipeline)
    
    # 3. Permutation Importance (Random Forest)
    plot_permutation_importance(rf_pipeline, X_test, y_test)
    
    logging.info("Todos os gráficos de avaliação foram gerados com sucesso.")
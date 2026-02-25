# Relatório Executivo

## Objetivo

Prever o canal aduaneiro (Verde, Amarelo, Vermelho ou Cinza) atribuído a uma Declaração de Importação (DI), auxiliando na antecipação de custos e prazos logísticos.

## Dados

Dataset com ~100k DIs registradas entre Jan/2023 e Jan/2025. 14 variáveis, predominantemente categóricas. Desbalanceamento extremo: 97% das DIs caem no canal Verde.

## Principais Insights

**O dataset atual não possui variáveis com poder discriminativo suficiente para prever o canal com confiança.** A correlação entre as features categóricas e o target é negligível (Cramér's V < 0.03 para todas). Três arquiteturas distintas (Random Forest, HistGradientBoosting, XGBoost) convergem para o mesmo teto de performance nas classes de risco, reforçando que o limite é dos dados e não dos algoritmos.

**Setores de maior risco:**  Produtos  químicos, bebidas alcóolicas e certos materiais (fios/cabos elétricos e obras de borracha).

**Sazonalidade clara:** Volume de DIs cai drasticamente em Janeiro e Julho e atinge picos em Abril e Outubro. Finais de semana têm volume significativamente menor.

**Modal aéreo é o mais fiscalizado** (2.44% de canal vermelho), seguido por meios próprios (2.17%).

**Empresas menores sofrem mais fiscalização:** Micro empresas e empresas de pequeno porte possuem taxa de canal vermelho (~2.8%) superior à categoria DEMAIS (~2.0%), mas é necessário cautela pois +95% das DIs pertencem à classe DEMAIS.

## Modelo

O melhor resultado foi a Random Forest com oversampling e class_weight='balanced'. Recall de 29% para Vermelho e 72% para Verde. Não é um classificador forte, mas funciona como ranqueador: DIs com maior probabilidade de risco podem ser priorizadas para revisão.

## O que falta para melhorar

O principal gargalo é a **falta de variáveis informativas**. Dados como valor da mercadoria, histórico do importador, peso/volume, e informações sobre anuências e licenças provavelmente podem futuramente ajudar na decisão do modelo.

Além disso, cross-validation temporal e tuning de hiperparâmetros (via Optuna ou GridSearch) poderiam extrair mais performance das variáveis existentes, mas não foram explorados por restrição de tempo.
# Case FIESC — Classificação de espectros (5 classes)

# Link dos dados: https://huggingface.co/datasets/Willgnner-Santos/Prova-Pratica-Case-Online-FIESC

## Visão geral
Desenvolvi um pipeline completo para **análise e classificação supervisionada** de espectros (faixa aproximada **780–1080 nm**), usando **5 arquivos CSV** (uma classe por arquivo).  
Implementei também uma etapa **não supervisionada (clustering)** e três arquiteturas de **aprendizado profundo** para comparar desempenho.

O notebook cobre:
- carga e consolidação das classes em um único DataFrame;
- diagnóstico de estrutura, granularidade e **qualidade (NaNs, duplicatas)**;
- análise espectral (curvas médias por classe, boxplots em λ específicos, **ANOVA/F-score** por comprimento de onda);
- modelagem supervisionada (clássicos) com **pipeline de pré-processamento**;
- seleção e salvamento do melhor pipeline;
- clustering com métricas (ARI e silhouette);
- deep learning (MLP, CNN 1D e CNN 1D + Self-Attention) e comparação final.

---

## Estrutura esperada dos dados
Organize os dados em uma pasta (ex.: `Dados/`) com os arquivos:

```
Dados/
  Classe_1.csv
  Classe_2.csv
  Classe_3.csv
  Classe_4.csv
  Classe_5.csv
```

Cada `Classe_i.csv` contém as intensidades do espectro (features) por amostra.  
Como os equipamentos podem ter **granularidades diferentes**, alguns arquivos têm mais colunas que outros — ao concatenar, eu permiti a união das colunas e tratei os faltantes com imputação.

---

## Como executar
### 1) Ambiente (recomendado)
Faça um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn joblib
pip install xgboost lightgbm catboost hdbscan
pip install tensorflow
pip install jupyter
```

> Se você estiver no Google Colab, use a célula de instalação (com `!pip install ...`) diretamente.

### 2) Ajuste os caminhos no notebook
No início do notebook, **edite o caminho** da pasta de dados:

```python
DATA_DIR = r"CAMINHO/DA/SUA/PASTA/Dados"
```

Quando for salvar o pipeline vencedor, **ajuste o caminho do artefato**:

```python
model_path = r"CAMINHO/ONDE/SALVAR/pipeline_espectro_best.pkl"
```

### 3) Rode o notebook
Faça isso:
- Abra o `.ipynb` no Jupyter/VS Code/Colab
- Execute as células em ordem (Run All)

---

## O que eu implementei (resumo técnico)
### Pré-processamento (base para modelos clássicos e DL)
- Imputação de NaNs com **mediana por coluna** (`SimpleImputer(strategy="median")`)
- Padronização com **StandardScaler**
- Split **estratificado** treino/teste (80/20)

### Análise exploratória e espectral
- Distribuição de classes e estatísticas globais
- Curvas médias por classe (no eixo de λ gerado por `np.linspace(780,1080,n_features)`)
- Boxplots em comprimentos de onda específicos
- Relevância por λ com **ANOVA/F-score**

### Modelos supervisionados (clássicos)
- Validação cruzada estratificada (5-fold) e ranking por **F1-macro**
- Treino no conjunto de treino e avaliação no teste (acc, precision_macro, recall_macro, f1_macro)
- Relatório por classe e **matriz de confusão** do melhor modelo
- Salvamento do **pipeline vencedor** com `joblib.dump(...)`

> Observação: no notebook eu incluí bibliotecas adicionais (XGBoost/LightGBM/CatBoost). Caso você remova esses modelos, as demais etapas continuam funcionando com scikit-learn.

### Não supervisionado (clustering)
- PCA para reduzir para **10 componentes** antes do clustering
- Algoritmos: **K-Means**, Agglomerative, MeanShift, DBSCAN e HDBSCAN
- Métricas: **ARI** (comparando clusters vs. rótulos reais) e **silhouette**
- Visualização do K-Means em espaço 2D (PCA apenas para plot)

### Aprendizado profundo
- MLP (baseline) com Dense + Dropout
- CNN 1D para capturar padrões locais do espectro
- CNN 1D + **Self-Attention** (MultiHeadAttention) para capturar dependências globais
- Early stopping e comparação final (clássicos vs DL) em **F1-macro**

---

## Saídas e artefatos
Durante a execução eu gerei:
- gráficos e tabelas de EDA e análise espectral;
- ranking de modelos supervisionados (CV + teste);
- resultados de clustering (ARI/silhouette) e visualizações;
- **arquivo `.pkl`** com o pipeline supervisionado vencedor (pré-processamento + modelo).

---

## Tecnologias
- Python, Jupyter
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (Pipeline, Imputer, StandardScaler, PCA, métricas, modelos)
- xgboost, lightgbm, catboost (opcional)
- hdbscan (clustering)
- TensorFlow/Keras (MLP, CNN 1D, MultiHeadAttention)

---

## Reprodutibilidade
- Use `random_state=42` (como no notebook) para reproduzir splits e PCA/clustering.
- Garanta que os CSVs estejam consistentes (mesma codificação e cabeçalhos).
- Se a faixa de λ do seu equipamento for diferente, atualize `wl_min` e `wl_max`.

---

## Próximos passos 
- Consolidar os melhores hiperparâmetros com GridSearch/Optuna
- Exportar também um `requirements.txt`
- Criar um script (CLI) para treinar e inferir sem depender do notebook

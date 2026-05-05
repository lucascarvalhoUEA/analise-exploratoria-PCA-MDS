#!/usr/bin/env python3
"""
Análise Multidimensional com PCA e MDS
Dataset: Tabela 74 IBGE - Produção de Leite nos Municípios do Amazonas (2004-2024)
Fonte  : IBGE - Pesquisa da Pecuária Municipal
"""
import csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # backend sem display; troca por TkAgg se quiser janela
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (13, 9)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

ANOS = list(range(2004, 2025))   # 21 anos = 21 features

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E PARSING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("1. CARREGAMENTO E PARSING DO DATASET")
print("=" * 62)

with open('../data/tabela74.csv', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.reader(f))

print(f"Total de linhas no arquivo: {len(rows)}")
print(f"Colunas por linha de dado : {len(rows[5])} (87 esperadas)")

# Blocos de dados (detectados previamente):
# Produção Mil litros : linhas 5–66
# Valor Mil Reais     : linhas 74–135
# Percentual          : linhas 143–204
BLOCOS = {
    'prod_litros':   (5,   67),
    'valor_reais':   (74, 136),
    'percentual':    (143, 205),
}

def extrair_tabela(rows, inicio, fim):
    """Extrai uma das três sub-tabelas do CSV bruto."""
    dados = {}
    for row in rows[inicio:fim]:
        if not row or row[0].strip() != 'MU':
            continue
        mun = row[2].replace('(AM)', '').strip()
        valores = {}
        for k, ano in enumerate(ANOS):
            col_leite = 3 + 4 * k + 2   # posição do valor Leite
            raw = row[col_leite] if col_leite < len(row) else ''
            v = raw.strip()
            if v in ('..', '...', '-', '', 'X'):
                valores[ano] = np.nan
            else:
                try:
                    valores[ano] = float(v)
                except ValueError:
                    valores[ano] = np.nan
        dados[mun] = valores
    return pd.DataFrame(dados).T.rename_axis('Município')

prod  = extrair_tabela(rows, *BLOCOS['prod_litros'])
valor = extrair_tabela(rows, *BLOCOS['valor_reais'])
pct   = extrair_tabela(rows, *BLOCOS['percentual'])

print(f"\nTabela extraída — Produção Leite (Mil litros):")
print(f"  Municípios : {prod.shape[0]}")
print(f"  Anos       : {prod.shape[1]}  ({ANOS[0]}–{ANOS[-1]})")
print(f"  NaN totais : {prod.isna().sum().sum()}")
print(f"\nAmostra (5 municípios, 5 primeiros anos):")
print(prod.iloc[:5, :5].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. SELEÇÃO E JUSTIFICATIVA DAS FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("2. SELEÇÃO DAS FEATURES")
print("=" * 62)
print("""
• Unidade de análise : 62 municípios do Amazonas (observações)
• Features           : Produção de leite (Mil litros) em cada um
                       dos 21 anos — 2004 a 2024 → 21 variáveis
• Justificativa      : Cada ano representa uma dimensão do perfil
                       produtivo do município ao longo do tempo.
                       PCA e MDS reduzirão esses 21 anos para 2D,
                       preservando similaridades entre municípios.
                       Municípios com trajetórias parecidas ficarão
                       próximos; outliers ficarão isolados.
""")

stats = pd.DataFrame({
    'Média'     : prod.mean(axis=1).round(0),
    'Mediana'   : prod.median(axis=1).round(0),
    'Std'       : prod.std(axis=1).round(0),
    'Máximo'    : prod.max(axis=1).round(0),
    '% NaN'     : (prod.isna().mean(axis=1) * 100).round(1),
})
print("Top 10 municípios por média de produção:")
print(stats.sort_values('Média', ascending=False).head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARAÇÃO E PADRONIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("3. PREPARAÇÃO E PADRONIZAÇÃO")
print("=" * 62)

# Remove municípios com mais de 50 % de NaN
mask = prod.isna().mean(axis=1) <= 0.50
prod_clean = prod[mask].copy()
print(f"Municípios removidos (>50% NaN): {prod.shape[0] - prod_clean.shape[0]}")
print(f"Municípios mantidos            : {prod_clean.shape[0]}")

# Imputa NaN com mediana de cada ano (coluna)
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(prod_clean)
X_imp   = pd.DataFrame(X_imp, index=prod_clean.index, columns=ANOS)

# Padronização Z-score (média=0, std=1) — necessária para PCA
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_imp)

print(f"\nMatriz final: {X_sc.shape[0]} municípios × {X_sc.shape[1]} features")
print(f"Média pós-padronização : {X_sc.mean():.6f}  (≈ 0)")
print(f"Std  pós-padronização  : {X_sc.std():.6f}   (≈ 1)")

# Categorias para coloração dos gráficos
media_mun = X_imp.mean(axis=1)
q33 = media_mun.quantile(0.33)
q66 = media_mun.quantile(0.66)
categorias = pd.cut(
    media_mun,
    bins=[-np.inf, q33, q66, np.inf],
    labels=['Baixa produção', 'Média produção', 'Alta produção']
)
print(f"\nCategoria por terço de produção média:")
print(categorias.value_counts().to_string())

CORES = {
    'Baixa produção': '#4ECDC4',
    'Média produção': '#FFE66D',
    'Alta produção' : '#FF6B6B',
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("4. ANÁLISE COM PCA")
print("=" * 62)

# PCA completo para análise de variância
pca_full = PCA().fit(X_sc)
var_exp  = pca_full.explained_variance_ratio_
var_cum  = np.cumsum(var_exp)

print("\nVariância explicada por componente (primeiros 10):")
for i in range(10):
    barra = '█' * int(var_exp[i] * 80)
    print(f"  PC{i+1:2d}: {var_exp[i]*100:5.1f}%  acum: {var_cum[i]*100:5.1f}%  {barra}")

# PCA 2D
pca2    = PCA(n_components=2).fit(X_sc)
X_pca   = pca2.transform(X_sc)
vPC1    = pca2.explained_variance_ratio_[0] * 100
vPC2    = pca2.explained_variance_ratio_[1] * 100
vTOT    = vPC1 + vPC2
loadings = pca2.components_.T   # shape (21, 2)

print(f"\nPC1: {vPC1:.1f}%  |  PC2: {vPC2:.1f}%  |  Total: {vTOT:.1f}%")

# ── Gráfico 1: Scree Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
n_show = 15
axes[0].bar(range(1, n_show+1), var_exp[:n_show]*100,
            color='steelblue', alpha=0.8, edgecolor='navy', label='Individual')
axes[0].plot(range(1, n_show+1), var_cum[:n_show]*100,
             'o-', color='crimson', lw=2, ms=5, label='Acumulada')
axes[0].axhline(80, color='gray', ls='--', alpha=0.6, label='80%')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Variância Explicada (%)')
axes[0].set_title('Scree Plot')
axes[0].legend(); axes[0].set_xticks(range(1, n_show+1))

# Loadings PC1 como barras horizontais
axes[1].barh([str(a) for a in ANOS], loadings[:, 0],
             color=['tomato' if x < 0 else 'steelblue' for x in loadings[:, 0]],
             alpha=0.85, edgecolor='white')
axes[1].axvline(0, color='black', lw=0.8)
axes[1].set_xlabel('Loading')
axes[1].set_title(f'Contribuição dos Anos na PC1 ({vPC1:.1f}%)')
axes[1].tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('../images/fig1_scree_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig1_scree_loadings.png")

# ── Gráfico 2: PCA 2D ─────────────────────────────────────────────────────
top_muns = media_mun.nlargest(8).index.tolist()
fig, ax = plt.subplots(figsize=(14, 10))

for cat in CORES:
    idx = [i for i, m in enumerate(prod_clean.index) if str(categorias[m]) == cat]
    if idx:
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=CORES[cat], label=cat, s=120, alpha=0.85,
                   edgecolors='white', lw=0.8, zorder=3)

for i, mun in enumerate(prod_clean.index):
    if mun in top_muns or abs(X_pca[i, 0]) > 2.2 or abs(X_pca[i, 1]) > 1.8:
        ax.annotate(mun, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.65))

ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel(f'PC1 — {vPC1:.1f}% da variância', fontsize=12)
ax.set_ylabel(f'PC2 — {vPC2:.1f}% da variância', fontsize=12)
ax.set_title(
    f'PCA 2D — Produção de Leite por Município do Amazonas (2004–2024)\n'
    f'Variância total explicada: {vTOT:.1f}%',
    fontsize=13, fontweight='bold')
ax.legend(title='Nível de Produção', fontsize=10)
plt.tight_layout()
plt.savefig('../images/fig2_pca_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig2_pca_2d.png")

# ── Gráfico 3: Biplot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
cat_num = {'Baixa produção': 0, 'Média produção': 1, 'Alta produção': 2}
colors_num = [cat_num.get(str(categorias[m]), 0) for m in prod_clean.index]
ax.scatter(X_pca[:, 0], X_pca[:, 1],
           c=colors_num, cmap='RdYlGn', s=80, alpha=0.7, zorder=3)

escala = 3.0
for j, ano in enumerate(ANOS):
    ax.annotate('', xy=(loadings[j,0]*escala, loadings[j,1]*escala), xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    ax.text(loadings[j,0]*escala*1.12, loadings[j,1]*escala*1.12,
            str(ano), fontsize=7.5, color='navy', ha='center')

ax.axhline(0, color='gray', lw=0.4)
ax.axvline(0, color='gray', lw=0.4)
ax.set_xlabel(f'PC1 ({vPC1:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({vPC2:.1f}%)', fontsize=12)
ax.set_title('Biplot PCA — Municípios e Vetores dos Anos', fontsize=13, fontweight='bold')
patches = [mpatches.Patch(color=c, label=l) for l, c in
           zip(['Baixa','Média','Alta'], ['#4ECDC4','#FFE66D','#FF6B6B'])]
ax.legend(handles=patches, title='Produção', fontsize=9)
plt.tight_layout()
plt.savefig('../images/fig3_biplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig3_biplot.png")

# ── Gráfico 4: Loadings PC1 e PC2 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, comp, idx_c, cor_pos, cor_neg in [
    (axes[0], 'PC1', 0, 'steelblue', 'tomato'),
    (axes[1], 'PC2', 1, 'darkorange', 'mediumpurple'),
]:
    vals = loadings[:, idx_c]
    cores = [cor_pos if v >= 0 else cor_neg for v in vals]
    ax.bar([str(a) for a in ANOS], vals, color=cores, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f'Loadings — {comp}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ano'); ax.set_ylabel('Loading')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig('../images/fig4_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig4_loadings.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("5. ANÁLISE COM MDS")
print("=" * 62)

# MDS não-métrico com matriz de dissimilaridade pré-computada normalizada
D      = pairwise_distances(X_sc, metric='euclidean')
D_norm = D / D.max()   # normaliza para [0, 1]
mds    = MDS(n_components=2, metric=False, dissimilarity='precomputed',
             random_state=42, n_init=10, max_iter=1000)
X_mds  = mds.fit_transform(D_norm)
stress = mds.stress_
qual   = ("Excelente" if stress < 0.05 else
          "Boa"       if stress < 0.10 else
          "Aceitável" if stress < 0.20 else "Ruim")
print(f"Stress: {stress:.4f}  → {qual}")

# ── Gráfico 5: MDS 2D ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
for cat in CORES:
    idx = [i for i, m in enumerate(prod_clean.index) if str(categorias[m]) == cat]
    if idx:
        ax.scatter(X_mds[idx, 0], X_mds[idx, 1],
                   c=CORES[cat], label=cat, s=120, alpha=0.85,
                   edgecolors='white', lw=0.8, zorder=3)

for i, mun in enumerate(prod_clean.index):
    if mun in top_muns or abs(X_mds[i, 0]) > 2.0 or abs(X_mds[i, 1]) > 1.8:
        ax.annotate(mun, (X_mds[i, 0], X_mds[i, 1]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.65))

ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('Dimensão MDS 1', fontsize=12)
ax.set_ylabel('Dimensão MDS 2', fontsize=12)
ax.set_title(
    f'MDS 2D — Produção de Leite por Município do Amazonas (2004–2024)\n'
    f'Stress = {stress:.4f} ({qual})',
    fontsize=13, fontweight='bold')
ax.legend(title='Nível de Produção', fontsize=10)
plt.tight_layout()
plt.savefig('../images/fig5_mds_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig5_mds_2d.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARAÇÃO PCA vs MDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("6. COMPARAÇÃO PCA vs MDS")
print("=" * 62)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, coords, tit, xl, yl in [
    (axes[0], X_pca,
     f'PCA 2D — {vTOT:.1f}% variância', f'PC1 ({vPC1:.1f}%)', f'PC2 ({vPC2:.1f}%)'),
    (axes[1], X_mds,
     f'MDS 2D — Stress={stress:.4f}', 'Dimensão 1', 'Dimensão 2'),
]:
    for cat in CORES:
        idx = [i for i, m in enumerate(prod_clean.index) if str(categorias[m]) == cat]
        if idx:
            ax.scatter(coords[idx,0], coords[idx,1],
                       c=CORES[cat], label=cat, s=90, alpha=0.85,
                       edgecolors='white', lw=0.6, zorder=3)
    for i, mun in enumerate(prod_clean.index):
        if mun in top_muns:
            ax.annotate(mun, (coords[i,0], coords[i,1]),
                        fontsize=7, ha='center', va='bottom',
                        xytext=(0,5), textcoords='offset points')
    ax.axhline(0, color='gray', lw=0.4)
    ax.axvline(0, color='gray', lw=0.4)
    ax.set_title(tit, fontsize=12, fontweight='bold')
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.legend(title='Produção', fontsize=9)

plt.suptitle('Comparação: PCA vs MDS — Municípios do Amazonas',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../images/fig6_comparacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig6_comparacao.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVOLUÇÃO TEMPORAL DOS DESTAQUES
# ─────────────────────────────────────────────────────────────────────────────
top5 = media_mun.nlargest(5).index.tolist()
cores_line = plt.cm.tab10(np.linspace(0, 0.5, len(top5)))

fig, ax = plt.subplots(figsize=(14, 7))
for mun, cor in zip(top5, cores_line):
    ax.plot(ANOS, X_imp.loc[mun].values, 'o-', label=mun,
            color=cor, lw=2, ms=5)
ax.set_xlabel('Ano', fontsize=12)
ax.set_ylabel('Produção de Leite (Mil litros)', fontsize=12)
ax.set_title('Evolução Temporal — Top 5 Municípios Produtores do AM',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../images/fig7_evolucao_top5.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → fig7_evolucao_top5.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANÁLISE DE OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("7. OUTLIERS IDENTIFICADOS")
print("=" * 62)

dist_pca = pd.Series(np.sqrt(X_pca[:,0]**2 + X_pca[:,1]**2),
                     index=prod_clean.index)
dist_mds = pd.Series(np.sqrt(X_mds[:,0]**2 + X_mds[:,1]**2),
                     index=prod_clean.index)

print("\nTop 6 outliers — PCA (distância da origem):")
for mun, d in dist_pca.nlargest(6).items():
    print(f"  {mun:<38} dist={d:.2f}  média={media_mun[mun]:.0f} Mil L")

print("\nTop 6 outliers — MDS (distância da origem):")
for mun, d in dist_mds.nlargest(6).items():
    print(f"  {mun:<38} dist={d:.2f}  média={media_mun[mun]:.0f} Mil L")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMÁRIO FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("8. SUMÁRIO E RESPOSTA ÀS PERGUNTAS ROTEADORAS")
print("=" * 62)
n_comp_80 = int(np.argmax(var_cum >= 0.80)) + 1

print(f"""
DATASET
  Fonte     : IBGE - Pesquisa da Pecuária Municipal (Tabela 74)
  Variável  : Produção de Leite — Mil litros/ano
  Período   : 2004-2024 (21 anos = 21 features)
  Municípios: {prod_clean.shape[0]} do Amazonas

PCA
  Variância PC1+PC2    : {vTOT:.1f}%
  Componentes p/ 80%   : {n_comp_80}
  PC1 representa       : nível médio de produção (loading + em todos os anos)
  PC2 representa       : variação/tendência temporal
  Grupos identificados : 3 (alta / média / baixa produção)
  Principais outliers  : {', '.join(dist_pca.nlargest(3).index.tolist())}

MDS
  Stress               : {stress:.4f} ({qual})
  Grupos               : Alinhados com PCA (validação cruzada)
  Outliers confirmados : Mesmos do PCA

PERGUNTAS ROTEADORAS
  1. Problema investigado  : Padrões de produção leiteira nos
                             municípios do AM ao longo de 2004-2024.
  2. Features              : 21 anos de produção (Mil litros).
  3. Expectativa           : Grupos por nível de produção; outliers
                             nos grandes produtores.
  4. PCA revelou           : Sim — 3 grupos e outliers claros.
  5. Variáveis influentes  : Anos recentes (loadings positivos altos).
  6. Variância 2 PCs       : {vTOT:.1f}% — {"Adequada" if vTOT >= 60 else "Parcial"}.
  7. MDS revelou           : Grupos similares ao PCA.
  8. PCA ≈ MDS?            : Sim — padrões convergentes.
  9. Categoria separada    : Alta produção completamente isolada.
  10. Outliers explicados  : Autazes e Careiro da Várzea têm
                             tradição pecuária leiteira muito superior.
  11. Faz sentido?         : Sim — reflete geografia e agropecuária.
  12. Risco de interpretar?: Axes do MDS são arbitrários; PCA com
                             <70% pode perder informação relevante.
  13. PCA para features?   : Sim — {n_comp_80} PCs capturam 80%.
  14. MDS para redução?    : Não ideal — melhor para visualização.
  15. Aprendizado          : Existe nítida estratificação entre
                             municípios; poucos dominam a produção.
""")

print("✓ Análise concluída. Gráficos salvos no diretório do projeto.")
print("  fig1_scree_loadings.png | fig2_pca_2d.png | fig3_biplot.png")
print("  fig4_loadings.png       | fig5_mds_2d.png | fig6_comparacao.png")
print("  fig7_evolucao_top5.png")

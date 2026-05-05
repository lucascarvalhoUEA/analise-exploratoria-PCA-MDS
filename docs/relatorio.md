# Relatório: Compreensão de Dados Multidimensionais com PCA e MDS
**Dataset:** Tabela 74 — IBGE Pesquisa da Pecuária Municipal  
**Tema:** Produção de Leite nos Municípios do Amazonas (2004–2024)

---

## 1. Explicação Teórica do PCA

A **Análise de Componentes Principais** (PCA, do inglês *Principal Component Analysis*) é uma técnica estatística de redução de dimensionalidade linear. Seu objetivo é transformar um conjunto de variáveis possivelmente correlacionadas em um conjunto menor de variáveis não correlacionadas, chamadas de **componentes principais**, que capturam a maior parte da variância (informação) presente nos dados originais.

O algoritmo funciona através da decomposição espectral da matriz de covariância dos dados padronizados. A primeira componente principal (PC1) é o eixo de maior variância no espaço multidimensional; a segunda (PC2) é perpendicular à primeira e captura a segunda maior variância; e assim por diante. Matematicamente:

```
C = (1/n) · Xᵀ · X        (Matriz de covariância)
C · v = λ · v              (Problema de autovalores)
PC_k = X · v_k             (Projeção dos dados)
```

Onde `v_k` são os **autovetores** (direções dos componentes) e `λ_k` são os **autovalores** (magnitude da variância capturada). Os pesos de cada variável original em cada componente são denominados **loadings**, e permitem interpretar o significado de cada componente.

A variância explicada por cada componente é calculada como `λ_k / Σλ`. Em aplicações práticas, busca-se o número mínimo de componentes que explique ao menos 80% da variância total — ponto de equilíbrio entre redução e preservação de informação.

---

## 2. Explicação Teórica do MDS

O **Escalonamento Multidimensional** (MDS, do inglês *Multidimensional Scaling*) é uma família de técnicas que busca representar a **estrutura de dissimilaridades** entre observações em um espaço de menor dimensão, geralmente 2D ou 3D. Ao contrário do PCA, que parte das variáveis originais, o MDS parte de uma **matriz de distâncias** entre pares de observações.

O objetivo do MDS é encontrar uma configuração de pontos em baixa dimensão tal que as distâncias entre eles no espaço reduzido se aproximem o máximo possível das distâncias originais. Existem duas variantes principais:

- **MDS Métrico:** preserva os valores das distâncias originais.
- **MDS Não-Métrico:** preserva apenas a *ordem* (rank) das distâncias, sendo mais robusto a não-linearidades nos dados.

A qualidade do ajuste é medida pelo índice **Stress**:

```
Stress = √( Σ(d̂_ij - d_ij)² / Σ d_ij² )
```

Onde `d_ij` são as distâncias originais e `d̂_ij` as distâncias no espaço reduzido. Valores de Stress abaixo de 0,05 indicam representação excelente; acima de 0,20, representação ruim.

---

## 3. Descrição do Dataset

O dataset utilizado é a **Tabela 74 da Pesquisa da Pecuária Municipal (PPM)** do Instituto Brasileiro de Geografia e Estatística (IBGE). O arquivo CSV original contém três sub-tabelas concatenadas referentes à **produção física** (Mil litros), ao **valor da produção** (Mil reais) e ao **percentual estadual** da produção de origem animal, por município do Amazonas.

Para este trabalho, foi extraída exclusivamente a subtabela de **produção de leite (Mil litros)**, abrangendo os 62 municípios do estado do Amazonas ao longo de 21 anos consecutivos (2004 a 2024).

O arquivo apresenta desafios consideráveis de parsing: usa BOM UTF-8, possui colunas intercaladas por tipo de produto (Total + Leite + Unidade de medida) dentro de cada ano, e registra valores ausentes com símbolos especiais do IBGE (`..`, `...`, `-`), que precisaram ser convertidos para `NaN` antes do processamento.

---

## 4. Seleção e Justificativa das Features

A matriz de dados construída para a análise possui a seguinte estrutura:

| Elemento | Descrição |
|---|---|
| **Observações (linhas)** | 62 municípios do Amazonas |
| **Features (colunas)** | Produção de leite (Mil litros) em cada um dos 21 anos: 2004, 2005, …, 2024 |

A escolha dos 21 anos como features é justificada pela natureza longitudinal do problema: cada município acumula um **perfil temporal de produção** que o caracteriza de forma única. Ao tratar cada ano como uma dimensão do espaço de features, o PCA e o MDS são capazes de identificar municípios com **trajetórias de produção similares**, agrupá-los e destacar aqueles cujo comportamento foge do padrão estadual.

Essa formulação também atende ao requisito de ao menos cinco variáveis numéricas — neste caso, são 21 variáveis contínuas para cada observação —, tornando a aplicação de técnicas de redução de dimensionalidade plenamente justificada.

---

## 5. Preparação e Padronização dos Dados

O pipeline de pré-processamento seguiu as etapas abaixo:

**5.1 Limpeza e filtragem**  
Quatro municípios com mais de 50% dos valores ausentes ao longo da série histórica foram removidos da análise, resultando em uma matriz final de **58 municípios × 21 anos**. Os valores ausentes restantes foram imputados pela **mediana anual** de cada coluna, estratégia robusta a outliers.

**5.2 Padronização Z-score**  
A padronização foi aplicada à matriz imputada, transformando cada variável (ano) para média zero e desvio padrão unitário:

```
z_ij = (x_ij − μ_j) / σ_j
```

Essa etapa é indispensável no PCA: sem ela, municípios como Autazes (~9.700 Mil litros/ano) dominariam completamente as componentes em detrimento dos municípios com produções menores (~10 Mil litros/ano), distorcendo os padrões estruturais de agrupamento.

**5.3 Categorias para visualização**  
Os municípios foram classificados em três grupos por terço de produção média: *Alta produção* (20 municípios), *Média produção* (19) e *Baixa produção* (19). Essa categorização foi usada para colorir os gráficos e apoiar a interpretação visual.

---

## 6. Aplicação do PCA e Resultados

### 6.1 Variância Explicada

O PCA foi aplicado à matriz padronizada (58×21). O resultado do Scree Plot revelou uma estrutura de dados **dominada por uma única componente**:

| Componente | Variância Individual | Variância Acumulada |
|---|---|---|
| PC1 | 91,3% | 91,3% |
| PC2 | 4,4% | 95,7% |
| PC3 | 1,9% | 97,6% |

Apenas **1 componente** já supera a marca de 80% da variância total, e as duas primeiras explicam **95,7%** — qualidade extraordinária para uma projeção 2D.

### 6.2 Interpretação dos Loadings

Os loadings do PC1 são todos **positivos e de magnitude semelhante** para todos os 21 anos. Isso significa que a PC1 representa o **nível absoluto de produção** do município, independente do período. Não há ano que se destaque em relação aos demais — o potencial produtivo é um atributo estrutural e consistente ao longo do tempo.

A PC2 captura variações temporais sutis — municípios com produção crescente ou decrescente ao longo da série aparecem deslocados ao longo deste eixo.

### 6.3 Projeção 2D e Agrupamentos

A projeção bidimensional evidencia **três grupos bem separados**, alinhados com as categorias de produção pré-definidas. O grupo de alta produção encontra-se completamente isolado dos demais no extremo positivo do eixo PC1, enquanto os grupos de média e baixa produção formam uma nuvem compacta na região negativa/central.

---

## 7. Aplicação do MDS e Resultados

O MDS Não-Métrico foi aplicado sobre uma **matriz de dissimilaridade euclidiana normalizada** (58×58). O valor de stress obtido foi:

```
Stress = 0,0075  →  Qualidade: EXCELENTE (< 0,05)
```

Esse resultado indica que a projeção 2D preserva de forma quase perfeita as distâncias originais no espaço de 21 dimensões. A representação visual é, portanto, altamente confiável.

Os agrupamentos identificados pelo MDS são **idênticos** aos do PCA: os mesmos três clusters são visíveis, e os mesmos municípios ocupam posições de destaque (outliers). Isso constitui uma **validação cruzada independente** dos padrões encontrados pelo PCA.

---

## 8. Padrões, Agrupamentos e Outliers

### 8.1 Agrupamentos identificados

| Grupo | Municípios representativos | Produção média |
|---|---|---|
| **Alta produção** | Autazes, Careiro da Várzea, Parintins, Apuí, Itacoatiara | > 1.500 Mil litros/ano |
| **Média produção** | Barreirinha, Boca do Acre, Manicorê, Guajará | 200–1.500 Mil litros/ano |
| **Baixa produção** | Demais 38 municípios | < 200 Mil litros/ano |

### 8.2 Outliers

Os principais outliers identificados em ambos os métodos foram:

| Município | Dist. PCA | Dist. MDS | Média (Mil L/ano) |
|---|---|---|---|
| **Autazes** | 20,95 | 3,13 | 9.730 |
| **Careiro da Várzea** | 17,58 | 2,65 | 8.226 |
| **Parintins** | 10,27 | 1,83 | 4.906 |
| **Apuí** | 9,38 | 1,65 | 4.766 |
| **Itacoatiara** | 8,70 | 1,50 | 4.022 |

**Autazes** e **Careiro da Várzea** são os principais outliers em ambas as técnicas. Do ponto de vista geográfico e agropecuário, isso faz pleno sentido: Autazes é historicamente a maior bacia leiteira do Amazonas, com tradição pecuária consolidada e infraestrutura de beneficiamento. Careiro da Várzea se beneficia das condições naturais das terras de várzea do Rio Solimões, altamente produtivas para a criação bovina.

### 8.3 Tendências temporais

A análise da evolução temporal dos 5 maiores produtores revela que, embora o ranking geral tenha se mantido estável, alguns municípios apresentam trajetórias de crescimento recente expressivas. **Boca do Acre**, por exemplo, saltou de ~961 Mil litros (2010) para ~9.912 Mil litros (2024), indicando uma transformação produtiva significativa que o algoritmo captura através do deslocamento no eixo PC2.

---

## 9. Discussão: Eficácia das Técnicas

### PCA
O PCA se mostrou **altamente eficaz** para este dataset. A captura de 91,3% da variância em apenas 1 componente revela que os dados possuem uma estrutura intrínseca de baixa dimensionalidade: o nível de produção é o único fator que realmente diferencia os municípios. Em um cenário de redução de features para modelos preditivos, apenas 1 PC seria suficiente para representar 80% da informação, reduzindo de 21 para 1 variável sem perda relevante.

### MDS
O MDS complementou o PCA ao confirmar os mesmos padrões por um caminho metodológico completamente independente (baseado em distâncias, não em variância). O Stress excepcionalmente baixo (0,0075) garante que a visualização é confiável. O MDS é, no entanto, computacionalmente mais custoso e **não é adequado como técnica de redução de features** para pipelines de aprendizado de máquina — seu papel aqui é exclusivamente de **validação visual e exploratória**.

### Limitações
- O PCA é uma técnica **linear** e pode não capturar relações não-lineares entre anos.
- Os eixos do MDS são **arbitrários** (sem interpretação semântica direta, ao contrário dos loadings do PCA).
- A imputação pela mediana, embora robusta, pode suavizar variações reais em municípios com muitos dados ausentes.

---

## 10. Conclusão

A análise revelou que a produção de leite no Amazonas é **altamente concentrada** em um pequeno número de municípios, com Autazes e Careiro da Várzea como outliers consistentes ao longo de todo o período analisado. A estrutura dos dados é inerentemente unidimensional: o volume médio de produção explica quase toda a variação observada.

Tanto o PCA quanto o MDS produziram resultados **convergentes e mutuamente validados**, demonstrando que os agrupamentos identificados são robustos e não são artefatos de nenhuma das duas técnicas em particular. A atividade evidencia o potencial dessas ferramentas para revelar padrões estruturais em séries históricas multivariadas, mesmo em contextos com grande assimetria entre as observações.

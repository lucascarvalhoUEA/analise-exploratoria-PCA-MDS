# Análise Multidimensional com PCA e MDS

Este projeto realiza a redução de dimensionalidade e análise exploratória de um dataset multidimensional referente à produção de leite nos municípios do estado do Amazonas entre 2004 e 2024.

A análise utiliza **PCA (Análise de Componentes Principais)** e **MDS (Escalonamento Multidimensional)** para extrair padrões e agrupar municípios baseado em seu histórico longitudinal de produção.

## Estrutura do Repositório

- `/data/`: Contém os dados brutos (`tabela74.csv` - extraído do IBGE PPM).
- `/notebooks/`: Jupyter Notebook completo executado com todo o processo da atividade e as explicações teóricas.
- `/scripts/`: Contém o script em Python (`analise_pca_mds.py`) para quem deseja gerar os gráficos diretamente por linha de comando.
- `/images/`: Todos os gráficos gerados durante o processo.

## Como Executar

### Notebook
Você pode abrir o arquivo `notebooks/Analise_PCA_MDS.ipynb` em qualquer editor de notebooks como Jupyter Lab, Google Colab ou no VS Code. 

### Script 
Para rodar diretamente via script e exportar as imagens:
```bash
python scripts/analise_pca_mds.py
```
*(Certifique-se de instalar as dependências necessárias listadas acima no seu ambiente).*

## Ferramentas e Bibliotecas Utilizadas
- `pandas` e `numpy`: Manipulação e tratamento da matriz de dados.
- `scikit-learn`: Implementação dos algoritmos `PCA` e `MDS`, além da normalização de dados (`StandardScaler`).
- `matplotlib` e `seaborn`: Visualização dos gráficos e plots 2D.

# Pipeline Integrado de Inteligência Artificial para Predição e Inferência de Risco de Queimadas Ambientais

**Disciplina:** Trabalho Integrado de Fundamentos de Inteligência Artificial  
**Instituição:** Instituto de Computação (ICOMP) — Universidade Federal do Amazonas (UFAM)  
**Domínio de Aplicação:** P4 — Meio Ambiente (Focos de Queimadas, Risco de Fogo e Meteorologia)  

---

##  Equipe de Desenvolvimento
* Nathan V. A. Maciel
* John K. de A. Castro
* Victor S. V. de Souza
* Heitor O. Rocha
* Lorenzo A. da C. Freitas

---

## Visão Geral do Projeto
Este repositório contém o código-fonte, datasets e a documentação completa de um pipeline de Inteligência Artificial modularizado em 6 etapas teóricas e práticas. O projeto utiliza dados meteorológicos e de sensoriamento remoto (baseados no histórico do INPE e INMET) para prever, inferir e tomar decisões sobre o risco de incêndios florestais e focos de queimadas.

###  Estrutura do Pipeline em 6 Etapas

1. **Etapa 1 — Indução de Árvores de Decisão:** Algoritmo `LEARN-DECISION-TREE` implementado do zero com cálculo manual de Entropia de Shannon e Ganho de Informação, validação cruzada 5-fold vs. `scikit-learn` e análise de não-monotonicidade.
2. **Etapa 2 — Aprendizado Linear e Logístico:** Mínimos Quadrados Ordinários (MNO), Gradiente Descendente Multivariado com regularização Ridge ($L_2$) e Regressão Logística para classificação de emergência ambiental.
3. **Etapa 3 — Teoria da Probabilidade Aplicada:** Tabela de Distribuição Conjunta, verificação dos Axiomas de Kolmogorov, probabilidades condicionais/marginais, validação da Regra da Cadeia e diagnóstico via Teorema de Bayes.
4. **Etapa 4 — Aprendizado Probabilístico e Algoritmo EM:** Estimadores MLE e MAP (prior Beta), Naive Bayes Gaussiano do zero e algoritmo Expectation-Maximization (EM / GMM) para identificação da variável oculta microclimática ($Z$).
5. **Etapa 5 — Redes Bayesianas e Inferência Exata:** Grafo Acíclico Dirigido (DAG) de 5 nós, inferência por enumeração manual (`EnumerationInference`) vs. `pgmpy`, demonstração de d-separabilidade/nó colisor (*explaining away*) e redução de $58\%$ na complexidade de parâmetros.
6. **Etapa 6 — Sistema de Inferência Fuzzy:** Sistema Fuzzy de Mamdani alimentado pelas probabilidades a posteriori da Rede Bayesiana e velocidade do vento, fornecendo tomada de decisão suave via defuzzificação por Centroide.

---

##  Estrutura do Repositório

```text
trabalho-ia-ufam/
│
├── data/                                 # Datasets do projeto
│   └── queimadas_meteorologia.csv        # Dataset sintético parametrizado (N=1000)
│
├── notebooks/                            # Notebooks Jupyter auto-contidos
│   ├── etapa1_arvore_decisao.ipynb
│   ├── etapa2_aprendizado_linear.ipynb
│   ├── etapa3_probabilidade.ipynb
│   ├── etapa4_naive_bayes_em.ipynb
│   ├── etapa5_redes_bayesianas.ipynb
│   └── etapa6_logica_fuzzy.ipynb
│
├── src/                                  # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── utils.py                          # Geração de dados, semente e cross-validation
│   ├── em.py                             # Algoritmo Expectation-Maximization (GMM)
│   └── fuzzy.py                          # Sistema de Inferência Fuzzy (Mamdani)
│
├── docs/                                 # Documentação formal do artigo
│   └── relatorio_final.tex               # Relatório científico em LaTeX (Artigo 6-8 págs)
│
├── requirements.txt                      # Lista de dependências Python
└── README.md                             # Guia de reprodução do projeto

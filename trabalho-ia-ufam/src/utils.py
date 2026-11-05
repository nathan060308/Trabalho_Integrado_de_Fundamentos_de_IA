# src/utils.py
import numpy as np
import pandas as pd


def fix_seed(seed=42):
    """Garante a reprodutibilidade dos resultados em todos os experimentos."""
    np.random.seed(seed)


def gerar_dados_sinteticos_queimadas(n_samples=1000, seed=42):
    """Gera um DataFrame sintético que simula a relação entre variáveis meteorológicas

    e o risco de fogo do INMET/INPE para testes do pipeline.
    """
    fix_seed(seed)

    dias_sem_chuva = np.random.poisson(lam=12, size=n_samples)
    precipitacao = np.maximum(
        0, 50 - dias_sem_chuva * 3 + np.random.normal(0, 10, n_samples)
    )
    temperatura_max = np.random.normal(loc=33, scale=4, size=n_samples)
    umidade_relativa = np.clip(
        75 - dias_sem_chuva * 2.5 + np.random.normal(0, 8, n_samples), 10, 95
    )
    velocidade_vento = np.random.gamma(shape=2, scale=5, size=n_samples)

    # Fórmula empírica para o risco de fogo (0.0 a 1.0)
    score_risco = (
        0.35 * (dias_sem_chuva / 30)
        + 0.25 * (temperatura_max / 45)
        + 0.25 * (1 - umidade_relativa / 100)
        + 0.15 * (velocidade_vento / 30)
        - 0.20 * (precipitacao / 50)
    )
    risco_fogo = np.clip(score_risco + np.random.normal(0, 0.05, n_samples), 0, 1)

    # Variável-alvo binária: Risco Crítico (1 se risco_fogo >= 0.8, senao 0)
    risco_critico = (risco_fogo >= 0.8).astype(int)

    df = pd.DataFrame(
        {
            "dias_sem_chuva": dias_sem_chuva,
            "precipitacao": precipitacao,
            "temperatura_max": temperatura_max,
            "umidade_relativa": umidade_relativa,
            "velocidade_vento": velocidade_vento,
            "risco_fogo": risco_fogo,
            "risco_critico": risco_critico,
        }
    )

    return df


def cross_validation_split(X, y, k=5, seed=42):
    """Divide os dados em K Folds de forma aleatória e reprodutível."""
    fix_seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    fold_sizes = np.full(k, len(X) // k, dtype=int)
    fold_sizes[: len(X) % k] += 1

    current = 0
    folds = []
    for size in fold_sizes:
        start, stop = current, current + size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop

    return folds
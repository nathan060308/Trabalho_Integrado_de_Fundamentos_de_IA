# src/fuzzy.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def criar_sistema_fuzzy_queimadas():
    """Cria e constrói o Sistema de Controle Fuzzy (Mamdani) para tomada de decisão ambiental."""
    # 1. Definir Variáveis de Entrada (Antecedentes) e Saída (Consequente)
    prob_bayesiana = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01), "prob_bayesiana"
    )
    vento = ctrl.Antecedent(np.arange(0, 50.1, 0.5), "vento")
    alerta_ambiental = ctrl.Consequent(np.arange(0, 100.1, 1), "alerta")

    # 2. Funções de Pertinência (Triangulares)
    prob_bayesiana["baixa"] = fuzz.trimf(
        prob_bayesiana.universe, [0.0, 0.0, 0.4]
    )
    prob_bayesiana["media"] = fuzz.trimf(
        prob_bayesiana.universe, [0.2, 0.5, 0.8]
    )
    prob_bayesiana["alta"] = fuzz.trimf(
        prob_bayesiana.universe, [0.6, 1.0, 1.0]
    )

    vento["fraco"] = fuzz.trimf(vento.universe, [0, 0, 15])
    vento["moderado"] = fuzz.trimf(vento.universe, [10, 25, 40])
    vento["forte"] = fuzz.trimf(vento.universe, [30, 50, 50])

    alerta_ambiental["baixo"] = fuzz.trimf(
        alerta_ambiental.universe, [0, 0, 40]
    )
    alerta_ambiental["medio"] = fuzz.trimf(
        alerta_ambiental.universe, [30, 50, 70]
    )
    alerta_ambiental["critico"] = fuzz.trimf(
        alerta_ambiental.universe, [60, 100, 100]
    )

    # 3. Base de Regras Se-Então (Mamdani)
    regra1 = ctrl.Rule(
        prob_bayesiana["alta"] & vento["forte"], alerta_ambiental["critico"]
    )
    regra2 = ctrl.Rule(
        prob_bayesiana["alta"] & vento["moderado"], alerta_ambiental["critico"]
    )
    regra3 = ctrl.Rule(
        prob_bayesiana["media"] & vento["forte"], alerta_ambiental["critico"]
    )
    regra4 = ctrl.Rule(
        prob_bayesiana["media"] & vento["moderado"], alerta_ambiental["medio"]
    )
    regra5 = ctrl.Rule(
        prob_bayesiana["baixa"] & ~vento["forte"], alerta_ambiental["baixo"]
    )

    sistema = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
    simulador = ctrl.ControlSystemSimulation(sistema)

    return simulador
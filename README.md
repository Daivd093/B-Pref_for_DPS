# Dueling Posterior Sampling – Versión Adaptada con Preferencias B-Pref

Este repositorio contiene una versión adaptada y extendida del código original de:

**Dueling Posterior Sampling for Preference-Based Reinforcement Learning**<br/>
*Conference on Uncertainty in Artificial Intelligence (UAI), 2020*<br/>
Ellen Novoseller, Yibing Wei, Yanan Sui, Yisong Yue y Joel W. Burdick<br/>
[Paper DPS](https://arxiv.org/abs/1908.01289)

Implementando un profesor simulado según las descripciones del benchmark:

**B-Pref: Benchmarking Preference-Based Reinforcement Learning**<br/>
*NeurIPS 2021 Datasets and Benchmarks Track*<br/>
Kimin Lee, Laura Smith, Anca Dragan, Pieter Abbeel<br/>
[Paper B-Pref](https://openreview.net/pdf?id=ps95-mkHF_)

## Estructura del repositorio

* `Learning_algorithms/`: Algoritmos implementados (DPS, EPMC, PSRL).
* `Envs/`: Entornos simulados, incluyendo versiones modificadas para BPref.
* Scripts principales en el formato `algoritmo_in_entorno.py` para ejecutar los distintos algoritmos de aprendizaje en la versión BPref de cada entorno.

## Reproducción de Resultados

Para reproducir el entorno de desarrollo:

```
conda env create -f environment.yml
conda activate bpref-dps-env
```

Luego se ejecuta el script que corresponde con el algoritmo de aprendizaje que se quiere probar en el ambiente que corresponda. La versión actual tiene todos los ambientes implementados para entregar preferencias B-Pref, pero solo se adjuntan scripts para correr los experimentos en RiverSwim.

## Referencias

\[1] E. Novoseller, Y. Wei, Y. Sui, Y. Yue y J. Burdick. *Dueling Posterior Sampling for Preference-Based Reinforcement Learning*. arXiv:1908.01289, 2020.<br/>
\[2] C. Wirth y J. Fürnkranz. *A policy iteration algorithm for learning from preference-based feedback*, 2013.<br/>
\[3] C. Wirth. *Efficient Preference-Based Reinforcement Learning*. Tesis doctoral, 2017.<br/>
\[4] I. Osband, D. Russo y B. Van Roy. *(More) efficient reinforcement learning via posterior sampling*, 2013.<br/>
\[5] K. Lee, L. Smith, A. Dragan y P. Abbeel. *B-Pref: Benchmarking Preference-Based Reinforcement Learning*. NeurIPS Datasets and Benchmarks Track, 2021.<br/>
[Repositorio B-Pref](https://github.com/rll-research/B-Pref)

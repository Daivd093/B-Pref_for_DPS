# Dueling Posterior Sampling – Versión Adaptada con Preferencias B-Pref

Este repositorio contiene una versión adaptada y extendida del código original de:

**Dueling Posterior Sampling for Preference-Based Reinforcement Learning**
*Conference on Uncertainty in Artificial Intelligence (UAI), 2020*
Ellen Novoseller, Yibing Wei, Yanan Sui, Yisong Yue y Joel W. Burdick
[Paper DPS](https://arxiv.org/abs/1908.01289)

Implementando un profesor simulado según las descripciones del benchmark:

**B-Pref: Benchmarking Preference-Based Reinforcement Learning**
*NeurIPS 2021 Datasets and Benchmarks Track*
Kimin Lee, Laura Smith, Anca Dragan, Pieter Abbeel
[Paper B-Pref] (https://openreview.net/pdf?id=ps95-mkHF_)

---

## Contenidos del Repositorio

Este código fue desarrollado como parte de un trabajo de titulación para la carrera de Ingeniería Civil Electrónica, con el objetivo de:

* Replicar el algoritmo Dueling Posterior Sampling (DPS) para Aprendizaje Reforzado Basado en Preferencias.
* Implementar una versión modificada de los entornos para entregar preferencias utilizando el benchmark BPref.
* Se implementó manualmente un profesor simulado inspirado en el benchmark B-Pref (Lee et al., NeurIPS 2021), con el objetivo de evaluar la robustez del algoritmo frente a preferencias imperfectas. Esto permite emular comportamientos humanos no racionales, como inconsistencias o ruido en las comparaciones.

---

## Sobre Dueling Posterior Sampling

DPS es un enfoque bayesiano que permite aprender simultáneamente:

* La dinámica del entorno.
* La función de utilidad que origina el feedback en forma de preferencias (en lugar de recompensas absolutas).

Se presentan varias estrategias de asignación de crédito para interpretar esas preferencias, incluyendo:

* Regresión lineal bayesiana
* Regresión logística bayesiana
* Modelos de preferencia con proceso gaussiano (GP)
* Regresión con GP

---

## 📁 Estructura del repositorio

* `Learning_algorithms/`: Algoritmos implementados (DPS, EPMC, PSRL).
* `Envs/`: Entornos simulados, incluyendo versiones modificadas para BPref.
* Scripts principales en el formato `algoritmo_in_entorno.py` para ejecutar los distintos algoritmos de aprendizaje en la versión BPref de cada entorno.

---

## Reproducción de Resultados

Para reproducir el entorno de desarrollo:

```
conda env create -f environment.yml
conda activate bpref-dps-env
```

Luego se ejecuta el script que corresponde con el algoritmo de aprendizaje que se quiere probar en el ambiente que corresponda. La versión actual tiene todos los ambientes implementados para entregar preferencias B-Pref, pero solo se adjuntan scripts para correr los experimentos en RiverSwim.

---

## 📄 Referencias

\[1] E. Novoseller, Y. Wei, Y. Sui, Y. Yue y J. Burdick. *Dueling Posterior Sampling for Preference-Based Reinforcement Learning*. arXiv:1908.01289, 2020.
\[2] C. Wirth y J. Fürnkranz. *A policy iteration algorithm for learning from preference-based feedback*, 2013.
\[3] C. Wirth. *Efficient Preference-Based Reinforcement Learning*. Tesis doctoral, 2017.
\[4] I. Osband, D. Russo y B. Van Roy. *(More) efficient reinforcement learning via posterior sampling*, 2013.
\[5] K. Lee, L. Smith, A. Dragan y P. Abbeel. *B-Pref: Benchmarking Preference-Based Reinforcement Learning*. NeurIPS Datasets and Benchmarks Track, 2021.
[Repositorio B-Pref](https://github.com/rll-research/B-Pref)

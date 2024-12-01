# Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis

This repository contains the code for the paper "Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis" by Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros and Catarina Moreira. The paper is accepted at journal [Intelligent Systems with Applications](https://www.sciencedirect.com/journal/intelligent-systems-with-applications). The preprint version of the paper can be found on [arXiv](https://arxiv.org/abs/2407.11463).

## Abstract

Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.


## Data Profiling

| Dataset       	| Data Type 	| Total Inst. 	| Train/Test<br>(80%:20%) 	| Batch/Adv Inst.<br>(batch_size=64) 	| Total Feat. 	| Categorical Feat. 	| Numerical Feat. 	| Total Categorical Feat.<br>after One Hot Enc. 	|
|---------------	|:---------:	|:-----------:	|:-----------------------:	|:----------------------------------:	|:-----------:	|:-----------------:	|:---------------:	|:---------------------------------------------:	|
| Adult/Income  	|   Mixed   	|    32651    	|        26048/6513       	|              101/6464              	|     12      	|         8         	|        4        	|                       98                      	|
| Breast Cancer 	|    Num    	|     569     	|         455/114         	|                1/64                	|      30     	|         0         	|        30       	|                       0                       	|
| COMPAS        	|   Mixed   	|     7214    	|        5771/1443        	|               22/1408              	|      11     	|         7         	|        4        	|                       19                      	|
| Diabetes      	|    Num    	|     768     	|         614/154         	|                2/128               	|      8      	|         0         	|        8        	|                       0                       	|
| German Credit 	|   Mixed   	|     1000    	|         800/200         	|                3/192               	|      20     	|         15        	|        5        	|                       58                      	|


## Predictive Models

- All model parameters can found in [utils/models.py](./utils/models.py)
- Model training in [1_AE_model_training.ipynb](./1_AE_model_training.ipynb)

## Adversarial Attacks

- Attack methods and parameters in:
    - [DeepFool](./utils/deepfool.py)
    - [LowProFool](./utils/lowprofool.py)
    - [C&W](./utils/carlini.py)
    - [FGSM](./utils/fgsm.py)
    <!-- - [BIM](./utils/bim.py)
    - [MIM](./utils/mim.py) -->
    - [PGD](./utils/pgd.py)
- Generate adversarial examples by [2_generate_ae.py](./2_generate_ae.py) and [generate_ae.bat](./generate_ae.bat)
- Adversarial examples in folder [results](./results/)

## Evaluation

- Evaluated in [3_AE_performance_metrics.ipynb](./3_AE_performance_metrics.ipynb)
- Raw results can be found in folder [results](./results/)
- formatted result tables of qualitative analysis can be found in [qualitative analysis](./qualitative_analysis.md)
- Some visualisations are produced in [5_visualisation.ipynb](./5_visualisation.ipynb), [5_visualisation2.ipynb](./5_visualisation2.ipynb) and [5_visualisation_poster.ipynb](./5_visualisation_poster.ipynb). The visualisations are saved in folder [Visualisation](./Visualisation/).

## Poster

- The poster can be found in [Poster](./Poster/)
- Check more visualisations in markdown file [visualisation.md](./visualisation.md)

## XAMI Lab

- [XAMI Lab](https://www.xami-lab.org/)



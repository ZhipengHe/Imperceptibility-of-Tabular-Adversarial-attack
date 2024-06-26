# Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis


## Abstract

Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. Adversarial attacks are extensively explored in the literature for image data but not for tabular data, which is low-dimensional and heterogeneous. To address this gap, we propose a set of properties for evaluating the imperceptibility of adversarial attacks on tabular data. These properties are defined to capture four perspectives of perturbed data: *proximity* to the original input, *sparsity* of alterations, *deviation* to datapoints in the original dataset, and *sensitivity* of altering sensitive features. We evaluate the performance and imperceptibility of seven white-box adversarial attack methods and their variants using different machine learning models on tabular data. As an insightful finding from our evaluation, it is challenging to craft adversarial examples that are both the most effective and least perceptible due to a trade-off between imperceptibility and performance. Furthermore, optimization-based attacks, such as the C\&W $\ell_2$ attack, are preferred as the primary choice for crafting imperceptible adversarial examples on tabular data.


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
- formatted result tables can be found in [Supplementary_Material](./Supplementary_Material.pdf) and [qualitative analysis](./qualitative_analysis.md)
- Some visualisations are produced in [5_visualisation.ipynb](./5_visualisation.ipynb) and [5_visualisation2.ipynb](./5_visualisation2.ipynb). The visualisations are saved in folder [Visualisation](./Visualisation/)

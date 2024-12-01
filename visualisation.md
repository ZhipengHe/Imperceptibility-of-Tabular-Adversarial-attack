## Visualisation

This markdown file contains the visualisations produced in the notebooks [5_visualisation.ipynb](./5_visualisation.ipynb), [5_visualisation2.ipynb](./5_visualisation2.ipynb) and [5_visualisation_poster.ipynb](./5_visualisation_poster.ipynb). The visualisations are saved in folder [Visualisation](./Visualisation/).

### Visualisation 1: Effectiveness of Adversarial Attacks on Tabular Data - Attack Success Rate

The following visualisation shows the attack success rate of different adversarial attacks on tabular data. The attack success rate is calculated as the percentage of adversarial examples generated by the attack that are misclassified by the target model.

![Attack Success Rate](./Visualisation/asr_rgb.png)

### Visualisation 2: Imperceptibility of Adversarial Attacks on Tabular Data - Proximity to Original Data

The following visualisation shows the proximity of adversarial examples generated by different attacks to the original data. The proximity is calculated as the $\ell_2$ and $\ell_\infty$ distance between the original data and the adversarial example.

#### $\ell_2$ Distance

![L2 Distance](./Visualisation/l2_rgb.png)

#### $\ell_\infty$ Distance

![Linf Distance](./Visualisation/linf_rgb.png)

### Visualisation 3: Imperceptibility of Adversarial Attacks on Tabular Data - Sparsity of Altered Features

The following visualisation shows the sparsity of altered features in adversarial examples generated by different attacks. The sparsity is calculated as the percentage of features that are altered in the adversarial example.

![Sparsity of Altered Features](./Visualisation/sparsity_rgb.png)

For three mixed datasets, the sparsity of altered features is in a very low range, which is different from the other two numerical datasets. Here we break down the sparsity of altered features for both categorical and numerical features.

#### Adult/Income Dataset

![Sparsity of Altered Features - Adult/Income](./Visualisation/sparsity_adult.png)

#### COMPAS Dataset

![Sparsity of Altered Features - COMPAS](./Visualisation/sparsity_compas.png)

#### German Credit Dataset

![Sparsity of Altered Features - German Credit](./Visualisation/sparsity_german.png)

### Visualisation 4: Imperceptibility of Adversarial Attacks on Tabular Data - Deviation from Original Data Distribution (Deviation)

The following visualisation shows the deviation of adversarial examples generated by different attacks from the original data distribution. The deviation is calculated as the MD distance between the original data distribution and the adversarial example.

![Deviation from Original Data Distribution](./Visualisation/deviation_rgb.png)

### Visualisation 5: Imperceptibility of Adversarial Attacks on Tabular Data - Sensitivity in Perturbing Features with Narrow Distribution (Sensitivity)

The following visualisation shows the sensitivity of adversarial attacks in perturbing features with narrow distribution. The sensitivity is calculated as the percentage of features with a narrow distribution that are altered in the adversarial example.

![Sensitivity in Perturbing Features with Narrow Distribution](./Visualisation/sensitivity_rgb.png)


### Visualisation 6: Imperceptibility of Adversarial Attacks on Tabular Data - Immutability of Certain Features (Immutability) & Feature Interdependencies (Interdependencies)

The following visualisation shows the immutability of certain features and the feature interdependencies in adversarial examples generated by different attacks. 

![Immutability of Certain Features & Feature Interdependencies](./Visualisation/tab-immutable.png)

Weights of Logistic Regression Model for Compas Dataset:

![Weights of Logistic Regression Model for Compas Dataset](./Visualisation/COMPAS_Weights.png)

### Visualisation 7: Imperceptibility of Adversarial Attacks on Tabular Data - Feasibility of Specific Feature Values (Feasibility)

The following visualisation shows the feasibility of specific feature values in adversarial examples generated by different attacks. Here use case-based examples are provided to illustrate the feasibility of specific feature values.

![Feasibility of Specific Feature Values](./Visualisation/tab-feasibility.png)

Corresponding feature values are provided in the following table:

![Feasibility of Specific Feature Values - Table](./Visualisation/appendix-diabetes.png)

Weights of Logistic Regression Model for Diabetes Dataset:

![Weights of Logistic Regression Model for Diabetes Dataset](./Visualisation/Diabetes_Weights.png)

### Visualisation 8: The trade-off between imperceptibility and effectiveness of adversarial attacks

We compare the successful adversarial examples vs unsuccessful adversarial examples in terms of imperceptibility metrics. The following visualisation shows the trade-off between imperceptibility and effectiveness of adversarial attacks on tabular data.

#### Proximity $\ell_2$ Distance

![Trade-off - Proximity L2 Distance](./Visualisation/l2_distance_boxplot.png)

#### Deviation

![Trade-off - Deviation](./Visualisation/deviation_boxplot.png)

#### Sensitivity

![Trade-off - Sensitivity](./Visualisation/sensitivity_boxplot.png)
# Essential information for attack packages

## Adversarial Robustness Toolbox (ART) v1.12.2

- [Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)
- [Github](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/1.12.2)


## Which predictive models can be used in ART?

|                     	| **Type**          	|  **Decision Tree** 	|  **Random Forest** 	|   **Linear SVC**   	| **Logistic Regression** 	| **Neural Networks** 	|
|---------------------	|-------------------	|:------------------:	|:------------------:	|:------------------:	|:-----------------------:	|:-------------------:	|
| **DeepFool**        	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **LowProFool**      	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **C&W Attack**      	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **Boundary Attack** 	| Black Box Attack   	| :heavy_check_mark: 	| :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **HopSkipJump Attack**| Black Box Attack   	| :heavy_check_mark:    | :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
<!-- | **Sign-OPT Attack**   | Black Box Attack   	| :heavy_check_mark:    | :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	| -->

<details><summary>Classifier types in ART</summary>
<p>

```python
CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
    ClassifierClassLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
    ClassifierNeuralNetwork,
    DetectorClassifier,
    EnsembleClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=C0103
    ClassifierDecisionTree,
    LightGBMClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    XGBoostClassifier,
]

CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
    Classifier,
    BlackBoxClassifier,
    CatBoostARTClassifier,
    DetectorClassifier,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    JaxClassifier,
    LightGBMClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
    XGBoostClassifier,
    CLASSIFIER_NEURALNETWORK_TYPE,
]
```

</p>
</details>

## Current problems in ART

### DeepFool

When the neural network only have one output node, it has the following error:

```
ValueError: This attack has not yet been tested for binary classification with a single output classifier.
```

The error is from ([source code](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/deepfool.py#L104-L107)):

```python
if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
    raise ValueError(  # pragma: no cover
        "This attack has not yet been tested for binary classification with a single output classifier."
    )
```

https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons

Hence, we set up two neural networks in our project, "nn" and "nn_2".

```python
nn = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(24,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation(tf.nn.sigmoid),
        ]
    )
nn.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

nn_2 = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(24,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation(tf.nn.softmax),
        ]
    )
nn_2.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_2.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)
```

Two neural networks have similar settings but the actiavtion function of last layer and loss fuction used for compiling models. 

### LowProFool

From the experiment settings from original paper (https://arxiv.org/abs/1911.03274), `LowProFool` can only work on numerical features and continous features, all non-ordered categorical features need to be dropped. Hence, we only use dataset with numerical features `["diabetes", "breast_cancer"]`.

Another problem is that `art.attacks.evasion.LowProFool` doesn't work for $\ell_p$ norm when $p \in (0,2)$. However, it works when $p\ge 2$ or $p=\infty$. I have report this bug to ART. The detail of this problem can check the [issue page](https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/1970) on ART GitHub.


**To Reproduce**
Steps to reproduce the behavior:
1. Go to my notebook ([gist](https://gist.github.com/ZhipengHe/ff432a368f720c3504ec64398137bb39))
2. When set the `norm>=2` or  `norm='inf'` , the attack model works well. For example,
  ```python
  success_rate=test_general_cancer_lr(breast_cancer_dataset(splitter()), norm=2)
  print(success_rate)
  ```
Result is :
  ```
  1.0
  ```
3. When set the `0<norm<2`, the attack model doesn't work. For example,
```python
success_rate=test_general_cancer_lr(breast_cancer_dataset(splitter()), norm=1)
print(success_rate)
```
Error:
```
/usr/local/lib/python3.8/dist-packages/art/attacks/evasion/lowprofool.py:159: RuntimeWarning: divide by zero encountered in power
  self.importance_vec * self.importance_vec * perturbations * np.power(np.abs(perturbations), norm - 2)
/usr/local/lib/python3.8/dist-packages/art/attacks/evasion/lowprofool.py:159: RuntimeWarning: invalid value encountered in multiply
  self.importance_vec * self.importance_vec * perturbations * np.power(np.abs(perturbations), norm - 2)
...
...
ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
```
From `RuntimeWarning: divide by zero encountered in power`,
In `LowProFool` [L307-L313](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/lowprofool.py#L307-L313) , the attack model initialized the perturbation with `np.zero`
```python
# Initialize perturbation vectors and learning rate.
perturbations = np.zeros(samples.shape, dtype=np.float64)
eta = self.eta

# Initialize 'keep-the-best' variables.
best_norm_losses = np.inf * np.ones(samples.shape[0], dtype=np.float64)
best_perturbations = perturbations.copy()
```
In `LowProFool` [L148-L171](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/lowprofool.py#L148-L171) , when `0< norm <2`, it will encounter ` divide by zero` error.

```python
  numerator = (
      self.importance_vec * self.importance_vec * perturbations * np.power(np.abs(perturbations), norm - 2)
  )
```


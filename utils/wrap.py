from art.estimators.classification import (
    SklearnClassifier, 
    KerasClassifier, 
    PyTorchClassifier
    )
from art.estimators.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC
    )



def art_wrap_models(models, feature_range):
    '''
    Wrap the model to meet the requirements to art.attacks.evasion
    '''

    return {
        'lr': ScikitlearnLogisticRegression(models['lr'], clip_values=feature_range),
        'svc': ScikitlearnSVC(models['svc'], clip_values=feature_range),
        'nn_2': KerasClassifier(models['nn_2'], clip_values=feature_range),
        
    }
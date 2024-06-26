from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier


def generate_deepfool(model, feature_range, num_instances, X_test, y_test):

    print("Feature range:" )
    print(feature_range)

    X_test_re=X_test[0:num_instances]
    y_test_re=y_test[0:num_instances]

    deepfool_ae = DeepFool(classifier=KerasClassifier(model), max_iter=1000, verbose= True, batch_size=64)
    ae = deepfool_ae.generate(X_test_re,y_test_re) 

    return ae

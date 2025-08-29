from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    weights = model.linear.weight.data.numpy().flatten()
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': weights})
    feature_importance = feature_importance.reindex(feature_importance.Importance.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

def get_feature_importance(model, X):
    weights = model.linear.weight.data.numpy().flatten()
    return pd.Series(weights, index=X.columns)
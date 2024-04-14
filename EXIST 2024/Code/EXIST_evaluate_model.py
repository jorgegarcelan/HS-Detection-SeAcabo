import itertools
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def eval_data(run_id, best_model, X_test, y_test, labels_names):
    y_pred = best_model.predict(X_test)
    acc_global = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, target_names=labels_names)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels_names))
    plt.xticks(tick_marks, labels_names, rotation=45)
    plt.yticks(tick_marks, labels_names)

    # Annotate the matrix with text
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    plt.savefig(f"../Conf_matrix/{run_id}.png")
    plt.close()

    return acc_global, report
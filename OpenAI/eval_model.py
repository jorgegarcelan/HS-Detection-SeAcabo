import itertools
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



def eval_data(run_id, y_test, y_pred, labels_names):

    # Ensure y_test and y_pred are of type int
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
    
    # Acc
    acc_global = accuracy_score(y_test, y_pred)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_names)

    # Calculate per-class accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[~np.isfinite(per_class_acc)] = 0  # Replace infinities and NaNs with 0

    # SD
    std_global = np.std(per_class_acc)

    # SD
    std_global = np.std(per_class_acc)

    # F1
    # f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, output_dict=True, target_names=labels_names)

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
    plt.savefig(f"Conf_matrix/{run_id}.png")
    plt.close()

    return acc_global, std_global, report
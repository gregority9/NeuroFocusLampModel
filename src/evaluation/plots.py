from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt

def save_confusion_matrix_plot(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["rest", "task"]
        )

    disp.plot(values_format="d")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_roc_curve_plot(y_true, y_score, output_path):
    """
    Save ROC curve plot for binary classification.
    y_score should contain probabilities or decision scores for class 1.
    """

    if y_score is None:
        return

    if len(set(y_true)) < 2:
        return

    auc = roc_auc_score(y_true, y_score)

    RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_score=y_score,
        name="ROC Curve"
    )

    plt.title("ROC Curve (AUC = " + str(round(auc,3)) + ")")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, auc
from sklearn.preprocessing import LabelEncoder, label_binarize


# def init_ROC(self):
#     df_train = pd.DataFrame(self.reduced_data)
#
#     # print(len(self.reduced_y_train))
#     df_train['labels'] = self.labels
#     # print(df_train.head())
#     df_test = pd.DataFrame(self.reduced_y_test)
#     df_test['labels'] = self.test_labels
#
#     self.y_pred = self.knn.predict(df_test.drop(columns='labels'))
#     self.y_distanced, _ = self.knn.kneighbors(df_test.drop(columns='labels'))
#     print(f"self.y_distanced {self.y_distanced.shape}")
#     y_true = np.array(df_test['labels'])
#     y_distances = np.array(self.y_distanced)
#     label_encoder = LabelEncoder()
#     y_true_binary = label_encoder.fit_transform(y_true)
#     # print(len(y_true, y_distances))
#
#     self.draw_roc_multi_curves(y_true_binary, y_distances, n_classes=5)
#     cr = classification_report(df_test['labels'], self.y_pred)
#     print(cr)


def compute_roc_curve(self, y_true, y_distances):
    # Sort instances based on distances in descending order
    sorted_indices = np.argsort(y_distances)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_distances_sorted = y_distances[sorted_indices]

    # Since positive instances are represented as 1 and negative instances as 0
    # Then, the total number of positive instances
    total_positives = np.sum(y_true)
    # And, the total number of negative instances
    total_negatives = len(y_true) - total_positives

    TPR_values = []  # True Positive Rate
    FPR_values = []  # False Positive Rate

    # Calculate TPR and FPR for different classification thresholds
    T_positive_count = 0
    F_positive_count = 0

    for i in range(len(y_distances_sorted)):
        if y_true_sorted[i] == 1:
            T_positive_count += 1
        else:
            F_positive_count += 1

        TPR = T_positive_count / total_positives
        FPR = F_positive_count / total_negatives

        TPR_values.append(TPR)
        FPR_values.append(FPR)

    return FPR_values, TPR_values


def generate_random_color(self):
    """
    Generate a random color in RGB format. Used later for plotting the ROC curve.

    Returns:
        tuple: A tuple representing the RGB values of the random color.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


def draw_roc_multi_curves(self, y_true, y_distances, n_classes):
    # Binarize the true labels, label_binarize will create three binary columns, one for each class.
    # Each row in these columns will have a 1 if the original label was that class and 0 otherwise.
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute the ROC curve and AUC (area under the curve) for each class
    TPR = dict()
    FPR = dict()
    roc_auc = dict()
    for i in range(n_classes):
        FPR[i], TPR[i] = self.compute_roc_curve(y_true_bin[:, i], y_distances[:, i])
        roc_auc[i] = auc(FPR[i], TPR[i])

    # Plot the ROC curves for each class
    colors = []
    plt.figure()
    # Generate colors for each class and plot it
    for i in range(n_classes):
        generated_color = self.generate_random_color()
        plt.plot(FPR[i], TPR[i], color=generated_color, lw=2, label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))
        # colors.append(color)
    # Choose colors for each class
    # for i in range(n_classes):
    #     plt.plot(FPR[i], TPR[i], color=colors[i], lw=2, label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
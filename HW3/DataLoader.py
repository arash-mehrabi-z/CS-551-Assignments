import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Do not change the parameters.
RANDOM_STATE = 123
DATASET_NAME = 'mnist_784'
TEST_SIZE = 0.2


def get_data(display_details: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
        This method fetches the corresponding dataset and splits train and test sets. Also, it displays the details
        of the dataset.
        DO NOT CHANGE THE METHOD!
    :return: X_train, X_test, Y_train, Y_test as NumPy Array.
    """
    X, Y = fetch_openml(DATASET_NAME, version=1, return_X_y=True, as_frame=False)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=True,
                                                        random_state=RANDOM_STATE, stratify=Y)

    if display_details:
        print("Train Size:", X_train.shape[0], "\tTest Size:", X_test.shape[0])
        print("Feature Size:", X_train.shape[1])

        train_unique, train_count = np.unique(Y_train, return_counts=True)
        train_labels = {label: train_count[i] for i, label in enumerate(train_unique)}
        test_unique, test_count = np.unique(Y_test, return_counts=True)
        test_labels = {label: test_count[i] for i, label in enumerate(test_unique)}

        all_labels = sorted(list(train_labels.keys() | test_labels.keys()))

        print("\t\tClass Distribution:")
        for label in all_labels:
            print(f"Label: {label}\tTrain Set: {train_labels.get(label, 0)}\tTest Set: {test_labels.get(label, 0)}")

        plt.bar(train_labels.keys(), train_labels.values(), label="Train")
        plt.bar(test_labels.keys(), test_labels.values(), label="Test")
        plt.title("Class Distribution")
        plt.xticks(all_labels)
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig("class_distribution.png")
        plt.show()
        plt.close()

    return X_train, X_test, Y_train, Y_test


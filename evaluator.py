from knn import classify


def evaluate(training_data, test_data, k):
    """
    Evaluates k-NN classifier on the test data.

    If test samples have labels:
        - Compares predictions with true labels
        - Prints correct, errors, accuracy
        - Returns accuracy as float (0.0 - 100.0)

    If test samples have no labels (label=None):
        - Only prints predictions
        - Returns None

    Parameters:
        training_data : list of dicts with 'attributes' and 'label'
        test_data     : list of dicts with 'attributes' and 'label' or None
        k             : number of neighbours

    Returns:
        float : accuracy percentage, or None if no labels
    """

    has_labels = test_data[0]["label"] is not None

    if has_labels:
        return _evaluate_with_labels(training_data, test_data, k)
    else:
        return _evaluate_without_labels(training_data, test_data, k)


def _evaluate_with_labels(training_data, test_data, k):
    """
    Evaluates when test data has true labels.
    Prints full report: correct, errors, accuracy.
    """
    total   = len(test_data)
    correct = 0
    errors  = 0

    print()
    print(f"{'#':<5} {'True Label':<25} {'Predicted':<25} {'Result'}")
    print("-" * 65)

    for i, sample in enumerate(test_data, 1):
        predicted  = classify(training_data, sample, k)
        true_label = sample["label"]

        if predicted == true_label:
            correct += 1
            status = "correct"
        else:
            errors += 1
            status = "WRONG"

        print(f"{i:<5} {true_label:<25} {predicted:<25} {status}")

    accuracy = (correct / total) * 100

    print("-" * 65)
    print(f"k                   : {k}")
    print(f"Total samples       : {total}")
    print(f"Correctly classified: {correct}")
    print(f"Errors              : {errors}")
    print(f"Accuracy            : {accuracy:.2f}%")
    print("-" * 65)

    return accuracy


def _evaluate_without_labels(training_data, test_data, k):
    """
    Evaluates when test data has no labels.
    Only prints predicted class for each sample.
    """
    print()
    print(f"{'#':<5} {'Attributes':<40} {'Predicted'}")
    print("-" * 65)

    for i, sample in enumerate(test_data, 1):
        predicted  = classify(training_data, sample, k)
        attrs      = str(sample["attributes"])
        print(f"{i:<5} {attrs:<40} {predicted}")

    print("-" * 65)
    print("(No accuracy available — test data has no labels)")
    print("-" * 65)
import matplotlib.pyplot as plt
from data_loader import load_data, get_num_attributes
from evaluator import evaluate
from knn import classify

# ── CONFIG ────
ASK_FILE_PATHS = False  # True  = ask user for file paths at runtime
                    # False = use hardcoded paths below
TRAINING_FILE  = "iris_training 1.txt"
TEST_FILE      = "iris_test 1.txt"
HAS_LABEL      = True   # False if test file has no decision attribute
CHART_MAX_K    = None   # None = full training set size, or e.g. 20
CHART_ENABLED  = True   # False to skip chart entirely
DECIMAL_SEP    = ","    # decimal separator in data files ("," or ".")
GUI_ENABLED    = True  # True = launch GUI, False = terminal mode
# ────


def get_k_from_user(max_k):
    while True:
        try:
            k = int(input(f"Enter k (number of neighbours, max {max_k}): "))
            if k <= 0:
                print("  k must be a positive integer. Try again.")
            elif k > max_k:
                print(f"  k cannot exceed training set size ({max_k}). Try again.")
            else:
                return k
        except ValueError:
            print("  Invalid input. Please enter a whole number.")


def get_file_paths():

    while True:
        training_file = input("Enter training file path: ").strip()
        if training_file:
            break
        print("  File path cannot be empty.")

    while True:
        test_file = input("Enter test file path    : ").strip()
        if test_file:
            break
        print("  File path cannot be empty.")

    return training_file, test_file


def get_new_sample(num_attributes):
    attributes = []
    print()
    for i in range(1, num_attributes + 1):
        while True:
            try:
                raw   = input(f"  Enter value for attribute {i}: ")
                value = float(raw.replace(DECIMAL_SEP, "."))
                attributes.append(value)
                break
            except ValueError:
                print(f"  Invalid input. Please enter a number (e.g. 5{DECIMAL_SEP}1).")
    return {"attributes": attributes, "label": None}


def ask_yes_no(prompt):
    '''

     while True:
        answer = input(prompt).strip().lower()
        if answer in ("yes", "y"):
            return True
        elif answer in ("no", "n"):
            return False
        else:
            print("  Please answer yes or no.")
        '''

    answer = input(prompt).strip().lower()
    while answer not in ["yes", "no"]:
        print("Please answer yes or no.")
        answer = input(prompt).strip().lower() not in ("yes", "no")
    else:
        return answer in ("yes", "y")


def plot_accuracy_chart(training_data, test_data, current_k):
    if test_data[0]["label"] is None:
        print("  (No labels in test data — cannot plot accuracy chart)")
        return

    max_k = CHART_MAX_K if CHART_MAX_K is not None else len(training_data)
    max_k = min(max_k, len(training_data))
    ks    = list(range(1, max_k + 1))

    print(f"\n  Calculating accuracy for k=1 to {max_k}...")
    accuracies = []
    for k in ks:
        correct = sum(
            1 for s in test_data
            if classify(training_data, s, k) == s["label"]
        )
        accuracies.append((correct / len(test_data)) * 100)

    plt.figure(figsize=(12, 5))

    plt.plot(ks, accuracies,
             marker="o",
             color="steelblue",
             linewidth=2,
             markersize=4,
             label="Accuracy per k")

    if current_k <= max_k:
        plt.plot(current_k, accuracies[current_k - 1],
                 marker="o",
                 markersize=12,
                 color="red",
                 zorder=5,
                 label=f"Current k={current_k} ({accuracies[current_k - 1]:.1f}%)")

    min_acc = min(accuracies) - 5
    max_acc = min(100, max(accuracies) + 5)

    plt.title("k-NN Classification Accuracy vs k", fontsize=14)
    plt.xlabel("k (number of neighbours)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(min_acc, max_acc)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("        k-NN Classifier")
    print("=" * 60)

    # ── Step 1: Load data ────
    if ASK_FILE_PATHS:
        training_file, test_file = get_file_paths()
    else:
        training_file = TRAINING_FILE
        test_file     = TEST_FILE

    training_data  = load_data(training_file)
    test_data      = load_data(test_file, has_label=HAS_LABEL)
    num_attributes = get_num_attributes(training_data)
    max_k          = len(training_data)

    print(f"\nTraining file    : {training_file}")
    print(f"Test file        : {test_file}")
    print(f"Training samples : {len(training_data)}")
    print(f"Test samples     : {len(test_data)}")
    print(f"Attributes       : {num_attributes}")

    # ── Step 2: Get k from user ────
    print()
    k = get_k_from_user(max_k)

    # ── Step 3: Evaluate on test set ────
    print(f"\n--- Evaluating test set with k={k} ---")
    evaluate(training_data, test_data, k)

    # ── Step 4: Accuracy chart ────
    if CHART_ENABLED:
        chart_max = CHART_MAX_K if CHART_MAX_K is not None else max_k
        chart_max = min(chart_max, max_k)
        if ask_yes_no(f"\nShow accuracy chart for k=1 to {chart_max}? (yes/no): "):
            plot_accuracy_chart(training_data, test_data, current_k=k)

    # ── Step 5: Classify new samples (loop) ────
    print("\n--- Classify new samples ---")
    while True:
        sample = get_new_sample(num_attributes)
        result = classify(training_data, sample, k)

        print(f"\n  Predicted class : {result}")



        print()
        if not ask_yes_no("Classify another sample? (yes/no): "):
            break

    print("\nGoodbye!")
    print("=" * 60)


if __name__ == "__main__":
    if GUI_ENABLED:
        from knn_gui import KNNApp
        app = KNNApp()
        app.mainloop()
    else:
        main()
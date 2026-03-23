from knn import classify
from data_loader import load_data

#train = load_data('noisy_training.txt')
#test  = load_data('noisy_test.txt')
train = load_data('iris_training 1.txt')
test  = load_data('iris_test 1.txt')
k = int(input("Enter k: "))

print()
print(f"{'#':<5} {'True Label':<25} {'Predicted':<25} {'Result'}")
print("-" * 65)

correct = 0
for i, sample in enumerate(test, 1):
    predicted = classify(train, sample, k)
    true_label = sample["label"]
    status = "✓" if predicted == true_label else "✗"
    if predicted == true_label:
        correct += 1
    print(f"{i:<5} {true_label:<25} {predicted:<25} {status}")

total = len(test)
errors = total - correct
accuracy = (correct / total) * 100

print("-" * 65)
print(f"Total samples    : {total}")
print(f"Correct          : {correct}")
print(f"Errors           : {errors}")
print(f"Accuracy         : {accuracy:.2f}%")

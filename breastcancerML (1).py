
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the dataset.
data = pd.read_csv("wdbc.data", header=None)

print(data.head())
print("Dataset shape:", data.shape)

# Separate features and target.
X = data.iloc[:, 2:]
y = data.iloc[:, 1].map({"B": "Benign", "M": "Malignant"})

print("Features shape:", X.shape)
print("Target distribution:")
print(y.value_counts())

# Test proportions: 20%, 30%, and 25%.
split_ratios = [0.20, 0.30, 0.25]
os.makedirs("outputs", exist_ok=True)

results = []

for test_size in split_ratios:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    train_percent = int((1 - test_size) * 100)
    test_percent = int(test_size * 100)
    split_name = str(train_percent) + ":" + str(test_percent)

    print("\nTrain-test split:", split_name)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Scale using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the five classifiers.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(),
    }

    for name, model in models.items():

        # Train and predict.
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Calculate performance metrics.
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred,
            average="weighted", zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="weighted", zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            average="weighted", zero_division=0
        )

        print("\nModel:", name)
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        # Generate the confusion matrix.
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["Benign", "Malignant"]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )

        display.plot()
        plt.title(name + " - " + split_name)
        plt.tight_layout()

        image_name = (
            name.lower().replace(" ", "_")
            + "_" + str(train_percent)
            + "_" + str(test_percent)
            + ".png"
        )

        plt.savefig("outputs/" + image_name, dpi=300)
        plt.close()

        print("Confusion matrix saved:", image_name)

        # Store the four metrics.
        results.append({
            "Split": split_name,
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        })

# Create the final comparison table.
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nComparison saved to outputs/model_comparison.csv")

# Compare accuracy and F1 score across all splits.
comparison_charts = [
    ("Accuracy", "accuracy_comparison.png"),
    ("F1 Score", "f1_score_comparison.png"),
]

for metric, filename in comparison_charts:

    chart_data = results_df.pivot(
        index="Model",
        columns="Split",
        values=metric
    )

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(metric + " Comparison Across Train-Test Splits")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel(metric)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        ha="right"
    )

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Train:Test Split")
    plt.tight_layout()

    plt.savefig("outputs/" + filename, dpi=300)
    plt.close()

    print(filename, "saved successfully.")

# Compare mean accuracy across the three splits.
mean_accuracy = (
    results_df.groupby("Model", sort=False)["Accuracy"].mean() * 100
).sort_values(ascending=False)

ax = mean_accuracy.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Mean Accuracy Across All Three Splits")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Mean Accuracy (%)")

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=20,
    ha="right"
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    "outputs/mean_accuracy_comparison.png",
    dpi=300
)

plt.close()

print("mean_accuracy_comparison.png saved successfully.")
print("\nAll confusion matrices and comparison charts have been generated.")

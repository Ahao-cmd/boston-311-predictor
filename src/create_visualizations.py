from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Results from the 2024 Oct-Dec test set
metrics = [
    "Accuracy",
    "OVERDUE Recall",
    "OVERDUE F2",
    "ROC-AUC"
]

logistic_regression = [
    0.7822,
    0.697,
    0.592,
    0.8555
]

random_forest = [
    0.7803,
    0.711,
    0.600,
    0.8414
]


# Create output folder
output_dir = Path("assets")
output_dir.mkdir(exist_ok=True)


# Draw comparison chart
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    logistic_regression,
    width,
    label="Logistic Regression"
)

ax.bar(
    x + width / 2,
    random_forest,
    width,
    label="Random Forest"
)

ax.set_title("Boston 311 Predictor - Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

output_path = output_dir / "model_comparison.png"

plt.savefig(output_path, dpi=200)
plt.close()

print("Chart saved to:", output_path)

# Confusion Matrix Visualizations
confusion_matrices = [
    (
        "Logistic Regression",
        np.array([
            [35252, 8990],
            [2293, 5280]
        ]),
        "confusion_matrix_logistic.png"
    ),
    (
        "Random Forest",
        np.array([
            [35051, 9191],
            [2192, 5381]
        ]),
        "confusion_matrix_random_forest.png"
    )
]


for model_name, matrix, filename in confusion_matrices:

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw confusion matrix
    image = ax.imshow(matrix, cmap="Blues")

    # Labels
    labels = ["ONTIME", "OVERDUE"]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    # Display values inside the matrix
    for i in range(2):
        for j in range(2):

            count = matrix[i, j]

            percentage = (
                count / matrix[i].sum() * 100
            )

            text_color = (
                "white"
                if count > matrix.max() / 2
                else "black"
            )

            ax.text(
                j,
                i,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )

    fig.colorbar(image, ax=ax, label="Number of Cases")

    plt.tight_layout()

    output_path = output_dir / filename

    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Chart saved to:", output_path)
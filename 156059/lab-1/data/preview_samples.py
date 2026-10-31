"""Visualize sample images from BloodMNIST dataset."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "0: Basophil",
    "1: Eosinophil",
    "2: Erythroblast",
    "3: Immature Granulocyte",
    "4: Lymphocyte",
    "5: Monocyte",
    "6: Neutrophil",
    "7: Platelet",
]


def create_preview(data_path: Path, output_path: Path, samples_per_class: int = 4) -> None:
    data = np.load(data_path)
    train_images = data["train_images"]
    train_labels = data["train_labels"].flatten()

    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 2, num_classes * 2))

    for cls_idx in range(num_classes):
        match_indices = np.where(train_labels == cls_idx)[0]
        selected = match_indices[:samples_per_class]

        for col_idx, img_idx in enumerate(selected):
            ax = axes[cls_idx, col_idx]
            ax.imshow(train_images[img_idx])
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(CLASS_NAMES[cls_idx], fontsize=10, loc="left", weight="bold")

    plt.suptitle("BloodMNIST Sample Visualizations (28x28 RGB)", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Sample preview successfully saved to: {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "bloodmnist.npz"
    output_file = base_dir / "bloodmnist_preview.png"

    if not data_file.exists():
        print(f"Error: {data_file} not found. Please run dataset_downloader.py first.")
    else:
        create_preview(data_file, output_file)


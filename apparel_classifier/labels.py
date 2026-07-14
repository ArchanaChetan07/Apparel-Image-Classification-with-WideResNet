"""Fashion-MNIST class catalog."""

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

NUM_CLASSES = len(CLASS_NAMES)


def class_name(index: int) -> str:
    if index < 0 or index >= NUM_CLASSES:
        raise ValueError(f"class index {index} out of range [0, {NUM_CLASSES})")
    return CLASS_NAMES[index]

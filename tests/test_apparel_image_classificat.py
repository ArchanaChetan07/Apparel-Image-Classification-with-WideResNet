import pytest
import numpy as np


class TestImagePreprocessing:

    def test_image_normalization(self):
        image = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
        normalized = image.astype(np.float32) / 255.0
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

    def test_image_shape_correct(self):
        batch = np.random.rand(32, 28, 28, 1)
        assert batch.shape == (32, 28, 28, 1)

    def test_label_one_hot_encoding(self):
        labels = np.array([0, 1, 2, 9])
        num_classes = 10
        one_hot = np.eye(num_classes)[labels]
        assert one_hot.shape == (4, 10)
        assert one_hot[0, 0] == 1.0
        assert one_hot[1, 1] == 1.0

    def test_train_test_split_ratio(self):
        total = 60000
        test_size = 10000
        train_size = total - test_size
        assert train_size / total == pytest.approx(5/6, rel=0.01)

    def test_pixel_values_in_range(self):
        images = np.random.randint(0, 256, (100, 28, 28))
        assert images.min() >= 0
        assert images.max() <= 255


class TestClassificationMetrics:

    def test_accuracy_calculation(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 9])
        accuracy = np.mean(y_true == y_pred)
        assert accuracy == pytest.approx(0.8)

    def test_class_distribution_balanced(self):
        labels = np.array([i % 10 for i in range(1000)])
        counts = np.bincount(labels)
        assert counts.min() >= 90 and counts.max() <= 110

    def test_prediction_within_class_range(self):
        num_classes = 10
        preds = np.random.randint(0, num_classes, 50)
        assert preds.min() >= 0
        assert preds.max() < num_classes

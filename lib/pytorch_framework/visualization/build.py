from .test_visualizer import TestVisualizer
from .featuremap_visualizer import FeatureMapVisualizer
from .default_config import DEFAULT_CONFIG


def build_visualizer(node=DEFAULT_CONFIG):
    test_visualizer = TestVisualizer(
        basefld=node.BASE_FLD,
        image_channels=node.IMAGE_CHANNELS,
        num_classes=node.NUM_CLASSES,
        export_rendergraph=node.EXPORT_RENDERGRAPH,
        export_errormap=node.EXPORT_FEATUREMAP,
        mean=node.IMAGE_MEAN,
        std=node.IMAGE_STD,
    )
    if node.EXPORT_FEATUREMAP:
        feature_visualizer = FeatureMapVisualizer(
            basefld=node.BASE_FLD,
            **node.FEATUREMAP
        )
    else:
        feature_visualizer = None

    return test_visualizer, feature_visualizer
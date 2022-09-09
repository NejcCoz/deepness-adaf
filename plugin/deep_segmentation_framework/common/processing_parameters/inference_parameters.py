import enum
from dataclasses import dataclass
from typing import Optional

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper


# TODO: rename to SegmentationParameters
@dataclass
class InferenceParameters(MapProcessingParameters):
    """
    Parameters for Inference of model (including pre/post processing) obtained from UI.
    """

    postprocessing_dilate_erode_size: int  # dilate/erode operation size, once we have a single class map. 0 if inactive
    model: ModelWrapper  # wrapper of the loaded model

    pixel_classification__probability_threshold: float  # Minimum required class probability for pixel. 0 if disabled

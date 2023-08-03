import enum
from dataclasses import dataclass
from typing import Optional

from newdeepness.common.channels_mapping import ChannelsMapping
from newdeepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from newdeepness.processing.models.model_base import ModelBase


@dataclass
class SuperresolutionParameters(MapProcessingParameters):
    """
    Parameters for Inference of Super Resolution model (including pre/post-processing) obtained from UI.
    """

    output_scaling: float  # scaling factor for the model output (keep 1 if maximum model output value is 1)
    model: ModelBase  # wrapper of the loaded model
    scale_factor: int  # scale factor for the model output size

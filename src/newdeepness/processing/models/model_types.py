import enum
from dataclasses import dataclass

from newdeepness.common.processing_parameters.detection_parameters import DetectionParameters
from newdeepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from newdeepness.common.processing_parameters.regression_parameters import RegressionParameters
from newdeepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from newdeepness.common.processing_parameters.superresolution_parameters import SuperresolutionParameters
from newdeepness.processing.map_processor.map_processor_detection import MapProcessorDetection
from newdeepness.processing.map_processor.map_processor_regression import MapProcessorRegression
from newdeepness.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from newdeepness.processing.map_processor.map_processor_superresolution import MapProcessorSuperresolution
from newdeepness.processing.models.detector import Detector
from newdeepness.processing.models.regressor import Regressor
from newdeepness.processing.models.segmentor import Segmentor
from newdeepness.processing.models.superresolution import Superresolution


class ModelType(enum.Enum):
    SEGMENTATION = Segmentor.get_class_display_name()
    REGRESSION = Regressor.get_class_display_name()
    DETECTION = Detector.get_class_display_name()
    SUPERRESOLUTION = Superresolution.get_class_display_name()


@dataclass
class ModelDefinition:
    model_type: ModelType
    model_class: type
    parameters_class: type
    map_processor_class: type

    @classmethod
    def get_model_definitions(cls):
        return [
            cls(
                model_type=ModelType.SEGMENTATION,
                model_class=Segmentor,
                parameters_class=SegmentationParameters,
                map_processor_class=MapProcessorSegmentation,
            ),
            cls(
                model_type=ModelType.REGRESSION,
                model_class=Regressor,
                parameters_class=RegressionParameters,
                map_processor_class=MapProcessorRegression,
            ),
            cls(
                model_type=ModelType.DETECTION,
                model_class=Detector,
                parameters_class=DetectionParameters,
                map_processor_class=MapProcessorDetection,
            ),  # superresolution
            cls(
                model_type=ModelType.SUPERRESOLUTION,
                model_class=Superresolution,
                parameters_class=SuperresolutionParameters,
                map_processor_class=MapProcessorSuperresolution,
            )

        ]

    @classmethod
    def get_definition_for_type(cls, model_type: ModelType):
        model_definitions = cls.get_model_definitions()
        for model_definition in model_definitions:
            if model_definition.model_type == model_type:
                return model_definition
        raise Exception(f"Unknown model type: '{model_type}'!")

    @classmethod
    def get_definition_for_params(cls, params: MapProcessingParameters):
        """ get model definition corresponding to the specified parameters """
        model_definitions = cls.get_model_definitions()
        for model_definition in model_definitions:
            if type(params) == model_definition.parameters_class:
                return model_definition

        for model_definition in model_definitions:
            if isinstance(params, model_definition.parameters_class):
                return model_definition
        raise Exception(f"Unknown model type for parameters: '{params}'!")

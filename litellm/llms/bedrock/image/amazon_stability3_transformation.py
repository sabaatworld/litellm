import types
from typing import List, Optional

from openai.types.image import Image

from litellm.types.llms.bedrock import (
    AmazonStability3TextToImageRequest,
    AmazonStability3TextToImageResponse,
)
from litellm.types.utils import ImageResponse


class AmazonStability3Config:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0

    Stability API Ref: https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post
    """

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @classmethod
    def get_supported_openai_params(cls, model: Optional[str] = None) -> List:
        """
        No additional OpenAI params are mapped for stability 3
        """
        return ["n", "response_format", "size"]

    @classmethod
    def _is_stability_3_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a Stability 3 model

        Stability 3 models follow this pattern:
            sd3-large
            sd3-large-turbo
            sd3-medium
            sd3.5-large
            sd3.5-large-turbo

        Stability ultra models
            stable-image-ultra-v1
        """
        if model:
            if "sd3" in model or "sd3.5" in model:
                return True
            if "stable-image-ultra-v1" in model:
                return True
        return False

    @classmethod
    def transform_request_body(
        cls, prompt: str, optional_params: dict
    ) -> AmazonStability3TextToImageRequest:
        """
        Transform the request body for the Stability 3 models
        
        Removes aws_region_name from params since it's not part of the model's request format
        """
        # Create a copy of optional_params to avoid modifying the original
        params = optional_params.copy()
        
        # Remove aws_region_name if it exists
        params.pop('aws_region_name', None)
        
        data = AmazonStability3TextToImageRequest(prompt=prompt, **params)
        return data

    @classmethod
    def map_openai_params(cls, non_default_params: dict, optional_params: dict) -> dict:
        """
        Map the OpenAI params to the Bedrock params for Stability 3 models
        
        Parameters that need mapping:
        - size: Convert from "widthxheight" format to an appropriate aspect_ratio
        """
        _size = non_default_params.get("size")
        if _size is not None:
            width, height = map(int, _size.split("x"))
            
            # Calculate the aspect ratio based on width and height
            # Map to the closest supported aspect ratio
            aspect_ratios = {
                "16:9": 16/9,
                "1:1": 1/1,
                "21:9": 21/9,
                "2:3": 2/3,
                "3:2": 3/2,
                "4:5": 4/5,
                "5:4": 5/4,
                "9:16": 9/16,
                "9:21": 9/21
            }
            
            requested_ratio = width / height
            
            # Find the closest aspect ratio
            closest_ratio = min(aspect_ratios.items(), key=lambda x: abs(x[1] - requested_ratio))
            optional_params["aspect_ratio"] = closest_ratio[0]
        
        return optional_params

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        """
        Transform the response dict to the OpenAI response
        """

        stability_3_response = AmazonStability3TextToImageResponse(**response_dict)
        openai_images: List[Image] = []
        for _img in stability_3_response.get("images", []):
            openai_images.append(Image(b64_json=_img))

        model_response.data = openai_images
        return model_response

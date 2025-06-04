import numpy as np
from pydantic import BaseModel, field_validator
from numpydantic import NDArray
from typing import Annotated

class Pose(BaseModel):
    translation: Annotated[NDArray, 3]
    rotation: Annotated[NDArray, 3]

    @field_validator('translation', 'rotation')
    @classmethod
    def check_3d_arrays(cls, v):
        if not isinstance(v, np.ndarray) or v.shape != (3, 1):
            raise ValueError("Each element muss be a three dimensional numpy array!")
        return v

    def get_translation(self) -> NDArray:
        return self.translation

    def get_translation_as_homogeneous(self) -> NDArray:
        translation = self.get_translation()
        return np.array([[translation[0, 0]], [translation[1, 0]], [translation[2, 0]], [1.0]])

    def get_rotation(self) -> NDArray:
        return self.rotation

    def set_translation(self, translation: NDArray) -> None:
        if translation.shape != (3, 1):
            raise ValueError("Translation must be a three dimensional numpy array!")
        self.translation = translation

    def set_rotation(self, rotation: NDArray) -> None:
        if rotation.shape != (3, 1):
            raise ValueError("Rotation must be a three dimensional numpy array!")
        self.rotation = rotation

    def t(self) -> NDArray:
        return self.get_translation()

    def r(self) -> NDArray:
        return self.get_rotation()
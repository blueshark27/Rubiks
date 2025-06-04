import numpy as np
from pydantic import BaseModel, field_validator
from numpydantic import NDArray


class Scaling(BaseModel):
    x: float
    y: float
    z: float

    @field_validator('*')
    @classmethod
    def check_values(cls, v):
        if not isinstance(v, float):
            raise ValueError("Each element muss be a float!")
        return v

    def get_scaling(self) -> NDArray:
        return np.array([[self.x],[self.y],[self.z]])

    def set_scaling(self, scaling: NDArray) -> None:
        if scaling.shape != (3, 1):
            raise ValueError("Rotation must be a three dimensional numpy array!")
        self.x = scaling[0]
        self.y = scaling[1]
        self.z = scaling[2]

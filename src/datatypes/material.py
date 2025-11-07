"""
Material system for object appearance.

This module provides a Material class for defining the visual appearance of objects
using the Phong reflection model. Materials can specify ambient, diffuse, specular
colors and shininess for realistic lighting.
"""

import numpy as np
from typing import List, Optional, Tuple
from pydantic import BaseModel, field_validator


class Material(BaseModel):
    """
    Material class using Phong reflection model.

    A material defines how an object reflects light using:
    - Ambient: Base color in shadow
    - Diffuse: Main surface color under direct light
    - Specular: Highlight/reflection color
    - Shininess: How sharp/focused the specular highlights are

    Colors are RGBA format [r, g, b, a] where each component is in range [0, 1].

    Example:
        >>> # Red plastic material
        >>> material = Material(
        ...     ambient=[0.2, 0.0, 0.0, 1.0],
        ...     diffuse=[1.0, 0.0, 0.0, 1.0],
        ...     specular=[0.5, 0.5, 0.5, 1.0],
        ...     shininess=32.0
        ... )
    """

    ambient: List[float]
    diffuse: List[float]
    specular: List[float]
    shininess: float

    class Config:
        arbitrary_types_allowed = True

    @field_validator('ambient', 'diffuse', 'specular')
    def validate_color(cls, v):
        """Validate that colors are RGBA with 4 components in [0, 1]"""
        if len(v) != 4:
            raise ValueError(f"Color must have 4 components (RGBA), got {len(v)}")

        for i, component in enumerate(v):
            if not 0.0 <= component <= 1.0:
                raise ValueError(f"Color component {i} must be in range [0, 1], got {component}")

        return v

    @field_validator('shininess')
    def validate_shininess(cls, v):
        """Validate that shininess is positive"""
        if v < 0:
            raise ValueError(f"Shininess must be >= 0, got {v}")
        return v

    @classmethod
    def from_color(cls, color: List[float], shininess: float = 32.0) -> 'Material':
        """
        Create a material from a single color.

        This is a convenience method that creates a material with:
        - Ambient: 20% of the color
        - Diffuse: The specified color
        - Specular: White highlights
        - Shininess: Specified (default 32.0)

        Args:
            color: RGBA color [r, g, b, a]
            shininess: Specular shininess exponent (default: 32.0)

        Returns:
            Material with properties based on the color
        """
        if len(color) != 4:
            raise ValueError(f"Color must have 4 components (RGBA), got {len(color)}")

        # Ambient is darker version of diffuse
        ambient = [c * 0.2 for c in color]

        # Specular is typically white or light gray
        specular = [0.5, 0.5, 0.5, 1.0]

        return cls(
            ambient=ambient,
            diffuse=color,
            specular=specular,
            shininess=shininess
        )

    @classmethod
    def default(cls) -> 'Material':
        """
        Create a default gray material.

        Returns:
            Material with neutral gray appearance
        """
        return cls.from_color([0.8, 0.8, 0.8, 1.0])

    @classmethod
    def red(cls) -> 'Material':
        """Create a red material"""
        return cls.from_color([1.0, 0.0, 0.0, 1.0])

    @classmethod
    def green(cls) -> 'Material':
        """Create a green material"""
        return cls.from_color([0.0, 1.0, 0.0, 1.0])

    @classmethod
    def blue(cls) -> 'Material':
        """Create a blue material"""
        return cls.from_color([0.0, 0.0, 1.0, 1.0])

    @classmethod
    def yellow(cls) -> 'Material':
        """Create a yellow material"""
        return cls.from_color([1.0, 1.0, 0.0, 1.0])

    @classmethod
    def cyan(cls) -> 'Material':
        """Create a cyan material"""
        return cls.from_color([0.0, 1.0, 1.0, 1.0])

    @classmethod
    def magenta(cls) -> 'Material':
        """Create a magenta material"""
        return cls.from_color([1.0, 0.0, 1.0, 1.0])

    @classmethod
    def white(cls) -> 'Material':
        """Create a white material"""
        return cls.from_color([1.0, 1.0, 1.0, 1.0])

    @classmethod
    def black(cls) -> 'Material':
        """Create a black material"""
        return cls.from_color([0.0, 0.0, 0.0, 1.0])

    def get_ambient(self) -> List[float]:
        """Get ambient color (RGBA)"""
        return self.ambient

    def get_diffuse(self) -> List[float]:
        """Get diffuse color (RGBA)"""
        return self.diffuse

    def get_specular(self) -> List[float]:
        """Get specular color (RGBA)"""
        return self.specular

    def get_shininess(self) -> float:
        """Get shininess value"""
        return self.shininess

    def set_ambient(self, color: List[float]) -> None:
        """Set ambient color"""
        self.ambient = color

    def set_diffuse(self, color: List[float]) -> None:
        """Set diffuse color"""
        self.diffuse = color

    def set_specular(self, color: List[float]) -> None:
        """Set specular color"""
        self.specular = color

    def set_shininess(self, value: float) -> None:
        """Set shininess value"""
        self.shininess = value

    def set_color(self, color: List[float]) -> None:
        """
        Set the main color (diffuse) and adjust ambient accordingly.

        This convenience method updates:
        - Diffuse: To the specified color
        - Ambient: To 20% of the specified color

        Args:
            color: RGBA color [r, g, b, a]
        """
        self.diffuse = color
        self.ambient = [c * 0.2 for c in color]

    def copy(self) -> 'Material':
        """
        Create a copy of this material.

        Returns:
            New Material instance with the same properties
        """
        return Material(
            ambient=self.ambient.copy(),
            diffuse=self.diffuse.copy(),
            specular=self.specular.copy(),
            shininess=self.shininess
        )

    def __repr__(self) -> str:
        return (f"Material(diffuse={self.diffuse}, "
                f"ambient={self.ambient}, "
                f"specular={self.specular}, "
                f"shininess={self.shininess})")


# Predefined material presets
class MaterialPresets:
    """
    Collection of predefined materials for common use cases.

    These materials follow real-world material properties from
    various OpenGL and rendering resources.
    """

    @staticmethod
    def emerald() -> Material:
        """Emerald gemstone"""
        return Material(
            ambient=[0.0215, 0.1745, 0.0215, 1.0],
            diffuse=[0.07568, 0.61424, 0.07568, 1.0],
            specular=[0.633, 0.727811, 0.633, 1.0],
            shininess=76.8
        )

    @staticmethod
    def jade() -> Material:
        """Jade gemstone"""
        return Material(
            ambient=[0.135, 0.2225, 0.1575, 1.0],
            diffuse=[0.54, 0.89, 0.63, 1.0],
            specular=[0.316228, 0.316228, 0.316228, 1.0],
            shininess=12.8
        )

    @staticmethod
    def ruby() -> Material:
        """Ruby gemstone"""
        return Material(
            ambient=[0.1745, 0.01175, 0.01175, 1.0],
            diffuse=[0.61424, 0.04136, 0.04136, 1.0],
            specular=[0.727811, 0.626959, 0.626959, 1.0],
            shininess=76.8
        )

    @staticmethod
    def gold() -> Material:
        """Metallic gold"""
        return Material(
            ambient=[0.24725, 0.1995, 0.0745, 1.0],
            diffuse=[0.75164, 0.60648, 0.22648, 1.0],
            specular=[0.628281, 0.555802, 0.366065, 1.0],
            shininess=51.2
        )

    @staticmethod
    def silver() -> Material:
        """Metallic silver"""
        return Material(
            ambient=[0.19225, 0.19225, 0.19225, 1.0],
            diffuse=[0.50754, 0.50754, 0.50754, 1.0],
            specular=[0.508273, 0.508273, 0.508273, 1.0],
            shininess=51.2
        )

    @staticmethod
    def bronze() -> Material:
        """Metallic bronze"""
        return Material(
            ambient=[0.2125, 0.1275, 0.054, 1.0],
            diffuse=[0.714, 0.4284, 0.18144, 1.0],
            specular=[0.393548, 0.271906, 0.166721, 1.0],
            shininess=25.6
        )

    @staticmethod
    def chrome() -> Material:
        """Shiny chrome metal"""
        return Material(
            ambient=[0.25, 0.25, 0.25, 1.0],
            diffuse=[0.4, 0.4, 0.4, 1.0],
            specular=[0.774597, 0.774597, 0.774597, 1.0],
            shininess=76.8
        )

    @staticmethod
    def plastic_red() -> Material:
        """Red plastic"""
        return Material(
            ambient=[0.1, 0.0, 0.0, 1.0],
            diffuse=[1.0, 0.0, 0.0, 1.0],
            specular=[0.5, 0.5, 0.5, 1.0],
            shininess=32.0
        )

    @staticmethod
    def plastic_green() -> Material:
        """Green plastic"""
        return Material(
            ambient=[0.0, 0.1, 0.0, 1.0],
            diffuse=[0.0, 1.0, 0.0, 1.0],
            specular=[0.5, 0.5, 0.5, 1.0],
            shininess=32.0
        )

    @staticmethod
    def plastic_blue() -> Material:
        """Blue plastic"""
        return Material(
            ambient=[0.0, 0.0, 0.1, 1.0],
            diffuse=[0.0, 0.0, 1.0, 1.0],
            specular=[0.5, 0.5, 0.5, 1.0],
            shininess=32.0
        )

    @staticmethod
    def rubber_black() -> Material:
        """Black rubber"""
        return Material(
            ambient=[0.02, 0.02, 0.02, 1.0],
            diffuse=[0.01, 0.01, 0.01, 1.0],
            specular=[0.4, 0.4, 0.4, 1.0],
            shininess=10.0
        )

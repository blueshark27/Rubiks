from abc import ABC, abstractmethod
from typing import List

from src.datatypes.pose import Pose


class SceneObject(ABC):

    def __init__(self, name: str, pose: Pose):
        self.name = name
        self.pose = pose

    def get_pose(self) -> Pose:
        return self.pose

    def set_pose(self, pose: Pose):
        self.pose = pose

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def get_parent(self):
        pass

    @abstractmethod
    def set_parent(self, parent):
        pass

    @abstractmethod
    def get_children(self) -> List:
        pass

    @abstractmethod
    def add_child(self, child):
        pass

    @abstractmethod
    def remove_child(self, child):
        pass

"""
Keyframe-based animation system.

This module provides classes for creating and playing back animations using keyframes.
Animations can be applied to scene objects to create smooth motion over time.
"""

import numpy as np
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass

from src.datatypes.pose import Pose, PoseQuat


InterpolationMode = Literal['linear', 'step', 'smooth']


@dataclass
class Keyframe:
    """
    A keyframe representing a pose at a specific time.

    Attributes:
        time: Time in seconds (must be >= 0)
        pose: Pose or PoseQuat at this keyframe
    """
    time: float
    pose: Pose | PoseQuat

    def __post_init__(self):
        if self.time < 0:
            raise ValueError(f"Keyframe time must be >= 0, got {self.time}")


class Animation:
    """
    Animation track containing keyframes for a single object.

    Supports multiple interpolation modes:
    - 'linear': Linear interpolation (LERP for translation, SLERP for rotation if using PoseQuat)
    - 'step': No interpolation, snap to keyframe values
    - 'smooth': Smooth interpolation with easing (ease-in-out)

    Example:
        >>> # Create animation with keyframes
        >>> anim = Animation('my_object')
        >>> anim.add_keyframe(0.0, start_pose)
        >>> anim.add_keyframe(1.0, middle_pose)
        >>> anim.add_keyframe(2.0, end_pose)
        >>>
        >>> # Evaluate at time t
        >>> current_pose = anim.evaluate(1.5)
    """

    def __init__(self, name: str = "animation", interpolation: InterpolationMode = 'linear'):
        """
        Initialize animation.

        Args:
            name: Name of the animation
            interpolation: Interpolation mode ('linear', 'step', or 'smooth')
        """
        self.name = name
        self.interpolation = interpolation
        self.keyframes: List[Keyframe] = []
        self._sorted = True

    def add_keyframe(self, time: float, pose: Pose | PoseQuat) -> None:
        """
        Add a keyframe to the animation.

        Args:
            time: Time in seconds
            pose: Pose or PoseQuat at this time
        """
        keyframe = Keyframe(time=time, pose=pose)
        self.keyframes.append(keyframe)
        self._sorted = False

    def _ensure_sorted(self) -> None:
        """Ensure keyframes are sorted by time"""
        if not self._sorted:
            self.keyframes.sort(key=lambda k: k.time)
            self._sorted = True

    def get_duration(self) -> float:
        """
        Get total duration of the animation.

        Returns:
            Duration in seconds, or 0.0 if no keyframes
        """
        if not self.keyframes:
            return 0.0
        self._ensure_sorted()
        return self.keyframes[-1].time

    def get_keyframe_count(self) -> int:
        """Get number of keyframes"""
        return len(self.keyframes)

    def _find_surrounding_keyframes(self, time: float) -> Tuple[Optional[Keyframe], Optional[Keyframe], float]:
        """
        Find the keyframes surrounding a given time.

        Args:
            time: Time to query

        Returns:
            Tuple of (keyframe_before, keyframe_after, normalized_t)
            where normalized_t is [0, 1] between the two keyframes
        """
        self._ensure_sorted()

        if not self.keyframes:
            return None, None, 0.0

        # Before first keyframe
        if time <= self.keyframes[0].time:
            return self.keyframes[0], self.keyframes[0], 0.0

        # After last keyframe
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1], self.keyframes[-1], 1.0

        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            kf_before = self.keyframes[i]
            kf_after = self.keyframes[i + 1]

            if kf_before.time <= time <= kf_after.time:
                # Normalize time to [0, 1] between keyframes
                duration = kf_after.time - kf_before.time
                if duration < 1e-10:
                    normalized_t = 0.0
                else:
                    normalized_t = (time - kf_before.time) / duration

                return kf_before, kf_after, normalized_t

        # Shouldn't reach here
        return self.keyframes[-1], self.keyframes[-1], 1.0

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """
        Smooth ease-in-out interpolation curve.

        Args:
            t: Input in [0, 1]

        Returns:
            Smoothed value in [0, 1]
        """
        # Smoothstep function: 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)

    def evaluate(self, time: float) -> Optional[Pose | PoseQuat]:
        """
        Evaluate the animation at a given time.

        Args:
            time: Time in seconds

        Returns:
            Interpolated pose at the given time, or None if no keyframes
        """
        if not self.keyframes:
            return None

        kf_before, kf_after, t = self._find_surrounding_keyframes(time)

        if kf_before is None or kf_after is None:
            return None

        # If before and after are the same, return that pose
        if kf_before is kf_after:
            return kf_before.pose

        # Apply interpolation mode
        if self.interpolation == 'step':
            # No interpolation, snap to keyframe
            return kf_before.pose if t < 0.5 else kf_after.pose

        elif self.interpolation == 'smooth':
            # Apply easing to t
            t = self._ease_in_out(t)

        # Interpolate based on pose type
        pose_before = kf_before.pose
        pose_after = kf_after.pose

        # Check if both poses are the same type
        if type(pose_before) != type(pose_after):
            raise TypeError(
                f"Cannot interpolate between different pose types: "
                f"{type(pose_before).__name__} and {type(pose_after).__name__}"
            )

        # Interpolate PoseQuat (uses SLERP for rotation)
        if isinstance(pose_before, PoseQuat):
            return pose_before.interpolate(pose_after, t, method='slerp')

        # Interpolate Pose (axis-angle)
        elif isinstance(pose_before, Pose):
            # Linear interpolation for translation and rotation
            interp_translation = pose_before.translation * (1.0 - t) + pose_after.translation * t
            interp_rotation = pose_before.rotation * (1.0 - t) + pose_after.rotation * t
            return Pose(translation=interp_translation, rotation=interp_rotation)

        else:
            raise TypeError(f"Unknown pose type: {type(pose_before)}")

    def clear(self) -> None:
        """Remove all keyframes"""
        self.keyframes.clear()
        self._sorted = True


class AnimationClip:
    """
    Collection of animations that can be played together.

    An AnimationClip contains multiple Animation tracks, allowing you to animate
    multiple objects synchronously.

    Example:
        >>> clip = AnimationClip('scene_animation')
        >>> clip.add_animation('object1', anim1)
        >>> clip.add_animation('object2', anim2)
        >>>
        >>> # Evaluate all animations at time t
        >>> poses = clip.evaluate(1.5)
        >>> for obj_name, pose in poses.items():
        >>>     scene_objects[obj_name].set_pose(pose)
    """

    def __init__(self, name: str = "clip"):
        """
        Initialize animation clip.

        Args:
            name: Name of the animation clip
        """
        self.name = name
        self.animations: dict[str, Animation] = {}

    def add_animation(self, object_name: str, animation: Animation) -> None:
        """
        Add an animation track for an object.

        Args:
            object_name: Name of the object to animate
            animation: Animation to apply to this object
        """
        self.animations[object_name] = animation

    def remove_animation(self, object_name: str) -> None:
        """
        Remove an animation track.

        Args:
            object_name: Name of the object whose animation to remove
        """
        if object_name in self.animations:
            del self.animations[object_name]

    def get_animation(self, object_name: str) -> Optional[Animation]:
        """
        Get animation for a specific object.

        Args:
            object_name: Name of the object

        Returns:
            Animation for that object, or None if not found
        """
        return self.animations.get(object_name)

    def get_duration(self) -> float:
        """
        Get total duration (longest animation).

        Returns:
            Duration in seconds
        """
        if not self.animations:
            return 0.0
        return max(anim.get_duration() for anim in self.animations.values())

    def evaluate(self, time: float) -> dict[str, Pose | PoseQuat]:
        """
        Evaluate all animations at a given time.

        Args:
            time: Time in seconds

        Returns:
            Dictionary mapping object names to their interpolated poses
        """
        result = {}
        for obj_name, animation in self.animations.items():
            pose = animation.evaluate(time)
            if pose is not None:
                result[obj_name] = pose
        return result

    def clear(self) -> None:
        """Remove all animations"""
        self.animations.clear()

    def get_object_names(self) -> List[str]:
        """Get list of all animated object names"""
        return list(self.animations.keys())


class AnimationPlayer:
    """
    Player for controlling animation playback.

    Handles playing, pausing, looping, and time management for animations.

    Example:
        >>> player = AnimationPlayer(clip)
        >>> player.play()
        >>>
        >>> # In update loop
        >>> dt = 0.016  # 60 FPS
        >>> player.update(dt)
        >>> poses = player.get_current_poses()
    """

    def __init__(self, clip: AnimationClip, loop: bool = False):
        """
        Initialize animation player.

        Args:
            clip: AnimationClip to play
            loop: Whether to loop the animation
        """
        self.clip = clip
        self.loop = loop
        self.current_time = 0.0
        self.playing = False
        self.speed = 1.0

    def play(self) -> None:
        """Start playing the animation"""
        self.playing = True

    def pause(self) -> None:
        """Pause the animation"""
        self.playing = False

    def stop(self) -> None:
        """Stop and reset the animation"""
        self.playing = False
        self.current_time = 0.0

    def set_time(self, time: float) -> None:
        """
        Set current playback time.

        Args:
            time: Time in seconds
        """
        self.current_time = max(0.0, time)

    def get_time(self) -> float:
        """Get current playback time"""
        return self.current_time

    def set_speed(self, speed: float) -> None:
        """
        Set playback speed.

        Args:
            speed: Speed multiplier (1.0 = normal, 2.0 = double speed, etc.)
        """
        self.speed = speed

    def is_playing(self) -> bool:
        """Check if animation is currently playing"""
        return self.playing

    def is_finished(self) -> bool:
        """Check if animation has finished (only relevant if not looping)"""
        if self.loop:
            return False
        duration = self.clip.get_duration()
        return self.current_time >= duration

    def update(self, delta_time: float) -> None:
        """
        Update animation time.

        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        if not self.playing:
            return

        self.current_time += delta_time * self.speed

        # Handle looping
        duration = self.clip.get_duration()
        if duration > 0:
            if self.loop:
                # Wrap around
                self.current_time = self.current_time % duration
            elif self.current_time >= duration:
                # Clamp to end
                self.current_time = duration
                self.playing = False

    def get_current_poses(self) -> dict[str, Pose | PoseQuat]:
        """
        Get current poses for all animated objects.

        Returns:
            Dictionary mapping object names to their current poses
        """
        return self.clip.evaluate(self.current_time)

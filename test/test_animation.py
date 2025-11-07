"""
Unit tests for the animation system.
"""

import unittest
import numpy as np
import math

from src.animation.animation import Keyframe, Animation, AnimationClip, AnimationPlayer
from src.datatypes.pose import Pose, PoseQuat
from src.datatypes.quaternion import Quaternion


class TestKeyframe(unittest.TestCase):

    def test_keyframe_creation(self):
        """Test creating a keyframe"""
        pose = Pose(
            translation=np.array([[1], [2], [3]]),
            rotation=np.array([[0], [0], [0]])
        )
        kf = Keyframe(time=1.0, pose=pose)

        self.assertEqual(kf.time, 1.0)
        self.assertEqual(kf.pose, pose)

    def test_keyframe_negative_time_raises_error(self):
        """Test that negative time raises ValueError"""
        pose = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )

        with self.assertRaises(ValueError):
            Keyframe(time=-1.0, pose=pose)


class TestAnimation(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.pose1 = PoseQuat.from_translation_axis_angle(
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [1]]),
            0.0
        )
        self.pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )
        self.pose3 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [10], [0]]),
            np.array([[0], [0], [1]]),
            np.pi
        )

    def test_animation_creation(self):
        """Test creating an animation"""
        anim = Animation('test')
        self.assertEqual(anim.name, 'test')
        self.assertEqual(anim.get_keyframe_count(), 0)
        self.assertEqual(anim.get_duration(), 0.0)

    def test_add_keyframes(self):
        """Test adding keyframes"""
        anim = Animation('test')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(1.0, self.pose2)
        anim.add_keyframe(2.0, self.pose3)

        self.assertEqual(anim.get_keyframe_count(), 3)
        self.assertAlmostEqual(anim.get_duration(), 2.0)

    def test_evaluate_empty_animation(self):
        """Test evaluating animation with no keyframes"""
        anim = Animation('test')
        result = anim.evaluate(0.5)
        self.assertIsNone(result)

    def test_evaluate_single_keyframe(self):
        """Test evaluating animation with single keyframe"""
        anim = Animation('test')
        anim.add_keyframe(1.0, self.pose1)

        # Before keyframe
        result = anim.evaluate(0.5)
        np.testing.assert_array_almost_equal(result.translation, self.pose1.translation)

        # At keyframe
        result = anim.evaluate(1.0)
        np.testing.assert_array_almost_equal(result.translation, self.pose1.translation)

        # After keyframe
        result = anim.evaluate(1.5)
        np.testing.assert_array_almost_equal(result.translation, self.pose1.translation)

    def test_evaluate_linear_interpolation(self):
        """Test linear interpolation between keyframes"""
        anim = Animation('test', interpolation='linear')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(2.0, self.pose2)

        # At t=1.0 (middle), translation should be halfway
        result = anim.evaluate(1.0)
        expected_translation = np.array([[5], [0], [0]])
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

    def test_evaluate_step_interpolation(self):
        """Test step interpolation (no smoothing)"""
        anim = Animation('test', interpolation='step')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(2.0, self.pose2)

        # Before midpoint, should snap to first keyframe
        result = anim.evaluate(0.9)
        np.testing.assert_array_almost_equal(result.translation, self.pose1.translation)

        # After midpoint, should snap to second keyframe
        result = anim.evaluate(1.1)
        np.testing.assert_array_almost_equal(result.translation, self.pose2.translation)

    def test_evaluate_smooth_interpolation(self):
        """Test smooth interpolation with easing"""
        anim = Animation('test', interpolation='smooth')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(2.0, self.pose2)

        # Evaluate at multiple points
        result_start = anim.evaluate(0.0)
        result_mid = anim.evaluate(1.0)
        result_end = anim.evaluate(2.0)

        # Check endpoints
        np.testing.assert_array_almost_equal(result_start.translation, self.pose1.translation)
        np.testing.assert_array_almost_equal(result_end.translation, self.pose2.translation)

        # Middle should be interpolated (but not exactly halfway due to easing)
        self.assertGreater(result_mid.translation[0, 0], self.pose1.translation[0, 0])
        self.assertLess(result_mid.translation[0, 0], self.pose2.translation[0, 0])

    def test_evaluate_before_first_keyframe(self):
        """Test evaluating before first keyframe"""
        anim = Animation('test')
        anim.add_keyframe(1.0, self.pose1)
        anim.add_keyframe(2.0, self.pose2)

        result = anim.evaluate(0.5)
        np.testing.assert_array_almost_equal(result.translation, self.pose1.translation)

    def test_evaluate_after_last_keyframe(self):
        """Test evaluating after last keyframe"""
        anim = Animation('test')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(1.0, self.pose2)

        result = anim.evaluate(2.0)
        np.testing.assert_array_almost_equal(result.translation, self.pose2.translation)

    def test_unsorted_keyframes(self):
        """Test that keyframes get sorted by time"""
        anim = Animation('test')
        anim.add_keyframe(2.0, self.pose3)
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(1.0, self.pose2)

        # Should still interpolate correctly despite being added out of order
        # At t=0.5, halfway between pose1 (t=0, pos=[0,0,0]) and pose2 (t=1, pos=[10,0,0])
        result = anim.evaluate(0.5)
        expected_translation = np.array([[5.0], [0], [0]])
        np.testing.assert_array_almost_equal(result.translation, expected_translation, decimal=4)

    def test_clear_keyframes(self):
        """Test clearing all keyframes"""
        anim = Animation('test')
        anim.add_keyframe(0.0, self.pose1)
        anim.add_keyframe(1.0, self.pose2)

        self.assertEqual(anim.get_keyframe_count(), 2)

        anim.clear()
        self.assertEqual(anim.get_keyframe_count(), 0)
        self.assertIsNone(anim.evaluate(0.5))

    def test_mixed_pose_types_raises_error(self):
        """Test that mixing Pose and PoseQuat types raises error"""
        anim = Animation('test')

        pose_axis_angle = Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        )

        anim.add_keyframe(0.0, pose_axis_angle)
        anim.add_keyframe(1.0, self.pose1)  # PoseQuat

        with self.assertRaises(TypeError):
            anim.evaluate(0.5)


class TestAnimationClip(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.pose1 = PoseQuat.identity()
        self.pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )

        self.anim1 = Animation('anim1')
        self.anim1.add_keyframe(0.0, self.pose1)
        self.anim1.add_keyframe(2.0, self.pose2)

        self.anim2 = Animation('anim2')
        self.anim2.add_keyframe(0.0, self.pose2)
        self.anim2.add_keyframe(3.0, self.pose1)

    def test_clip_creation(self):
        """Test creating an animation clip"""
        clip = AnimationClip('test_clip')
        self.assertEqual(clip.name, 'test_clip')
        self.assertEqual(clip.get_duration(), 0.0)

    def test_add_animations(self):
        """Test adding animations to clip"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)
        clip.add_animation('object2', self.anim2)

        self.assertEqual(len(clip.get_object_names()), 2)
        self.assertIn('object1', clip.get_object_names())
        self.assertIn('object2', clip.get_object_names())

    def test_get_duration(self):
        """Test getting clip duration (longest animation)"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)  # Duration 2.0
        clip.add_animation('object2', self.anim2)  # Duration 3.0

        self.assertAlmostEqual(clip.get_duration(), 3.0)

    def test_evaluate_clip(self):
        """Test evaluating all animations in clip"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)
        clip.add_animation('object2', self.anim2)

        poses = clip.evaluate(1.0)

        self.assertEqual(len(poses), 2)
        self.assertIn('object1', poses)
        self.assertIn('object2', poses)

    def test_remove_animation(self):
        """Test removing an animation from clip"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)
        clip.add_animation('object2', self.anim2)

        clip.remove_animation('object1')

        self.assertEqual(len(clip.get_object_names()), 1)
        self.assertNotIn('object1', clip.get_object_names())

    def test_get_animation(self):
        """Test getting specific animation from clip"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)

        retrieved = clip.get_animation('object1')
        self.assertEqual(retrieved, self.anim1)

        not_found = clip.get_animation('nonexistent')
        self.assertIsNone(not_found)

    def test_clear_clip(self):
        """Test clearing all animations from clip"""
        clip = AnimationClip('test_clip')
        clip.add_animation('object1', self.anim1)
        clip.add_animation('object2', self.anim2)

        clip.clear()
        self.assertEqual(len(clip.get_object_names()), 0)


class TestAnimationPlayer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        pose1 = PoseQuat.identity()
        pose2 = PoseQuat.from_translation_axis_angle(
            np.array([[10], [0], [0]]),
            np.array([[0], [0], [1]]),
            np.pi/2
        )

        anim = Animation('anim')
        anim.add_keyframe(0.0, pose1)
        anim.add_keyframe(2.0, pose2)

        self.clip = AnimationClip('test_clip')
        self.clip.add_animation('object1', anim)

    def test_player_creation(self):
        """Test creating animation player"""
        player = AnimationPlayer(self.clip)
        self.assertFalse(player.is_playing())
        self.assertEqual(player.get_time(), 0.0)

    def test_play_pause_stop(self):
        """Test play, pause, and stop controls"""
        player = AnimationPlayer(self.clip)

        # Start playing
        player.play()
        self.assertTrue(player.is_playing())

        # Pause
        player.pause()
        self.assertFalse(player.is_playing())

        # Play again
        player.play()
        self.assertTrue(player.is_playing())

        # Stop resets time
        player.set_time(1.0)
        player.stop()
        self.assertFalse(player.is_playing())
        self.assertEqual(player.get_time(), 0.0)

    def test_update_advances_time(self):
        """Test that update advances time"""
        player = AnimationPlayer(self.clip)
        player.play()

        player.update(0.5)
        self.assertAlmostEqual(player.get_time(), 0.5)

        player.update(0.3)
        self.assertAlmostEqual(player.get_time(), 0.8)

    def test_update_respects_speed(self):
        """Test that playback speed affects time advancement"""
        player = AnimationPlayer(self.clip)
        player.set_speed(2.0)
        player.play()

        player.update(0.5)
        self.assertAlmostEqual(player.get_time(), 1.0)  # 0.5 * 2.0 speed

    def test_update_when_paused(self):
        """Test that update does nothing when paused"""
        player = AnimationPlayer(self.clip)
        # Don't call play()

        player.update(1.0)
        self.assertEqual(player.get_time(), 0.0)

    def test_non_looping_stops_at_end(self):
        """Test that non-looping animation stops at end"""
        player = AnimationPlayer(self.clip, loop=False)
        player.play()

        # Update past the end
        player.update(5.0)

        self.assertAlmostEqual(player.get_time(), self.clip.get_duration())
        self.assertFalse(player.is_playing())
        self.assertTrue(player.is_finished())

    def test_looping_wraps_time(self):
        """Test that looping animation wraps time"""
        player = AnimationPlayer(self.clip, loop=True)
        player.play()

        duration = self.clip.get_duration()

        # Update past the end
        player.update(duration + 0.5)

        self.assertAlmostEqual(player.get_time(), 0.5)
        self.assertTrue(player.is_playing())
        self.assertFalse(player.is_finished())

    def test_set_time(self):
        """Test manually setting time"""
        player = AnimationPlayer(self.clip)

        player.set_time(1.5)
        self.assertAlmostEqual(player.get_time(), 1.5)

        # Negative time should be clamped to 0
        player.set_time(-1.0)
        self.assertEqual(player.get_time(), 0.0)

    def test_get_current_poses(self):
        """Test getting current poses"""
        player = AnimationPlayer(self.clip)
        player.set_time(1.0)

        poses = player.get_current_poses()

        self.assertIn('object1', poses)
        # Should be interpolated between start and end
        self.assertGreater(poses['object1'].translation[0, 0], 0)
        self.assertLess(poses['object1'].translation[0, 0], 10)


if __name__ == '__main__':
    unittest.main()

# Rubiks - 3D Hierarchical Transformation Framework

A Python framework for 3D object hierarchies with parent-child transformations, animations, and OpenGL rendering.

## Features

- **Hierarchical Transformations**: Parent-child relationships with automatic transform propagation
- **High Performance**: 140x speedup with intelligent transform caching
- **Animation System**: Keyframe-based animations with smooth interpolation (SLERP)
- **Quaternion Support**: Smooth rotations without gimbal lock
- **Material System**: Phong lighting with 11+ realistic material presets (metals, gems, plastics)
- **Collision Detection**: Bounding box computation and spatial queries
- **Profiling Tools**: Performance monitoring and optimization recommendations
- **3D Primitives**: Spheres, cylinders, cones with OpenGL rendering

## Installation

### Requirements

- Python 3.13+
- NumPy
- PyOpenGL
- GLFW
- Pydantic
- numpydantic

### Install Dependencies

```bash
pip install numpy PyOpenGL glfw pydantic numpydantic
```

## Quick Start

### Creating Objects with Materials

```python
import numpy as np
from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose
from src.datatypes.material import Material, MaterialPresets

# Create sphere with a simple color
pose = Pose(
    translation=np.array([[0], [0], [0]]),
    rotation=np.array([[0], [0], [0]])
)
sphere1 = Sphere(pose=pose, radius=1.0, color=[1.0, 0.0, 0.0, 1.0])

# Use a preset material
sphere2 = Sphere(pose=pose, radius=1.0, material=MaterialPresets.gold())

# Create custom material
custom_material = Material(
    ambient=[0.1, 0.0, 0.0, 1.0],
    diffuse=[0.8, 0.0, 0.0, 1.0],
    specular=[1.0, 1.0, 1.0, 1.0],
    shininess=128.0
)
sphere3 = Sphere(pose=pose, radius=1.0, material=custom_material)

# Change material dynamically
sphere1.set_material(MaterialPresets.emerald())
sphere1.set_color([0.0, 1.0, 0.0, 1.0])  # Or just change color
```

### Basic Scene with Parent-Child Hierarchy

```python
import numpy as np
from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose

# Create parent sphere at origin
parent_pose = Pose(
    translation=np.array([[0], [0], [0]]),
    rotation=np.array([[0], [0], [0]])
)
parent = Sphere(pose=parent_pose, radius=1.0, name="Parent")

# Create child sphere (relative to parent)
child_pose = Pose(
    translation=np.array([[3], [0], [0]]),  # 3 units to the right of parent
    rotation=np.array([[0], [0], [0]])
)
child = Sphere(pose=child_pose, radius=0.5, name="Child", parent=parent)

# When you rotate the parent, the child follows automatically!
new_parent_pose = Pose(
    translation=np.array([[0], [0], [0]]),
    rotation=np.array([[0], [0], [np.pi/4]])  # Rotate 45 degrees
)
parent.set_pose(new_parent_pose)
# Child now orbits around parent!
```

### Animation Example

```python
from src.animation.animation import Animation, AnimationPlayer, AnimationClip
from src.datatypes.pose import PoseQuat

# Create keyframe animation
anim = Animation('my_animation', interpolation='smooth')

# Add keyframes
start_pose = PoseQuat.from_translation_axis_angle(
    translation=np.array([[0], [0], [0]]),
    axis=np.array([[0], [0], [1]]),
    angle=0.0
)
end_pose = PoseQuat.from_translation_axis_angle(
    translation=np.array([[10], [0], [0]]),
    axis=np.array([[0], [0], [1]]),
    angle=np.pi  # 180 degree rotation
)

anim.add_keyframe(0.0, start_pose)
anim.add_keyframe(2.0, end_pose)

# Create animation clip
clip = AnimationClip('my_clip')
clip.add_animation('my_object', anim)

# Create player
player = AnimationPlayer(clip, loop=True)
player.play()

# In your game/render loop:
delta_time = 0.016  # ~60 FPS
player.update(delta_time)
poses = player.get_current_poses()
my_object.set_pose(poses['my_object'].to_pose())
```

### Collision Detection

```python
from src.utils.bounding_box import check_collision, compute_hierarchy_bounds

# Check if two objects collide
if check_collision(sphere1, sphere2):
    print("Collision detected!")

# Get bounding box for an entire hierarchy
bbox = compute_hierarchy_bounds(root_object, include_children=True)
print(f"Scene volume: {bbox.get_volume():.2f}")
print(f"Scene center: {bbox.get_center().flatten()}")
```

### Performance Profiling

```python
from src.utils.hierarchy_profiler import create_profiling_report

# One-line profiling with automatic report
create_profiling_report(root_object, "My Scene")

# Output:
# [HIERARCHY STATISTICS]
#   Total Objects:       121
#   Maximum Depth:       4
#   Cache Hit Rate:      99.01%
#   Status:              [EXCELLENT]
```

## Examples

Run the included examples to see the framework in action:

```bash
# Material system showcase - 9 spheres with different materials
python -m examples.material_example

# Basic sphere with material
python -m examples.sphere_example

# Basic hierarchy example
python -m examples.hierarchy_example

# Solar system with 3-level hierarchy
python -m examples.solar_system_example

# Robot arm with joint control
python -m examples.robot_arm_example

# Keyframe animation demo
python -m examples.animation_example

# Profiling and optimization tools
python -m examples.profiling_example
```

## Core Concepts

### 1. Poses

A **Pose** represents an object's position and rotation:

```python
from src.datatypes.pose import Pose

pose = Pose(
    translation=np.array([[x], [y], [z]]),  # Position
    rotation=np.array([[rx], [ry], [rz]])   # Axis-angle rotation
)
```

For smooth animations, use **PoseQuat** (quaternion-based):

```python
from src.datatypes.pose import PoseQuat

pose = PoseQuat.from_translation_euler(
    translation=np.array([[0], [0], [0]]),
    roll=0.0, pitch=0.0, yaw=np.pi/4
)
```

### 2. Parent-Child Hierarchies

Objects can have parents and children:

```python
# Method 1: Set parent during creation
child = Sphere(pose=child_pose, parent=parent_sphere)

# Method 2: Add child to parent
parent.add_child(child)

# Method 3: Set parent after creation
child.set_parent(parent)

# Get all children
children = parent.get_children()

# Remove from hierarchy
parent.remove_child(child)
```

The framework automatically:
- Propagates transformations through the hierarchy
- Prevents circular references
- Invalidates caches when needed
- Warns about deep hierarchies

### 3. World vs Local Transforms

```python
# Local transform (relative to parent)
local_matrix = obj.get_local_transform()

# World transform (absolute position in world)
world_matrix = obj.get_world_transform()

# These are cached for performance!
stats = obj.get_cache_statistics()
print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
```

### 4. Transformations

The framework uses **T × R × S** order (Translation × Rotation × Scale):

```python
from src.datatypes import transform

# Build transformation matrix
matrix = transform.compose_transform(
    translation=np.array([[x], [y], [z]]),
    rotation=np.array([[rx], [ry], [rz]]),
    scaling=np.array([[sx], [sy], [sz]])
)

# Convert pose to matrix
matrix = transform.pose_to_matrix(pose, scaling)
```

### 5. Quaternions

For smooth rotations without gimbal lock:

```python
from src.datatypes.quaternion import Quaternion, slerp

# Create quaternion from axis-angle
q = Quaternion.from_axis_angle(
    axis=np.array([[0], [0], [1]]),
    angle=np.pi/2
)

# Smooth interpolation between rotations
q1 = Quaternion.from_euler(0, 0, 0)
q2 = Quaternion.from_euler(0, 0, np.pi/2)
q_mid = slerp(q1, q2, 0.5)  # Halfway between

# Convert to rotation matrix
R = q.to_rotation_matrix()
```

### 6. Materials

Define visual appearance using the Phong reflection model:

```python
from src.datatypes.material import Material, MaterialPresets

# Quick color materials
material = Material.red()
material = Material.blue()

# Realistic presets
material = MaterialPresets.gold()      # Metallic gold
material = MaterialPresets.emerald()   # Gemstone
material = MaterialPresets.chrome()    # Shiny metal

# Custom material
material = Material(
    ambient=[0.2, 0.0, 0.0, 1.0],   # Base color in shadow
    diffuse=[1.0, 0.0, 0.0, 1.0],   # Main surface color
    specular=[0.5, 0.5, 0.5, 1.0],  # Highlight color
    shininess=32.0                   # Highlight sharpness
)

# Easy color conversion
material = Material.from_color([1.0, 0.5, 0.0, 1.0], shininess=64.0)

# Available presets:
# Metals: gold, silver, bronze, chrome
# Gems: emerald, jade, ruby
# Plastics: plastic_red, plastic_green, plastic_blue
# Other: rubber_black
```

## Testing

Run the test suite to verify everything works:

```bash
# All tests (177 tests)
python -m unittest discover test -v

# Specific test suites
python -m unittest test.test_hierarchy -v       # Hierarchy tests
python -m unittest test.test_transform -v       # Transform math tests
python -m unittest test.test_quaternion -v      # Quaternion tests
python -m unittest test.test_animation -v       # Animation tests
python -m unittest test.test_bounding_box -v    # Collision tests
python -m unittest test.test_material -v        # Material system tests (31 tests)
python -m unittest test.test_performance -v     # Performance benchmarks
```

Expected results: **176/177 tests passing** (1 pre-existing unrelated test)

## Performance

### Transform Caching
- **Uncached**: ~0.014 ms per access
- **Cached**: ~0.0001 ms per access
- **Speedup**: 140x faster! 🚀

### Scalability
- **Depth 10**: 0.003 ms
- **Depth 50**: 0.010 ms (still fast!)
- **100 children**: 0.0002 ms per child
- **Cache hit rate**: 90-99% in typical usage

### Animation System
- Keyframe evaluation: ~0.001 ms per object
- SLERP interpolation: ~0.01 ms
- 100+ keyframes: Still sub-millisecond

## Best Practices

### 1. Hierarchy Design
✅ **Do**: Keep hierarchy depth < 15 levels when possible
✅ **Do**: Group objects logically (e.g., body parts, solar systems)
✅ **Do**: Monitor cache hit rates (aim for >90%)
❌ **Don't**: Create circular parent-child references (framework prevents this)
❌ **Don't**: Have >100 children on a single parent without grouping

### 2. Animation
✅ **Do**: Use `PoseQuat` for smooth rotations
✅ **Do**: Use `interpolation='smooth'` for natural motion
✅ **Do**: Batch keyframes for better performance
❌ **Don't**: Mix `Pose` and `PoseQuat` in the same animation
❌ **Don't**: Create thousands of very short animations

### 3. Performance
✅ **Do**: Use profiling tools to identify bottlenecks
✅ **Do**: Leverage transform caching (it's automatic!)
✅ **Do**: Use bounding boxes for collision detection
❌ **Don't**: Call `get_world_transform()` unnecessarily
❌ **Don't**: Modify poses every frame if not needed

## Architecture

### Module Structure

```
src/
├── datatypes/          # Core data types
│   ├── pose.py         # Pose and PoseQuat classes
│   ├── quaternion.py   # Quaternion implementation
│   ├── transform.py    # Transformation mathematics
│   ├── scaling.py      # Scaling data type
│   └── material.py     # Material system (Phong lighting)
├── primitives/         # 3D primitive shapes
│   ├── sphere.py       # Sphere primitive
│   ├── cylinder.py     # Cylinder primitive
│   ├── cone.py         # Cone primitive
│   ├── base_scene_object.py
│   └── opengl_primitives_wrapper.py  # OpenGL rendering
├── animation/          # Animation system
│   └── animation.py    # Keyframes, clips, player
├── utils/              # Utilities
│   ├── hierarchy_profiler.py  # Performance profiling
│   └── bounding_box.py        # Collision detection
├── scene_object.py     # Base class for all scene objects
└── lights/             # Lighting system

test/                   # Comprehensive test suite (177 tests)
examples/               # Working examples
```

### Key Classes

- **`SceneObject`**: Base class with hierarchy support
- **`Pose` / `PoseQuat`**: Position and rotation representation
- **`Quaternion`**: Rotation mathematics
- **`Material`**: Phong lighting properties (ambient, diffuse, specular, shininess)
- **`MaterialPresets`**: Realistic material library (11+ presets)
- **`Animation`**: Keyframe-based animation track
- **`AnimationPlayer`**: Playback control
- **`BoundingBox`**: Collision detection and spatial queries
- **`HierarchyProfiler`**: Performance analysis

## Documentation

- **`IMPLEMENTATION_COMPLETE.md`**: Complete system overview
- **`MATERIAL_SYSTEM_COMPLETE.md`**: Material system documentation
- **`PHASE_1_COMPLETE.md`**: Core transform system
- **`PHASE_2_COMPLETE.md`**: Rendering integration
- **`PHASE_3_COMPLETE.md`**: Optimization tools
- **`PHASE_4_COMPLETE.md`**: Advanced features

## Features by Phase

### Phase 1: Core System ✅
- Parent-child relationships
- Transform caching (140x speedup)
- Circular reference prevention
- Matrix mathematics (T × R × S)

### Phase 2: Integration ✅
- OpenGL rendering
- Example scenes
- Comprehensive testing (51 tests)

### Phase 3: Optimization ✅
- Performance profiling
- Hierarchy validation
- Cache monitoring
- Smart recommendations

### Phase 4: Advanced ✅
- Quaternion support
- Keyframe animations
- Bounding boxes
- Collision detection

### Phase 5: Materials ✅
- Phong reflection model
- Material presets (metals, gems, plastics)
- Automatic OpenGL integration
- Color convenience methods

## License

[Your License Here]

## Contributing

[Contribution Guidelines Here]

## Support

For issues or questions:
- Check the documentation in the project root
- Review the examples in `examples/`
- Run the test suite to verify your setup

## Credits

Developed as a comprehensive 3D transformation and animation framework for Python/OpenGL applications.

---

**Status**: Production-ready ✅
**Version**: 1.1
**Tests**: 176/177 passing
**Performance**: 140x speedup with caching
**Features**: Complete hierarchical transformation, animation, and material system

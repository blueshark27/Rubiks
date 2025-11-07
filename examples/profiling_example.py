"""
Profiling Example - Demonstrates hierarchy analysis and optimization tools

This example shows how to use the profiling utilities to:
1. Analyze hierarchy structure
2. Monitor cache performance
3. Get optimization recommendations
4. Visualize the hierarchy tree
"""

import numpy as np

from src.primitives.sphere import Sphere
from src.datatypes.pose import Pose
from src.utils.hierarchy_profiler import HierarchyProfiler, create_profiling_report


def create_test_hierarchy(depth: int, branching: int, name_prefix: str = ""):
    """
    Create a balanced tree hierarchy for testing.

    Args:
        depth: Depth of the tree
        branching: Number of children per node
        name_prefix: Prefix for object names

    Returns:
        Root object
    """
    def create_node(level: int, parent_name: str = ""):
        pose = Pose(
            translation=np.array([[level * 1.0], [0], [0]]),
            rotation=np.array([[0], [0], [level * 0.1]])
        )

        name = f"{name_prefix}L{level}" if not parent_name else f"{parent_name}_C"
        node = Sphere(pose=pose, radius=0.5, subdivision=0, name=name)  # subdivision=0 for speed

        if level < depth:
            for i in range(branching):
                child = create_node(level + 1, f"{name}_{i}")
                node.add_child(child)

        return node

    return create_node(0)


def example_basic_profiling():
    """Demonstrate basic profiling of a hierarchy"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Profiling")
    print("=" * 70)

    # Create a test hierarchy
    root = create_test_hierarchy(depth=4, branching=3, name_prefix="Node")

    # Profile using convenience function
    result = create_profiling_report(root, "Test Hierarchy (depth=4, branching=3)")

    # Print hierarchy structure
    print("\n[HIERARCHY STRUCTURE]")
    root.print_hierarchy()


def example_cache_monitoring():
    """Demonstrate cache performance monitoring"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Cache Performance Monitoring")
    print("=" * 70)

    # Create hierarchy
    root = create_test_hierarchy(depth=3, branching=4, name_prefix="CacheTest")

    # Collect all objects
    def get_all_objects(obj):
        objects = [obj]
        for child in obj.get_children():
            objects.extend(get_all_objects(child))
        return objects

    all_objects = get_all_objects(root)

    # Reset cache statistics
    for obj in all_objects:
        obj.reset_cache_statistics()

    print(f"\nCreated hierarchy with {len(all_objects)} objects")

    # Simulate some work - access transforms multiple times
    print("\nSimulating 100 transform accesses per object...")
    for _ in range(100):
        for obj in all_objects:
            obj.get_world_transform()

    # Show cache statistics for each object
    print("\n[CACHE STATISTICS PER OBJECT]")
    print(f"{'Object':<25} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10}")
    print("-" * 55)

    for obj in all_objects[:10]:  # Show first 10 objects
        stats = obj.get_cache_statistics()
        print(f"{obj.get_name():<25} {stats['hits']:8} {stats['misses']:8} "
              f"{stats['hit_rate_percent']:9.1f}%")

    if len(all_objects) > 10:
        print(f"... and {len(all_objects) - 10} more objects")


def example_hierarchy_validation():
    """Demonstrate hierarchy validation and warnings"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hierarchy Validation")
    print("=" * 70)

    # Create a deep hierarchy that will trigger warnings
    print("\n[1] Creating DEEP hierarchy (depth=12)...")
    deep_root = create_test_hierarchy(depth=12, branching=2, name_prefix="Deep")

    # Set custom threshold
    deep_root.set_depth_warning_threshold(10)

    # Validate
    warnings = deep_root.validate_hierarchy()
    if warnings:
        print(f"\n[WARNING] Found {len(warnings)} warnings:")
        for i, warning in enumerate(warnings[:5], 1):  # Show first 5
            print(f"  {i}. {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
    else:
        print("\n[OK] No warnings found")

    # Create a wide hierarchy
    print("\n[2] Creating WIDE hierarchy (120 children)...")
    wide_root = Sphere(
        pose=Pose(
            translation=np.array([[0], [0], [0]]),
            rotation=np.array([[0], [0], [0]])
        ),
        subdivision=0,
        name="WideParent"
    )

    # Add 120 children
    for i in range(120):
        child = Sphere(
            pose=Pose(
                translation=np.array([[i * 0.1], [0], [0]]),
                rotation=np.array([[0], [0], [0]])
            ),
            subdivision=0,
            name=f"Child{i}"
        )
        wide_root.add_child(child)

    warnings = wide_root.validate_hierarchy()
    if warnings:
        print(f"\n[WARNING] Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def example_hierarchy_statistics():
    """Demonstrate hierarchy statistics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Hierarchy Statistics")
    print("=" * 70)

    # Create hierarchy
    root = create_test_hierarchy(depth=5, branching=2, name_prefix="Stats")

    # Get statistics for root
    stats = root.get_hierarchy_statistics()

    print(f"\n[STATISTICS FOR ROOT]")
    print(f"  Depth from root:     {stats['depth']}")
    print(f"  Direct children:     {stats['children']}")
    print(f"  Total descendants:   {stats['descendants']}")
    print(f"  Subtree depth:       {stats['subtree_depth']}")

    # Get statistics for a leaf
    def get_leaf(obj):
        if not obj.get_children():
            return obj
        return get_leaf(obj.get_children()[0])

    leaf = get_leaf(root)
    leaf_stats = leaf.get_hierarchy_statistics()

    print(f"\n[STATISTICS FOR DEEPEST LEAF: {leaf.get_name()}]")
    print(f"  Depth from root:     {leaf_stats['depth']}")
    print(f"  Direct children:     {leaf_stats['children']}")
    print(f"  Total descendants:   {leaf_stats['descendants']}")
    print(f"  Subtree depth:       {leaf_stats['subtree_depth']}")


def example_benchmark():
    """Demonstrate performance benchmarking"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Performance Benchmarking")
    print("=" * 70)

    profiler = HierarchyProfiler()

    # Test different hierarchy configurations
    configs = [
        (3, 4, "Small (3x4)"),
        (5, 3, "Medium (5x3)"),
        (7, 2, "Deep (7x2)")
    ]

    results = []

    for depth, branching, name in configs:
        print(f"\n[BENCHMARK] Testing {name}...")
        root = create_test_hierarchy(depth, branching, name.split()[0])

        # Benchmark transform updates
        bench = profiler.benchmark_transform_updates(root, iterations=50)

        print(f"  Total time:          {bench['total_time_ms']:.2f} ms")
        print(f"  Per iteration:       {bench['avg_per_iteration_ms']:.4f} ms")
        print(f"  Objects updated:     {bench['objects_updated']}")

        # Profile for statistics
        result = profiler.profile(root, name)
        results.append(result)

    # Compare first and last
    if len(results) >= 2:
        print("\n")
        profiler.compare_results(results[0], results[-1])


def example_visualization():
    """Demonstrate hierarchy visualization"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Hierarchy Visualization")
    print("=" * 70)

    # Create a small hierarchy for clear visualization
    root = create_test_hierarchy(depth=3, branching=2, name_prefix="Visual")

    # Access transforms to generate cache statistics
    def access_all(obj, times=10):
        for _ in range(times):
            obj.get_world_transform()
        for child in obj.get_children():
            access_all(child, times)

    access_all(root)

    print("\n[HIERARCHY TREE] (with cache statistics)")
    root.print_hierarchy(show_stats=True)


def main():
    """Run all profiling examples"""
    print("\n" + "=" * 70)
    print("HIERARCHY PROFILING & OPTIMIZATION EXAMPLES")
    print("=" * 70)

    example_basic_profiling()
    example_cache_monitoring()
    example_hierarchy_validation()
    example_hierarchy_statistics()
    example_benchmark()
    example_visualization()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  [*] Use profiling to identify performance bottlenecks")
    print("  [*] Monitor cache hit rates (aim for >90%)")
    print("  [*] Keep hierarchy depth reasonable (<15 levels)")
    print("  [*] Validate hierarchies to catch issues early")
    print("  [*] Use benchmarking to compare different structures")
    print()


if __name__ == "__main__":
    main()

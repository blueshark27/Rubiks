"""
Hierarchy Profiler - Performance Analysis and Optimization Tools

This module provides utilities for profiling, analyzing, and optimizing
scene hierarchies. Use these tools to identify performance bottlenecks
and get recommendations for optimization.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProfileResult:
    """Result of a profiling operation"""
    name: str
    total_objects: int
    max_depth: int
    avg_depth: float
    total_cache_hits: int
    total_cache_misses: int
    cache_hit_rate: float
    warnings: List[str]
    execution_time_ms: float


class HierarchyProfiler:
    """
    Profiler for analyzing scene hierarchies and providing optimization recommendations.
    """

    def __init__(self):
        self.results: List[ProfileResult] = []

    def profile(self, root_object, name: str = "Unnamed Hierarchy") -> ProfileResult:
        """
        Profile a hierarchy starting from a root object.

        Args:
            root_object: The root of the hierarchy to profile
            name: Name for this profiling session

        Returns:
            ProfileResult with comprehensive statistics
        """
        start_time = time.perf_counter()

        # Collect all objects in hierarchy
        all_objects = self._collect_all_objects(root_object)

        # Calculate statistics
        total_objects = len(all_objects)
        depths = [obj.get_depth() for obj in all_objects]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0

        # Cache statistics
        total_hits = sum(obj._cache_hits for obj in all_objects)
        total_misses = sum(obj._cache_misses for obj in all_objects)
        total_accesses = total_hits + total_misses
        hit_rate = (total_hits / total_accesses * 100) if total_accesses > 0 else 0

        # Validate and collect warnings
        warnings = root_object.validate_hierarchy()

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # ms

        result = ProfileResult(
            name=name,
            total_objects=total_objects,
            max_depth=max_depth,
            avg_depth=avg_depth,
            total_cache_hits=total_hits,
            total_cache_misses=total_misses,
            cache_hit_rate=hit_rate,
            warnings=warnings,
            execution_time_ms=execution_time
        )

        self.results.append(result)
        return result

    def _collect_all_objects(self, obj) -> List:
        """Recursively collect all objects in hierarchy"""
        objects = [obj]
        for child in obj.get_children():
            objects.extend(self._collect_all_objects(child))
        return objects

    def print_report(self, result: ProfileResult):
        """
        Print a detailed profiling report.

        Args:
            result: The ProfileResult to report on
        """
        print("\n" + "=" * 70)
        print(f"HIERARCHY PROFILING REPORT: {result.name}")
        print("=" * 70)

        print(f"\n[HIERARCHY STATISTICS]")
        print(f"  Total Objects:       {result.total_objects}")
        print(f"  Maximum Depth:       {result.max_depth}")
        print(f"  Average Depth:       {result.avg_depth:.2f}")

        print(f"\n[CACHE PERFORMANCE]")
        print(f"  Cache Hits:          {result.total_cache_hits}")
        print(f"  Cache Misses:        {result.total_cache_misses}")
        print(f"  Hit Rate:            {result.cache_hit_rate:.2f}%")

        # Interpret cache performance
        if result.cache_hit_rate > 90:
            print(f"  Status:              [EXCELLENT]")
        elif result.cache_hit_rate > 70:
            print(f"  Status:              [GOOD]")
        elif result.cache_hit_rate > 50:
            print(f"  Status:              [FAIR] (consider optimization)")
        else:
            print(f"  Status:              [POOR] (needs optimization)")

        print(f"\n[PROFILING TIME]        {result.execution_time_ms:.4f} ms")

        if result.warnings:
            print(f"\n[WARNINGS] ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print(f"\n[OK] NO WARNINGS - Hierarchy looks good!")

        print("\n" + "=" * 70)

    def get_optimization_recommendations(self, result: ProfileResult) -> List[str]:
        """
        Generate optimization recommendations based on profiling results.

        Args:
            result: The ProfileResult to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Deep hierarchy check
        if result.max_depth > 20:
            recommendations.append(
                f"[FIX] Deep hierarchy detected (depth={result.max_depth}). "
                f"Consider flattening by grouping objects or restructuring."
            )
        elif result.max_depth > 10:
            recommendations.append(
                f"[TIP] Moderate hierarchy depth ({result.max_depth}). "
                f"Monitor performance if it increases further."
            )

        # Cache performance check
        if result.cache_hit_rate < 50:
            recommendations.append(
                f"[FIX] Low cache hit rate ({result.cache_hit_rate:.1f}%). "
                f"Objects may be updating too frequently. Consider batching updates."
            )
        elif result.cache_hit_rate < 70:
            recommendations.append(
                f"[TIP] Cache hit rate could be improved ({result.cache_hit_rate:.1f}%). "
                f"Review update patterns."
            )

        # Large hierarchy check
        if result.total_objects > 1000:
            recommendations.append(
                f"[FIX] Large hierarchy ({result.total_objects} objects). "
                f"Consider spatial partitioning or level-of-detail techniques."
            )
        elif result.total_objects > 500:
            recommendations.append(
                f"[TIP] Growing hierarchy ({result.total_objects} objects). "
                f"Monitor memory usage as it scales."
            )

        # Specific warnings
        if result.warnings:
            recommendations.append(
                f"[ACTION] Address {len(result.warnings)} validation warnings above."
            )

        if not recommendations:
            recommendations.append("[OK] Hierarchy is well-optimized! No recommendations.")

        return recommendations

    def compare_results(self, result1: ProfileResult, result2: ProfileResult):
        """
        Compare two profiling results.

        Args:
            result1: First profiling result
            result2: Second profiling result
        """
        print("\n" + "=" * 70)
        print(f"COMPARISON: '{result1.name}' vs '{result2.name}'")
        print("=" * 70)

        print(f"\n[SIZE COMPARISON]")
        print(f"  Objects:      {result1.total_objects:6} vs {result2.total_objects:6}")
        print(f"  Max Depth:    {result1.max_depth:6} vs {result2.max_depth:6}")
        print(f"  Avg Depth:    {result1.avg_depth:6.2f} vs {result2.avg_depth:6.2f}")

        print(f"\n[CACHE COMPARISON]")
        print(f"  Hit Rate:     {result1.cache_hit_rate:6.2f}% vs {result2.cache_hit_rate:6.2f}%")
        cache_diff = result2.cache_hit_rate - result1.cache_hit_rate
        if cache_diff > 0:
            print(f"  Improvement:  +{cache_diff:.2f}% [BETTER]")
        elif cache_diff < 0:
            print(f"  Change:       {cache_diff:.2f}% [WORSE]")
        else:
            print(f"  Change:       No change")

        print(f"\n[WARNINGS]")
        print(f"  {result1.name}: {len(result1.warnings)} warnings")
        print(f"  {result2.name}: {len(result2.warnings)} warnings")

        print("\n" + "=" * 70)

    def benchmark_transform_updates(self, root_object, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark transform update performance.

        Args:
            root_object: Root of hierarchy to benchmark
            iterations: Number of update iterations

        Returns:
            Dictionary with benchmark results
        """
        import numpy as np
        from src.datatypes.pose import Pose

        # Reset cache statistics
        all_objects = self._collect_all_objects(root_object)
        for obj in all_objects:
            obj.reset_cache_statistics()

        # Benchmark updates
        start = time.perf_counter()
        for i in range(iterations):
            new_pose = Pose(
                translation=np.array([[i * 0.1], [0], [0]]),
                rotation=np.array([[0], [0], [i * 0.01]])
            )
            root_object.set_pose(new_pose)

            # Force recalculation by getting world transforms
            for obj in all_objects:
                obj.get_world_transform()

        end = time.perf_counter()
        total_time = (end - start) * 1000  # ms

        return {
            'total_time_ms': total_time,
            'avg_per_iteration_ms': total_time / iterations,
            'iterations': iterations,
            'objects_updated': len(all_objects)
        }


def create_profiling_report(root_object, name: str = "Scene") -> ProfileResult:
    """
    Convenience function to quickly profile a hierarchy and print report.

    Args:
        root_object: Root of hierarchy to profile
        name: Name for the hierarchy

    Returns:
        ProfileResult
    """
    profiler = HierarchyProfiler()
    result = profiler.profile(root_object, name)
    profiler.print_report(result)

    print("\n[OPTIMIZATION RECOMMENDATIONS]")
    recommendations = profiler.get_optimization_recommendations(result)
    for rec in recommendations:
        print(f"  {rec}")

    return result

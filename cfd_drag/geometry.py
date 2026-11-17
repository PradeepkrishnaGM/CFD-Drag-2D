"""Obstacle geometry, mirroring the conventions in cfd_engine/solver.cpp.

Local frame: rect is size1 wide (x) by size2 tall (y); tri is an isosceles triangle with
base size1 (along y, on the left) and length size2 (tip pointing +x); circle has radius
size1. A shape rotated by `angle` degrees (counter-clockwise) maps local (lx, ly) to
x = lx cos t - ly sin t, y = lx sin t + ly cos t.
"""
import math


def vertices(shape: str, size1: float, size2: float) -> list[tuple[float, float]]:
    if shape == "rect":
        return [(-size1 / 2, -size2 / 2), (size1 / 2, -size2 / 2),
                (size1 / 2, size2 / 2), (-size1 / 2, size2 / 2)]
    if shape == "tri":
        return [(size2 / 2, 0.0), (-size2 / 2, size1 / 2), (-size2 / 2, -size1 / 2)]
    raise ValueError(f"no vertices for shape {shape!r}")


def extents(shape: str, size1: float, size2: float, angle: float) -> tuple[float, float]:
    """Bounding-box size (dx, dy) of the rotated shape. dy is the frontal height D."""
    if shape == "circle":
        return 2.0 * size1, 2.0 * size1
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    xs = [lx * c - ly * s for lx, ly in vertices(shape, size1, size2)]
    ys = [lx * s + ly * c for lx, ly in vertices(shape, size1, size2)]
    return max(xs) - min(xs), max(ys) - min(ys)


def frontal_height(shape: str, size1: float, size2: float, angle: float) -> float:
    return extents(shape, size1, size2, angle)[1]

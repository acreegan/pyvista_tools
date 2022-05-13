from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
import itertools

"""
geometry_tools is a module that provides functions for making geometric calculations in support of pyvista_tools. These
functions should not rely on pyvista specific types or data structures.
"""


def find_sequence(array: ArrayLike, sequence: ArrayLike, check_reverse=False) -> int:
    """
    Find the start index of a subsequence in an array.

    Parameters
    ----------
    array
        Array in which to search for sequence
    sequence
        Sequence to search for
    check_reverse
        Also search for the reverse of the sequence. The forward sequence is still given precedence.

    Returns
    -------
    Location
        Start index of sequnce in array. -1 represents not found.

    """
    location = -1
    # hstack array so we can find sequences that wrap around
    search_array = np.hstack((array, array))
    for i in range(len(search_array) - len(sequence) + 1):
        if np.all(search_array[i:i + len(sequence)] == sequence):
            location = i
            break

    if location == -1 and check_reverse:
        location = find_sequence(array, sequence[::-1], check_reverse=False)

    return location


def winding_order_agrees_with_normal(points: ArrayLike, normal) -> bool:
    """
    Calculate whether the normal vector implied by a list of points agrees with the given normal vector

    Parameters
    ----------
    points
    normal

    Returns
    -------
    agrees:
        True if the dot product between expected normal and given normal is positive. False otherwise

    """
    expected_normal = compute_normal(points)
    dot = np.dot(expected_normal, normal)
    agrees = dot > 0

    return agrees


def compute_normal(points: ArrayLike) -> np.ndarray:
    """
    Compute a vector that is normal to a plane defined by a list of three points

    Parameters
    ----------
    points
        Points defining a plane

    Returns
    -------
    normal
        Vector that is normal to the plane defined by the input points

    """
    if len(points) < 3:
        raise ValueError("Need at least three points to compute a normal")

    normal = np.cross(points[1] - points[0], points[2] - points[1])
    return normal


def find_loops_and_chains(lines: ArrayLike):
    """
    Classify connected "loops" and "chains" in a list of line segments represented by 2 Tuples. Chains are lists of
    connected segments with two loose ends. Loops are lists of connected segments without loose ends.

    Parameters
    ----------
    lines: Nx2 ArrayLike
    """
    edges = []
    for line in lines:
        line_in_loops = [line[0] in itertools.chain(*edge) or line[1] in itertools.chain(*edge) for edge in edges]
        # If either end of the line is already in a loop, add the line to that loop
        if np.any(line_in_loops):
            edges[np.argmax(line_in_loops)].add(tuple(line))
        # Otherwise, start a new loop
        else:
            s = set()
            s.add(tuple(line))
            edges.append(s)

    # Before sorting, classify into loops and chains
    # Loops have all nodes exactly twice. Chains have one line with a unique node 0, and one line with a unique node 1
    # To sort chains, we need to start with the line with the unique node 0
    loops = []
    chains = []
    for edge in edges:
        starts, ends = tuple(zip(*edge))
        if set(starts) == set(ends):
            # To guarantee consistent behavior, arbitarily set the start node of a loop to the minimum node index
            loops.append({"start": min(starts), "edge": edge})
        else:
            chains.append({"start": list(set(starts) - set(ends))[0], "edge": edge})

    # Sort
    sorted_loops = [sort_edge(loop["edge"], loop["start"]) for loop in loops]
    sorted_chains = [sort_edge(chain["edge"], chain["start"]) for chain in chains]

    return sorted_loops, sorted_chains


def sort_edge(edge: ArrayLike, start_node=None) -> ArrayLike:
    """
    Sort an edge represented by a list of 2 Tuples such that connected nodes are sequential in the list.

    Parameters
    ----------
    edge
    start_node

    Returns
    -------
    sorted_edge

    """
    sorted_edge = []
    edge = list(edge)

    if start_node is not None:
        start_index = np.argmax([line[0] == start_node for line in edge])
    else:
        start_index = 0

    sorted_edge.append(edge.pop(start_index))  # Start with first item
    for _ in range(len(edge)):
        # Next item in loop is index where the start of the line is the end of the current line
        next_index = np.argmax([line[0] == sorted_edge[-1][1] for line in edge])
        sorted_edge.append(edge.pop(next_index))

    return sorted_edge


def dihedral_angle(normal_a: ArrayLike, normal_b: ArrayLike, plane_normal: ArrayLike = None, degrees=False) -> float:
    """
    Calculate dihedral angle between two faces specified by their normal vectors, with 0 < angle < pi. Optionally, an
    additional normal can be given, defining a plane on which normal_a and normal_b lie. With this information, the
    dihedral angle can be given as 0 < angle < 2*pi

    Parameters
    ----------
    normal_a
        Normal vector A
    normal_b
        Normal vector B
    plane_normal
        Vector that is normal to the plane that normal_a and normal_b lie on (it is perpendicular to both). The direction
        of this vector will be used to determine if the dihedral angle is positive or negative, thus allowing the output
        to be between 0 and 2pi
    degrees
        Return the angle in degrees

    Returns
    -------
    angle
        Dihedral angle in radians (or optionally degrees)

    """
    length_product = np.linalg.norm(normal_a) * np.linalg.norm(normal_b)
    dot_product = np.dot(normal_a, normal_b)
    cosine = dot_product / length_product
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))  # Avoid rounding errors resulting in nan

    if plane_normal is not None:
        cross_product = np.cross(normal_a, normal_b)
        direction = np.dot(plane_normal, cross_product)
        if direction < 0:
            angle = 2 * np.pi - angle

    if degrees:
        angle = np.rad2deg(angle)

    return angle

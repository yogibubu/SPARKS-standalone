"""Differentiable local R/A/D displacements of a frozen SMITH definition.

This is an evaluation adapter, not a coordinate selector. Unsupported
primitive functions fail explicitly. Periodic displacements are centered at
the supplied reference before forming the frozen linear combinations.
"""
import numpy as np


def valence_displacement_torch(definition, reference_coordinates):
    """Return a Torch callable with the existing SMITH dihedral convention.

    Atom indices and linear combinations come exclusively from the frozen
    definition. Derivatives of every order are supplied by Torch; no finite
    displacement of the electronic potential is involved.
    """
    import torch
    reference = np.asarray(reference_coordinates, float)
    if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all():
        raise ValueError('finite reference Cartesian coordinates required')
    primitives = {p.identifier: p for p in definition.primitives}
    selected = tuple(dict.fromkeys(pid for row in definition.gics
        for pid, _ in (row.coefficients or ((row.primitive_id, 1.),))))
    items = tuple(primitives[pid] for pid in selected)
    if any(p.function not in {'R', 'A', 'D'} for p in items):
        raise ValueError('differentiable valence adapter supports frozen R/A/D primitives only')
    for p in items:
        if len(p.atoms) != {'R':2,'A':3,'D':4}[p.function] or any(
                not 1 <= i <= len(reference) for i in p.atoms):
            raise ValueError('valid frozen valence atom indices required')
        x = reference[np.array(p.atoms)-1]
        if p.function == 'R' and np.linalg.norm(x[1]-x[0]) <= 1e-12:
            raise ValueError('coincident reference atoms')
        if p.function in {'A','D'}:
            if np.linalg.norm(np.cross(x[0]-x[1],x[2]-x[1])) <= 1e-10:
                raise ValueError('reference valence angle is singular')
        if p.function == 'D' and np.linalg.norm(np.cross(x[1]-x[2],x[3]-x[2])) <= 1e-10:
            raise ValueError('reference valence dihedral is singular')
    indices = {pid: i for i, pid in enumerate(selected)}
    coefficients = np.zeros((len(definition.gics), len(items)))
    for i, row in enumerate(definition.gics):
        for pid, value in (row.coefficients or ((row.primitive_id, 1.),)):
            coefficients[i, indices[pid]] += value

    def values(x):
        result = []
        for p in items:
            positions = [x[i-1] for i in p.atoms]
            if p.function == 'R':
                value = torch.linalg.vector_norm(positions[1]-positions[0])
            elif p.function == 'A':
                a, b, c = positions; u, v = a-b, c-b
                value = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u, v)), torch.dot(u, v))
            else:
                a, b, c, d = positions
                axis = (c-b)/torch.linalg.vector_norm(c-b)
                v = a-b; v = v-torch.dot(v, axis)*axis
                w = d-c; w = w-torch.dot(w, axis)*axis
                value = torch.atan2(-torch.dot(torch.linalg.cross(axis, v), w), -torch.dot(v, w))
            result.append(value)
        return torch.stack(result)

    origin = values(torch.tensor(reference, dtype=torch.float64)).detach()
    periodic = tuple(p.function == 'D' for p in items)

    def displacement(x):
        delta = values(x)-origin.to(x)
        delta = torch.stack([torch.atan2(torch.sin(v), torch.cos(v)) if periodic[i] else v
                             for i, v in enumerate(delta)])
        return torch.as_tensor(coefficients, dtype=x.dtype, device=x.device)@delta
    return displacement

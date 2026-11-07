"""Gaussian Gram coefficients on the exact lowering closure of a finite basis."""
import numpy as np
from numba import njit


@njit(cache=True)
def _closed_gram(rows,lower,q,r,h):
    size,width=rows.shape;gram=np.empty((size,size))
    gram[0,0]=1.
    for i in range(size):
        for j in range(i+1):
            if i==0:continue
            axis=0
            while rows[i,axis]==0:axis+=1
            previous=lower[i,axis]
            value=h[axis]*gram[previous,j]
            for k in range(width):
                left=rows[previous,k]
                if left:value+=q[axis,k]*np.sqrt(left)*gram[lower[previous,k],j]
                right=rows[j,k]
                if right:value+=r[axis,k]*np.sqrt(right)*gram[previous,lower[j,k]]
            value/=np.sqrt(rows[i,axis])
            gram[i,j]=value;gram[j,i]=value
    return gram


def gaussian_gram_on_lowering_closure(basis,q,r,h,*,maximum_entries=8000000):
    """Exact Mehler Gram matrix without a Cartesian bounding-box tensor.

    The symmetric generating matrix is [[q,r],[r.T,q]], with repeated linear
    vector h. Every needed lower occupation is retained; nothing is pruned by
    coefficient size. The resource guard applies to the actual closure square.
    """
    basis=np.asarray(basis);q=np.asarray(q,float);r=np.asarray(r,float);h=np.asarray(h,float)
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or np.any(basis<0)
            or not len(basis) or q.shape!=(basis.shape[1],)*2 or r.shape!=q.shape
            or h.shape!=(basis.shape[1],) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(h))
            or not np.allclose(q,q.T,atol=1e-12,rtol=1e-12)
            or not np.allclose(r,r.T,atol=1e-12,rtol=1e-12)
            or not isinstance(maximum_entries,int) or maximum_entries<1):
        raise ValueError('finite Gaussian matrices, nonnegative occupations and a positive budget required')
    from math import isqrt
    rows,lower,take=gaussian_lowering_closure(basis,maximum_states=isqrt(maximum_entries))
    gram=_closed_gram(rows,lower,q,r,h)
    return gram[np.ix_(take,take)],dict(initial_moment_grid_entries=len(rows)**2,
        lowering_closure_size=len(rows),moment_algorithm='exact lowering-closure Gaussian Gram',
        coefficient_pruning_used=False)


def gaussian_lowering_closure(basis, *, maximum_states):
    """Shared exact downward closure, its lower neighbors and original row map."""
    basis=np.asarray(basis)
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or not len(basis) or np.any(basis<0)
            or isinstance(maximum_states,(bool,np.bool_))
            or not isinstance(maximum_states,(int,np.integer)) or maximum_states<1):
        raise ValueError('nonnegative integer occupations and a positive closure budget required')
    original=[tuple(int(x) for x in row) for row in basis]
    closure=set(original);pending=list(closure)
    if len(closure)>maximum_states:raise MemoryError('Gaussian lowering closure exceeds declared state budget')
    while pending:
        row=pending.pop()
        for axis,count in enumerate(row):
            if count:
                child=row[:axis]+(count-1,)+row[axis+1:]
                if child not in closure:
                    closure.add(child);pending.append(child)
                    if len(closure)>maximum_states:
                        raise MemoryError('Gaussian lowering closure exceeds declared state budget')
    ordered=sorted(closure,key=lambda row:(sum(row),row));lookup={row:i for i,row in enumerate(ordered)}
    rows=np.asarray(ordered,dtype=np.int64);lower=np.full(rows.shape,-1,dtype=np.int64)
    for i,row in enumerate(ordered):
        for axis,count in enumerate(row):
            if count:lower[i,axis]=lookup[row[:axis]+(count-1,)+row[axis+1:]]
    return rows, lower, np.array([lookup[row] for row in original])


@njit(cache=True)
def _rectangular_overlap(initial, ilower, final, flower, a, b, cross, hi, hf, ground):
    table = np.empty((len(final), len(initial)), dtype=np.complex128)
    table[0, 0] = ground
    for f in range(1, len(final)):
        axis = 0
        while final[f, axis] == 0:
            axis += 1
        previous = flower[f, axis]
        value = hf[axis] * table[previous, 0]
        for k in range(final.shape[1]):
            count = final[previous, k]
            if count:
                value += b[axis, k] * np.sqrt(count) * table[flower[previous, k], 0]
        table[f, 0] = value / np.sqrt(final[f, axis])
    for i in range(1, len(initial)):
        axis = 0
        while initial[i, axis] == 0:
            axis += 1
        previous = ilower[i, axis]
        for f in range(len(final)):
            value = hi[axis] * table[f, previous]
            for k in range(initial.shape[1]):
                count = initial[previous, k]
                if count:
                    value += a[axis, k] * np.sqrt(count) * table[f, ilower[previous, k]]
                count = final[f, k]
                if count:
                    value += cross[axis, k] * np.sqrt(count) * table[flower[f, k], previous]
            table[f, i] = value / np.sqrt(initial[i, axis])
    return table


def gaussian_overlap_on_lowering_closures(initial, final, matrix, linear, ground, *, maximum_entries):
    """Exact rectangular two-surface Gaussian overlap on downward closures.

    The Gaussian generator retains all modes, including vacuum spectators.
    This avoids the Cartesian occupation bounding box without pruning terms.
    """
    initial, final = np.asarray(initial), np.asarray(final)
    if initial.ndim != 2 or final.ndim != 2 or initial.shape[1] != final.shape[1]:
        raise ValueError('compatible initial and final occupation rows required')
    n = initial.shape[1]
    matrix, linear = np.asarray(matrix, complex), np.asarray(linear, complex)
    if (matrix.shape != (2*n, 2*n) or linear.shape != (2*n,)
            or not np.isfinite(matrix).all() or not np.isfinite(linear).all()
            or not np.isfinite(ground) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
            or isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int) or maximum_entries < 1):
        raise ValueError('finite symmetric Gaussian generator and positive entry budget required')
    f, fl, ftake = gaussian_lowering_closure(final, maximum_states=max(1, maximum_entries//len(initial)))
    i, il, itake = gaussian_lowering_closure(initial, maximum_states=max(1, maximum_entries//len(f)))
    if len(i)*len(f) > maximum_entries:
        raise MemoryError('Gaussian rectangular closure exceeds declared entry budget')
    table = _rectangular_overlap(i, il, f, fl, matrix[:n,:n], matrix[n:,n:],
                                 matrix[:n,n:], linear[:n], linear[n:], complex(ground))
    return table[np.ix_(ftake, itake)], dict(gaussian_grid_entries=len(i)*len(f),
        initial_lowering_closure_size=len(i), final_lowering_closure_size=len(f),
        overlap_algorithm='exact rectangular lowering closures')

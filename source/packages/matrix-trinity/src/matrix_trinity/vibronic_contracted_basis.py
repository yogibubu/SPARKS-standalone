"""Energy-selected products of a retained core spectrum and one modal basis.

The contraction is a numerical subspace, not a change of potential. Its
energy ceiling requires independent convergence checks on the observable.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ContractedModeProduct:
    core_energies_cm: np.ndarray
    core_vectors: np.ndarray
    extra_energies_cm: np.ndarray
    energy_ceiling_cm: float
    maximum_states: int=10000
    maximum_workspace_bytes: int=8*1024**3

    def __post_init__(self):
        e=np.asarray(self.core_energies_cm,float);u=np.asarray(self.core_vectors)
        x=np.asarray(self.extra_energies_cm,float)
        if (e.ndim!=1 or not len(e) or x.ndim!=1 or not len(x) or
                u.ndim!=2 or u.shape[1]!=len(e) or u.shape[0]<len(e) or
                not np.all(np.isfinite(e)) or not np.all(np.isfinite(x)) or
                not np.all(np.isfinite(u)) or np.any(np.diff(e)<0) or
                not np.isfinite(self.energy_ceiling_cm) or self.maximum_states<1):
            raise ValueError('finite ordered core spectrum and extra modal energies required')
        self.counts=np.array([np.count_nonzero(e+value<=self.energy_ceiling_cm) for value in x])
        self.offsets=np.r_[0,np.cumsum(self.counts)]
        self.size=int(self.offsets[-1]);self.full_size=u.shape[0]*len(x)
        if not 0<self.size<=self.maximum_states:raise MemoryError('contracted product state budget exceeded')
        dtype=np.result_type(u.dtype,np.float64)
        self.workspace_bytes=4*self.size**2*np.dtype(dtype).itemsize+u.nbytes
        if self.workspace_bytes>self.maximum_workspace_bytes:raise MemoryError('contracted product workspace exceeded')
        used=int(self.counts.max())
        if not np.allclose(u[:,:used].conj().T@u[:,:used],np.eye(used),rtol=1e-10,atol=1e-10):
            raise ValueError('orthonormal core states required')
        self.core_energies_cm=e;self.core_vectors=u;self.extra_energies_cm=x
        self.diagnostics=dict(contracted_states=self.size,full_product_states=self.full_size,
            core_states_by_extra_level=self.counts.tolist(),energy_ceiling_cm=float(self.energy_ceiling_cm),
            potential_modified=False,contraction_convergence_certified=False,
            numeric_workspace_estimate_bytes=self.workspace_bytes)

    def lift(self, coefficients):
        coefficients=np.asarray(coefficients)
        if coefficients.ndim!=2 or coefficients.shape[0]!=self.size:raise ValueError('contracted coefficient columns required')
        dtype=np.result_type(coefficients,self.core_vectors)
        if self.full_size*coefficients.shape[1]*np.dtype(dtype).itemsize>self.maximum_workspace_bytes:
            raise MemoryError('lifted coefficients exceed workspace')
        result=np.zeros((self.full_size,coefficients.shape[1]),dtype=dtype)
        n=len(self.core_vectors)
        for level,count in enumerate(self.counts):
            if count:result[level*n:(level+1)*n]=self.core_vectors[:,:count]@coefficients[self.offsets[level]:self.offsets[level+1]]
        return result

    def project(self, vectors):
        vectors=np.asarray(vectors)
        if vectors.ndim!=2 or vectors.shape[0]!=self.full_size:raise ValueError('full product coefficient columns required')
        result=np.empty((self.size,vectors.shape[1]),dtype=np.result_type(vectors,self.core_vectors))
        n=len(self.core_vectors)
        for level,count in enumerate(self.counts):
            if count:result[self.offsets[level]:self.offsets[level+1]]=self.core_vectors[:,:count].conj().T@vectors[level*n:(level+1)*n]
        return result

    def project_matrix(self, matrix, *, progress=None):
        """Reuse complete occupation blocks; never construct the dense embedding."""
        if matrix.shape!=(self.full_size,self.full_size):raise ValueError('matrix must use the complete original product row order')
        from scipy.sparse import issparse
        from .vibronic_sparse import _checked_matrix,_dense_hermitian_errors
        if issparse(matrix):matrix=_checked_matrix(matrix)
        else:
            finite,scale,error=_dense_hermitian_errors(np.asarray(matrix))
            if not finite or error>1e-12*scale:raise ValueError('finite Hermitian matrix required')
        n=len(self.core_vectors);result=np.zeros((self.size,self.size),dtype=np.result_type(matrix.dtype,self.core_vectors))
        total=sum(count>0 for count in self.counts);done=0
        for i,ni in enumerate(self.counts):
            if not ni:continue
            si=slice(self.offsets[i],self.offsets[i+1]);left=self.core_vectors[:,:ni]
            for j in range(i,len(self.counts)):
                nj=self.counts[j]
                if not nj:continue
                sj=slice(self.offsets[j],self.offsets[j+1])
                block=matrix[i*n:(i+1)*n,j*n:(j+1)*n]
                if hasattr(block,'toarray'):block=block.toarray()
                value=left.conj().T@(block@self.core_vectors[:,:nj])
                result[si,sj]=value
                if i!=j:result[sj,si]=value.conj().T
            done+=1
            if progress:progress(dict(completed_occupation_blocks=done,total_occupation_blocks=int(total)))
        if not np.all(np.isfinite(result)):raise FloatingPointError('nonfinite contracted Hamiltonian')
        return result


    def project_quadrature(self, core_functions, extra_functions, residual_cm, *, progress=None):
        """Project the complete native correction on the same contracted products.

        The core may be multivariate. Tables already contain square-root
        quadrature weights. No potential term, grid value or coupling is pruned.
        """
        core=np.asarray(core_functions);extra=np.asarray(extra_functions)
        residual=np.asarray(residual_cm)
        used=int(self.counts.max())
        if (core.ndim!=2 or core.shape[0]<used or extra.ndim!=2
                or len(extra)!=len(self.extra_energies_cm)
                or residual.shape!=(core.shape[1],extra.shape[1])
                or not np.all(np.isfinite(core)) or not np.all(np.isfinite(extra))
                or not np.all(np.isfinite(residual)) or np.iscomplexobj(residual)):
            raise ValueError('finite weighted core/extra tables and real residual grid required')
        if not np.allclose(core[:used].conj()@core[:used].T,np.eye(used),rtol=1e-10,atol=1e-10):
            raise ValueError('orthonormal core quadrature table required')
        if not np.allclose(extra@extra.conj().T,np.eye(len(extra)),rtol=1e-10,atol=1e-10):
            raise ValueError('orthonormal extra quadrature table required')
        dtype=np.result_type(core,extra,residual)
        required=(self.workspace_bytes+core.nbytes+residual.nbytes
                  +2*used*core.shape[1]*np.dtype(dtype).itemsize)
        if required>self.maximum_workspace_bytes:raise MemoryError('contracted quadrature projection exceeds workspace')
        diagonal=np.concatenate([self.core_energies_cm[:count]+value
            for value,count in zip(self.extra_energies_cm,self.counts) if count])
        result=np.diag(diagonal).astype(dtype)
        for i,ni in enumerate(self.counts):
            if not ni:continue
            si=slice(self.offsets[i],self.offsets[i+1])
            for j in range(i,len(self.counts)):
                nj=self.counts[j]
                if not nj:continue
                sj=slice(self.offsets[j],self.offsets[j+1])
                weight=residual@(extra[i].conj()*extra[j])
                value=core[:ni].conj()@(weight[:,None]*core[:nj].T)
                result[si,sj]+=value
                if i!=j:result[sj,si]+=value.conj().T
            if progress:progress(dict(completed_extra_levels=i+1,total_extra_levels=len(self.counts)))
        return result

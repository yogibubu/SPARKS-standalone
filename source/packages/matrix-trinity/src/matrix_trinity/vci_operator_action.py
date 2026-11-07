"""Compiled sparse action of resident canonical modal operator strings.

Local q/p matrix elements come from the existing compiler. Packed integer
occupation keys reduce allocation during accumulation; public states remain
ordinary tuples. Larger occupations or mode spaces use the resident compiler.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .curvilinear_vpt2 import _local_operator_transitions, _compiled_operator_couplings

_KEY = types.UniTuple(types.int64, 2)


@njit(cache=True)
def _action(state, modes, coefficients, counts, outputs, factors, strings):
    result = Dict.empty(key_type=_KEY, value_type=types.complex128)
    lo = np.int64(0); hi = np.int64(0)
    for mode in range(len(state)):
        if mode < 15:
            lo |= state[mode] << (4*mode)
        else:
            hi |= state[mode] << (4*(mode-15))
    for term in range(len(coefficients)):
        occupations = np.zeros(4, np.int64)
        lengths = np.ones(4, np.int64)
        for slot in range(4):
            mode = modes[term, slot]
            if mode >= 0:
                occupations[slot] = state[mode]
            lengths[slot] = counts[strings[term, slot], occupations[slot]]
        for a in range(lengths[0]):
            for b in range(lengths[1]):
                for c in range(lengths[2]):
                    for d in range(lengths[3]):
                        choices = (a,b,c,d)
                        left = lo; right = hi
                        value = complex(coefficients[term])
                        for slot in range(4):
                            string = strings[term, slot]
                            occupation = occupations[slot]
                            choice = choices[slot]
                            value *= factors[string, occupation, choice]
                            mode = modes[term, slot]
                            if mode >= 0:
                                target = outputs[string, occupation, choice]
                                if mode < 15:
                                    shift = 4*mode
                                    left = (left & ~(np.int64(15) << shift)) | (target << shift)
                                else:
                                    shift = 4*(mode-15)
                                    right = (right & ~(np.int64(15) << shift)) | (target << shift)
                        key = (left,right)
                        result[key] = result.get(key, 0j) + value
    return result


@njit(cache=True)
def _unpack(packed, width):
    rows = np.empty((len(packed), width), np.int64)
    values = np.empty(len(packed), np.complex128)
    index = 0
    for key, value in packed.items():
        left, right = key
        for mode in range(width):
            rows[index, mode] = ((left >> (4*mode)) & 15 if mode < 15 else
                                 (right >> (4*(mode-15))) & 15)
        values[index] = value
        index += 1
    return rows, values


class CompiledModalAction:
    """Exact cached local ladder products, with a general Python fallback."""
    def __init__(self, terms):
        self.terms = tuple(terms)
        self.supported = all(len(key) <= 4 and all(len(ops) <= 4 for _,ops in key)
                             for key,_ in self.terms)
        if not self.supported:
            return
        ops = [()]
        for key,_ in self.terms:
            for _,string in key:
                if string not in ops:
                    ops.append(string)
        lookup = {string:i for i,string in enumerate(ops)}
        self.modes = np.full((len(terms),4), -1, np.int64)
        self.strings = np.zeros((len(terms),4), np.int64)
        self.coefficients = np.array([value for _,value in terms], float)
        for i,(key,_) in enumerate(terms):
            for j,(mode,string) in enumerate(key):
                self.modes[i,j] = mode; self.strings[i,j] = lookup[string]
        self.counts = np.zeros((len(ops),12), np.int64)
        self.outputs = np.zeros((len(ops),12,5), np.int64)
        self.factors = np.zeros((len(ops),12,5), complex)
        for i,string in enumerate(ops):
            for n in range(12):
                transitions = _local_operator_transitions(string,n)
                self.counts[i,n] = len(transitions)
                for j,(target,value) in enumerate(transitions):
                    self.outputs[i,n,j] = target; self.factors[i,n,j] = value

    def __call__(self, state):
        state = tuple(state)
        if not self.supported or len(state)>30 or max(state,default=0)>11:
            return _compiled_operator_couplings(state,self.terms)
        packed = _action(np.asarray(state,np.int64),self.modes,self.coefficients,
                         self.counts,self.outputs,self.factors,self.strings)
        rows, values = _unpack(packed, len(state))
        return dict(zip(map(tuple, rows.tolist()), values.tolist()))

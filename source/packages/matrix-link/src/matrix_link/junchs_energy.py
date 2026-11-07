"""junChS single-point energies for RHF and ROHF reference states.

The published junChS energy formula and packaged basis artifacts are reused.
This energy-only contract is separate from the frozen singlet composite-gradient
protocol: it does not claim an open-shell analytic or numerical gradient.
"""
import hashlib
import json
from .composite_protocol import load_composite_protocol
from .scan import QMScanBackend, QMBackendTerm


def junchs_molpro_energy_backend(*, charge=0, multiplicity=1, executable=None,
                                 processors=1, memory_gb=32, timeout=14400,
                                 env=None):
    """Five signed canonical energy terms, with an explicit ROHF open shell.

    Molpro RHF-RCCSD(T) and RMP2 use the same spin-restricted open-shell
    reference convention. The frozen-core and all-electron MP2 terms share
    cc-pwCVTZ on heavy atoms and cc-pVTZ on H/He. No local CC approximation,
    empirical CV correction, electronic gradient or geometry change is used.
    """
    if isinstance(multiplicity,bool) or int(multiplicity)!=multiplicity or multiplicity<1:
        raise ValueError('positive integer spin multiplicity required')
    protocol=load_composite_protocol('junchs-molpro')
    open_shell=int(multiplicity)>1
    cc='RCCSD(T)' if open_shell else 'CCSD(T)'
    mp='RMP2' if open_shell else 'MP2'
    reference='ROHF' if open_shell else 'RHF'
    contract=dict(schema='matrix.link.junchs_energy.molpro.v1',
        parent_energy_formula_protocol=protocol.protocol_id,
        parent_protocol_sha256=protocol.sha256,formula=protocol.payload['formula'],
        reference=reference,coupled_cluster_command=cc,MP2_command=mp,
        properties=['energy'],derivatives_qualified=False,
        reference_documentation='https://www.molpro.com/manual/doku.php?id=open-shell_coupled_cluster_theories')
    contract['energy_contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    common=dict(name='molpro',charge=int(charge),multiplicity=int(multiplicity),
        executable=executable,processors=processors,memory_gb=memory_gb,
        timeout=timeout,env=env,properties=('energy',),gradient_mode='energy',
        resolution=contract)
    def term(label,coefficient,method,basis,basis_file=None):
        return QMBackendTerm(label,coefficient,QMScanBackend(method=method,basis=basis,
            basis_file=basis_file,**common))
    terms=(term('fc-CCSD(T)-junTZ',1.,cc,'jun-cc-pVTZ',protocol.basis_path),
        term('fc-MP2-junQZ',64/37,mp,'jun-cc-pVQZ',protocol.cbs_upper_basis_path),
        term('fc-MP2-junTZ',-64/37,mp,'jun-cc-pVTZ',protocol.cbs_lower_basis_path),
        term('ae-MP2-wCVTZ',1.,mp+';core','cc-pwCVTZ'),
        term('fc-MP2-wCVTZ',-1.,mp,'cc-pwCVTZ'))
    return QMScanBackend(name='linear_composite',charge=int(charge),
        multiplicity=int(multiplicity),properties=('energy',),gradient_mode='energy',
        composite_terms=terms,composite_parallel_workers=1,
        resume_completed_composite_terms=True,resolution=contract)

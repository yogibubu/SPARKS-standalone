"""Compute-host checks must reject the laptop despite network aliases."""
import pytest
from matrix_sparks import execution


def test_oracle_identity(monkeypatch):
    monkeypatch.setattr(execution.socket, 'getfqdn', lambda: 'oracle.sns.it')
    assert execution.require_compute_host() == 'oracle.sns.it'


def test_studio_dhcp_name(monkeypatch):
    monkeypatch.setattr(execution.socket, 'getfqdn', lambda: 'mac-2.homenet.telecomitalia.it')
    monkeypatch.setattr(execution.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(execution.subprocess, 'check_output',
        lambda args, **kw: 'Mac Studio di Vincenzo' if args[0]=='scutil' else 'Mac15,14')
    assert 'Mac15,14' in execution.require_compute_host()


@pytest.mark.parametrize('name,model', [('MacBook Pro', 'Mac15,14'), ('Mac Studio', 'Mac15,7')])
def test_name_and_hardware_are_both_required(monkeypatch,name,model):
    monkeypatch.setattr(execution.socket, 'getfqdn', lambda: 'mac.homenet.telecomitalia.it')
    monkeypatch.setattr(execution.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(execution.subprocess, 'check_output',
        lambda args, **kw: name if args[0]=='scutil' else model)
    with pytest.raises(RuntimeError): execution.require_compute_host()

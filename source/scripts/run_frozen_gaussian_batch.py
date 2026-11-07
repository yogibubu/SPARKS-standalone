"""Run an explicitly authorized finite GDV batch, preserving completed jobs."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import socket
import subprocess
from time import perf_counter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_ept_summary(text):
    """Normal termination alone does not establish a complete EPT export."""
    if not all(label in text for label in ('Orbital energies by symmetry in eV',
            'Final roots for representation', 'Eigenvalue (eV)', 'polestrength')):
        raise ValueError('Gaussian terminated without a complete native EPT root/pole/MO summary; preserve the saved files')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--gdv", type=Path, required=True)
    p.add_argument("--formchk", type=Path, required=True)
    p.add_argument('--require-ept-roots', action='store_true')
    a = p.parse_args()
    host = socket.getfqdn().lower()
    allowed = host == "oracle.sns.it"
    if not allowed and platform.system() == "Darwin":
        name = subprocess.check_output(["scutil", "--get", "ComputerName"],text=True).lower()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"],text=True).strip()
        allowed = "mac studio" in name and model in {
            "Mac13,1", "Mac13,2", "Mac14,13", "Mac14,14", "Mac15,14", "Mac16,9"}
    if not allowed:
        raise RuntimeError("Verified Oracle or MAC-STUDIO required; laptops are forbidden")
    root = a.plan.resolve().parent
    plan = json.loads(a.plan.read_text())
    auth = json.loads(a.authorization.read_text())
    if auth.get("authorized") is not True or auth.get("plan_sha256") != digest(a.plan):
        raise ValueError("Explicit matching finite authorization required")
    jobs = [(state["name"], point) for state in plan["states"] for point in state["jobs"]]
    if len(jobs) != auth["new_electronic_jobs"] or len(jobs) != plan["total_new_electronic_jobs"]:
        raise ValueError("Frozen job count differs")
    started = perf_counter()
    history = []

    def save(status):
        value = dict(
            schema="matrix.gaussian.frozen_batch_execution.v1",
            host=host,
            plan_sha256=digest(a.plan),
            authorization_sha256=digest(a.authorization),
            status=status,
            concurrent_jobs=1,
            total_jobs=len(jobs),
            completed_jobs=len(history),
            elapsed_seconds=perf_counter() - started,
            jobs=history,
        )
        temp = root / "execution.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(root / "execution.json")

    for state, point in jobs:
        inp = (root / point["input"]).resolve()
        if root not in inp.parents:
            raise ValueError("Input escapes finite plan directory")
        text = inp.read_text()
        route = " ".join(
            line.strip() for line in text.splitlines() if line.lstrip().startswith("#")
        )
        approved_route = point.get("route", plan.get("route"))
        threads = plan.get("threads_per_job", 32)
        memory = plan.get("memory_gb_per_job", 128)
        if (type(threads) is not int or threads <= 0
                or type(memory) is not int or memory <= 0):
            raise ValueError("Finite plan requires positive integer thread and memory limits")
        if (route != approved_route
                or not re.search(rf"(?mi)^%mem={memory}GB\s*$",text)
                or not re.search(rf"(?mi)^%nprocshared={threads}\s*$",text)):
            raise ValueError("Unapproved route/resources")
        folder = inp.parent
        log = folder / "job.log"
        chk = folder / "job.chk"
        fchk = folder / "job.fchk"
        meta = folder / "result.json"
        identity = digest(inp)
        if point.get("input_sha256", identity) != identity:
            raise ValueError("Input content differs from frozen finite plan")
        if meta.exists():
            existing = json.loads(meta.read_text())
            if (
                existing["input_sha256"] != identity
                or not log.exists()
                or "Normal termination of Gaussian" not in log.read_text(errors="replace")
                or not fchk.exists()
            ):
                raise ValueError("Incompatible completed checkpoint")
            if a.require_ept_roots:
                require_ept_summary(log.read_text(errors='replace'))
            history.append(existing)
            save("running")
            continue
        begin = perf_counter()
        if log.exists() and "Normal termination of Gaussian" not in log.read_text(errors="replace"):
            save("incomplete_job_preserved")
            raise RuntimeError(f"Preserve incomplete job/checkpoint: {folder}")
        if not log.exists():
            scratch = folder / "scratch"
            scratch.mkdir(exist_ok=True)
            env = dict(os.environ, GAUSS_SCRDIR=str(scratch))
            with inp.open("rb") as stream, log.open("wb") as output:
                result = subprocess.run(
                    [str(a.gdv)],
                    stdin=stream,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    cwd=folder,
                    env=env,
                )
            if result.returncode or "Normal termination of Gaussian" not in log.read_text(
                errors="replace"
            ):
                save("failed_job_preserved")
                raise RuntimeError(f"GDV failed; preserve checkpoint: {folder}")
        if not fchk.exists():
            with (folder / "formchk.log").open("wb") as output:
                subprocess.run(
                    [str(a.formchk), str(chk), str(fchk)],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=folder,
                )
        if a.require_ept_roots:
            try:
                require_ept_summary(log.read_text(errors='replace'))
            except ValueError:
                save('incomplete_native_ept_export_preserved')
                raise
        match = re.search(r"^Total Energy\s+R\s+([-+\d.EeDd]+)", fchk.read_text(), re.M)
        if match is None:
            raise ValueError("Completed FCHK has no total correlated energy")
        row = dict(
            state=state,
            name=point["name"],
            role=point["role"],
            input_sha256=identity,
            energy_hartree=float(match[1].replace("D", "E")),
            seconds=perf_counter() - begin,
            log_sha256=digest(log),
            fchk_sha256=digest(fchk),
            normal_termination=True,
        )
        meta.write_text(json.dumps(row, indent=2) + "\n")
        history.append(row)
        save("running")
        print(json.dumps(row), flush=True)
    save("complete")
    print(
        json.dumps(dict(status="complete", jobs=len(history), seconds=perf_counter() - started)),
        flush=True,
    )


if __name__ == "__main__":
    main()

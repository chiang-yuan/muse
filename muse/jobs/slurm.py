"""SLURM job submission utilities."""

from __future__ import annotations

import subprocess


def submit_job(
    cmd: str,
    time: str,
    partition: str,
    nodes: int,
    ntasks_per_node: int,
    job_name: str,
    account: str | None = None,
    **kwargs: str | None,
) -> subprocess.CompletedProcess:
    """Submit a SLURM batch job using ``sbatch --wrap``.

    Constructs and executes an ``sbatch`` command with the given
    resource parameters, wrapping the provided command string.

    Args:
        cmd: The command to run inside the SLURM job.
        time: Wall time limit (e.g., ``"2:00:00"``).
        partition: SLURM partition/QOS name.
        nodes: Number of nodes to request.
        ntasks_per_node: Number of tasks per node.
        job_name: Name for the SLURM job.
        account: SLURM account/project for billing. Defaults to None.
        **kwargs: Additional sbatch options. Supported keys:
            - ``constraint``: Node constraint string.
            - ``exclude``: Nodes to exclude.

    Returns:
        The completed process result from ``subprocess.run``.

    Raises:
        subprocess.CalledProcessError: If sbatch returns a non-zero exit code.
    """
    scmd = [
        *f"""sbatch
        --time={time}
        --qos={partition}
        --job-name={job_name}
        --output={job_name}.out
        --error={job_name}.err
        --nodes={nodes}
        --ntasks-per-node={ntasks_per_node}""".replace(
            "'", ""
        ).split(),
    ]

    if account is not None:
        scmd += [f"--account={account}"]

    if kwargs.get("constraint"):
        scmd += [f"--constraint={kwargs['constraint']}"]

    if kwargs.get("exclude"):
        scmd += ["--exclude"]

    scmd += ["--wrap", cmd]

    return subprocess.run(scmd, check=True)

import subprocess


def submit_job(
    cmd: str,
    time: str,
    partition: str,
    nodes: int,
    ntasks_per_node: int,
    job_name: str,
    account: str,
    **kwargs,
):
    scmd = [
        *f"""sbatch
        --time={time}
        --account={account}
        --qos={partition}
        --job-name={job_name}
        --output={job_name}.out
        --error={job_name}.err
        --nodes={nodes}
        --ntasks-per-node={ntasks_per_node}""".replace(
            "'", ""
        ).split(),
    ]

    if kwargs.get("constraint", None):
        scmd += [f"--constraint={kwargs['constraint']}"]

    if kwargs.get("exclude", None):
        scmd += ["--exclude"]

    scmd += ["--wrap", cmd]

    return subprocess.run(scmd, check=True)

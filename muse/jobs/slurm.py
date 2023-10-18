import subprocess


def submit_job(
        cmd: str,
        time: str,
        partition: str,
        nodes: int,
        ntasks_per_node: int,
        job_name: str,
        ):

    scmd = [ 
        *f"""sbatch
        --time={time} 
        --partition={partition}
        --job-name={job_name}
        --output={job_name}.out
        --error={job_name}.err
        --nodes={nodes}
        --ntasks-per-node={ntasks_per_node}"""
        .replace("'", "").split(),
        "--wrap", cmd
    ]

    return subprocess.run(scmd, check=True)
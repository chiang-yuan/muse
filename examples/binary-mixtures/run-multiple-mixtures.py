import hashlib
import json
import os
import re
import subprocess

from ase.formula import Formula

from muse.jobs.slurm import submit_job

temperature = 1100
pressure = 0.0
tolerance = 0.2
rattle = 0.1
scale = 1.05
shuffle = True
seed = 1

a = Formula("NaCl")
b = Formula("CsCl")

pre_cmd = "module load anaconda3; source activate ms"
post_cmd = ""

nsamples = 11

for i in range(nsamples):
    recipe = {str(a): nsamples-1-i, str(b): i}

    cmd = re.sub(r'\s+', ' ', f"""
        python binary-mixture.py '{json.dumps(recipe)}' {temperature} {pressure} 
        --tolerance {tolerance} 
        --rattle {rattle} 
        --scale {scale} {"--no-shuffle" if shuffle else ""} --seed {seed}
        --log
        --root {os.getcwd()}"""
    )

    sha256 = hashlib.sha256()
    sha256.update(cmd.encode("utf-8"))

    job_name = f"bm-{sha256.hexdigest()[:8]}"

    print(recipe, job_name)

    status = submit_job(
        cmd=f"{pre_cmd}; {cmd}; {post_cmd}",
        time="2:00:00",
        partition="RM",
        nodes=1,
        ntasks_per_node=64,
        job_name=job_name,
    )

    print(status)
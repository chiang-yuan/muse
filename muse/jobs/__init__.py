"""HPC job submission utilities.

Provides :func:`submit_job` for submitting SLURM batch jobs.
"""

from muse.jobs.slurm import submit_job

__all__ = ["submit_job"]

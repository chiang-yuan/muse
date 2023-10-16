import subprocess

temperature = 1100
pressure = 0.0

cmd = [
    "python",
    "binary-mixture.py",
    '{"Al": 0.5, "Ni": 0.5}',
    f"{temperature}",
    f"{pressure}",
    "--tolerance",
    "0.1",
    "--rattle",
    "0.1",
    "--scale",
    "1.0",
    "--shuffle",
    "--seed",
    "42",
]

result = subprocess.run(cmd, check=True)

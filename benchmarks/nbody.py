import subprocess

r = subprocess.call("./benchmarks/nbody -benchmark -numbodies=819200", shell=True)
exit(r)

import subprocess
r = subprocess.call("./nbody -benchmark -numbodies=819200", shell=True)
exit(r)

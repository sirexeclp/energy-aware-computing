import subprocess
r = subprocess.call("~/samples/bin/x86_64/linux/release/nbody -benchmark -numbodies=819200", shell=True)
exit(r)

from pathlib import Path
import time
g = list(Path("/sys/class/powercap").glob("intel-rapl:[0-9]"))
g += list(Path("/sys/class/powercap").glob("intel-rapl:[0-9]/intel-rapl:[0-9]:[0-9]"))
last ={}
while True:
    for p in g:
        name_file = p / "name"
        energy_file = p / "energy_uj"
        name = name_file.read_text().strip()
        energy = int(energy_file.read_text().strip())/1_000_000
        print(f"{name}: {energy - last.get(name,0):.3f}")
        last[name] = energy
    print()
    time.sleep(1)

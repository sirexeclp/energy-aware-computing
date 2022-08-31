with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "rb") as f:
    # f.seek(0x611)
    test = int(
        f.read()
    )  # 1**-9 * int.from_bytes(f.read(8), byteorder='big', signed=False)
    print(test * 10**-6)

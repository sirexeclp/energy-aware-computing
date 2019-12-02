import numpy as np
import subprocess
import numpy.testing as npt
import matplotlib.pyplot as plt#
from colorama import Fore, Style
x = np.load("test_data.npy")
y = np.load("expected_result.npy")

y_hat = np.load("output.npy")
print(f"y_hat: {len(y_hat)}")
print(f"y: {len(y)}")
plt.plot(np.abs(y_hat))
plt.plot(np.abs(y))
plt.show()
npt.assert_allclose(y_hat,y,rtol=.2)

print(f"{Fore.GREEN}Validation Passed!{Style.RESET_ALL}")
# result = subprocess.run(["./dft","fft","10"],stdout=subprocess.PIPE)
# output = result.stdout.decode('utf-8')
# data = []
# for line in output.splitlines():
#     data.append(complex(*eval(line)))    
# #test  = np.fromstring(output, dtype=complex, sep=',')
# npt.assert_allclose(data,y)
# print(np.array(data))
       
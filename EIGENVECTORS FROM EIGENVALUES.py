# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # EIGENVECTORS FROM EIGENVALUES
# PETER B. DENTON, STEPHEN J. PARKE, TERENCE TAO, AND XINING ZHANG
#
# [arXiv:1908.03795](https://arxiv.org/abs/1908.03795) [math.RA]

import numpy as np
mat = np.array([[1, -1], [1, 1]])#np.diag((1, 5, 3))
w, v = np.linalg.eig(mat)

result = np.zeros((len(w),len(w)), dtype=complex)
print(result.shape)
for i in range(len(w)):
    for j in range(len(w)):
        prod1=1
        prod2=1
        for k in range(len(w)):
            if k!=i:
                prod1*=w[i]-w[k]
            if k+1 < len(w):
                prod2*=w[i]-np.linalg.eig(np.delete(np.delete(mat, j,axis=0),j,axis=1))[0][k]
        result[i,j]=np.sqrt(prod2/prod1)

result

v



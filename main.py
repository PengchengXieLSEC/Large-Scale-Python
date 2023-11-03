#Codes of a model-based method for solving large-scale DFO
#Copyright: Pengcheng Xie & Ya-xiang Yuan 
#Connect: xpc@lsec.cc.ac.cn



import numpy as np
from python.solver import solvetry


def func(x: np.ndarray, name: str):
    INFINITY = 1.0e308
    f = INFINITY
    info = 0  # info = 0 means successful evaluation
    y = np.r_[0e0, x, 0e0]

    name = name.lower()

    if name == 'dqrtic':

        f = 0.0e0
        temp = np.arange(1, len(x) + 1)
        f += ((y[1:-1] - temp) ** 4).sum()

    else:
        # info = 1 means unknown function name
        info = 1

    return f, info


if __name__ == '__main__':
    result = solvetry(lambda x: func(x, 'dqrtic')[0], np.zeros(10))
    print(result)

"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

import random

def draw(p_k):
    k_uni = random.random()
    for i in range(len(p_k)):
        k_uni = k_uni - p_k[i]
        if k_uni < 0:
            return i
    return len(p_k) - 1
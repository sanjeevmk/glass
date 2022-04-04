import csv
import numpy as np
import sys
sys.path.append("../")
import random
random.seed(10)

class gaussian:
    def __init__(self,mu,cov,covinv):
        self.mu = mu
        self.cov = cov
        self.covinv = covinv

    def get(self):
        return self.mu,self.cov,self.covinv
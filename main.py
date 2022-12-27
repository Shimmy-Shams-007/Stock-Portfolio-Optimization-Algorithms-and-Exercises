# This document does not hold any main programs
# This is just a practise document for basic python exercises


import time
import numpy as np

from math import exp

class CouponBonds:
    def __init__(self, principal, maturity, rate, interest):
        self.principal = principal
        self.maturity = maturity
        self.rate = rate / 100
        self.interest = interest / 100

    def present_value(self, x, n):
        return x * exp (-self.interest * n)

    def calculate_price(self):
        price = 0

        for t in range (1, self.maturity + 1):
            price = price + self.present_value(self.principal * self.rate, t)

        price = price + self.present_value(self.principal, self.maturity)

        return price

if __name__ == '__main__':
    bond = CouponBonds(1000, 10, 2, 4)
    print (bond.calculate_price())
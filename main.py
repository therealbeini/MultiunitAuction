import math
import random

import numpy as np

from AuctionCalculator import AuctionCalculator


def input_k():
    print('Input: Please input k as an integer to calculate the auction result.')
    return int(input())


def input_num_items():
    print(
        'Input: Welcome to the two buyer sequential multiunit auction simulator. Please enter the amount of items as an integer.')
    while True:
        try:
            num_items = int(input())
        except ValueError:
            print('Error: Please enter an integer.')
            continue
        if num_items <= 0:
            print('Error: Please enter an integer greater than 0.')
            continue
        return num_items


def input_buyer_valuations(num_items: int, buyer: int) -> list:
    example_sequence = ""
    for i in range(num_items, 0, -1):
        example_sequence += str(i) + " "
    example_sequence = example_sequence[:-1]
    print(
        f'Input: The auction has {num_items} identical items. Please enter valuations for buyer {buyer} as a non-increasing '
        f'sequence of integers, seperated with a blank space. As example: {example_sequence}')
    while True:
        try:
            values = np.array(list((map(int, input().split()))))
        except ValueError:
            print('Error: Please enter only integers seperated by a blank space.')
            continue
        if len(values) != num_items:
            print(f'Error: Please enter the correct number integers seperated by a blank space. '
                  f'Expected were {num_items} integers.')
            continue
        if all(x <= 0 for x in values):
            print('Error: Please enter only integers greater than 0.')
            continue
        failed = False
        for i in range(len(values) - 1):
            if values[i] < values[i + 1]:
                print('Error: The sequence of valuations have to be non-increasing.')
                failed = True
                break
        if failed:
            continue
        return values


def test():
    num_items = 4
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    for i in range(10):
        for j in range(0, i + 1):
            for m in range(0, j + 1):
                for n in range(0, m + 1):
                    for a in range(10):
                        for b in range(0, a + 1):
                            for c in range(0, b + 1):
                                for d in range(0, c + 1):
                                    fb_values = np.array([2,2,0,0])
                                    sb_values = np.array([5,5,5,0])
                                    print(fb_values)
                                    print(sb_values)
                                    calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                                             sb_values=sb_values, k=2)
                                    welfare = 0
                                    for x in range(num_items):
                                        cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
                                        if cur_welfare > welfare:
                                            welfare = cur_welfare
                                    if welfare == 0:
                                        continue
                                    utility = calc.find_equilibrium()
                                    cur_welfare_ratio = float(utility / welfare)
                                    if cur_welfare_ratio < min_welfare_ratio:
                                        min_welfare_ratio = cur_welfare_ratio
                                        min_fb_values = fb_values
                                        min_sb_values = sb_values
    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)


def test2():
    num_items = 6
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    z = 2
    a = 5
    counter = 0
    for b in range(0, a + 1):
        for c in range(0, b + 1):
            for d in range(0, c + 1):
                for e in range(d + 1):
                    for f in range(0, e + 1):
                        for g in range(0, a + 1):
                            for h in range(0, g + 1):
                                for i in range(0, h + 1):
                                    for j in range(0, i + 1):
                                        for k in range(0, j + 1):
                                            for l in range(0, k + 1):
                                                # fb_values = np.array([int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h)])
                                                # sb_values = np.array([int(i),int(j), int(k), int(l), int(m), int(n), int(o), int(p)])
                                                fb_values = np.array(
                                                    [int(a), int(b), int(c), int(d), int(e), int(f)])
                                                sb_values = np.array(
                                                    [int(g), int(h), int(i), int(j), int(k), int(l)])
                                                calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                                                         sb_values=sb_values, k=2)
                                                welfare = 0
                                                for x in range(num_items):
                                                    cur_welfare = sum(fb_values[:x]) + sum(
                                                        sb_values[:num_items - x])
                                                    if cur_welfare > welfare:
                                                        welfare = cur_welfare
                                                if welfare == 0:
                                                    continue
                                                utility = calc.find_equilibrium()
                                                cur_welfare_ratio = float(utility / welfare)
                                                if cur_welfare_ratio < min_welfare_ratio:
                                                    min_welfare_ratio = cur_welfare_ratio
                                                    min_fb_values = fb_values
                                                    min_sb_values = sb_values
                                                if counter % 100 == 0:
                                                    print(counter)
                                                counter += 1
                                                print(fb_values)
                                                print(sb_values)
    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)


def test3():
    num_items = 6
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    a = 1000
    z = 3
    counter = 0
    for b in np.arange(a / z, a + 1, a / z):
        for c in np.arange(b / z, b + 1, b / z):
            for d in np.arange(c / z, c + 1, c / z):
                for e in np.arange(d / z, d + 1, d / z):
                    for f in np.arange(e / z, e + 1, e / z):
                        # for g in numpy.arange(f / z, f + 1, f / z):
                        #     for h in numpy.arange(g / z, g + 1, g / z):
                        for i in np.arange(a / z, a, a / z):
                            for j in np.arange(i / z + 1, i, i / z):
                                for k in np.arange(j / z, j, j / z):
                                    for l in np.arange(k / z, k, k / z):
                                        for m in np.arange(l / z, l, l / z):
                                            for n in np.arange(m / z, m, m / z):
                                                        # for o in numpy.arange(n / z, n + 1, n / z):
                                                        #     for p in numpy.arange(o / z, o + 1, o / z):
                                                #fb_values = np.array([int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h)])
                                                #sb_values = np.array([int(i),int(j), int(k), int(l), int(m), int(n), int(o), int(p)])
                                                #fb_values = np.array([int(a), int(b), int(c), int(d), int(e), int(f)])
                                                #sb_values = np.array([int(i), int(j), int(k), int(l), int(m), int(n)])
                                                fb_values = np.array(
                                                    [1000,333,111,37,24,8])
                                                sb_values = np.array(
                                                    [333,223,148,99,33,11])

                                                print(fb_values)
                                                print(sb_values)
                                                calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                                                         sb_values=sb_values, k=2)
                                                welfare = 0
                                                for x in range(num_items):
                                                    cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
                                                    if cur_welfare > welfare:
                                                        welfare = cur_welfare
                                                if welfare == 0:
                                                    continue
                                                utility = calc.find_equilibrium()
                                                cur_welfare_ratio = float(utility / welfare)
                                                if cur_welfare_ratio < min_welfare_ratio:
                                                    min_welfare_ratio = cur_welfare_ratio
                                                    min_fb_values = fb_values
                                                    min_sb_values = sb_values
                                                if counter % 100 == 0:
                                                    print(counter)
                                                counter += 1
    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)

def test4():
    num_items = 8
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    a = 10000
    counter = 0
    for i in range(1, 100):
        fb_values = np.array([a, a, a, a, a, a, a, a])
        sb_values = np.zeros([8], dtype=int)
        sb_values[0] = 10000
        for j in range(1, 8):
            sb_values[j] = math.floor(sb_values[j - 1] * i / 100)
        # fb_values = np.array([int(a), int(b), int(c), int(d), int(e), int(f)])
        # sb_values = np.array([int(i), int(j), int(k), int(l), int(m), int(n)])
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=2)
        welfare = 0
        for x in range(num_items + 1):
            cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
            if cur_welfare > welfare:
                welfare = cur_welfare
        if welfare == 0:
            continue
        utility = calc.find_equilibrium()
        cur_welfare_ratio = float(utility / welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio
            min_fb_values = fb_values
            min_sb_values = sb_values
        counter += 1
        print(counter)
    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)

def test5():
    num_items = 6
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    z = 3
    fa = 1000
    counter = 0
    for b in range(0, z + 1):
        fb = math.floor(fa * b / z)
        for c in range(0, z + 1):
            fc = math.floor(fb * c / z)
            for d in range(0, z + 1):
                fd = math.floor(fc * d / z)
                for e in range(0, z + 1):
                    fe = math.floor(fd * e / z)
                    for f in range(0, z + 1):
                        ff = math.floor(fe * f / z)
                        for g in range(0, z + 1):
                            fg = math.floor(fa * g / z)
                            for h in range(0, z + 1):
                                fh = math.floor(fg * h / z)
                                for i in range(0, z + 1):
                                    fi = math.floor(fh * i / z)
                                    for j in range(0, z + 1):
                                        fj = math.floor(fi * j / z)
                                        for k in range(0, z + 1):
                                            fk = math.floor(fj * k / z)
                                            for l in range(0, z + 1):
                                                fl = math.floor(fk * l / z)
                                                # fb_values = np.array([int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h)])
                                                # sb_values = np.array([int(i),int(j), int(k), int(l), int(m), int(n), int(o), int(p)])
                                                fb_values = np.array(
                                                    [int(fa), int(fb), int(fc), int(fd), int(fe), int(ff)])
                                                sb_values = np.array(
                                                    [int(fg), int(fh), int(fi), int(fj), int(fk), int(fl)])
                                                print(fb_values)
                                                print(sb_values)
                                                print(min_welfare_ratio)
                                                calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                                                         sb_values=sb_values, k=2)
                                                welfare = 0
                                                for x in range(num_items):
                                                    cur_welfare = sum(fb_values[:x]) + sum(
                                                        sb_values[:num_items - x])
                                                    if cur_welfare > welfare:
                                                        welfare = cur_welfare
                                                if welfare == 0:
                                                    continue
                                                utility = calc.find_equilibrium()
                                                cur_welfare_ratio = float(utility / welfare)
                                                if cur_welfare_ratio < min_welfare_ratio:
                                                    min_welfare_ratio = cur_welfare_ratio
                                                    min_fb_values = fb_values
                                                    min_sb_values = sb_values
                                                if counter % 100 == 0:
                                                    print(counter)
                                                counter += 1
    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)

def vcg_test():
    num_items = 4
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    for i in range(10):
        for j in range(0, i + 1):
            for m in range(0, j + 1):
                for n in range(0, m + 1):
                    for a in range(10):
                        for b in range(0, a + 1):
                            for c in range(0, b + 1):
                                for d in range(0, c + 1):
                                    fb_values = np.array([5,5,0,0])
                                    sb_values = np.array([9,9,9,9])
                                    print(fb_values)
                                    print(sb_values)
                                    calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                                             sb_values=sb_values, k=2)
                                    current_welfare = calc.get_vcg_prices(k=2)
                                    max_welfare = 0
                                    for x in range(num_items):
                                        cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
                                        if cur_welfare > max_welfare:
                                            max_welfare = cur_welfare
                                    if max_welfare == 0:
                                        continue
                                    cur_welfare_ratio = float(current_welfare / max_welfare)
                                    if cur_welfare_ratio < min_welfare_ratio:
                                        min_welfare_ratio = cur_welfare_ratio
                                        min_fb_values = fb_values
                                        min_sb_values = sb_values

    print(min_welfare_ratio)
    print(min_fb_values)
    print(min_sb_values)

def random_test():
    num_items = 10
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    count = 0
    while True:
        fb_values = np.empty(10)
        sb_values = np.empty(10)
        for i in range(10):
            fb_values[i] = random.randint(0, 100)
            sb_values[i] = random.randint(0, 100)
        fb_values[::-1].sort()
        sb_values[::-1].sort()
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=2)
        current_welfare = calc.get_vcg_prices(k=2)
        max_welfare = 0
        for x in range(num_items):
            cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
            if cur_welfare > max_welfare:
                max_welfare = cur_welfare
        if max_welfare == 0:
            continue
        cur_welfare_ratio = float(current_welfare / max_welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio
            min_fb_values = fb_values
            min_sb_values = sb_values

        if count % 1000 == 0:
            print(min_welfare_ratio)
            print(min_fb_values)
            print(min_sb_values)

        count += 1

def gaussian_test():
    mu, sigma = 100, 5
    factor = 1 - (1/math.e)
    max_bid = 100
    num_items = 10
    min_welfare_ratio = 1.0
    min_fb_values = None
    min_sb_values = None
    count = 0
    while True:
        sb_values = np.empty(100)
        fb_values = np.random.normal(mu, sigma, 100)
        for i in range(100):
            if fb_values[i] > 100:
                fb_values[i] = round(100 + (100 - fb_values[i]))
            else:
                fb_values[i] = round(fb_values[i])
            fb_values[i] = 100
            base = max_bid - ((max_bid - max_bid * factor) / ((100 - i) / 100))
            sb_values[i] = int(max(0, min(100, round(np.random.normal(max(base, 0), sigma, 1)[0]))))
            sb_values[i] = int(max(0,round(base)))
        fb_values[::-1].sort()
        sb_values[::-1].sort()
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=2)
        current_welfare = calc.get_vcg_prices(k=2)
        max_welfare = 0
        for x in range(num_items):
            cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
            if cur_welfare > max_welfare:
                max_welfare = cur_welfare
        if max_welfare == 0:
            continue
        cur_welfare_ratio = float(current_welfare / max_welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio
            min_fb_values = fb_values
            min_sb_values = sb_values

        if count % 1000 == 0:
            print(min_welfare_ratio)
            print(min_fb_values)
            print(min_sb_values)

        count += 1

if __name__ == '__main__':
    gaussian_test()

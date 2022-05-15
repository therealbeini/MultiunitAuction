import math
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from AuctionCalculator import AuctionCalculator

import sqlite3
con = sqlite3.connect('promille.db')

cur = con.cursor()

# Create table
cur.execute('''CREATE TABLE IF NOT EXISTS final3_2_20_10
               (promille integer, count integer)''')


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

def gaussian_test():
    mu, sigma = 100, 5
    factor = 1 - (1/math.e)
    max_bid = 100
    num_items = 100
    min_welfare_ratio = 1.0
    count = 0
    promilles = np.zeros(1001)
    while True:
        sb_values = np.empty(100)
        fb_values = np.random.normal(mu, sigma, 100)
        for i in range(100):
            fb_values[i] = 100
            base = max_bid - (max_bid - max_bid * factor) / ((100 - i) / 100)
            sb_values[i] = int(max(0, min(100, round(np.random.normal(max(base, 0), sigma, 1)[0]))))
        fb_values[::-1].sort()
        sb_values[::-1].sort()
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=2)
        current_welfare = calc.get_vcg_prices(k=2)
        max_welfare = 0
        for x in range(num_items + 1):
            cur_welfare = sum(fb_values[:x]) + sum(sb_values[:num_items - x])
            if cur_welfare > max_welfare:
                max_welfare = cur_welfare
        if max_welfare == 0:
            continue
        cur_welfare_ratio = float(current_welfare / max_welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio
        promilles[math.ceil(cur_welfare_ratio * 1000)] += 1

        if count % 100 == 0:
            print(min_welfare_ratio)
            for i in range(1000):
                cur.execute("SELECT * FROM promilles_gaussian_1000 WHERE promille=?", (i,))
                row = cur.fetchone()
                if row is None:
                    cur.execute('INSERT INTO promilles_gaussian_1000 values(?,?)', (i, promilles[i]))
                else:
                    new_count = row[1] + promilles[i]
                    cur.execute('''UPDATE promilles_gaussian_1000 SET count = ? WHERE promille= ? ''', (new_count, i))
            con.commit()
            promilles = np.zeros(1001)

        count += 1

def draw_histogram():
    cur.execute("SELECT * FROM promilles")
    data = cur.fetchall()
    histo = np.zeros(200)
    label = np.zeros(200)
    sum = 0
    for i in range(800, 1000, 1):
        sum += data[i][1]
        label[i - 800] = i
    for e in data:
        if e[0] < 800: continue
        histo[e[0] - 800] = e[1] / sum
    plt.bar(label, histo, alpha=0.5, label=f'100items, 189511663226 samples')

    cur.execute("SELECT * FROM promilles_1000")
    data = cur.fetchall()
    histo = np.zeros(200)
    label = np.zeros(200)
    sum = 0
    for i in range(800, 1000, 1):
        sum += data[i][1]
        label[i - 800] = i
    for e in data:
        if e[0] < 800: continue
        histo[e[0] - 800] = e[1] / sum
    plt.bar(label, histo, alpha=0.5, label=f'1000items, {sum} samples')
    plt.legend(loc='upper left')
    plt.savefig('promille_overlayed_zoomed.png')
    plt.show()

def random_2_buyer_test(table, k, num_items, max_value):
    min_welfare_ratio = 1.0
    count = 0
    promilles = np.zeros(1001)
    while True:
        fb_values = np.empty(num_items)
        sb_values = np.empty(num_items)
        for i in range(num_items):
            fb_values[i] = random.randint(0, max_value)
            sb_values[i] = random.randint(0, max_value)
        fb_values[::-1].sort()
        sb_values[::-1].sort()
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=k)
        current_welfare = calc.get_vcg_prices(k=k)
        combined = np.concatenate((fb_values, sb_values),axis=None)
        combined[::-1].sort()
        max_welfare = sum(combined[:num_items])
        if max_welfare == 0:
            continue
        cur_welfare_ratio = float(current_welfare / max_welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio

        promilles[math.ceil(cur_welfare_ratio * 1000)] += 1

        if count % 1000 == 0:
            print(min_welfare_ratio)
            for i in range(1001):
                cur.execute(f"SELECT * FROM {table} WHERE promille=?", (i,))
                row = cur.fetchone()
                if row is None:
                    cur.execute(f'INSERT INTO {table} values(?,?)', (i, promilles[i]))
                else:
                    new_count = row[1] + promilles[i]
                    cur.execute(f'''UPDATE {table} SET count = ? WHERE promille= ? ''', (new_count, i))
            con.commit()
            promilles = np.zeros(1001)

        count += 1

def random_3_or_more_buyer_test(table, buyers, k, num_items, max_value):
    min_welfare_ratio = 1.0
    count = 0
    promilles = np.zeros(1001)
    while True:
        values = np.empty((buyers, num_items), dtype=int)
        for i in range(buyers):
            for j in range(num_items):
                values[i][j] = random.randint(0, max_value)
            values[i][::-1].sort()
        calc = AuctionCalculator(num_items=num_items, values=values, k=k, num_buyers=buyers)
        current_welfare = calc.get_vcg_prices_for_3_or_more_buyers(k=k)
        flat_array = values.flatten()
        flat_array[::-1].sort()
        max_welfare = sum(flat_array[:num_items])
        cur_welfare_ratio = float(current_welfare / max_welfare)
        if cur_welfare_ratio < min_welfare_ratio:
            min_welfare_ratio = cur_welfare_ratio

        promilles[math.ceil(cur_welfare_ratio * 1000)] += 1

        if count % 100 == 0:
            print(min_welfare_ratio)
            for i in range(1001):
                cur.execute(f"SELECT * FROM {table} WHERE promille=?", (i,))
                row = cur.fetchone()
                if row is None:
                    cur.execute(f'INSERT INTO {table} values(?,?)', (i, promilles[i]))
                else:
                    new_count = row[1] + promilles[i]
                    cur.execute(f'''UPDATE {table} SET count = ? WHERE promille= ? ''', (new_count, i))
            con.commit()
            promilles = np.zeros(1001)

        count += 1

def random_test_iterative():
    num_items = 4
    min_welfare_ratio = 1.0
    while True:
        fb_values = np.empty(num_items)
        sb_values = np.empty(num_items)
        for i in range(num_items):
            fb_values[i] = random.randint(0, 10)
            sb_values[i] = random.randint(0, 10)
        fb_values[::-1].sort()
        sb_values[::-1].sort()
        calc = AuctionCalculator(num_items=num_items, fb_values=fb_values,
                                 sb_values=sb_values, k=2)
        current_welfare = calc.find_equilibrium()
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

def print_latex():
    cur.execute("SELECT * FROM final3_2_20_100")
    data = cur.fetchall()
    y = np.zeros(100)
    samples = 0
    for i in range(len(data) - 1):
        samples += int(data[i][1])
        y[int(i / 10)] += int(data[i][1])
    samples += int(data[1000][1])
    y[99] += int(data[1000][1])
    s = ""
    for i in range(100):
        val = '%.10f' % (y[i] / samples)
        s += f'({i + 1}, {val}) '
    print(s)

def fit_gaussian():
    cur.execute("SELECT * FROM final3_2_20_100")
    data = cur.fetchall()
    x = np.arange(1, 101)
    y = np.zeros(100)
    samples = 0
    for i in range(len(data) - 1):
        samples += int(data[i][1])
        y[int(i / 10)] += int(data[i][1])
    y[99] += int(data[1000][1])
    # weighted arithmetic mean (corrected - check the section below)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    popt, pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    plt.plot(x, y, 'b+:', label=f'data, {samples} samples')
    plt.plot(x, Gauss(x, *popt), 'r-', label=f"fit, sigma = {round(sigma, 3)}, mean = {round(mean, 3)}")
    plt.legend()
    plt.title('Welfare for 3 buyers, n=4 and k=2')
    plt.xlabel('Welfare in promille')
    plt.ylabel('Total frequency')
    plt.savefig('2_buyers_4_items.png')
    plt.show()

    print(mean)
    print(sigma)


if __name__ == '__main__':
    random_3_or_more_buyer_test('final3_2_20_10', 3, 2, 20, 10)
    random_2_buyer_test("final_10_1000_10", 10, 1000, 100)
    #random_test_iterative()
    #print_latex()
    #fit_gaussian()

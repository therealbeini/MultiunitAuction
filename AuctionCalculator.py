import numpy as np


class AuctionCalculator:
    def __init__(self, num_items=None, fb_values=None, sb_values=None):
        self.num_items = num_items
        self.fb_values = fb_values
        self.sb_values = sb_values
        self.sfb = None
        self.ssb = None
        print(sb_values)
        print(fb_values)
        a = self.calculate_single(k=1)
        b = self.calculate_multiple(k=2)
        if b > a:
            print('Different utilities.')
            exit()

    def calculate_single(self, k) -> (np.ndarray, np.ndarray):
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        self.sfb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        self.ssb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        if k == 1:
            for i in range(self.num_items - k, -1, -k):
                for j in range(i + 1):
                    self.sfb[i][j] = np.sum(self.fb_values[i - j:i - j + k]) + ffu[i + k][j] - \
                               ffu[i + k][j + k]
                    self.ssb[i][j] = np.sum(self.sb_values[j:j + k]) + sfu[i + k][j + k] - \
                               sfu[i + k][j]
                    if self.sfb[i][j] >= self.ssb[i][j]:
                        ffu[i][j] = np.sum(self.fb_values[i - j:i - j + k]) + ffu[i + k][j] - self.ssb[i][j]
                        sfu[i][j] = sfu[i + k][j]
                    else:
                        ffu[i][j] = ffu[i + k][j + k]
                        sfu[i][j] = np.sum(self.sb_values[j:j + k]) + sfu[i + k][j + k] - self.sfb[i][j]

            self.sfb = self.sfb
            self.ssb = self.ssb

        position = 0
        for i in range(0, self.num_items, k):
            print(f"In round {i}, buyer 1 bids {self.sfb[i][position]}, buyer 2 bids {self.ssb[i][position]}. "
                  f"Buyer {1 if self.sfb[i][position] >= self.ssb[i][position] else 2} wins the item.")
            if self.sfb[i][position] < self.ssb[i][position]:
                position += k

        fb_items = int((self.num_items - position) / k)
        sb_items = int(position / k)
        fb_utility = ffu[0][0]
        sb_utility = sfu[0][0]

        print(f"Buyer 1 won {fb_items} items and buyer 2 won {sb_items} items.")
        print(f"Buyer 1 has an utility of {fb_utility} and buyer 2 has an utility of {sb_utility}.")

        print(ffu)
        print(sfu)
        print(self.sfb)
        print(self.ssb)

        return fb_utility

    def calculate_multiple(self, k) -> (np.ndarray, np.ndarray):
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        fb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)
        sb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        for i in range(int(self.num_items / k) - 1, -1, -1):
            for j in range(i * k + 1):
                s_current = 0
                for a in range(k):
                    sb[i * k][j][a] = self.ssb[i * k + a][j + s_current]
                    if self.ssb[i * k][j] > self.sfb[i * k][j]:
                        s_current += 1
                    sb[i * k][j][::-1].sort()

        for i in range(int(self.num_items / k) - 1, -1, -1):
            for j in range(i * k + 1):
                current_max = 0
                max_index = 0
                for a in range(k + 1):
                    if a == 0:
                        current_next = ffu[(i + 1) * k][j + (k - a)]
                    else:
                        current_next = ffu[(i + 1) * k][j + (k - a)] + sum(self.fb_values[(i * k) - j:(i * k) - j + a]) - (sb[i * k][j][k - a] * a)
                    if current_next > current_max:
                        current_max = current_next
                        max_index = a
                ffu[i * k][j] = current_max
                if max_index != 0:
                    for a in range(k):
                        fb[i * k][j][a] = sb[i * k][j][k - max_index]

        print(ffu)
        print(fb)

        fb_utility = ffu[0][0]

        return fb_utility
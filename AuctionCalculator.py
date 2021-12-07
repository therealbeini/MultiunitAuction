import numpy as np


class AuctionCalculator:
    def __init__(self, k=None, num_items=None, fb_values=None, sb_values=None):
        self.k = k
        self.num_items = num_items
        self.fb_values = fb_values
        self.sb_values = sb_values
        self.tree = None

    def transform_into_single(self):
        new_num = int((self.num_items + 1) / self.k)
        new_fb = np.empty([new_num])
        new_sb = np.empty([new_num])
        for i in range(new_num):
            new_fb[i] = sum(self.fb_values[i * self.k:(i + 1) * self.k])
            new_sb[i] = sum(self.sb_values[i * self.k:(i + 1) * self.k])
        return AuctionCalculator(k=1, num_items=new_num, fb_values=new_fb, sb_values=new_sb)

    def transform_test(self):
        new_num = int((self.num_items + 1) / self.k)
        new_fb = np.empty([new_num])
        new_sb = np.empty([new_num])
        for i in range(new_num):
            new_fb[i] = sum(self.fb_values[i * self.k:(i + 1) * self.k])
            new_sb[i] = sum(self.sb_values[i * self.k:(i + 1) * self.k])
        single = AuctionCalculator(k=1, num_items=new_num, fb_values=new_fb, sb_values=new_sb)
        multiple_result = self.calculate()
        single_result = single.calculate()
        for i in range(4):
            if multiple_result[i] != single_result[i]:
                print('We got a different result from our single and multiple item auction.')
                exit()

    def input_and_calculate(self):
        self.input_k()
        self.input_num_items()
        self.fb_values = self.input_buyer_valuations(1)
        self.sb_values = self.input_buyer_valuations(2)
        if self.k > 1:
            single = self.transform_into_single()
        multiple_result = self.calculate()
        single_result = single.calculate()
        for i in range(4):
            if multiple_result[i] != single_result[i]:
                print('We got a different result from our single and multiple item auction.')
                exit()

    def input_k(self):
        print('Input: Welcome to the two buyer sequential multiunit auction simulator. Please input k as an integer.')
        self.k = int(input())

    def input_num_items(self):
        print(
            'Input: Please enter the amount of items as an integer.')
        while True:
            try:
                self.num_items = int(input())
            except ValueError:
                print('Error: Please enter an integer.')
                continue
            if self.num_items <= 0:
                print('Error: Please enter an integer greater than 0.')
                continue
            return

    def input_buyer_valuations(self, buyer: int) -> list:
        example_sequence = ""
        for i in range(self.num_items, 0, -1):
            example_sequence += str(i) + " "
        example_sequence = example_sequence[:-1]
        print(
            f'Input: The auction has {self.num_items} identical items. Please enter valuations for buyer {buyer} as a non-increasing '
            f'sequence of integers, seperated with a blank space. As example: {example_sequence}')
        while True:
            try:
                values = np.array(list((map(int, input().split()))))
            except ValueError:
                print('Error: Please enter only integers seperated by a blank space.')
                continue
            if len(values) != self.num_items:
                print(f'Error: Please enter the correct number integers seperated by a blank space. '
                      f'Expected were {self.num_items} integers.')
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

    def print_tree(self):
        print(self.tree)

    def calculate(self) -> (np.ndarray, np.ndarray):
        self.tree = np.zeros([self.num_items + 1, self.num_items + 1, 4], dtype=int)

        for i in range(self.num_items + 1):
            self.tree[self.num_items][i][0] = 0
            self.tree[self.num_items][i][1] = -1
            self.tree[self.num_items][i][2] = 0
            self.tree[self.num_items][i][3] = -1

        for i in range(self.num_items - self.k, -1, -self.k):
            for j in range(i + 1):
                self.tree[i][j][1] = np.sum(self.fb_values[i - j:i - j + self.k]) + self.tree[i + self.k][j][0] - self.tree[i + self.k][j + self.k][0]
                self.tree[i][j][3] = np.sum(self.sb_values[j:j + self.k]) + self.tree[i + self.k][j + self.k][2] - self.tree[i + self.k][j][2]
                if self.tree[i][j][1] >= self.tree[i][j][3]:
                    self.tree[i][j][0] = np.sum(self.fb_values[i - j:i - j + self.k]) + self.tree[i + self.k][j][0] - self.tree[i][j][3]
                    self.tree[i][j][2] = self.tree[i + self.k][j][2]
                else:
                    self.tree[i][j][0] = self.tree[i + self.k][j + self.k][0]
                    self.tree[i][j][2] = np.sum(self.sb_values[j:j + self.k]) + self.tree[i + self.k][j + self.k][2] - self.tree[i][j][1]

        position = 0
        for i in range(0, self.num_items, self.k):
            print(f"In round {i}, buyer 1 bids {self.tree[i][position][1]}, buyer 2 bids {self.tree[i][position][3]}. "
                  f"Buyer {1 if self.tree[i][position][1] >= self.tree[i][position][3] else 2} wins the item.")
            if self.tree[i][position][1] < self.tree[i][position][3]:
                position += self.k

        fb_items = int((self.num_items - position) / self.k)
        sb_items = int(position / self.k)
        fb_utility = self.tree[0][0][0]
        sb_utility = self.tree[0][0][2]

        print(f"Buyer 1 won {fb_items} items and buyer 2 won {sb_items} items.")
        print(f"Buyer 1 has an utility of {fb_utility} and buyer 2 has an utility of {sb_utility}.")

        return fb_items, sb_items, fb_utility, sb_utility

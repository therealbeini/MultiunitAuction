import numpy as np

from AuctionCalculator import AuctionCalculator


def input_k():
    print('Input: Please input k as an integer to calculate the auction result.')
    return int(input())

def input_num_items():
    print('Input: Welcome to the two buyer sequential multiunit auction simulator. Please enter the amount of items as an integer.')
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
    for i in range(10):
        for j in range(0, i + 1):
            for m in range(0, j + 1):
                for n in range(0, m + 1):
                    for a in range(i + 1):
                        for b in range(0, a + 1):
                            for c in range(0, b + 1):
                                for d in range(0, c + 1):
                                    AuctionCalculator(num_items=num_items, fb_values=np.array([i, j, m, n]),
                                                      sb_values=np.array([a, b, c, d]))

if __name__ == '__main__':
    # num_items = input_num_items()
    # fb_values = input_buyer_valuations(num_items, 1)
    # sb_values = input_buyer_valuations(num_items, 2)
    # calculator = AuctionCalculator(num_items=num_items, fb_values=fb_values, sb_values=sb_values)
    # while True:
    #     k = input_k()
    #     calculator.calculate_multiple(k)

    test()

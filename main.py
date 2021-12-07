import numpy as np

from AuctionCalculator import AuctionCalculator


def test():
    num_items = 4
    k = 2
    for i in range(10):
        for j in range(0, i + 1):
            for m in range(0, j + 1):
                for n in range(0, m + 1):
                    for a in range(i + 1):
                        for b in range(0, a + 1):
                            for c in range(0, b + 1):
                                for d in range(0, c + 1):
                                    AuctionCalculator(k=k, num_items=num_items, fb_values=np.array([i, j, m, n]),
                                                      sb_values=np.array([a, b, c, d])).transform_test()


if __name__ == '__main__':
    # calculator = AuctionCalculator()
    # calculator.input_and_calculate()

    test()

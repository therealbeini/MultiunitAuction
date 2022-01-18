import numpy as np


class AuctionCalculator:
    def __init__(self, num_items=None, fb_values=None, sb_values=None):
        self.num_items = num_items
        # valuations
        self.fb_values = fb_values
        self.sb_values = sb_values
        # bids for multi-unit auction with k = 1
        self.sfb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        self.ssb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

    def find_equilibrium(self):
        # first calculate the result of the k = 1 auction,
        # then use these results to create k != 1 auction with same bids as the k = 1 auction
        self.calculate_single(k=1)
        # assume that the first buyer just bids 0 everywhere first
        fb_bids = np.zeros([self.num_items + 1, self.num_items + 1, 2], dtype=int)
        # find best response for the second buyer --> the second buyer has fixed his bids now
        sb_utility, fb_counter_utility, sb_bids = self.find_best_response_s(2, fb_bids)
        while True:
            # now find the best response for the first buyer again,
            # we also output the utility of the second buyer if the first buyer bids according to THIS best response
            fb_utility, sb_counter_utility, fb_bids = self.find_best_response_f(2, sb_bids)
            # if the utility of the second buyer is equal from his earlier best response
            # and this best response of the first buyer, then we have found an equilibrium
            if fb_utility == fb_counter_utility:
                print('Equilibrium found.')
                break
            # now find the best response for the second buyer again,
            # we also output the utility of the first buyer if the second buyer bids according to THIS best response
            sb_utility, fb_counter_utility, sb_bids = self.find_best_response_s(2, fb_bids)
            # if the utility of the first buyer is equal from his earlier best response
            # and this best response of the second buyer, then we have found an equilibrium
            if sb_utility == sb_counter_utility:
                print('Equilibrium found.')
                break

    def calculate_single(self, k) -> (np.ndarray, np.ndarray):
        """
        Calculate the bids and utilities for a multi-unit auction with k = 1. At the end print the results in
        human-readable format. The input parameter of k is a fragment of an earlier version and this function should
        only be used with k = 1.

        @param k: k should always be 1 for a multi-unit auction selling a single item sequentially.
        """
        # forward utilities
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        if k == 1:
            # iterating from the bottom of the tree, disregarding the terminal row
            for i in range(self.num_items - k, -1, -k):
                # from left to right, the leftmost means the first buyer won all items until this round
                # and the rightmost means the second buyer won all items
                for j in range(i + 1):
                    # use the formula for forward utility + utility for getting the item
                    self.sfb[i][j] = np.sum(self.fb_values[i - j:i - j + k]) + ffu[i + k][j] - \
                                     ffu[i + k][j + k]
                    self.ssb[i][j] = np.sum(self.sb_values[j:j + k]) + sfu[i + k][j + k] - \
                                     sfu[i + k][j]
                    # the first buyer wins the bid
                    if self.sfb[i][j] >= self.ssb[i][j]:
                        # update forward utility
                        ffu[i][j] = np.sum(self.fb_values[i - j:i - j + k]) + ffu[i + k][j] - self.ssb[i][j]
                        sfu[i][j] = sfu[i + k][j]
                    # the second buyer wins the bid
                    else:
                        # update forward utility
                        ffu[i][j] = ffu[i + k][j + k]
                        sfu[i][j] = np.sum(self.sb_values[j:j + k]) + sfu[i + k][j + k] - self.sfb[i][j]

        # output
        current_second_buyer_items = 0
        for i in range(0, self.num_items, k):
            print(f"In round {i}, buyer 1 bids {self.sfb[i][current_second_buyer_items]}, buyer 2 bids {self.ssb[i][current_second_buyer_items]}. "
                  f"Buyer {1 if self.sfb[i][current_second_buyer_items] >= self.ssb[i][current_second_buyer_items] else 2} wins the item.")
            if self.sfb[i][current_second_buyer_items] < self.ssb[i][current_second_buyer_items]:
                current_second_buyer_items += k

    def calculate_multiple(self, k) -> np.ndarray:
        """
        Calculate the bids and utilities for a multi-unit auction with k != 1. Note that we are in the testing phase
        and our results are based on the (false) assumption that an auction with k != 1 behaves in the same way like
        a k = 1 auction.

        @param k: number of items sold at the same time
        @return: the bids of the first buyer under our assumption
        """
        # bids of the second buyer
        fb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        # This code block transforms the bids for a multi-unit auction with k = 1 to the k != 1 auction for the
        # second buyer. The idea is that any buyer at any point we will bid the same way he would've bid in the k = 1
        # auction. TODO may be able to change definition of i to reduce complexity
        for i in range(int(self.num_items / k) - 1, -1, -1):
            for j in range(i * k + 1):
                # track how many times we have to go right
                sb_items = 0
                for a in range(k):
                    fb[i * k][j][a] = self.sfb[i * k + a][j + sb_items]
                    # second buyer has won the item
                    if self.ssb[i * k][j] > self.sfb[i * k][j]:
                        sb_items += 1
                # sort at the end to have the highest bid at the beginning
                fb[i * k][j][::-1].sort()

        return fb

    def find_best_response_f(self, k, sb) -> (np.ndarray, np.ndarray):
        """
        Find the best response of the first buyer given the bids of the second buyer.

        @param k: number of items sold at the same time
        @param sb: bids of the second buyer
        @return: the best response bids of the first buyer
        """
        # forward utilities
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        # first buyer bids to be calculated
        fb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        # traverse the tree from the bottom
        for i in range(int(self.num_items / k) - 1, -1, -1):
            # traverse the tree from left to right
            for j in range(i * k + 1):
                # index for the path with the best utility,
                # also means the number of items the second buyer gets in this round
                max_index = 0
                # current best utility
                current_max = 0
                # the best utility path of the first buyer --> the utility of the second buyer in this path
                current_su = 0
                # go through all the cases, a means how many items the second buyer gets,
                # a.k.a which index we need to take
                # TODO can be refactored
                for a in range(k + 1):
                    # the first buyer gets 0 items
                    if a == k:
                        current_next = ffu[(i + 1) * k][j + a]
                        su = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - (
                                fb[i * k][j][k - a] * a)
                    # the first buyer gets all items
                    elif a == 0:
                        current_next = ffu[(i + 1) * k][j + a] + sum(
                            self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - (sb[i * k][j][a] * (k - a))
                        su = sfu[(i + 1) * k][j + a]
                    # both buyers get some items
                    else:
                        current_next = ffu[(i + 1) * k][j + a] + sum(
                            self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - (sb[i * k][j][a] * (k - a))
                        su = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - (
                                fb[i * k][j][k - a] * a)
                    # we got a new best path
                    if current_next > current_max:
                        current_max = current_next
                        current_su = su
                        max_index = a
                # at the end set the forward utilities
                ffu[i * k][j] = current_max
                sfu[i * k][j] = current_su
                # set bids of the first buyer if he got some items, else they are 0
                if max_index != k:
                    # currently use identical bids for all items
                    # THE SAME as the second buyers bid that wants to be overbid
                    # TODO need to some stuff here. Maybe need to think about bids that make us indifferent about
                    #  winning and losing?
                    for a in range(k):
                        fb[i * k][j][a] = sb[i * k][j][max_index]

        # the end utility is the root utilty
        fb_utility = ffu[0][0]
        sb_utility = sfu[0][0]

        return fb_utility, sb_utility, fb


    def find_best_response_s(self, k, fb) -> (np.ndarray, np.ndarray):
        """
        Find the best response of the second buyer given the bids of the first buyer.
        Mostly identical to the previous function, with differences marked out.

        @param k: number of items sold at the same time
        @param sb: bids of the second buyer
        @return: the best response bids of the second buyer
        """
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        for i in range(int(self.num_items / k) - 1, -1, -1):
            for j in range(i * k + 1):
                # this now obviously is the index of the best path for the second buyer
                max_index = 0
                current_max = 0
                current_fu = 0
                # also means the items the second buyer has won
                # TODO pretty sure this is correct but for sure refactorable
                for a in range(k + 1):
                    if a == k:
                        fu = ffu[(i + 1) * k][j + a]
                        current_next = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - (
                                fb[i * k][j][k - a] * a)
                    elif a == 0:
                        fu = ffu[(i + 1) * k][j + a] + sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - (
                                sb[i * k][j][a] * (k - a))
                        current_next = sfu[(i + 1) * k][j + a]
                    else:
                        fu = ffu[(i + 1) * k][j + a] + sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - (
                                sb[i * k][j][a] * (k - a))
                        current_next = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - (
                                fb[i * k][j][k - a] * a)
                    if current_next > current_max:
                        current_max = current_next
                        current_fu = fu
                        max_index = a
                ffu[i * k][j] = current_fu
                sfu[i * k][j] = current_max
                if max_index != 0:
                    if max_index != k:
                        for a in range(k):
                            # bid +1 of the first buyer bids that we want to overbid
                            sb[i * k][j][a] = fb[i * k][j][max_index] + 1
                    else:
                        for a in range(k):
                            # bid 0 if we don't want to beat any first buyer bids
                            sb[i * k][j][a] = 0

        fb_utility = ffu[0][0]
        sb_utility = sfu[0][0]

        return sb_utility, fb_utility, sb

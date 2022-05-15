import math
from bisect import bisect_right, bisect_left, bisect

import numpy as np


class AuctionCalculator:

    def __init__(self, num_items=None, fb_values=None, sb_values=None, values=None, k=None, num_buyers=2):
        self.num_items = num_items
        self.k = k
        self.num_buyers = num_buyers
        # valuations
        self.fb_values = fb_values
        self.sb_values = sb_values
        # valuations for 3+ buyers
        self.values = values
        # bids for multi-unit auction with k = 1
        self.sfb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        self.ssb = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        # combinational table initialization
        self.table = self.build_combinatorial_table()

    def get_vcg_prices(self, k) -> int:
        """
        Calculate VCG prices for 2 buyers. Requires fb_values and sb_values to be given.
        """
        # forward utilities
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        f_prices = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        s_prices = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)

        possible_path = np.zeros([self.num_items + 1, self.num_items + 1, k + 1], dtype=bool)

        # traverse the tree from the bottom
        for i in range(int(self.num_items / k) - 1, -1, -1):
            # traverse the tree from left to right
            for j in range(i * k + 1):
                # index for the path with the best utility,
                # also means the number of items the second buyer gets in this round
                max_utility = -1
                max_f_utility = -1
                max_s_utility = -1
                max_without_f_utility = -1
                max_without_s_utility = -1
                for a in range(k + 1):
                    f_utility = sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) + ffu[(i + 1) * k][j + a]
                    s_utility = sum(self.sb_values[j:j + a]) + sfu[(i + 1) * k][j + a]
                    current_utility = f_utility + s_utility
                    if current_utility > max_utility:
                        max_utility = current_utility
                        max_f_utility = f_utility
                        max_s_utility = s_utility
                    if a == 0 and f_utility > max_without_s_utility:
                        max_without_s_utility = f_utility
                    elif a == k and s_utility > max_without_f_utility:
                        max_without_f_utility = s_utility
                for a in range(k + 1):
                    f_utility = sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) + ffu[(i + 1) * k][j + a]
                    s_utility = sum(self.sb_values[j:j + a]) + sfu[(i + 1) * k][j + a]
                    current_utility = f_utility + s_utility
                    if current_utility == max_utility:
                        possible_path[i * k][j][a] = True
                f_price = max_f_utility - (max_utility - max_without_f_utility)
                s_price = max_s_utility - (max_utility - max_without_s_utility)
                f_prices[i * k][j] = f_price
                s_prices[i * k][j] = s_price
                ffu[i * k][j] = max_f_utility - f_price
                sfu[i * k][j] = max_s_utility - s_price

        reachable = np.zeros([self.num_items + 1, self.num_items + 1], dtype=bool)
        lowest = np.full([self.num_items + 1, self.num_items + 1], np.inf)
        self.get_vcg_worst_welfare(possible_path, f_prices, s_prices, lowest, reachable)

        min_welfare = math.inf
        for j in range(self.num_items + 1):
            if reachable[self.num_items][j]:
                current_welfare = sum(self.fb_values[:self.num_items - j]) + sum(self.sb_values[:j])
                if current_welfare < min_welfare:
                    min_welfare = current_welfare

        #print(f'Welfare is {max_welfare}, buyer 1 gets {self.num_items - max_index} items and buyer 2 gets {max_index} items')
        return min_welfare

    def get_vcg_worst_welfare(self, possible_path, f_prices, s_prices, lowest, reachable):
        """
        Calculate worst efficiency for 2 buyer auction of all equilibria.
        """
        reachable[0][0] = True
        for i in range(0, self.num_items - self.k + 1, self.k):
            for j in range(i + 1):
                for a in range(self.k + 1):
                    if reachable[i][j] and possible_path[i][j][a]:
                        price = f_prices[i][j] + s_prices[i][j]
                        if lowest[i][j] < price:
                            return False
                        reachable[i + self.k][j + a] = True
                        if lowest[i + self.k][j + a] > price:
                            lowest[i + self.k][j + a] = price
        return True

    def get_vcg_prices_for_3_or_more_buyers(self, k) -> int:
        """
        Calculate VCG prices for 3 or more buyers. Requires values to be given.
        """
        # amount of permutations
        permutations = self.table[self.num_buyers][self.num_items]

        # forward utility array
        fu = np.zeros([self.num_items + 1, permutations, self.num_buyers], dtype=int)

        # VCG prices
        prices = np.zeros([self.num_items + 1, permutations, self.num_buyers], dtype=int)

        # path for lowest welfare finding
        possible_path = np.zeros([self.num_items + 1, permutations, self.table[self.num_buyers][k]], dtype=bool)

        # traverse the tree from the bottom
        for i in range(int(self.num_items / k) - 1, -1, -1):
            # traverse the tree from left to right
            for j in range(max(1, self.table[self.num_buyers][i * k])):
                sold_distr = self.convert_index_to_distribution(j, i * k)
                sold_occurrences = np.zeros(self.num_buyers, dtype=int)
                for num in sold_distr:
                    sold_occurrences[num] += 1
                # max utility for this node
                max_utility = -1
                # utilities of individual buyers in max utility case
                max_individual_utility = np.zeros(self.num_buyers)
                # utilities without certain buyer in max utility case
                max_without_utility = np.zeros(self.num_buyers)
                # loop through distributions
                for a in range(self.table[self.num_buyers][k]):
                    distr = self.convert_index_to_distribution(a, k)
                    occurrences = np.zeros(self.num_buyers, dtype=int)
                    for num in distr:
                        occurrences[num] += 1
                    # the index of the next node
                    next_index = self.convert_distribution_to_index(np.sort(np.concatenate((distr, sold_distr), axis=None)))
                    utilities = np.zeros(self.num_buyers, dtype=int)
                    # calculate utilities for individual buyers
                    for b in range(self.num_buyers):
                        utilities[b] = sum(self.values[b][sold_occurrences[b]:sold_occurrences[b] + occurrences[b]]) + \
                                       fu[(i + 1) * k][next_index][b]
                    current_utility = sum(utilities)
                    if current_utility > max_utility:
                        max_utility = current_utility
                        # set max utility of individual buyers for the max utility case
                        for b in range(self.num_buyers):
                            max_individual_utility[b] = utilities[b]
                    for b in range(self.num_buyers):
                        max_without_current_utility = current_utility - utilities[b]
                        if max_without_current_utility > max_without_utility[b]:
                            max_without_utility[b] = max_without_current_utility
                for a in range(self.table[self.num_buyers][k]):
                    distr = self.convert_index_to_distribution(a, k)
                    occurrences = np.zeros(self.num_buyers, dtype=int)
                    for num in distr:
                        occurrences[num] += 1
                    # the index of the next node
                    next_index = self.convert_distribution_to_index(np.sort(np.concatenate((distr, sold_distr), axis=None)))
                    utilities = np.zeros(self.num_buyers, dtype=int)
                    # calculate utilities for individual buyers
                    for b in range(self.num_buyers):
                        utilities[b] = sum(self.values[b][sold_occurrences[b]:sold_occurrences[b] + occurrences[b]]) + \
                                       fu[(i + 1) * k][next_index][b]
                    current_utility = sum(utilities)
                    if current_utility == max_utility:
                        possible_path[i * k][j][a] = True
                for b in range(self.num_buyers):
                    price = max_individual_utility[b] - (max_utility - max_without_utility[b])
                    prices[i * k][j][b] = price
                    fu[i * k][j][b] = max_individual_utility[b] - price

        reachable = np.zeros([self.num_items + 1, permutations], dtype=bool)
        lowest = np.full([self.num_items + 1, permutations], np.inf)
        self.get_vcg_worst_welfare_for_3_or_more_buyers(possible_path, prices, lowest, reachable)

        min_welfare = math.inf
        for j in range(permutations):
            if reachable[self.num_items][j]:
                distr = self.convert_index_to_distribution(j, self.num_items)
                occurrences = np.zeros(self.num_buyers, dtype=int)
                for num in distr:
                    occurrences[num] += 1
                current_welfare = 0
                for b in range(self.num_buyers):
                    current_welfare += sum(self.values[b][:occurrences[b]])
                if current_welfare < min_welfare:
                    min_welfare = current_welfare

        #print(min_welfare)
        return min_welfare

    def get_vcg_worst_welfare_for_3_or_more_buyers(self, possible_path, prices, lowest, reachable):
        """
        Calculate worst efficiency for 3 or more buyer auction of all equilibria.
        """
        reachable[0][0] = True
        for i in range(0, self.num_items - self.k + 1, self.k):
            for j in range(max(1, self.table[self.num_buyers][i])):
                sold_distr = self.convert_index_to_distribution(j, i)
                for a in range(self.table[self.num_buyers][self.k]):
                    distr = self.convert_index_to_distribution(a, self.k)
                    total_distr = np.sort(np.concatenate((sold_distr, distr), axis=None))
                    next_index = self.convert_distribution_to_index(total_distr)
                    if reachable[i][j] and possible_path[i][j][a]:
                        price = sum(prices[i][j])
                        if lowest[i][j] < price:
                            return False
                        reachable[i + self.k][next_index] = True
                        if lowest[i + self.k][next_index] > price:
                            lowest[i + self.k][next_index] = price
        return True

    def build_combinatorial_table(self):
        table = np.zeros([self.num_buyers + 1, self.num_items + 1], dtype=int)
        for n in range(1, self.num_buyers + 1):
            for m in range(1, self.num_items + 1):
                table[n, m] = math.factorial(m + n - 1) / (math.factorial(m) * math.factorial(n - 1))
        return table

    def convert_distribution_to_index(self, distr):
        current_items = len(distr)
        index = 0
        for i in range(current_items):
            index += self.table[self.num_buyers - distr[i] - 1][current_items - i]
        index = self.table[self.num_buyers][current_items] - index - 1
        return index

    def convert_index_to_distribution(self, index, current_items):
        index = self.table[self.num_buyers][current_items] - index - 1
        distr = np.zeros(current_items, dtype=int)
        for i in range(current_items, 0, -1):
            buyer_index = bisect(self.table[:, i], index) - 1
            distr[current_items - i] = self.num_buyers - 1 - buyer_index
            index -= self.table[buyer_index][i]
        return distr

    def find_equilibrium(self) -> int:
        # first calculate the result of the k = 1 auction,
        # then use these results to create k != 1 auction with same bids as the k = 1 auction
        self.calculate_single(k=1)
        fb_set = list()
        sb_set = list()
        # assume that the first buyer just bids 0 everywhere first
        fb_bids = np.zeros([self.num_items + 1, self.num_items + 1, self.k], dtype=int)
        old_sb_bids = np.zeros([self.num_items + 1, self.num_items + 1, self.k], dtype=int)
        # find best response for the second buyer --> the second buyer has fixed his bids now
        sb_bids = self.find_best_response_s(self.k, fb_bids, old_sb_bids)
        fb_counter_utility, sb_utility = self.find_utilities(fb_bids, sb_bids, self.k)
        while True:
            # now find the best response for the first buyer again,
            # we also output the utility of the second buyer if the first buyer bids according to THIS best response
            old_fb_bids = fb_bids
            fb_bids = self.find_best_response_f(self.k, sb_bids, old_fb_bids)
            for fb in fb_set:
                if (fb_bids == fb).all():
                    print('loop found')
                    return 0
            fb_set.append(fb_bids)
            fb_utility, sb_counter_utility = self.find_utilities(fb_bids, sb_bids, self.k)
            # if the utility of the second buyer is equal from his earlier best response
            # and this best response of the first buyer, then we have found an equilibrium
            if fb_utility == fb_counter_utility:
                return sum(self.find_utilities(old_fb_bids, sb_bids, self.k, True))
            # now find the best response for the second buyer again,
            # we also output the utility of the first buyer if the second buyer bids according to THIS best response
            old_sb_bids = sb_bids
            sb_bids = self.find_best_response_s(self.k, fb_bids, old_sb_bids)
            for sb in sb_set:
                if (sb_bids == sb).all():
                    print('loop found')
                    return 0
            fb_set.append(sb_bids)
            fb_counter_utility, sb_utility = self.find_utilities(fb_bids, sb_bids, self.k)
            # if the utility of the first buyer is equal from his earlier best response
            # and this best response of the second buyer, then we have found an equilibrium
            if sb_utility == sb_counter_utility:
                return sum(self.find_utilities(fb_bids, old_sb_bids, self.k, True))

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
            #print(
            #    f"In round {i}, buyer 1 bids {self.sfb[i][current_second_buyer_items]}, buyer 2 bids {self.ssb[i][current_second_buyer_items]}. "
            #    f"Buyer {1 if self.sfb[i][current_second_buyer_items] >= self.ssb[i][current_second_buyer_items] else 2} wins the item.")
            if self.sfb[i][current_second_buyer_items] < self.ssb[i][current_second_buyer_items]:
                current_second_buyer_items += k

    def find_best_response_f(self, k, sb, old_fb) -> (np.ndarray, np.ndarray):
        """
        Find the best response of the first buyer given the bids of the second buyer.

        @param k: number of items sold at the same time
        @param sb: bids of the second buyer
        @return: the best response bids of the first buyer
        """
        # forward utilities
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        # first buyer bids to be calculated
        fb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        # traverse the tree from the bottom
        for i in range(int(self.num_items / k) - 1, -1, -1):
            # traverse the tree from left to right
            for j in range(i * k + 1):
                # index for the path with the best utility,
                # also means the number of items the second buyer gets in this round
                max_index = -1
                # current best utility
                current_max = -1
                # the best utility path of the first buyer --> the utility of the second buyer in this path
                current_su = 0
                # number of zero bids from the second buyer
                # TODO leading zeros
                sb_zeros = len([elem for elem in sb[i * k][j] if elem == 0])
                unlosable_zeros = 0
                for a in range(sb_zeros):
                    if self.fb_values[(i * k) - j + a] > self.sb_values[j + k - 1 - a]:
                        unlosable_zeros += 1
                    elif self.fb_values[(i * k) - j + a] == self.sb_values[j + k - 1 - a]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) >= sum(self.sb_values[j:j + k]):
                            unlosable_zeros += 1
                # go through all the cases, a means how many items the second buyer gets,
                # a.k.a which index we need to take
                for a in range(k, -1, -1):
                    # first buyer has to get a certain amount of items because of zeros in the second buyers bids
                    if unlosable_zeros > k - a:
                        continue
                    # the first buyer gets 0 items
                    if a == k:
                        current_next = ffu[(i + 1) * k][j + a]
                    # the first buyer gets some items
                    else:
                        # disallow overbidding
                        overbid = False
                        for b in range(k - a):
                            if self.fb_values[(i * k) - j + b] < self.sb_values[j + a]:
                                if self.fb_values[(i * k) - j + b] <= sb[i * k][j][a]:
                                    overbid = True
                                    break
                            elif self.fb_values[(i * k) - j + b] == self.sb_values[j + a]:
                                if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) < sum(self.sb_values[j:j + k]):
                                    if self.fb_values[(i * k) - j + b] <= sb[i * k][j][a]:
                                        overbid = True
                                        break
                                else:
                                    if self.fb_values[(i * k) - j + b] < sb[i * k][j][a]:
                                        overbid = True
                                        break
                            else:
                                if self.fb_values[(i * k) - j + b] < sb[i * k][j][a]:
                                    overbid = True
                                    break
                        #if overbid:
                        #    continue
                        #current_next = ffu[(i + 1) * k][j + a] + \
                        #               sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - sum(sb[i * k][j][a:k])
                        if a != k:
                            current_next = ffu[(i + 1) * k][j + a] + \
                                           sum(self.fb_values[(i * k) - j:(i * k) - j + (k - a)]) - sb[i * k][j][a] * (k - a)
                        else:
                            current_next = ffu[(i + 1) * k][j + a]
                    # we got a new best path
                    if current_next > current_max:
                        current_max = current_next
                        max_index = a
                # at the end set the forward utilities
                ffu[i * k][j] = current_max
                # currently use identical bids for all items
                # THE SAME as the second buyers bid that wants to be overbid
                if max_index != k:
                    # find out if the last bid must overbid the second buyer -> all must overbid
                    if self.fb_values[(i * k) - j + k - 1] < self.sb_values[j + max_index]:
                        for a in range(k - max_index):
                            fb[i * k][j][a] = sb[i * k][j][max_index] + 1
                    elif self.fb_values[(i * k) - j + k - 1] == self.sb_values[j + max_index]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) < sum(self.sb_values[j:j + k]):
                            for a in range(k - max_index):
                                fb[i * k][j][a] = sb[i * k][j][max_index] + 1
                        else:
                            for a in range(k - max_index):
                                fb[i * k][j][a] = sb[i * k][j][max_index]
                    else:
                        for a in range(k - max_index):
                            fb[i * k][j][a] = sb[i * k][j][max_index]
                    # bid -1 of the first bid that wants to be underbid or real valuation
                    if self.fb_values[(i * k) - j + k - 1] < self.sb_values[j + max_index]:
                        for a in range(k - max_index, k, 1):
                            fb[i * k][j][a] = min(sb[i * k][j][max_index], self.fb_values[i * k - j + a])
                    elif self.fb_values[(i * k) - j + k - 1] == self.sb_values[j + max_index]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) < sum(self.sb_values[j:j + k]):
                            for a in range(k - max_index):
                                fb[i * k][j][a] = max(0, min(sb[i * k][j][max_index], self.fb_values[i * k - j + a],
                                                             0))
                        else:
                            for a in range(k - max_index):
                                fb[i * k][j][a] = max(0, min(sb[i * k][j][max_index] - 1, self.fb_values[i * k - j + a],
                                                             0))
                    else:
                        for a in range(k - max_index, k, 1):
                            fb[i * k][j][a] = max(0, min(sb[i * k][j][max_index] - 1, self.fb_values[i * k - j + a], 0))
                else:
                    if self.fb_values[(i * k) - j + k - 1] < self.sb_values[j + max_index - 1]:
                        for a in range(k - max_index, k, 1):
                            fb[i * k][j][a] = min(sb[i * k][j][k - 1], self.fb_values[i * k - j + a])
                    elif self.fb_values[(i * k) - j + k - 1] < self.sb_values[j + max_index - 1]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) < sum(self.sb_values[j:j + k]):
                            for a in range(k - max_index):
                                fb[i * k][j][a] = max(0, min(sb[i * k][j][k - 1], self.fb_values[i * k - j + a], 0))
                        else:
                            for a in range(k - max_index):
                                fb[i * k][j][a] = max(0, min(sb[i * k][j][k - 1] - 1, self.fb_values[i * k - j + a], 0))
                    else:
                        for a in range(k - max_index, k, 1):
                            fb[i * k][j][a] = max(0, min(sb[i * k][j][k - 1] - 1, self.fb_values[i * k - j + a], 0))

                # check if old bids were as good, if yes take old bids
                f_pos = 0
                s_pos = 0
                for a in range(k):
                    # always give the item to the buyer with the higher utility
                    # when the utilities are the same, give it to the first buyer
                    if self.fb_values[i * k - j + f_pos] > self.sb_values[j + s_pos]:
                        if old_fb[i * k][j][f_pos] >= sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                    elif self.fb_values[i * k - j + f_pos] == self.sb_values[j + s_pos]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) >= sum(self.sb_values[j:j + k]):
                            if old_fb[i * k][j][f_pos] >= sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                        else:
                            if old_fb[i * k][j][f_pos] > sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                    else:
                        if old_fb[i * k][j][f_pos] > sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                if s_pos == max_index:
                    fb[i * k][j] = old_fb[i * k][j]
        return fb

    def find_best_response_s(self, k, fb, old_sb) -> (np.ndarray, np.ndarray):
        """
        Find the best response of the second buyer given the bids of the first buyer.
        Mostly identical to the previous function, with differences marked out.

        @param k: number of items sold at the same time
        @param sb: bids of the second buyer
        @return: the best response bids of the second buyer
        """
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sb = np.zeros([self.num_items + 1, self.num_items + 1, k], dtype=int)

        for i in range(int(self.num_items / k) - 1, -1, -1):
            for j in range(i * k + 1):
                # this now obviously is the index of the best path for the second buyer
                max_index = -1
                current_max = -1
                # TODO leading zeros
                fb_zeros = len([elem for elem in fb[i * k][j] if elem == 0])
                unlosable_zeros = 0
                for a in range(fb_zeros):
                    if self.sb_values[j + a] > self.fb_values[(i * k) - j + k - 1 - a]:
                        unlosable_zeros += 1
                    elif self.sb_values[j + a] == self.fb_values[(i * k) - j + k - 1 - a]:
                        if sum(self.sb_values[j:j + k]) > sum(self.fb_values[(i * k) - j:(i * k) - j + k]):
                            unlosable_zeros += 1
                # also means the items the second buyer has won
                for a in range(k + 1):
                    if unlosable_zeros > a:
                        continue
                    if a == 0:
                        current_next = sfu[(i + 1) * k][j + a]
                    else:
                        # disallow overbidding
                        overbid = False
                        for b in range(a):
                            if self.sb_values[j + b] < self.fb_values[i * k - j + (k - a)]:
                                if self.sb_values[j + b] <= fb[i * k][j][k - a]:
                                    overbid = True
                                    break
                            elif self.sb_values[j + b] == self.fb_values[i * k - j + (k - a)]:
                                if sum(self.sb_values[j:j + k]) <= sum(self.fb_values[(i * k) - j:(i * k) - j + k]):
                                    if self.sb_values[j + b] <= fb[i * k][j][k - a]:
                                        overbid = True
                                        break
                                else:
                                    if self.sb_values[j + b] < fb[i * k][j][k - a]:
                                        overbid = True
                                        break
                            else:
                                if self.sb_values[j + b] < fb[i * k][j][k - a]:
                                    overbid = True
                                    break
                        #if overbid:
                        #    continue
                        # current_next = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - \
                        #                sum(fb[i * k][j][k - a:k])
                        if a != 0:
                            current_next = sfu[(i + 1) * k][j + a] + sum(self.sb_values[j:j + a]) - \
                                           fb[i * k][j][k - a] * a
                        else:
                            current_next = sfu[(i + 1) * k][j + a]
                    if current_next > current_max:
                        current_max = current_next
                        max_index = a
                sfu[i * k][j] = current_max
                if max_index != 0:
                    # find out if the last bid must overbid the first buyer -> all must overbid
                    if self.sb_values[j + max_index - 1] <= self.fb_values[i * k - j + (k - max_index)]:
                        for a in range(max_index):
                            sb[i * k][j][a] = fb[i * k][j][k - max_index] + 1
                    elif self.sb_values[j + max_index - 1] == self.fb_values[i * k - j + (k - max_index)]:
                        if sum(self.sb_values[j:j + k]) <= sum(self.fb_values[(i * k) - j:(i * k) - j + k]):
                            for a in range(max_index):
                                sb[i * k][j][a] = fb[i * k][j][k - max_index] + 1
                        else:
                            for a in range(max_index):
                                sb[i * k][j][a] = fb[i * k][j][k - max_index]
                    else:
                        for a in range(max_index):
                            sb[i * k][j][a] = fb[i * k][j][k - max_index]
                    # no overbidding
                    if self.sb_values[j + k - 1] <= self.fb_values[i * k - j + (k - max_index)]:
                        for a in range(max_index, k, 1):
                            sb[i * k][j][a] = min(fb[i * k][j][k - max_index], self.sb_values[j + a])
                    elif self.sb_values[j + k - 1] == self.fb_values[i * k - j + (k - max_index)]:
                        if sum(self.sb_values[j:j + k]) <= sum(self.fb_values[(i * k) - j:(i * k) - j + k]):
                            for a in range(max_index):
                                sb[i * k][j][a] = min(fb[i * k][j][k - max_index], self.sb_values[j + a])
                        else:
                            for a in range(max_index):
                                sb[i * k][j][a] = max(0, min(fb[i * k][j][k - max_index] - 1, self.sb_values[j + a]))
                    else:
                        for a in range(max_index, k, 1):
                            sb[i * k][j][a] = max(0, min(fb[i * k][j][k - max_index] - 1, self.sb_values[j + a]))
                else:
                    if self.sb_values[j + k - 1] <= self.fb_values[i * k - j + (k - 1)]:
                        for a in range(max_index, k, 1):
                            sb[i * k][j][a] = min(fb[i * k][j][k - 1], self.sb_values[j + a])
                    elif self.sb_values[j + k - 1] == self.fb_values[i * k - j + (k - 1)]:
                        if sum(self.sb_values[j:j + k]) <= sum(self.fb_values[(i * k) - j:(i * k) - j + k]):
                            for a in range(max_index):
                                sb[i * k][j][a] = min(fb[i * k][j][k - 1], self.sb_values[j + a])
                        else:
                            for a in range(max_index):
                                sb[i * k][j][a] = max(0, min(fb[i * k][j][k - 1] - 1, self.sb_values[j + a]))
                    else:
                        for a in range(max_index, k, 1):
                            sb[i * k][j][a] = max(0, min(fb[i * k][j][k - 1] - 1, self.sb_values[j + a]))

                # check if old bids were as good, if yes take old bids
                f_pos = 0
                s_pos = 0
                for a in range(k):
                    # always give the item to the buyer with the higher utility
                    # when the utilities are the same, give it to the first buyer
                    if self.fb_values[i * k - j + f_pos] > self.sb_values[j + s_pos]:
                        if fb[i * k][j][f_pos] >= old_sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                    elif self.fb_values[i * k - j + f_pos] == self.sb_values[j + s_pos]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) >= sum(self.sb_values[j:j + k]):
                            if fb[i * k][j][f_pos] >= old_sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                        else:
                            if fb[i * k][j][f_pos] > old_sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                    else:
                        if fb[i * k][j][f_pos] > old_sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                if s_pos == max_index:
                    sb[i * k][j] = old_sb[i * k][j]
        return sb

    def find_total_welfare(self, fb, sb, k) -> int:
        for a in range(k):
            if fb[0][0][a] > self.fb_values[a]:
                print('hi')
            if sb[0][0][a] > self.sb_values[a]:
                print('hi')

        current_position = 0
        for i in range(int(self.num_items / k)):
            s_items = 0
            f_pos = 0
            s_pos = 0
            for j in range(k):
                if fb[i * k][current_position][f_pos] >= sb[i * k][current_position][s_pos]:
                    f_pos += 1
                else:
                    s_pos += 1
                    s_items += 1
            current_position += s_items
        return sum(self.fb_values[:self.num_items - current_position]) + sum(self.sb_values[:current_position])

    def find_utilities(self, fb, sb, k, end=False) -> (int, int):
        if end:
            for a in range(k):
                if fb[0][0][a] > self.fb_values[a] + 2:
                    print('hi')
                if sb[0][0][a] > self.sb_values[a] + 2:
                    print('hi')
        # forward utilities
        ffu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        sfu = np.zeros([self.num_items + 1, self.num_items + 1], dtype=int)
        for i in range(int(self.num_items / k) - 1, -1, -1):
            # from left to right, the leftmost means the first buyer won all items until this round
            # and the rightmost means the second buyer won all items
            for j in range(i * k + 1):
                f_pos = 0
                s_pos = 0
                for a in range(k):
                    # always give the item to the buyer with the higher utility
                    # when the utilities are the same, give it to the first buyer
                    if self.fb_values[i * k - j + f_pos] > self.sb_values[j + s_pos]:
                        if fb[i * k][j][f_pos] >= sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                    elif self.fb_values[i * k - j + f_pos] == self.sb_values[j + s_pos]:
                        if sum(self.fb_values[(i * k) - j:(i * k) - j + k]) >= sum(self.sb_values[j:j + k]):
                            if fb[i * k][j][f_pos] >= sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                        else:
                            if fb[i * k][j][f_pos] > sb[i * k][j][s_pos]:
                                f_pos += 1
                            else:
                                s_pos += 1
                    else:
                        if fb[i * k][j][f_pos] > sb[i * k][j][s_pos]:
                            f_pos += 1
                        else:
                            s_pos += 1
                # ffu[i * k][j] = ffu[(i + 1) * k][j + s_pos] + sum(self.fb_values[(i * k) - j:(i * k) - j + (k - s_pos)]) \
                #                 - sum(sb[i * k][j][s_pos:k])
                # sfu[i * k][j] = sfu[(i + 1) * k][j + s_pos] + sum(self.sb_values[j:j + s_pos]) - \
                #                        sum(fb[i * k][j][k - s_pos:k])
                if s_pos != k:
                    ffu[i * k][j] = ffu[(i + 1) * k][j + s_pos] + sum(self.fb_values[(i * k) - j:(i * k) - j + (k - s_pos)]) \
                                    - sb[i * k][j][s_pos] * (k - s_pos)
                else:
                    ffu[i * k][j] = ffu[(i + 1) * k][j + s_pos]
                if s_pos != 0:
                    sfu[i * k][j] = sfu[(i + 1) * k][j + s_pos] + sum(self.sb_values[j:j + s_pos]) - \
                                           fb[i * k][j][k - s_pos] * s_pos
                else:
                    sfu[i * k][j] = sfu[(i + 1) * k][j + s_pos]

        return ffu[0][0], sfu[0][0]


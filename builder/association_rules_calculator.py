import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")

import django
django.setup()

from collections import defaultdict
from itertools import combinations
from datetime import datetime

from collector.models import Log
from recommender.models import SeededRecs


def build_association_rules():
    data = retrieve_buy_events()
    data = generate_transactions(data)
    #print(data)
    data = calculate_support_confidence(data, 0.01)
    save_rules(data)


def retrieve_buy_events():
    data = Log.objects.filter(event='buy').values()
    return data


def generate_transactions(data):
    transactions = dict()
    for transaction_item in data:
        # {'id': 99869, 'created': datetime.datetime(2022, 8, 1, 15, 57, 17), 'user_id': '400004',
        # 'content_id': '2869728', 'event': 'buy', 'session_id': '935090'}
        transaction_id = transaction_item["session_id"]
        if transaction_id not in transactions:
            transactions[transaction_id] = []
        transactions[transaction_id].append(transaction_item["content_id"])

    return transactions

def calculate_support_confidence(transactions, min_sup=0.01):
    """
    transactions  ==>  {session_id:[v1,v2...]}
    这里认为同一个session_id有过行为的item_id是有关联关系的。
    """

    N = len(transactions)
    print(N)
    one_itemsets = calculate_itemsets_one(transactions, min_sup)
    print(one_itemsets)
    two_itemsets = calculate_itemsets_two(transactions, one_itemsets)

    rules = calculate_association_rules(one_itemsets, two_itemsets, N)
    print(rules)
    return sorted(rules)


def calculate_itemsets_one(transactions, min_sup=0.01):
    """
    transactions  ==>  {session_id:[v1,v2...]}
    转换为movie_id:nums的字典形式
    """

    N = len(transactions) # 一共有N个记录，session_id。min_sup*N=有行为的电影的最小次数。

    temp = defaultdict(int)
    one_itemsets = dict()

    for key, items in transactions.items(): # session_id:[movie_id,...]
        for item in items:
            inx = frozenset({item}) # frozenset({'a'})
            temp[inx] += 1

    print("temp:")
    #print(temp)
    # remove all items that is not supported.
    for key, itemset in temp.items():
        #print(f"{key}, {itemset}, {min_sup}, {min_sup * N}")
        if itemset > min_sup * N:
            one_itemsets[key] = itemset

    return one_itemsets


def calculate_itemsets_two(transactions, one_itemsets):
    """
    transactions  ==>  {session_id:[v1,v2...]}
    转换为(movie_id1,movie_id2):nums的字典形式
    """
    two_itemsets = defaultdict(int)

    for key, items in transactions.items():
        items = list(set(items))  # remove duplications

        if (len(items) > 2):
            for perm in combinations(items, 2): # 两两组合
                if has_support(perm, one_itemsets):
                    two_itemsets[frozenset(perm)] += 1
        elif len(items) == 2:
            if has_support(items, one_itemsets):
                two_itemsets[frozenset(items)] += 1
    return two_itemsets


def calculate_association_rules(one_item_sets, two_item_sets, N):
    """
    关联推荐
    one_item_sets  =>  movie_id:nums
    two_item_sets  => (movie_id1,movie_id2):nums
    N:session_id的个数
    """
    timestamp = datetime.now()

    rules = []
    for source, source_freq in one_item_sets.items(): # movie_id:nums
        for key, group_freq in two_item_sets.items(): # (movie_id1,movie_id2):nums
            if source.issubset(key):
                target = key.difference(source) # 两个集合的差集，即返回key本身
                support = group_freq / N # 支持度是个百分比，它指的是某个商品组合出现的次数与总次数之间的比例。
                confidence = group_freq / source_freq  # 置信度是个条件概念，就是说在 A 发生的情况下，B 发生的概率是多少。
                # 提升度 (A→B)= 置信度 (A→B)/ 支持度 (B)
                rules.append((timestamp, next(iter(source)), next(iter(target)),
                              confidence, support))
    return rules


def has_support(perm, one_itemsets):
    return frozenset({perm[0]}) in one_itemsets and \
           frozenset({perm[1]}) in one_itemsets


def save_rules(rules):

    for rule in rules:
        SeededRecs(
            created=rule[0],
            source=str(rule[1]),
            target=str(rule[2]),
            support=rule[3],
            confidence=rule[4]
        ).save()


if __name__ == '__main__':
    print("Calculating association rules...")

    build_association_rules()



from collections import defaultdict
from email.policy import default


#initializes the probability. The initial prob considers for words that only apper
def initialize_prob(bitext):
    t = defaultdict(lambda: defaultdict(float))
    t_count = defaultdict(int)
    for (origin, target) in bitext:
        for o_w in origin:
            for t_w in target:
                t[o_w][t_w] = 1 / len(target)

        for t_w in target:
            t_count[t_w] += 1

    # divide the t by the count of the target word
    for o_w in t.keys():
        for t_w in t[o_w].keys():
            t[o_w][t_w] /= t_count[t_w]

    # normalize so that sum of all t(t | o) = 1
    for o_w in t.keys():
        total = sum(t[o_w].values())
        for t_w in t[o_w].keys():
            t[o_w][t_w] /= total
    return t

# method to initialize prob. Used in agreement-align paper.
def initialize_prob_by_appearance(bitext):
    t = defaultdict(lambda: defaultdict(float))
    target_count = defaultdict(int)
    for (origin, target) in bitext:
        for o_w in origin:
            for t_w in target:
                t[o_w][t_w] += 1

        for t_w in target:
            target_count[t_w] += 1

    for o_w in t.keys():
        for t_w in t[o_w].keys():
            t[o_w][t_w] /= target_count[t_w]

    return t
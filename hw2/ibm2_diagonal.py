#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import math

from better_initialization import initialize_prob
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iteration", dest="threshold", default=5, type="int", help="maximum number of iterations")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

f_data = f"{opts.train}.{opts.french}"
e_data = f"{opts.train}.{opts.english}"

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]


#Initializes probability for word alignment
def initialize_prob(bitext: list) -> dict:
    target_vocab = set()
    for (origin, target) in bitext:
        for word in target:
            target_vocab.add(word)

    t = defaultdict(lambda: defaultdict(lambda: 1 / len(target_vocab)))
    return t


# Diagonal Preference: Adds bias to favor alignments near the diagonal
def alignment_prob(i, j, m, n, p0=0.08, lambda_=4):
    """Computes the alignment probability with diagonal preference"""
    # h is the distance of (i,j) from the diagonal normalized by sentence lengths
    h = abs(i / m - j / n)
    if j == 0:
        return p0
    else:
        return (1 - p0) * math.exp(-lambda_ * h)


def agreement_e_step(bitext, t_ef, t_fe):
    count_ef = defaultdict(lambda: defaultdict(float))
    total_o_ef = defaultdict(float)
    count_fe = defaultdict(lambda: defaultdict(float))
    total_o_fe = defaultdict(float)

    for (origin, target) in bitext:
        target_total_ef = defaultdict(float)
        target_total_fe = defaultdict(float)

        for t_w in target:
            for o_w in origin:
                # Compute total alignment probability for E-to-F and F-to-E
                target_total_ef[t_w] += t_ef[o_w][t_w]
                target_total_fe[o_w] += t_fe[t_w][o_w]

        # Update counts for E-to-F
        for t_w in target:
            for o_w in origin:
                agreement_prob = (t_ef[o_w][t_w] * t_fe[t_w][o_w])  # Geometric mean for agreement
                count_ef[o_w][t_w] += agreement_prob / target_total_ef[t_w]
                total_o_ef[o_w] += agreement_prob / target_total_ef[t_w]

                # Similarly, update counts for F-to-E
                count_fe[t_w][o_w] += agreement_prob / target_total_fe[o_w]
                total_o_fe[t_w] += agreement_prob / target_total_fe[o_w]

    return count_ef, total_o_ef, count_fe, total_o_fe



def agreement_m_step(count_ef, total_o_ef, t_ef, count_fe, total_o_fe, t_fe):
    for o_w in count_ef.keys():
        for t_w in count_ef[o_w].keys():
            t_ef[o_w][t_w] = count_ef[o_w][t_w] / total_o_ef[o_w]

    for t_w in count_fe.keys():
        for o_w in count_fe[t_w].keys():
            t_fe[t_w][o_w] = count_fe[t_w][o_w] / total_o_fe[t_w]

    return t_ef, t_fe


def run_joint_ibm(bitext, num_iter=5):
    t_ef = initialize_prob(bitext)  # E-to-F
    t_fe = initialize_prob(bitext)  # F-to-E
    # t_ef = initialize_prob(bitext)
    # bitext_rev = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))][
    #          :opts.num_sents]
    # t_fe = initialize_prob(bitext_rev)

    sys.stderr.write("Training with IBM1 joint model...\n")
    for i in range(num_iter):
        sys.stderr.write(f"Iteration {i+1}...\n")
        count_ef, total_o_ef, count_fe, total_o_fe = agreement_e_step(bitext, t_ef, t_fe)
        t_ef, t_fe = agreement_m_step(count_ef, total_o_ef, t_ef, count_fe, total_o_fe, t_fe)

    sys.stderr.write("Done!\n")
    return t_ef, t_fe

def calculate_difference(t_old: dict, t_new: dict) -> float:
    total_diff = 0.0
    for o_w in t_old.keys():
        for t_w in t_old[o_w].keys():
            total_diff += abs(t_old[o_w][t_w] - t_new[o_w][t_w])
    return total_diff


def align_joint(bitext, t_ef, t_fe):
    for (origin, target) in bitext:
        for i, o_w in enumerate(origin):
            max_prob = 0
            max_index = 0
            for j, t_w in enumerate(target):
                combined_prob = (t_ef[o_w][t_w] * t_fe[t_w][o_w]) ** 0.5  # Agreement-based alignment
                if combined_prob > max_prob:
                    max_prob = combined_prob
                    max_index = j
            sys.stdout.write(f"{i}-{max_index} ")
        sys.stdout.write("\n")

def align_joint(bitext, t_ef, t_fe):
    for (origin, target) in bitext:
        for i, o_w in enumerate(origin):
            for j, t_w in enumerate(target):
                combined_prob = (t_ef[o_w][t_w] * t_fe[t_w][o_w]) ** 0.4# Agreement-based alignment
                if combined_prob > .1:

                    sys.stdout.write(f"{i}-{j} ")
        sys.stdout.write("\n")
if __name__ == "__main__":
    t_ef, t_fe = run_joint_ibm(bitext, opts.threshold)
    align_joint(bitext, t_ef, t_fe)

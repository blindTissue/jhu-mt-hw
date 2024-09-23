#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iteration", dest="threshold", default=5, type="int", help="maximum number of iterations")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)



#initializes probability for word alignment.
# Bitext is written in sentences of (Origin, Target) pairs.
# Sets the probability of alignment to equal to the size of vocabulary of the target language.
# The returned dict is formed with the following structure:
# {origin_word: {target_word: 1/len(target_vocab)}}
def initialize_prob(bitext: list) -> dict:
    target_vocab = set()
    for (o, t) in bitext:
        for word in t:
            target_vocab.add(word)

    t= defaultdict(lambda: defaultdict(lambda: 1/len(target_vocab)))
    return t


def e_step(bitext: list, t: dict):
    count = defaultdict(lambda: defaultdict(float))
    total_o = defaultdict(float)
    for (origin, target) in bitext:
        target_total = defaultdict(float)
        for t_w in target:
            for o_w in origin:
                target_total[t_w] += t[o_w][t_w]

        for t_w in target:
            for o_w in origin:
                count[o_w][t_w] += t[o_w][t_w] / target_total[t_w]
                total_o[o_w] += t[o_w][t_w] / target_total[t_w]

    return count, total_o

def m_step(count:dict, total_o: dict, t:dict):
    for o_w in count.keys():
        for t_w in count[o_w].keys():
            t[o_w][t_w] = count[o_w][t_w] / total_o[o_w]

    return t


def run_ibm(bitext, num_iter=5):
    t = initialize_prob(bitext)
    sys.stderr.write("Training with IBM1...\n")
    for i in range(num_iter):
        sys.stderr.write(f"Iteration {i+1}...\n")
        count, total_o = e_step(bitext, t)
        t = m_step(count, total_o, t)
    sys.stderr.write("Done!")
    return t


# Reverse=True if we are doing English to French alignment
def align(bitext: list, t: dict, reverse=False):
    for (origin, target) in bitext:
        for i, o_w in enumerate(origin):
            max_prob = 0
            max_index = 0
            for j, t_w in enumerate(target):
                if t[o_w][t_w] > max_prob:
                    max_prob = t[o_w][t_w]
                    max_index = j
            if reverse:
                sys.stdout.write(f"{max_index}-{i} ")
            else:
                sys.stdout.write(f"{i}-{max_index} ")
        sys.stdout.write("\n")

def align_output_list(bitext: list, t: dict, reverse=False):
    output = []
    for (origin, target) in bitext:
        sentence = []
        for i, o_w in enumerate(origin):
            max_prob = 0
            max_index = 0
            for j, t_w in enumerate(target):
                if t[o_w][t_w] > max_prob:
                    max_prob = t[o_w][t_w]
                    max_index = j
            if reverse:
                sentence.append((max_index, i))
            else:
                sentence.append((i, max_index))
        output.append(sentence)
    return output


if __name__ == "__main__":
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][
             :opts.num_sents]

    t = run_ibm(bitext, opts.threshold)
    align(bitext, t)




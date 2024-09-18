#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

# This is IBM Model 1 which aligns French words to English words


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iterations", dest="iterations", default=10, type="int", help="max threshold for iterations")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=10000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

def initialize_probs(bitext):
    english_words = set()
    for f, e in bitext:
        for e_i in e:
            english_words.add(e_i)

    t = defaultdict(lambda : defaultdict(lambda : 1.0/len(english_words)))
    return t


def e_step(t, bitext):
    count = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)
    for (f, e) in bitext:
        s_total_e = defaultdict(float)
        for e_i in e:
            for f_i in f:
                s_total_e[e_i] += t[f_i][e_i]

        for e_i in e:
            for f_i in f:
                count[f_i][e_i] += t[f_i][e_i] / s_total_e[e_i]
                total[f_i] += t[f_i][e_i] / s_total_e[e_i]
    return count, total


def m_step(t, count, total):
    for f in count.keys():
        for e in count[f].keys():
            t[f][e] = count[f][e] / total[f]
    return t


def training_loop(t, bitext, iterations=5):
    for i in range(iterations):
        sys.stderr.write("Training iteration %i...\n" % i)
        count, total = e_step(t, bitext)
        t = m_step(t, count, total)

    sys.stderr.write("Finished training\n")
    return t

sys.stderr.write("Training with IBM1...\n")
t = initialize_probs(bitext)
t = training_loop(t, bitext, opts.iterations)

sys.stderr.write("Starting alignment...\n")

for f, e in bitext:
    for i, f_i in enumerate(f):
        bestp = 0
        bestj = 0
        for j, e_j in enumerate(e):
            if t[f_i][e_j] > bestp:
                bestp = t[f_i][e_j]
                bestj = j
        sys.stdout.write("%i-%i " % (i, bestj))
    sys.stdout.write("\n")








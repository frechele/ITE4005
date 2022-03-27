import itertools
import sys


def parse_database(filename: str):
    data = []

    with open(filename, 'rt') as f:
        for line in f.readlines():
            data.append(frozenset(map(int, line.split())))

    return data


def generate_candidates(data, length: int, not_freq_patterns=None):
    if length == 1:
        candidates = frozenset()

        for trx in data:
            candidates = candidates.union(trx)

        return [frozenset([cand]) for cand in candidates]

    combinations = itertools.combinations(data, 2)

    candidates = []
    for comb in combinations:
        comb_a, comb_b = map(frozenset, comb)
        comb = comb_a.union(comb_b)

        if len(comb) != length:
            continue

        if length > 2:
            valid = True
            for not_freq in not_freq_patterns:
                if comb.issuperset(not_freq):
                    valid = False
                    break

            if not valid:
                continue

        candidates.append(comb)

    return candidates
    

def filter_count_candidates(data, candidates, min_support):
    counts = dict()
    not_freq = list()

    min_support = len(data) * 0.01 * min_support

    for candidate in candidates:
        count = 0

        for trx in data:
            if candidate.intersection(trx) == candidate:
                count += 1

        if count >= min_support:
            counts[candidate] = count
        else:
            not_freq.append(candidate)

    return counts, not_freq


def apriori(data, min_support):
    candidates = generate_candidates(data, 1)
    counts, not_freq = filter_count_candidates(data, candidates, min_support)
    L = [list(counts.keys())]
    total = counts

    length = 1
    while True:
        length += 1

        candidates = generate_candidates(L[-1], length, not_freq)
        counts, not_freq = filter_count_candidates(data, candidates, min_support)

        total.update(counts)
        
        if len(counts) == 0: # L_k == \Phi
            break

        L.append(list(counts.keys()))

        if len(candidates) == 1:
            break

    return total


def powerset(pattern: set):
    for length in range(1, len(pattern)):
        for comb in itertools.combinations(pattern, length):
            yield frozenset(comb)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python3 {} <min support> <input> <output>'.format(sys.argv[0]))
        sys.exit()

    min_support = float(sys.argv[1])
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    data = parse_database(input_filename)

    total = apriori(data, min_support)

    with open(output_filename, 'wt') as f:
        for pattern in total:
            if len(pattern) == 1:
                continue

            pat_sup = total[pattern]

            for X in powerset(pattern):
                Y = pattern.difference(X)

                x_sup = total[X]

                X_str = ','.join(map(str, X))
                Y_str = ','.join(map(str, Y))

                f.write('{%s}\t{%s}\t%0.2f\t%0.2f\n' % (X_str, Y_str, pat_sup/len(data)*100, pat_sup/x_sup*100))

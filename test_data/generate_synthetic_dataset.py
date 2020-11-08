import numpy as np
import argparse
from typing import List


def generate_abc_dataset(
        num_sequences: int,
        seq_length: int,
) -> List[str]:
    """
    Create a simple dataset using a Markov chain on 3 characters.

    :param num_sequences: Number of sequences to generate.
    :param seq_length: Length of each generated sequence.

    :return: list containing generated sequences
    """
    # Create a Markov chain on 3 characters
    chars = list('abc')
    prior_probs = [0.6, 0.2, 0.2]
    transition_probs = {
        'a': [0.3, 0.2, 0.5],
        'b': [0.4, 0.5, 0.1],
        'c': [0.05, 0.0, 0.95],
    }

    dataset = []
    for i in range(num_sequences):

        # Keep track of ith generated sequence
        seq_i = []

        # Sample from prior
        char = np.random.choice(chars, p=prior_probs)
        seq_i.append(char)

        for j in range(seq_length - 1):
            # Sample from transition distribution
            char = np.random.choice(chars, p=transition_probs[seq_i[-1]])
            seq_i.append(" ")
            seq_i.append(char)

        dataset.append("".join(seq_i))

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Synthetic dataset.')
    parser.add_argument('--dataset', type=str, choices=['abc'], help='Name of dataset.', default='abc')
    parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate.', default=300)
    parser.add_argument('--seq_length', type=int, help='Length of each sequence being generated.', default=10)
    parser.add_argument('--output_file', type=str, help='Name of output file.', required=True)

    args = parser.parse_args()

    if args.dataset == 'abc':
        dataset = generate_abc_dataset(num_sequences=args.num_sequences, seq_length=args.seq_length)
    else:
        raise NotImplementedError

    with open(args.output_file, 'w') as f:
        for seq in dataset:
            f.write(f'{{"text": "{seq}"}}\n')

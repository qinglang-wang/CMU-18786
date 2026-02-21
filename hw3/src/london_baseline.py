# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    eval_corpus_path = "birth_dev.tsv"
    
    with open(eval_corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    predictions = ["London"] * len(lines)
    
    total, correct = utils.evaluate_places(eval_corpus_path, predictions)
    
    if total > 0:
        accuracy = (correct / total) * 100.0
        print(f'Correct: {correct} out of {total}: {accuracy:.1f}%')
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")

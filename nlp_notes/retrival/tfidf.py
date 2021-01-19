"""Simple Tf-Idf implementation."""

import collections
import math
import re
from typing import Dict, Iterable, List


def tokenize(seq: str) -> List[str]:
    return re.findall(r"\w+", seq.strip().lower())


def flatten(docs: Iterable[List[str]]) -> Iterable:
    return (w for s in docs for w in s)


def build_vocab(texts: List[str]) -> Dict[str, int]:
    tokens = flatten(map(tokenize, texts))
    return collections.Counter(tokens)


def tfidf(texts: List[str]) -> Dict[str, float]:
    """Term Frequency Inverse Document Frequency.

    Tf-Idf score/weight reflects how important a word is to
    a document in a collection or corpus.
    """
    features = {}
    N = len(texts)
    vocab = build_vocab(texts)
    for idx in range(N):
        doc = tokenize(texts[idx])
        for w in vocab:
            tf = doc.count(w) / len(vocab)  # Term frequency.
            df = vocab.get(w, 0)  # Document frequency.
            idf = math.log((N + 1) / (df + 1))  # Inverse document frequency.
            features[w] = tf * idf

    return dict(sorted(features.items(), key=lambda k: k[1], reverse=True))


if __name__ == "__main__":
    text0 = (
        "A ConV neural network is a "
        + (" deep " * 67)
        + " learning algorithm that has the ability"
    )
    text1 = "A Neural Network is an algorithm which has a powerful ability"
    dataset = [text0, text1]
    features = tfidf(dataset)
    print(f"Score: {sum(features.values())/len(features):.6f}\n")
    print(list(features.items()))

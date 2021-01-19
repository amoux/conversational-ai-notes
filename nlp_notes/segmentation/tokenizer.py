"""From scratch code, practice + experiment with no help :)

```python
def psims(q, b):
    score = distance.cosine(test[q], train[b])
    print(f"similarity: {score}")
    print(f"q : {web.test[q]}\n\ndb : {web.train[b]}")
```
"""

import math
import random
from collections import Counter
from typing import Iterable, List, Union
from urllib.request import Request, URLError, urlopen

import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from sklearn.cluster import KMeans


def standardize(mx, mean, std):
    # NOTE: When using this method with Tf-Idf weighting, downstream
    # methods become sensative to sequence length.
    return (mx - mean) / std


def cosine(a, b) -> np.ndarray:
    return dot(a, b) / (norm(a) * norm(b))


def cluster_kmeans(mx: np.ndarray, data: List[str], n=8, **kwargs):
    kmeans = KMeans(n, **kwargs)
    kmeans.fit(mx)
    cluster = {k: [] for k in range(n)}
    for doc_id, k in enumerate(kmeans.labels_):
        cluster[k].append(data[doc_id])
    return cluster


class WebDataset:
    def __init__(self, ds: List[str], ratio=0.9, seed=1234) -> None:
        self.ratio = ratio
        random.seed(seed)
        index = list(range(len(ds)))
        random.shuffle(index)
        split = round(ratio * len(index))
        self._train = list(map(ds.__getitem__, index[:split]))
        self._test = list(map(ds.__getitem__, index[split:]))
        self.index = np.array(index)
        self.full = ds

    @property
    def train(self) -> List[str]:
        return self._train

    @property
    def test(self) -> List[str]:
        return self._test

    def get(self, item: int) -> str:
        return self.full.__getitem__(item)

    def __repr__(self):
        T = "{}(x: {}, y: {}, full: {}, r: {})"
        c = self.__class__.__name__
        return T.format(c, len(self.train), len(self.test), len(self.full), self.ratio)


def extract_texts_from_page(url: str, text_tag="p", features="lxml"):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        reqs = Request(url, headers=headers)
        page = urlopen(reqs)
        soup = BeautifulSoup(page, features=features)
        texts = list(map(lambda p: p.text, soup.find_all(text_tag)))
    except URLError as e:
        print(f"Url {url} raised an exception:\n\t{e}")
        return []
    else:
        return texts


def website_to_dataset(nlp, url, min_words=5, ratio=0.9, seed=1234) -> WebDataset:
    # min_words: Minimum number of words per sentence
    dataset = []
    texts = extract_texts_from_page(url)
    for doc in nlp.pipe(texts, cleanup=True):
        sents = []
        for sent in doc.sents:
            if len(sent) > min_words:
                sents.append(sent.text.strip())
        dataset.extend(sents)
    return WebDataset(dataset, ratio=ratio, seed=seed)


class Dictionary:
    oov_token = "<oov>"
    oov_token_id = 0

    def __init__(
        self,
        texts: List[str],
        min_count=1,
        preserve_case=False,
        reduce_len=True,
        strip_handles=True,
    ) -> None:
        self.min_count = min_count
        self.tokenizer = TweetTokenizer(preserve_case, reduce_len, strip_handles)
        c = Counter(self.flatten(map(self.tokenize, texts)))
        freqs = dict(filter(lambda k: k[1] > min_count, c.most_common()))
        vocab = {self.oov_token: self.oov_token_id}
        vocab.update(freqs)
        stoi = {w: id for id, w in enumerate(vocab)}
        m = sum(vocab.values())
        self.probs = {stoi[w]: c / m for w, c in vocab.items()}
        self.itos = list(stoi)
        self.stoi = stoi
        self.freqs = Counter(freqs)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) - 1

    @staticmethod
    def flatten(docs: list) -> Iterable:
        return (t for s in docs for t in s)

    def tokenize(self, seq: str) -> List[str]:
        return self.tokenizer.tokenize(seq)

    def decode(
        self, x, return_tokens=True, remove_special_tokens=False,
    ) -> Union[List[str], str]:
        token_ids = x if isinstance(x, list) else x.squeeze(0).tolist()
        tokens = [self.itos[id] for id in token_ids]
        oov = self.oov_token
        if remove_special_tokens and tokens.count(oov) > 0:
            while oov in tokens:
                tokens.pop(tokens.index(oov))
        if not return_tokens:
            string = "".join(tokens).strip()
            return string
        return tokens

    def encode(self, x, return_tensors="list") -> Union[List[int], np.ndarray]:
        tokens = self.tokenize(x) if isinstance(x, str) else x
        token_ids = [self.stoi.get(t, self.oov_token_id) for t in tokens]
        if return_tensors == "np":
            return np.array([token_ids])
        return token_ids

    def fit(self, texts: List[str], dtype=np.float):
        # - Scheme (2):
        #  - log(1+f_t,d)  : Document term weight.
        #  - log(1+N/n_t) : Query term weight.
        N = len(texts)
        features = np.zeros((N, self.vocab_size), dtype=dtype)
        max_length = 0
        for i in range(N):
            doc = self.tokenize(texts[i])
            max_length = max(max_length, len(doc))
            for j, w in enumerate(doc):
                # D: Log normalization.
                tf = 1 + math.log(doc.count(w))
                # Q: Inverse document frequency smooth.
                idf = np.log(1 + N / (1 + self.freqs.get(w, 0)))
                features[i][j] = tf * idf
        return {"tfidf": features, "max_length": max_length}

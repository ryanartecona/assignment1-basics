from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain, groupby, islice, pairwise
import json
from operator import itemgetter
from pathlib import Path
from posixpath import dirname
from typing import Iterable
import regex as re


@dataclass
class NaiveTokenizer:
    def encode_iterable(self, str_iter: Iterable[str]) -> list[int]:
        return [ord(c) for s in str_iter for c in s]

    def encode(self, text: str) -> list[int]:
        return self.encode_iterable(text)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(tok) for tok in tokens)


@dataclass
class BPE:
    PRETOKEN_PAT: str = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def train(
        self, text: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # count occurrences of adjacent byte pairs in pretokens
        docs = re.split("|".join(map(re.escape, special_tokens)), text)
        pretoken_occs = Counter()
        # bytes_occurring = set()
        for doc in docs:
            for m in re.finditer(self.PRETOKEN_PAT, doc):
                match_bytes = m.group().encode("utf-8")
                pretoken_occs[tuple(bytes([byte]) for byte in match_bytes)] += 1
                # for byte in match_bytes:
                #     bytes_occurring.add(bytes([byte]))
        
        # init the vocabulary
        init_vocab = list(bytes([byte]) for byte in range(256))
        init_vocab += [token.encode("utf-8") for token in special_tokens]
        enc = {i: tok for i, tok in enumerate(init_vocab)}
        dec = {v: k for k, v in enc.items()}
        next_i = len(enc)

        merges: list[tuple[bytes, bytes]] = []

        # find a new merge each iter until vocab full
        while len(enc) < vocab_size:
            pair_occs = Counter()
            pair_pretokens = defaultdict(lambda: defaultdict(list))
            for pretoken in pretoken_occs:
                for i, (t1, t2) in enumerate(pairwise(pretoken)):
                    pair_occs[t1, t2] += pretoken_occs[pretoken]
                    pair_pretokens[t1, t2][pretoken].append(i)

            # guard in case we exhaust all possible merges within vocab size
            if not pair_occs:
                break

            most_common_pairs = []
            for _, group in groupby(pair_occs.most_common(), key=itemgetter(1)):
                for pair, _ in group:
                    most_common_pairs.append(pair)
                break  # only look at 1st group, i.e. tied most common pairs

            most_common_pair = max(most_common_pairs)

            # record a vocab word for the new merged pair, and the merge itself
            t1, t2 = most_common_pair
            enc[next_i] = t1 + t2
            dec[t1 + t2] = next_i
            merges.append((t1, t2))
            pair_occs[most_common_pair] = 0
            next_i += 1

            # update the pretoken occurrences with the new merged tokens
            for pretoken_premerge, pair_indices in pair_pretokens[most_common_pair].items():
                pretoken_merged = list(pretoken_premerge)
                for pair_index in sorted(pair_indices, reverse=True):
                    pretoken_merged[pair_index : pair_index + 2] = [t1 + t2]
                    # update new pair occurrence counts for new pairs containing the now-merged token
                    # if pair_index > 0:
                    #     pair_occs[pretoken_premerge[pair_index - 1], t1 + t2] += pretoken_occs[pretoken_premerge]
                    # if pair_index + 2 < len(pretoken_premerge):
                    #     pair_occs[t1 + t2, pretoken_premerge[pair_index + 2]] += pretoken_occs[pretoken_premerge]
                pretoken_occs[tuple(pretoken_merged)] += pretoken_occs[pretoken_premerge]
                pretoken_occs[pretoken_premerge] = 0

        return enc, merges


if __name__ == "__main__":
    print('training on "corpus.en"...')
    enc, merges = BPE().train(
        (
            Path(dirname(__file__)) / ".." / "tests" / "fixtures" / "corpus.en"
        ).read_text(),
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    enc = {i: tok.decode("utf-8", errors="replace") for i, tok in enc.items()}
    merges = [(t1.decode("utf-8"), t2.decode("utf-8")) for t1, t2 in merges]
    json.dump(
        {"enc": enc, "merges": merges},
        open(Path(dirname(__file__)) / "test.json", "w"),
        indent=2,
    )

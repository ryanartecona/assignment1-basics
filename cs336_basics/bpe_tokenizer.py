from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain, groupby, islice, pairwise
import json
from multiprocessing import Pool
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

    def _count_doc(self, doc: str) -> Counter:
        doc_occs = Counter()
        for m in re.finditer(self.PRETOKEN_PAT, doc):
            match_bytes = m.group().encode("utf-8")
            doc_occs[tuple(bytes([byte]) for byte in match_bytes)] += 1
        return doc_occs

    def train(
        self, text: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # init the vocabulary
        init_vocab = list(bytes([byte]) for byte in range(256))
        init_vocab += [token.encode("utf-8") for token in special_tokens]
        enc = {i: tok for i, tok in enumerate(init_vocab)}
        dec = {v: k for k, v in enc.items()}
        next_i = len(enc)

        merges: list[tuple[bytes, bytes]] = []

        # count occurrences of pretokens themselves
        docs = re.split("|".join(map(re.escape, special_tokens)), text)
        pretoken_occs = Counter()
        with Pool() as pool:
            for doc_occs in pool.imap_unordered(self._count_doc, docs, chunksize=10):
                pretoken_occs += doc_occs

        # count occurrences of byte pairs within pretokens, and track the index
        # into each pretoken where each pair occurs, so we can efficiently
        # update counts after merging
        pair_occs = Counter()
        pair_pretokens = defaultdict(lambda: defaultdict(list))
        for pretoken in pretoken_occs:
            for i, (t1, t2) in enumerate(pairwise(pretoken)):
                pair_occs[t1, t2] += pretoken_occs[pretoken]
                pair_pretokens[t1, t2][pretoken].append(i)

        # find a new merge each iter until vocab full
        while len(enc) < vocab_size:

            # guard in case we exhaust all possible merges within vocab size
            if not pair_occs:
                break

            most_common_pairs = []
            for _, group in groupby(pair_occs.most_common(), key=itemgetter(1)):
                for pair, _ in group:
                    most_common_pairs.append(pair)
                break  # only look at 1st group, i.e. tied most common pairs

            # stable tiebreak
            most_common_pair = max(most_common_pairs)

            # record a vocab word for the new merged pair, and the merge itself
            t1, t2 = most_common_pair
            enc[next_i] = t1 + t2
            dec[t1 + t2] = next_i
            merges.append((t1, t2))
            pair_occs[most_common_pair] = 0
            next_i += 1

            # update the pretoken occurrences with the new merged tokens
            for pretoken_premerge, pair_indices in pair_pretokens[
                most_common_pair
            ].items():
                # as pretokens experience pair merges, we remove old pretoken
                # pair indices from pair_pretokens[t1,t2] but we don't delete
                # the empty index list
                if not pair_indices:
                    continue
                # inner loops below shadow these
                t1, t2 = most_common_pair

                # create a new pretoken with the newly merged pair
                pretoken_merged = list(pretoken_premerge)
                for i, pair_index in enumerate(sorted(pair_indices)):
                    merged_index = (
                        pair_index - i
                    )  # account for previously merged pairs shifting indices
                    pretoken_merged[merged_index : merged_index + 2] = [t1 + t2]
                pretoken_merged = tuple(pretoken_merged)

                # count occurrences of pairs in the old (premerge) pretoken and
                # remove each corresponding index from pair_pretokens
                premerge_pair_occs = Counter()
                for i, (t1, t2) in enumerate(pairwise(pretoken_premerge)):
                    premerge_pair_occs[t1, t2] += pretoken_occs[pretoken_premerge]
                    pair_pretokens[t1, t2][pretoken_premerge].remove(i)
                # count occurrences of pairs in the new (merged) pretoken and
                # add each corresponding index to pair_pretokens
                merged_pair_occs = Counter()
                for i, (t1, t2) in enumerate(pairwise(pretoken_merged)):
                    merged_pair_occs[t1, t2] += pretoken_occs[pretoken_premerge]
                    pair_pretokens[t1, t2][pretoken_merged].append(i)

                # update pair occurrence counts by subtracting old counts and adding new counts
                pair_occs -= premerge_pair_occs
                pair_occs += merged_pair_occs
                # update pretoken occurrence counts as if the byte pair was merged in place
                pretoken_occs[pretoken_merged] += pretoken_occs[pretoken_premerge]
                pretoken_occs[pretoken_premerge] = 0

        return enc, merges


if __name__ == "__main__":
    name = "TinyStoriesV2-GPT4-train"
    # name = "TinyStoriesV2-GPT4-valid"
    corpus_path = (
        Path(dirname(__file__)) / ".." / "data" / f"{name}.txt"
    )
    dump_path = Path(dirname(__file__)) / f"{name}-bpe.json"
    print(f"reading file {corpus_path.absolute().relative_to(Path.cwd())} ...")
    corpus_text = corpus_path.read_text()
    print("training BPE tokenizer...")
    enc, merges = BPE().train(
        corpus_text,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    enc = {i: tok.decode("utf-8", errors="replace") for i, tok in enc.items()}
    merges = [
        (t1.decode("utf-8", errors="replace"), t2.decode("utf-8", errors="replace"))
        for t1, t2 in merges
    ]
    json.dump(
        {"enc": enc, "merges": merges},
        open(dump_path, "w"),
        indent=2,
    )
    print(f"dumped json to {dump_path.relative_to(Path.cwd())}")

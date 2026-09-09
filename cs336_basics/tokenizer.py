import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path
from typing import ClassVar, Optional, TypeVar, cast

import regex as re
from line_profiler import profile
from sortedcontainers import SortedSet

PRETOKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass
class BPECodec:
    enc: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    @property
    def vocab(self):
        return self.enc

    @classmethod
    def from_json(cls, obj):
        enc = {
            int(i): tok.encode("utf-8", errors="replace") for i, tok in obj["enc"].items()
        }
        merges = [
            (t1.encode("utf-8", errors="replace"), t2.encode("utf-8", errors="replace"))
            for t1, t2 in obj["merges"]
        ]
        return cls(enc, merges)

    def to_json(self):
        enc_obj = {
            i: tok.decode("utf-8", errors="replace") for i, tok in self.enc.items()
        }
        merges_obj = [
            (t1.decode("utf-8", errors="replace"), t2.decode("utf-8", errors="replace"))
            for t1, t2 in self.merges
        ]
        return {"enc": enc_obj, "merges": merges_obj}

    @property
    def vocab_size(self):
        return len(self.enc)


# eq=False so the class is hashable (by identity) and supports @lru_cache method
@dataclass(eq=False)
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    # derived fields
    enc: dict[int, bytes] = field(init=False)
    dec: dict[bytes, int] = field(init=False)

    PRETOKEN_PAT: ClassVar[str] = PRETOKEN_PAT

    def __post_init__(self):
        next_vocab_index = max(self.vocab.keys())
        # in case some special tokens are substrings of other special tokens,
        # sort by length descending so we match the longest one first
        super().__setattr__(
            "special_tokens",
            sorted(self.special_tokens or [], key=lambda s: (len(s), s), reverse=True),
        )
        special_vocab = {
            i + next_vocab_index: tok.encode("utf-8")
            for i, tok in enumerate(self.special_tokens or [])
        }
        super().__setattr__("dec", self.vocab | special_vocab)
        super().__setattr__("enc", {b: i for i, b in self.vocab.items()})

    @classmethod
    def from_file(
        cls, codec_filepath: str | Path, special_tokens: Optional[list[str]] = None
    ):
        codec_filepath = Path(codec_filepath)
        with open(codec_filepath, "r") as f:
            codec_json = json.load(f)
        codec = BPECodec.from_json(codec_json)
        return cls(codec.vocab, codec.merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @lru_cache(maxsize=2**16)
    def _merge_pretoken(self, pretoken: bytes) -> list[bytes]:
        # b'foo' -> [b'f', b'o', b'o']
        pretoken = [bytes([byte]) for byte in pretoken]
        # set([(b'f', b'o'), (b'o', b'o')])
        present_pairs = set((t1, t2) for t1, t2 in pairwise(pretoken))
        # apply merges to each pretoken
        for m1, m2 in self.merges:
            # short circuit for irrelevant merge
            if (m1, m2) not in present_pairs:
                continue
            # apply this merge
            i = 0
            while i < len(pretoken) - 1:
                t1, t2 = pretoken[i], pretoken[i + 1]
                if (t1, t2) == (m1, m2):
                    pretoken[i : i + 2] = [t1 + t2]
                i += 1
            # update present_pairs
            present_pairs = set((t1, t2) for t1, t2 in pairwise(pretoken))
        return pretoken

    def encode_iterable(self, chunks: Iterable[str]) -> Iterator[int]:
        leftover_part = None
        for chunk in chunks:
            # splitting on an empty pattern splits character-wise, so avoid
            if leftover_part:
                chunk = leftover_part.decode("utf-8") + chunk
                leftover_part = None
            if self.special_tokens:
                docs = re.split(
                    "(" + "|".join(map(re.escape, self.special_tokens)) + ")", chunk
                )
            else:
                docs = [chunk]
            for i, doc in enumerate(docs):
                if doc in (self.special_tokens or []):
                    yield self.enc[doc.encode("utf-8")]
                    continue
                # split each doc into pretokens
                pretokens = [
                    m.group().encode("utf-8")
                    for m in re.finditer(self.PRETOKEN_PAT, doc)
                ]
                if i == len(docs) - 1:
                    # for the last doc in docs, treat last pretoken as leftover
                    # for next chunk
                    leftover_part = pretokens[-1] if pretokens else None
                    pretokens = pretokens[:-1]
                # then apply merges to each pretoken
                for pretoken in pretokens:
                    pretoken = self._merge_pretoken(pretoken)
                    for tok in pretoken:
                        yield self.enc[tok]
        if leftover_part:
            pretoken = self._merge_pretoken(leftover_part)
            for tok in pretoken:
                yield self.enc[tok]

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.dec[tok] for tok in tokens).decode(
            "utf-8", errors="replace"
        )

T = TypeVar('T')
    
class TopCounter[T]:
    '''A counter that keeps track of the top items by count. It supports adding 
    and removing items, and retrieving the top item.

    >>> tc = TopCounter[str]()
    >>> tc['apple'] = 3
    >>> tc['banana'] = 5
    >>> tc.top()
    (5, 'banana')
    >>> tc['apple'] += 2
    >>> tc.top()
    (5, 'banana')
    >>> tc['banana'] -= 5
    >>> tc.top()
    (5, 'apple')
    >>> bool(tc)
    True
    >>> len(tc)
    1
    >>> tc['apple'] -= 5
    >>> bool(tc)
    False
    '''

    def __init__(self):
        self._counts = Counter()
        self.topitems = SortedSet()

    def __getitem__(self, item: T):
        return self.counts[item]

    def __setitem__(self, item: T, count: int):
        old_count = self.counts[item]
        if (old_count, item) in self.topitems:
            self.topitems.remove((old_count, item))
        if count > 0:
            self.topitems.add((count, item))
            self.counts[item] = count
        else:
            del self.counts[item]

    def __iadd__(self, item: T, amount: int):
        new_count = self.counts[item] + amount
        assert new_count > 0, f"Cannot add negative amount ({amount}) to item with current count ({self.counts[item]})"
        self.__setitem__(item, self.counts[item] + amount)
        return self

    def __isub__(self, item: T, amount: int):
        new_count = self.counts[item] - amount
        assert new_count >= 0, f"Cannot subtract more than current count ({self.counts[item]}) amount ({amount})"
        self.__setitem__(item, new_count)
        return self

    def __len__(self):
        return len(self.counts)

    def __bool__(self):
        return bool(self.counts)

    def top(self) -> tuple[int, T]:
        '''Return the top item and its count. Raises ValueError if the
        TopCounter is empty. In the case of ties, returns the item corresponding
        to max(tie_items) as a stable tiebreak.
        '''
        if not self.topitems:
            raise ValueError("TopCounter is empty")
        return cast(tuple[int, T], self.topitems[-1])

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, new_counts):
        self._counts = new_counts
        self.topitems = SortedSet((count, item) for item, count in new_counts.items() if count > 0)

@dataclass
class BPE:
    PRETOKEN_PAT: ClassVar[str] = PRETOKEN_PAT

    def _count_doc(self, doc: str) -> Counter:
        doc_occs = Counter()
        for m in re.finditer(self.PRETOKEN_PAT, doc):
            match_bytes = m.group().encode("utf-8")
            doc_occs[tuple(bytes([byte]) for byte in match_bytes)] += 1
        return doc_occs

    @profile
    def train(self, text: str, vocab_size: int, special_tokens: list[str]) -> BPECodec:
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
        pair_occs = TopCounter[tuple[bytes, bytes]]()
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

            most_common_count, most_common_pair = pair_occs.top()

            # record a vocab word for the new merged pair, and the merge itself
            t1, t2 = most_common_pair
            enc[next_i] = t1 + t2
            dec[t1 + t2] = next_i
            merges.append((t1, t2))
            next_i += 1

            pair_occ_updates = pair_occs.counts
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
                    # account for previously merged pairs shifting indices
                    merged_index = pair_index - i
                    # merge the pair in place
                    pretoken_merged[merged_index : merged_index + 2] = [t1 + t2]
                pretoken_merged = tuple(pretoken_merged)

                # count occurrences of pairs in the old (premerge) pretoken and
                # remove each corresponding index from pair_pretokens
                for i, (t1, t2) in enumerate(pairwise(pretoken_premerge)):
                    pair_occ_updates[t1, t2] -= pretoken_occs[pretoken_premerge]
                    pair_pretokens[t1, t2][pretoken_premerge].remove(i)
                # count occurrences of pairs in the new (merged) pretoken and
                # add each corresponding index to pair_pretokens
                for i, (t1, t2) in enumerate(pairwise(pretoken_merged)):
                    # pair_occs[t1, t2] += pretoken_occs[pretoken_premerge]
                    pair_occ_updates[t1, t2] += pretoken_occs[pretoken_premerge]
                    pair_pretokens[t1, t2][pretoken_merged].append(i)

                # update pretoken occurrence counts as if the byte pair was merged in place
                pretoken_occs[pretoken_merged] += pretoken_occs[pretoken_premerge]
                pretoken_occs[pretoken_premerge] = 0

            # apply accumulated occurrence count changes from all pretoken merges
            pair_occs.counts = pair_occ_updates

        return BPECodec(enc, merges)

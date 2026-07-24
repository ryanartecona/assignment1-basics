from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby, pairwise
import json
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from posixpath import dirname
from typing import ClassVar, Iterable, Iterator, Optional
import regex as re

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
            i: tok.encode("utf-8", errors="replace") for i, tok in obj["enc"].items()
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
            for i, tok in enumerate(self.special_tokens)
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


@dataclass
class BPE:
    PRETOKEN_PAT: ClassVar[str] = PRETOKEN_PAT

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

        return BPECodec(enc, merges)


if __name__ == "__main__":
    # name = "TinyStoriesV2-GPT4-train"
    name = "TinyStoriesV2-GPT4-valid"
    corpus_path = Path(dirname(__file__)) / ".." / "data" / f"{name}.txt"
    dump_path = Path(dirname(__file__)) / f"{name}-bpe.json"

    # Train a BPECodec on the above corpus and save it
    # print(f"reading file {corpus_path.absolute().relative_to(Path.cwd())} ...")
    # corpus_text = corpus_path.read_text()
    # print("training BPE tokenizer...")
    # codec = BPE().train(
    #     corpus_text,
    #     vocab_size=10_000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # json.dump(
    #     codec.to_json(),
    #     open(dump_path, "w"),
    #     indent=2,
    # )
    # print(f"dumped json to {dump_path.relative_to(Path.cwd())}")

    # hacky roundtrip test of Tokenizer.from_file and BPECodec
    t = Tokenizer.from_file(dump_path)
    test = t.decode(t.encode("My name is Ryan!"))
    print("TEST roundtrip:", repr(test))

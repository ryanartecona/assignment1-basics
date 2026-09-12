import json

import click
import numpy as np

from pathlib import Path

import torch
from cs336_basics.model import AdamW, TransformerLM, cross_entropy_loss, get_batch, lr_schedule_cosine_annealing
from cs336_basics.tokenizer import BPE, Tokenizer


readable_file = click.Path(exists=True, readable=True, path_type=Path, dir_okay=False)
writable_file = click.Path(writable=True, path_type=Path, dir_okay=False)


@click.group()
def cli():
    """A simple CLI application."""
    pass


@cli.group()
def tokenizer():
    """Tokenizer related commands."""
    pass


@tokenizer.command(name="train")
@click.option("--vocab-size", default=10000, help="Vocabulary size for the tokenizer.")
@click.option("--corpus-path", default=None, help="Path to the training corpus.", type=readable_file)
@click.option("--output-path", default=None, help="Path to save the trained tokenizer.", type=writable_file)
@click.option(
    "--special-tokens", default=["<|endoftext|>"], help="Special tokens to include in the tokenizer.", multiple=True
)
def tokenizer_train(vocab_size, corpus_path, output_path, special_tokens):
    """Train a tokenizer."""
    click.echo(f"reading file {corpus_path.absolute().relative_to(Path.cwd())} ...")
    corpus_text = corpus_path.read_text()
    click.echo("training BPE tokenizer...")
    codec = BPE().train(
        corpus_text,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    if output_path.suffix == ".pkl":
        with open(output_path, "wb") as f:
            f.write(codec.to_pickle())
    else:
        json.dump(
            codec.to_json(),
            open(output_path, "w"),
            indent=2,
        )
    click.echo(f"saved tokenizer codec to {output_path}")


@tokenizer.command(name="test-roundtrip")
@click.option(
    "--codec-path",
    default=None,
    help="Path to the trained tokenizer codec.",
    type=readable_file,
)
@click.option("--test-string", default="Hello tokenizer!", help="String to test the tokenizer roundtrip.")
def tokenizer_test_roundtrip(codec_path: Path, test_string: str):
    # hacky roundtrip test of Tokenizer.from_file and BPECodec
    t = Tokenizer.from_file(codec_path)
    test = t.decode(t.encode(test_string))
    click.echo(f"test roundtrip output: {repr(test)}")


@tokenizer.command(name="encode-dataset")
@click.option("--codec-path", required=True, help="Path to the trained tokenizer codec.", type=readable_file)
@click.option("--corpus-path", required=True, help="Path to the training corpus.", type=readable_file)
@click.option(
    "--special-tokens", default=["<|endoftext|>"], help="Special tokens to include in the tokenizer.", multiple=True
)
@click.option("--output-path", required=True, help="Path to save the encoded dataset.", type=writable_file)
def tokenizer_encode_dataset(codec_path: Path, corpus_path: Path, special_tokens: list[str], output_path: Path):
    """Encode a dataset using the trained tokenizer."""
    click.echo(f"reading file {corpus_path.absolute().relative_to(Path.cwd())} ...")
    corpus_text = corpus_path.read_text()
    click.echo("loading tokenizer codec...")
    tokenizer = Tokenizer.from_file(codec_path, special_tokens=special_tokens)
    click.echo("encoding dataset...")
    encoded_dataset = tokenizer.encode(corpus_text)
    np.save(output_path, np.array(encoded_dataset, dtype=np.uint16))
    click.echo(f"saved encoded dataset to {output_path}")


@cli.command()
@click.option("--codec-path", required=True, help="Path to the trained tokenizer codec.", type=readable_file)
@click.option("--corpus-path", required=True, help="Path to the training corpus.", type=readable_file)
@click.option("--d-model", default=512, help="Dimension of the model.")
@click.option("--n-layers", default=6, help="Number of layers in the model.")
@click.option("--context-length", default=128, help="Context length for the model.")
@click.option("--d-ff", default=2048, help="Dimension of the feedforward network.")
@click.option("--n-heads", default=8, help="Number of attention heads in the feedforward network.")
@click.option("--rope-theta", default=10000, help="RoPE theta value.")
@click.option("--steps", default=100, help="Number of training steps.")
def train(codec_path, corpus_path, d_model, n_layers, context_length, d_ff, n_heads, rope_theta, steps):
    """Train a model with provided config."""
    click.echo("Loading tokenizer codec...")
    tokenizer = Tokenizer.from_file(codec_path)
    corpus_toks = np.memmap(dtype=np.uint16, filename=corpus_path, mode="r")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_device(device)
    click.echo("Initializing model and optimizer...")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        context_length=context_length,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        rope_theta=rope_theta,
    )
    optimizer = AdamW(model.parameters())
    click.echo("Model and optimizer initialized.")
    warmup_steps = steps // 10
    for i in range(steps):
        batch_x, batch_y = get_batch(corpus_toks, context_length=context_length, batch_size=32, device=str(device))
        logits = model.forward(batch_x)
        loss = cross_entropy_loss(logits, batch_y)
        loss.backward()
        lr = lr_schedule_cosine_annealing(i, lr_max=1e-2, lr_min=5e-4, warmup_period=warmup_steps, annealing_period=steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad()
        click.echo(f"Step {i+1}/10 completed. loss: {loss.mean().item():0.5f}, lr: {lr:0.3e}")


if __name__ == "__main__":
    cli()

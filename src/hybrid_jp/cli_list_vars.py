from pathlib import Path

import sdf_helper as sh
import typer
from rich import print

from .deck_parse import get_deck_constants
from .sdf_files import list_variables

hybrid_jp = typer.Typer()
list_vars_app = typer.Typer()
deck_app = typer.Typer()
hybrid_jp.add_typer(list_vars_app, name="sdf-vars")
hybrid_jp.add_typer(deck_app, name="deck")


@list_vars_app.command("read")
def list_vars_in_sdf_file(path_to_file: Path):
    if path_to_file.suffix != ".sdf":
        print("File is not an sdf")
        raise typer.Abort()
    print(f"Reading {path_to_file}...")
    data = sh.getdata(str(path_to_file), verbose=False)
    print("Variables:")
    vars = list_variables(data, show_type=False)
    max_name_len = max([len(v[0]) for v in vars])
    for v in vars:
        name = f"[green]{v[0]:<{max_name_len}}[/green]"
        print(f"{name} {v[1]}")


@deck_app.command("eval-consts")
def deck_eval(path_to_input_deck: Path):
    if not path_to_input_deck.exists():
        print(f"{path_to_input_deck} file does not exist")
        raise typer.Abort()
    if not path_to_input_deck.suffix == ".deck":
        print(f"{path_to_input_deck} is not a .deck file")

    consts = get_deck_constants(path_to_input_deck)

    print("[green]begin: constant[/green]")
    longest_name = max([len(i) for i in consts])
    for k, v in consts.items():
        k_str = f"[green]{k:>{longest_name}}[/green]"
        v_str = f"[blue]{v: .3E}[/blue]".replace("E", "e")
        print(f"  {k_str}  = {v_str}")
    print("[green]end: constant[/green]")

from pathlib import Path

import sdf_helper as sh
import typer
from rich import print

from .sdf_files import list_variables

list_vars_app = typer.Typer()


@list_vars_app.command()
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

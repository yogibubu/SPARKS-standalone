"""Dispatch SPARKS scientific commands without importing optional backends."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("photoelectron", "response"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "photoelectron":
        from matrix_sparks.photoelectron_cli import main as command
    else:
        from matrix_sparks.response_cli import main as command
    command(args.arguments)


if __name__ == "__main__":
    main()

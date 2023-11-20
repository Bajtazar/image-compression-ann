#!/usr/bin/env python3
from configparser import ConfigParser
from argparse import ArgumentParser, Namespace
from sys import argv


CONFIG_PATH: str = "config.ini"


def build_arg_parser(config: ConfigParser, args: list[str]) -> Namespace:
    parser = ArgumentParser()
    for key, section in config.items():
        for name, default in section.items():
            parser.add_argument(f"--{key}.{name}", default=default)
    return parser.parse_args(args)


def actualize_config(config: ConfigParser, args: Namespace) -> ConfigParser:
    dictified_args = args.__dict__
    for key, section in config.items():
        for name in section.keys():
            config[key][name] = dictified_args[f"{key}.{name}"]
    return config


def main(args: list[str]) -> None:
    config = ConfigParser()
    config.read(CONFIG_PATH)
    parsed = build_arg_parser(config, args[1:])
    config = actualize_config(config, parsed)
    with open(CONFIG_PATH, "w") as handle:
        config.write(handle, space_around_delimiters=False)


if __name__ == "__main__":
    main(argv)

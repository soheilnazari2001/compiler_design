import click

from jinja2 import Environment


class Filters:
    @staticmethod
    def to_set(value):
        return "{" + ", ".join(f'"{item}"' for item in value) + "}"


def get_context():
    return {
        "start": "program",
        "nonterminals": {
            "program": {
                "node_name": "program",
                "goes_to_epsilon": True,
                "follows": ("d", "e"),
                "derivations": {
                    ("a", "b"): [{"is_terminal": True, "value": "NUM"}],
                    ("b", "c"): [{"is_terminal": False, "value": "program"}],
                },
            },
            "declaration_list": {
                "node_name": "Declaration-list",
                "goes_to_epsilon": False,
                "follows": ("a", "b"),
                "derivations": {
                    ("b", "c"): [{"is_terminal": False, "value": "declaration_list"}],
                    ("a", "b"): [{"is_terminal": True, "value": "NUM"}],
                },
            }
        },
    }


@click.command()
@click.option(
    "--out",
    "-o",
    default="parser.py",
    show_default=True,
    type=click.File("w"),
    help="Path of the parser file to generate.",
)
@click.option(
    "--template",
    "-t",
    default="templates/parser.py.j2",
    show_default=True,
    type=click.File("r"),
    help="Path of the template file to generate parser from.",
)
def generate_parser(out, template):
    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["to_set"] = Filters.to_set
    template = environment.from_string(template.read())
    context = get_context()
    parser = template.render(context)
    out.write(parser)


if __name__ == "__main__":
    generate_parser()

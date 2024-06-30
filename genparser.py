import click
from jinja2 import Environment

from grammar import FOLLOW, PREDICTIVE_SET, StatementType


class Filters:
    @classmethod
    def add_filters(cls, environment, *filters):
        for filter in filters:
            environment.filters[filter] = getattr(cls, filter)

    @staticmethod
    def to_set(value):
        return "{" + ", ".join(f'"{item}"' for item in value) + "}"


class Globals:
    @classmethod
    def add_globals(cls, environment):
        environment.globals["StatementType"] = StatementType


def get_context():
    for name, nonterminal in PREDICTIVE_SET.items():
        nonterminal.follows = FOLLOW[name]

    return {"start": "program", "nonterminals": PREDICTIVE_SET}


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
    Filters.add_filters(environment, "to_set")
    Globals.add_globals(environment)
    template = environment.from_string(template.read())
    context = get_context()
    parser = template.render(context)
    out.write(parser)


if __name__ == "__main__":
    generate_parser()

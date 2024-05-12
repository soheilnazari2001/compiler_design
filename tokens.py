from enum import Enum, auto


class TokenType(Enum):
    NUM = auto()
    ID = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    START = auto()
    PANIC = auto()
    EOF = auto()


class Token:
    def __init__(self, token_type, token_value=""):
        self.token_type = token_type
        self.token_value = token_value

    def __repr__(self):
        return f"({self.token_type}, {self.token_value})"

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
    NAME_TYPES = {TokenType.NUM.name, TokenType.ID.name, TokenType.EOF.name}
    VALUE_TYPES = {TokenType.KEYWORD.name, TokenType.SYMBOL.name}

    def __init__(self, token_type, token_value=""):
        self.token_type = token_type
        self.token_value = token_value

    def __repr__(self):
        return f"({self.token_type}, {self.token_value})"

    @property
    def is_name_type(self):
        return self.token_type in self.NAME_TYPES

    @property
    def is_value_type(self):
        return self.token_type in self.VALUE_TYPES

    @property
    def terminal(self):
        if self.is_name_type:
            return self.token_type
        if self.is_value_type:
            return self.token_value

        raise NotImplementedError(
            f"terminal not implemented for type {self.token_type}"
        )

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
    EOF = "$"


class Token:
    NAME_TYPES = {TokenType.NUM.name, TokenType.ID.name, TokenType.EOF.name}
    VALUE_TYPES = {TokenType.KEYWORD.name, TokenType.SYMBOL.name}

    def __init__(self, type, lexeme=""):
        self.type = type
        self.lexeme = lexeme

    def __repr__(self):
        return f"({self.type}, {self.lexeme})"

    @property
    def is_name_type(self):
        return self.type in self.NAME_TYPES

    @property
    def is_value_type(self):
        return self.type in self.VALUE_TYPES

    @property
    def terminal(self):
        if self.is_name_type:
            return self.type
        if self.is_value_type:
            return self.lexeme

        raise NotImplementedError(f"terminal not implemented for type {self.type}")

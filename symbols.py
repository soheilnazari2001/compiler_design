from enum import Enum


class SymbolType(Enum):
    INT = "int"
    VOID = "void"
    ARRAY = "array"


class Symbol:
    def __init__(
        self, /, *, address=None, lexeme=None, type=None, size=0, param_count=0
    ):
        self.address = address
        self.lexeme = lexeme
        self.type = type
        self.size = size
        self.param_count = param_count
        self.param_symbols = []
        self.is_initialized = False
        self.is_function = False
        self.is_array = False

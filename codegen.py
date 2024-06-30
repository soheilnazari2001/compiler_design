from parser import Parser


def indirect_address(address):
    if address.startswith("#"):
        return address[1:]

    return f"@{address}"


class Instruction:
    ARGS_COUNT = 3

    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args + [""] * (self.ARGS_COUNT - len(args))

    def __repr__(self):
        return f"{tuple(self.opcode, *self.args)}"

    @classmethod
    def add(cls, in1, in2, out):
        return cls("ADD", in1, in2, out)

    @classmethod
    def mult(cls, in1, in2, out):
        return cls("MULT", in1, in2, out)

    @classmethod
    def sub(cls, in1, in2, out):
        return cls("SUB", in1, in2, out)

    @classmethod
    def eq(cls, in1, in2, out):
        return cls("EQ", in1, in2, out)

    @classmethod
    def lt(cls, in1, in2, out):
        return cls("LT", in1, in2, out)

    @classmethod
    def assign(cls, left, right):
        return cls("ASSIGN", left, right)

    @classmethod
    def jpf(cls, condition, address):
        address = indirect_address(address)
        return cls("JPF", condition, address)

    @classmethod
    def jp(cls, address):
        address = indirect_address(address)
        return cls("JP", address)

    @classmethod
    def print(cls, value):
        return cls("PRINT", value)


class CodeGenerator:
    WORD_SIZE = 4

    def __init__(self, parser: Parser):
        self.parser = parser
        self.semantic_stack = []
        self.instructions: list[Instruction] = []
        self.data_address = 20000
        self.temp_address = 60000
        self.return_address_address = self.get_next_data_address()
        self.return_value_address = self.get_next_data_address()
        self.stack_pointer_address = self.get_next_data_address()
    
    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def get_next_data_address(self, size=WORD_SIZE):
        next_data_address = self.data_address
        self.data_address += size
        return next_data_address

    def action_pnext(self):
        self.semantic_stack.append(self.parser.lookahead.token_value)

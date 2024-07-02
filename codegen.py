from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser import Parser

from symbols import Symbol, SymbolType
from tokens import Token, TokenType


def indirect_address(address):
    if address.startswith("#"):
        return address[1:]

    return f"@{address}"


class CodeGenerator:
    WORD_SIZE = 4

    class Instruction:
        ARGS_COUNT = 3

        def __init__(self, opcode, *args):
            self.opcode = opcode
            self.args = args + ("",) * (self.ARGS_COUNT - len(args))

        def __repr__(self):
            return f"({self.opcode}, {', '.join(str(arg) for arg in self.args)})"

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

    INSTRUCTION_FROM_OPERATION = {
        "+": Instruction.add,
        "-": Instruction.sub,
        "<": Instruction.lt,
        "==": Instruction.eq,
        "*": Instruction.mult,
    }

    def __init__(self, parser: "Parser"):
        self.parser = parser
        self.scopes = [[]]
        self.argument_counts = []
        self.current_declared_function_symbol = None
        self.current_id = None
        self.is_rhs = False
        self.current_type = None
        self.called_functions = []
        self.no_push_flag = False
        self.check_declaration_flag = False
        self.function_scope_flag = False
        self.breaks = []
        self.has_reached_main = False
        self.force_declaration_flag = False
        self.current_id = ""
        self.void_flag = False
        self.void_line_number = None
        self.found_arg_type_mismtach = []
        self.initialized_temp_addresses: set[str] = set()
        self.pushed_temp_addresses_stack: list[set[str]] = []
        self.semantic_stack = []
        self.execution_flow_stack = []
        self.instruction_index = 0
        self.instructions: dict[int, self.Instruction] = {}
        self.semantic_errors_list: list[str] = []
        self.data_address = 100000
        self.temp_address = 500004
        self.stack_start_address = self.temp_address - self.WORD_SIZE
        self.return_address_address = self.get_next_data_address()
        self.return_value_address = self.get_next_data_address()
        self.stack_pointer_address = self.get_next_data_address()
        self.add_instructions(
            self.Instruction.assign(
                f"#{self.stack_start_address}", f"{self.stack_pointer_address}"
            ),
            self.Instruction.assign("#0", f"{self.return_address_address}"),
            self.Instruction.assign("#0", f"{self.return_value_address}"),
        )
        self.jump_to_main_address = len(self.instructions)
        self.instruction_index += 1
        self.function_data_start_pointer = None
        self.function_temp_start_pointer = None
        self.generate_implicit_output()

    @property
    def generated_code(self):
        return (
            "\n".join(
                f"{index}\t{repr(self.instructions[index])}"
                for index in sorted(self.instructions)
            )
            if self.instructions and not self.semantic_errors_list
            else "The code has not been generated."
        )

    @property
    def semantic_errors(self):
        return (
            "\n".join(self.semantic_errors_list)
            if self.semantic_errors_list
            else "The input program is semantically correct."
        )

    def add_symbol(self, lexeme, address):
        symbol = Symbol(lexeme=lexeme, address=address)
        self.scopes[-1].append(symbol)
        return symbol

    def remove_symbol(self, lexeme):
        scope = self.scopes[-1]
        for i, symbol in enumerate(scope):
            if symbol.lexeme == lexeme:
                scope.pop(i)
                return

    def find_address_by_lexeme(
        self, lexeme, check_declaration=False, force_declaration=False
    ):
        symbol = self.find_symbol_by_lexeme(
            lexeme, check_declaration, force_declaration
        )
        return symbol.address if symbol else None

    def find_symbol_by_lexeme(
        self,
        lexeme,
        check_declaration=False,
        force_declaration=False,
        prevent_add=False,
    ):
        if not force_declaration:
            for scope in reversed(self.scopes):
                for symbol in scope:
                    if symbol.lexeme == lexeme:
                        return symbol

        if check_declaration:
            self.raise_undefined_semantic_error(lexeme)
            return None

        if not prevent_add:
            address = self.get_next_data_address()
            return self.add_symbol(lexeme, address)

        return None

    def find_symbol_by_address(self, address):
        for scope in reversed(self.scopes):
            for symbol in scope:
                if symbol.address == address:
                    return symbol
        return None

    def push_to_stack(self, data):
        self.add_instructions(
            self.Instruction.sub(
                self.stack_pointer_address,
                f"#{self.WORD_SIZE}",
                self.stack_pointer_address,
            ),
            self.Instruction.assign(data, f"@{self.stack_pointer_address}"),
        )

    def pop_from_stack(self, address):
        self.add_instructions(
            self.Instruction.assign(f"@{self.stack_pointer_address}", address),
            self.Instruction.add(
                self.stack_pointer_address,
                f"#{self.WORD_SIZE}",
                self.stack_pointer_address,
            ),
        )

    def raise_arg_type_mismatch_error(self, lexeme, index, expected, got):
        if not self.found_arg_type_mismtach or not self.found_arg_type_mismtach[-1]:
            if len(self.found_arg_type_mismtach) == 0:
                self.found_arg_type_mismtach.append(True)
            self.found_arg_type_mismtach[-1] = True
            self.raise_arg_type_mismatch_semantic_error(index, lexeme, expected, got)

    def pid(self, previous_token, current_token):
        self.current_id = previous_token.lexeme
        address = self.find_address_by_lexeme(
            previous_token.lexeme,
            self.check_declaration_flag,
            self.force_declaration_flag,
        )
        self.handle_main_function(previous_token)
        if not self.no_push_flag:
            self.semantic_stack.append(address)
        self.handle_operand_mismatch(current_token)
        self.handle_arg_mismatch(current_token, previous_token)

    def handle_main_function(self, previous_token):
        if previous_token.lexeme == "main":
            self.add_instruction(
                self.Instruction.jp(f"#{self.instruction_index}"),
                self.jump_to_main_address,
            )
            if not self.has_reached_main:
                for symbol in self.scopes[0]:
                    if not symbol.is_function:
                        self.add_instruction(
                            self.Instruction.assign("#0", symbol.address)
                        )
            self.has_reached_main = True

    def handle_operand_mismatch(self, current_token):
        if self.is_rhs:
            symbol = self.find_symbol_by_lexeme(self.current_id, prevent_add=True)
            if symbol.is_function:
                if symbol.type != SymbolType.INT.value:
                    self.raise_operand_type_mismatch_semantic_error(
                        SymbolType.INT.value, SymbolType.VOID.value
                    )
            else:
                if symbol.is_array:
                    if current_token.lexeme != "[" and not self.argument_counts:
                        if current_token.lexeme in self.INSTRUCTION_FROM_OPERATION:
                            self.raise_operand_type_mismatch_semantic_error(
                                SymbolType.ARRAY.value, SymbolType.INT.value
                            )
                        else:
                            self.raise_operand_type_mismatch_semantic_error(
                                SymbolType.INT.value, SymbolType.ARRAY.value
                            )

    def handle_arg_mismatch(self, current_token, previous_token):
        if len(self.argument_counts) > 0:
            index = self.argument_counts[-1]
            symbol: Symbol = self.find_symbol_by_lexeme(
                self.called_functions[-1], prevent_add=True
            )
            param_symbol: Symbol = symbol.param_symbols[index]
            current_symbol: Symbol = self.find_symbol_by_lexeme(
                previous_token.lexeme, prevent_add=True
            )
            if param_symbol.type == SymbolType.INT.value:
                if (
                    current_symbol.type == SymbolType.ARRAY.value
                    and current_token.lexeme != "["
                ):
                    self.raise_arg_type_mismatch_error(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.INT.value,
                        SymbolType.ARRAY.value,
                    )
                if current_symbol.type == SymbolType.VOID.value:
                    self.raise_arg_type_mismatch_error(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.INT.value,
                        SymbolType.VOID.value,
                    )
            if param_symbol.type == SymbolType.ARRAY.value:
                if current_symbol.type == SymbolType.INT.value:
                    self.raise_arg_type_mismatch_error(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.INT.value,
                    )
                if (
                    current_symbol.type == SymbolType.ARRAY.value
                    and current_token.lexeme == "["
                ):
                    self.raise_arg_type_mismatch_error(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.INT.value,
                    )
                if current_symbol.type == SymbolType.VOID.value:
                    self.raise_arg_type_mismatch_error(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.VOID.value,
                    )

    def pnum(self, previous_token, _):
        num = f"#{previous_token.lexeme}"
        if not self.no_push_flag:
            self.semantic_stack.append(num)
        if len(self.argument_counts) > 0:
            index = self.argument_counts[-1]
            symbol: Symbol = self.find_symbol_by_lexeme(
                self.called_functions[-1], prevent_add=True
            )
            param_symbol: Symbol = symbol.param_symbols[index]
            if param_symbol.type == SymbolType.ARRAY.value:
                self.raise_arg_type_mismatch_error(
                    index + 1,
                    symbol.lexeme,
                    SymbolType.ARRAY.value,
                    SymbolType.INT.value,
                )

    def label(self, *_):
        self.semantic_stack.append(f"#{self.instruction_index}")

    def save(self, *_):
        self.semantic_stack.append(f"#{self.instruction_index}")
        self.instruction_index += 1

    def push_operation(self, previous_token, _):
        self.semantic_stack.append(previous_token.lexeme)

    def execute(self, *_):
        temp_address = self.get_next_temp_address()
        operand2 = self.semantic_stack.pop()
        operation = self.semantic_stack.pop()
        operand1 = self.semantic_stack.pop()
        self.semantic_stack.append(temp_address)
        # self.initialized_temp_addresses.add(temp_address)
        instruction = self.INSTRUCTION_FROM_OPERATION[operation](
            operand1, operand2, temp_address
        )
        self.add_instruction(instruction)

    def start_argument_list(self, *_):
        self.argument_counts.append(0)
        self.called_functions.append(self.current_id)
        self.found_arg_type_mismtach.append(False)

    def end_argument_list(self, *_):
        self.found_arg_type_mismtach.pop()

    def jp_from_saved(self, *_):
        instruction = self.Instruction.jp(f"#{self.instruction_index}")
        destination = self.semantic_stack.pop()
        self.add_instruction(instruction, destination)

    def jpf_from_saved(self, *_):
        destination = self.semantic_stack.pop()
        condition = self.semantic_stack.pop()
        instruction = self.Instruction.jpf(condition, f"#{self.instruction_index}")
        self.add_instruction(instruction, destination)

    def save_and_jpf_from_last_save(self, *_):
        destination = self.semantic_stack.pop()
        condition = self.semantic_stack.pop()
        instruction = self.Instruction.jpf(condition, f"#{self.instruction_index + 1}")
        self.add_instruction(instruction, destination)
        self.semantic_stack.append(f"#{self.instruction_index}")
        self.instruction_index += 1

    def assign(self, *_):
        value = self.semantic_stack.pop()
        address = self.semantic_stack.pop()
        instruction = self.Instruction.assign(value, address)
        self.add_instruction(instruction)
        self.semantic_stack.append(value)
        symbol: Symbol = self.find_symbol_by_address(address)
        if symbol:
            symbol.is_initialized = True
        else:
            self.initialized_temp_addresses.add(address)

    def start_no_push(self, *_):
        if not self.function_scope_flag:
            self.no_push_flag = True

    def end_no_push(self, *_):
        self.no_push_flag = False

    def declare_array(self, *_):
        length = int(self.semantic_stack.pop()[1:])
        symbol: Symbol = self.scopes[-1][-1]
        symbol.is_array = True
        symbol.type = SymbolType.ARRAY.value
        size = length * self.WORD_SIZE
        array_start_address = self.get_next_data_address(size=size)
        self.add_instruction(
            self.Instruction.assign(f"#{array_start_address}", symbol.address)
        )
        if len(self.scopes) > 1:
            for address in range(
                array_start_address, array_start_address + size, self.WORD_SIZE
            ):
                self.add_instruction(self.Instruction.assign("#0", address))

    def array(self, *_):
        offset = self.semantic_stack.pop()
        temp = self.get_next_temp_address()
        array_start = self.semantic_stack.pop()
        self.add_instructions(
            self.Instruction.mult(offset, f"#{self.WORD_SIZE}", temp),
            self.Instruction.add(temp, f"{array_start}", temp),
        )
        self.semantic_stack.append(f"@{temp}")

    def until(self, *_):
        condition = self.semantic_stack.pop()
        destination = self.semantic_stack.pop()
        instruction = self.Instruction.jpf(condition, destination)
        self.add_instruction(instruction)

    def start_break_scope(self, *_):
        self.breaks.append([])

    def add_break(self, *_):
        if not self.breaks:
            self.raise_break_semantic_error()
            return
        self.breaks[-1].append(self.instruction_index)
        self.instruction_index += 1

    def handle_breaks(self, *_):
        for destination in self.breaks[-1]:
            instruction = self.Instruction.jp(f"#{self.instruction_index}")
            # insert method
            self.add_instruction(instruction, destination)
        self.breaks.pop()

    def pop(self, *_):
        self.semantic_stack.pop()

    def check_declaration(self, *_):
        self.check_declaration_flag = True

    def uncheck_declaration(self, *_):
        self.check_declaration_flag = False

    def set_function_scope_flag(self, *_):
        self.function_scope_flag = True

    def open_scope(self, *_):
        if not self.function_scope_flag:
            self.scopes.append([])
        self.function_scope_flag = False
        self.execution_flow_stack.append((self.data_address, self.temp_address))

    def close_scope(self, *_):
        self.scopes.pop()
        self.data_address, self.temp_address = self.execution_flow_stack.pop()

    def pop_param(self, previous_token, _):
        address = self.semantic_stack.pop()
        self.pop_from_stack(address)
        symbol: Symbol = self.find_symbol_by_address(address)
        symbol.type = self.current_type
        if previous_token and previous_token.lexeme == "]":
            symbol.type = SymbolType.ARRAY.value
            symbol.is_array = True
        self.current_declared_function_symbol.param_symbols.append(symbol)
        if symbol:
            symbol.is_initialized = True
            self.current_declared_function_symbol.param_count += 1
        else:
            self.initialized_temp_addresses.add(address)

    def declare_function(self, *_):
        symbol: Symbol = self.scopes[-1][-1]
        symbol.address = f"#{self.instruction_index}"
        symbol.is_function = True
        symbol.type = self.current_type
        symbol.param_count = 0
        self.current_declared_function_symbol = symbol
        self.void_flag = False
        self.function_data_start_pointer = self.data_address
        self.function_temp_start_pointer = self.temp_address

    def call(self, *_):
        self.store_execution_flow()
        self.push_addresses()

        arg_count = self.argument_counts.pop()
        self.save_return_address(arg_count)

        self.make_call(arg_count)

        self.pop_addresses()
        self.restore_execution_flow()

        self.retrieve_return_value()

        function_name = self.called_functions.pop()
        symbol = self.find_symbol_by_lexeme(function_name)
        if symbol.param_count != arg_count:
            self.raise_arg_count_mismatch_semantic_error(function_name)

    def retrieve_return_value(self):
        temp = self.get_next_temp_address()
        self.semantic_stack.append(temp)
        self.add_instruction(self.Instruction.assign(self.return_value_address, temp))

    def restore_execution_flow(self):
        pushed_temp_addresses = self.pushed_temp_addresses_stack.pop()
        print(
            "at restore, temp_address =",
            self.temp_address,
            "and function_temp_start_pointer =",
            self.function_temp_start_pointer,
        )
        for address in range(
            self.temp_address,
            self.function_temp_start_pointer - self.WORD_SIZE,
            -self.WORD_SIZE,
        ):
            temp_address = address - self.WORD_SIZE
            if (
                pushed_temp_addresses is not None
                and temp_address in pushed_temp_addresses
            ):
                self.pop_from_stack(temp_address)
                print("restoring", address)
            else:
                # print("is", address, "not initialized? - at restore")
                # self.add_instruction(self.Instruction.print(address))
                # self.add_instruction(self.Instruction.print(address))
                pass
        for address in range(
            self.data_address,
            self.function_data_start_pointer,
            -self.WORD_SIZE,
        ):
            symbol: Symbol = self.find_symbol_by_address(address - self.WORD_SIZE)
            if symbol and symbol.is_initialized:
                self.pop_from_stack(address - self.WORD_SIZE)

    def make_call(self, arg_count):
        for _ in range(arg_count):
            data = self.semantic_stack.pop()
            self.push_to_stack(data)
        address = self.semantic_stack.pop()
        instruction = self.Instruction.jp(address)
        self.add_instruction(instruction)

    def store_execution_flow(self):
        for address in range(
            self.function_data_start_pointer,
            self.data_address,
            self.WORD_SIZE,
        ):
            symbol: Symbol = self.find_symbol_by_address(address)
            if symbol and symbol.is_initialized:
                self.push_to_stack(address)
        self.pushed_temp_addresses_stack.append(self.initialized_temp_addresses.copy())
        print(
            "at store, temp_address =",
            self.temp_address,
            "and function_temp_start_pointer =",
            self.function_temp_start_pointer,
        )
        for address in range(
            self.function_temp_start_pointer,
            self.temp_address + self.WORD_SIZE,
            self.WORD_SIZE,
        ):
            if address in self.pushed_temp_addresses_stack[-1]:
                self.push_to_stack(address)
                print("storing", address)
            else:
                # print("is", address, "not initialized? - at store")
                # self.add_instruction(self.Instruction.print(address))
                pass

    def set_return_value(self, *_):
        value = self.semantic_stack.pop()
        self.save_return_value(value)

    def jump_back(self, *_):
        if not self.has_reached_main:
            instruction = self.Instruction.jp(str(self.return_address_address))
            self.add_instruction(instruction)

    def add_argument_count(self, *_):
        self.found_arg_type_mismtach[-1] = False
        self.argument_counts[-1] += 1

    def zero_initialize(self, *_):
        if len(self.scopes) > 1:
            symbol: Symbol = self.scopes[-1][-1]
            if not symbol.is_array:
                symbol.type = SymbolType.INT.value
            self.add_instruction(self.Instruction.assign("#0", symbol.address))

    def array_param(self, *_):
        symbol: Symbol = self.scopes[-1][-1]
        symbol.is_array = True
        symbol.type = SymbolType.ARRAY.value

    def set_force_declaration_flag(self, *_):
        self.force_declaration_flag = True

    def unset_force_declaration_flag(self, *_):
        self.force_declaration_flag = False

    def void_check(self, *_):
        self.void_flag = True
        self.void_line_number = self.parser.scanner.reader.line_number

    def void_check_throw(self, *_):
        if self.void_flag:
            self.void_flag = False
            self.remove_symbol(self.current_id)
            self.raise_illegal_void_type_semantic_error(
                self.current_id, line_number=self.void_line_number
            )
            self.void_line_number = None

    def save_type(self, previous_token, _):
        self.current_type = previous_token.lexeme

    def start_rhs(self, *_):
        self.is_rhs = True

    def end_rhs(self, *_):
        self.is_rhs = False

    def negate(self, *_):
        value = self.semantic_stack.pop()
        temp_address = self.get_next_temp_address()
        self.add_instruction(self.Instruction.sub("#0", value, temp_address))
        self.semantic_stack.append(temp_address)

    def for_jump_to_condition(self, *_):
        condition_address = self.semantic_stack[-5]
        self.add_instruction(self.Instruction.jp(condition_address))

    def for_body_start(self, *_):
        address = self.semantic_stack[-2]
        self.add_instruction(
            self.Instruction.jp(f"#{self.instruction_index}"), index=address
        )

    def for_body_end(self, *_):
        step_address = self.semantic_stack[-1]
        self.add_instruction(self.Instruction.jp(step_address))
        condition = self.semantic_stack[-4]
        condition_jump_instruction_index = self.semantic_stack[-3]
        self.add_instruction(
            self.Instruction.jpf(condition, f"#{self.instruction_index}"),
            index=condition_jump_instruction_index,
        )
        self.semantic_stack.pop()
        self.semantic_stack.pop()
        self.semantic_stack.pop()
        self.semantic_stack.pop()
        self.semantic_stack.pop()

    def raise_semantic_error(self, message, line_number=None):
        self.semantic_errors_list.append(
            f"#{self.parser.scanner.reader.line_number if line_number is None else line_number} : Semantic Error! {message}"
        )

    def raise_undefined_semantic_error(self, name, line_number=None):
        self.raise_semantic_error(f"'{name}' is not defined.", line_number=line_number)

    def raise_illegal_void_type_semantic_error(self, name, line_number=None):
        self.raise_semantic_error(
            f"Illegal type of void for '{name}'.", line_number=line_number
        )

    def raise_arg_count_mismatch_semantic_error(self, name, line_number=None):
        self.raise_semantic_error(
            f"Mismatch in numbers of arguments of '{name}'.", line_number=line_number
        )

    def raise_break_semantic_error(self, line_number=None):
        self.raise_semantic_error(
            "No 'for' found for 'break'.", line_number=line_number
        )

    def raise_operand_type_mismatch_semantic_error(
        self, actual, expected, line_number=None
    ):
        self.raise_semantic_error(
            f"Type mismatch in operands, Got {actual} instead of {expected}.",
            line_number=line_number,
        )

    def raise_arg_type_mismatch_semantic_error(
        self, at, arg_name, expected, actual, line_number=None
    ):
        self.raise_semantic_error(
            f"Mismatch in type of argument {arg_name} of '{at}'. Expected '{expected}' but got '{actual}' instead.",
            line_number=line_number,
        )

    def add_instruction(self, instruction, index=None):
        if index is None:
            index = self.instruction_index
            self.instruction_index += 1
        elif isinstance(index, str):
            index = int(index[1:])

        self.instructions[index] = instruction

    def add_instructions(self, *instructions):
        for instruction in instructions:
            self.add_instruction(instruction)

    def get_next_data_address(self, size=WORD_SIZE):
        next_data_address = self.data_address
        self.data_address += size
        return next_data_address

    def get_next_temp_address(self):
        next_temp_address = self.temp_address
        self.temp_address += self.WORD_SIZE
        return next_temp_address

    def push_addresses(self):
        self.push_to_stack(self.return_address_address)
        self.push_to_stack(self.stack_pointer_address)

    def pop_addresses(self):
        self.pop_from_stack(self.stack_pointer_address)
        self.pop_from_stack(self.return_address_address)

    def save_return_address(self, arg_count):
        self.add_instruction(
            self.Instruction.assign(
                f"#{self.instruction_index + 2 * (arg_count + 1)}",
                self.return_address_address,
            )
        )

    def save_return_value(self, value):
        self.add_instruction(self.Instruction.assign(value, self.return_value_address))

    def generate_implicit_output(self):
        self.do_action("#pid", Token(TokenType.ID, "output"), None)
        self.do_action("#declare_function", None, None)
        self.do_action("#open_scope", None, None)
        self.do_action("#set_function_scope_flag", None, None)
        self.do_action("#pid", Token(TokenType.ID, "a"), None)
        self.do_action("#pop_param", None, None)
        self.do_action("#pid", Token(TokenType.ID, "a"), None)
        self.do_action("#open_scope", None, None)
        self.add_instruction(self.Instruction.print(self.semantic_stack.pop()))
        self.do_action("#close_scope", None, None)
        self.do_action("#jump_back", None, None)

    def do_action(self, identifier, previous_token, current_token):
        try:
            getattr(self, identifier[1:])(previous_token, current_token)
        except Exception as exception:
            if not self.semantic_errors_list:
                raise exception

from parser import Parser


def main(input_file, output_tree_file, output_error_file):
    parser = Parser(input_file)
    parser.parse()
    parser.print_tree(output_tree_file)
    parser.print_errors(output_error_file)


if __name__ == "__main__":
    main("input.txt", "parse_tree.txt", "syntax_errors.txt")

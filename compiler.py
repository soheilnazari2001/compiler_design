from parser import Parser


def main(input_file_path, output_tree_file_path, output_error_file_path):
    with open(input_file_path, "r") as input_file:
        parser = Parser(input_file)
        parser.parse()
    parser.print_tree(output_tree_file_path)
    parser.print_errors(output_error_file_path)


if __name__ == "__main__":
    main("input.txt", "parse_tree.txt", "syntax_errors.txt")

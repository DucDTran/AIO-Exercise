def main():
    def count_chars(string):
        chars_dict = {}
        for i in range(len(string)):
            if string[i] not in chars_dict:
                chars_dict[string[i]] = 1
            else:
                chars_dict[string[i]] += 1
        return chars_dict
    print(count_chars('smiles'))


if __name__ == '__main__':
    main()

import os


def main():

    def read_file(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
            file.close()
        words_list = [word for sentence in data.split(
            '\n') for word in sentence.split(' ')]
        return words_list

    def word_count(file_path):
        data = read_file(file_path=file_path)
        words_dict = {}
        for word in data:
            if word.lower() not in words_dict:
                words_dict[word.lower()] = 1
            else:
                words_dict[word.lower()] += 1
        return words_dict

    print(word_count('/Users/dwctran/AIO-Exercise/data-structure-240610/P1_data.txt'))


if __name__ == '__main__':
    main()

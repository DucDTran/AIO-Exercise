def main():
    def sliding_window(num_list, k):
        max_list = []
        first_index = 0
        last_index = k
        while first_index < len(num_list) - k + 1:
            max_list.append(max(num_list[first_index:last_index]))
            first_index += 1
            last_index += 1
        return max_list
    print(sliding_window(num_list=[3, 4, 5, 1, -44, 5, 10, 12, 33, 1], k=3))


if __name__ == '__main__':
    main()

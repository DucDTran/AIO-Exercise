def main():
    def calc_lev_distance(source, target):
        rows = len(source) + 1
        cols = len(target) + 1
        matrix = [[None for _ in range(cols)] for _ in range(rows)]
        for row_idx in range(0, rows):
            matrix[row_idx][0] = row_idx
        for col_idx in range(0, cols):
            matrix[0][col_idx] = col_idx
        for row_idx in range(1, rows):
            for col_idx in range(1, cols):
                if source[row_idx-1] == target[col_idx-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[row_idx][col_idx] = min(matrix[row_idx-1][col_idx] + 1,
                                               matrix[row_idx][col_idx-1] + 1,
                                               matrix[row_idx-1][col_idx-1] + cost)
        return matrix[-1][-1]
    print(calc_lev_distance('yu', 'you'))


if __name__ == '__main__':
    main()

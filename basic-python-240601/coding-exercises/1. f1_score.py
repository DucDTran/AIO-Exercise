
def main():
    # f1-score classification model
    def calc_f1_score(tp, fp, fn):
        assert isinstance(tp, int), 'tp must be int'
        assert isinstance(fp, int), 'fp must be int'
        assert isinstance(fn, int), 'fn must be int'
        assert (tp > 0 and fp > 0 and fn >
                0), 'tp and fp and fn must be greater than zero'

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        print(f'precision score is {precision}',
              f'recall score is {recall}', f'f1-score is {f1_score}', sep='\n')

    try:
        calc_f1_score(tp=2, fp=3, fn=5)
    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    main()

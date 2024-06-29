# question 1
def max_kernel(num_list, k):
    result = []
    first_index = 0
    last_index = k
    while first_index < len(num_list) - k + 1:
        result.append(max(num_list[first_index:last_index]))
        first_index += 1
        last_index += 1
    return result


assert max_kernel([3, 4, 5, 1, -44], 3) == [5, 5, 5]
num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
print(max_kernel(num_list=num_list, k=3))

# question 2


def character_count(word):
    character_statistic = {}
    for i in range(len(word)):
        if word[i] not in character_statistic:
            character_statistic[word[i]] = 1
        else:
            character_statistic[word[i]] += 1
    return character_statistic


assert character_count("Baby") == {'B': 1, 'a': 1, 'b': 1, 'y': 1}
print(character_count('smiles'))

# question 3


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        file.close()
    words_list = [word for sentence in data.split(
        '\n') for word in sentence.split(' ')]
    return words_list


def count_word(file_path):
    data = read_file(file_path=file_path)
    counter = {}
    for word in data:
        if word.lower() not in counter:
            counter[word.lower()] = 1
        else:
            counter[word.lower()] += 1
    return counter


file_path = '/Users/dwctran/AIO-Exercise/data-structure-240610/P1_data.txt'
result = count_word(file_path=file_path)
assert result['who'] == 3
print(result['man'])

# question 4


def levenshtein_distance(token1, token2):
    rows = len(token1) + 1
    cols = len(token2) + 1
    matrix = [[None for _ in range(cols)] for _ in range(rows)]
    for row_idx in range(0, rows):
        matrix[row_idx][0] = row_idx
    for col_idx in range(0, cols):
        matrix[0][col_idx] = col_idx
    for row_idx in range(1, rows):
        for col_idx in range(1, cols):
            if token1[row_idx-1] == token2[col_idx-1]:
                cost = 0
            else:
                cost = 1
            matrix[row_idx][col_idx] = min(matrix[row_idx-1][col_idx] + 1,
                                           matrix[row_idx][col_idx-1] + 1,
                                           matrix[row_idx-1][col_idx-1] + cost)
    return matrix[-1][-1]


assert levenshtein_distance('h1', 'hello') == 4
print(levenshtein_distance('hola', 'hello'))


# question 5
def check_the_number(n):
    list_of_numbers = []
    results = ""
    for i in range(1, 5):
        list_of_numbers.append(i)
    if n in list_of_numbers:
        results = "True"
    if n not in list_of_numbers:
        results = "False"
    return results


N = 7
assert check_the_number(N) == "False"
N = 2
results = check_the_number(N)
print(results)

# question 6


def my_function(data, max, min):
    result = []
    for i in data:
        if i < min:
            result.append(min)
        elif i > max:
            result.append(max)
        else:
            result.append(i)
    return result


my_list = [5, 2, 5, 0, 1]
max = 1
min = 0
assert my_function(max=max, min=min, data=my_list) == [1, 1, 1, 0, 1]
my_list = [10, 2, 5, 0, 1]
max = 2
min = 1
print(my_function(max=max, min=min, data=my_list))


# question 7
def my_function(x, y):
    x.extend(y)
    return x


list_num1 = ['a', 2, 5]
list_num2 = [1, 1]
list_num3 = [0, 0]

assert my_function(list_num1, my_function(list_num2, list_num3)) == ['a', 2, 5, 1, 1,
                                                                     0, 0]

list_num1 = [1, 2]
list_num2 = [3, 4]
list_num3 = [0, 0]

print(my_function(list_num1, my_function(list_num2, list_num3)))


# question 8

def my_function(n):
    min_value = n[0]
    for i in n:
        if i < min_value:
            min_value = i
        else:
            continue
    return min_value


my_list = [1, 22, 93, -100]
assert my_function(my_list) == -100
my_list = [1, 2, 3, -1]
print(my_function(my_list))

# question 9


def my_function(n):
    max_value = n[0]
    for i in n:
        if i > max_value:
            max_value = i
        else:
            continue
    return max_value


my_list = [1001, 9, 100, 0]
assert my_function(my_list) == 1001
my_list = [1, 9, 9, 0]
print(my_function(my_list))

# question 10


def my_function(integers, number=1):
    bool_list = []
    for val in integers:
        if val == number:
            bool_list.append(True)
        else:
            bool_list.append(False)
    return any(bool_list)


my_list = [1, 3, 9, 4]
assert my_function(my_list, -1) == False

my_list = [1, 2, 3, 4]
print(my_function(my_list, 2))


# question 11
def my_function(list_nums=[0, 1, 2]):
    var = 0
    for i in list_nums:
        var += i
    return var / len(list_nums)


assert my_function([4, 6, 8]) == 6
print(my_function())


# question 12
def my_function(data):
    var = []
    for i in data:
        if i % 3 == 0:
            var.append(i)
    return var


assert my_function([3, 9, 4, 5]) == [3, 9]
print(my_function([1, 2, 3, 5, 6]))

# question 13


def my_function(y):
    var = 1
    while (y > 1):
        var *= y
        y -= 1
    return var


assert my_function(8) == 40320
print(my_function(4))


# question 14
def my_function(x):
    reverse_list = ''.join(reversed([element for element in x]))
    return reverse_list


x = 'I can do it'
assert my_function(x) == 'ti od nac I'

x = 'apricot'
print(my_function(x))


# question 15
def function_helper(x):
    if x > 0:
        return "T"
    else:
        return "N"


def my_function(data):
    res = [function_helper(x) for x in data]
    return res


data = [10, 0, -10, -1]
assert my_function(data=data) == ['T', 'N', 'N', 'N']

data = [2, 3, 5, -1]
print(my_function(data=data))

# question 16


def function_helper(x, data):
    for i in data:
        if x == i:
            return 0
    return 1


def my_function(data):
    res = []
    for i in data:
        if function_helper(i, res):
            res.append(i)

    return res


lst = [10, 10, 9, 7, 7]
assert my_function(lst) == [10, 9, 7]

lst = [9, 9, 8, 1, 1]
print(my_function(lst))

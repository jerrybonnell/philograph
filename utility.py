'''
  name: utility.py

  utility functions
'''

def load_file(filename):
    """
    load a file into memory
    """
    f_data = []
    # open the data-set file
    file = open(filename, "r")
    for line in file:
        row = line.strip()  # a row in the file
        f_data.append(row)  # append it to the 2D array

    return f_data

def write_file(filename, data):
    """
    writes to a file

    filename: name of file written to 
    data: contents to write 
    """
    file = open(filename, "a")
    file.write(data)
    file.close()

def convert_numerals(input_str):
    """
    convert a string containing roman numerals to a string
    containing those roman numerals as integers
    """
    # credit to: http://code.activestate.com/recipes/81611-roman-numerals/
    copy = input_str[:]
    copy = copy.split(" ")

    nums = ['m', 'd', 'c', 'l', 'x', 'v', 'i']
    ints = [1000, 500, 100, 50, 10, 5, 1]
    places = []

    for i in range(len(copy)):
        is_valid = True

        if "." in copy[i]:
            copy[i] = copy[i].replace(".", "")
        else:
            # . must be appended to end of string to signify it is a roman
            # numeral
            is_valid = False

        if "xix" in copy[i] or "xviii" in copy[i]:
            is_valid = True

        for c in copy[i].lower():
            if c not in nums:
                # return original
                is_valid = False

        if is_valid is False:
            continue

        for char_index in range(len(copy[i])):
            c = copy[i][char_index].lower()
            value = ints[nums.index(c)]
            # If the next place holds a larger number, this value is negative.
            try:
                nextvalue = ints[nums.index(copy[i][char_index + 1].lower())]
                if nextvalue > value:
                    value *= -1
            except IndexError:
                # there is no next place.
                pass
            places.append(value)

        out = 0

        for n in places:
            out += n

        copy[i] = str(out)

    return " ".join(copy)

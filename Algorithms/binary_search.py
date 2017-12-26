"""You're going to write a binary search function.
You should use an iterative approach - meaning
using loops.
Your function should take two inputs:
a Python list to search through, and the value
you're searching for.
Assume the list only has distinct elements,
meaning there are no repeated values, and 
elements are in a strictly increasing order.
Return the index of value, or -1 if the value
doesn't exist in the list."""

import math

def binary_search(input_array, value):
    """" Binary_search based on loop """

    sub_array = input_array
    offset = 0 
    while True:
        # get middle value and index
        middle_idx = int(math.floor(len(sub_array)/2.0))
        middle_value = sub_array[middle_idx]

        # check if value is found
        if middle_value == value:
            return middle_idx + offset
        
        # if sub_array only has one element, end
        if len(sub_array) == 1: 
            break   
        # else, get a sub array
        elif value < middle_value:
            sub_array = sub_array[:middle_idx]
        elif value > middle_value:
            sub_array = sub_array[middle_idx+1:]
            # update offset, to get the right index
            offset += (middle_idx+1)


    # if not found, return -1
    return -1

test_list = [1,3,9,11,15,19,29]
test_val1 = 25
test_val2 = 15
test_val3 = 1 # edge case, not found
test_val4 = 29
test_val5 = 0
print binary_search(test_list, test_val1)
print binary_search(test_list, test_val2)
print binary_search(test_list, test_val3)
print binary_search(test_list, test_val4)
print binary_search(test_list, test_val5)



"""Implement quick sort in Python.
Input a list.
Output a sorted list."""
def quicksort(array):

    # make a set of found numbers
    found = set()

    # keep looping until all numbers are found
    while len(found) < len(array):
        # the pivot will ne the last one
        pos = len(array)-1
        # unless the pivot is already in its right place. then look at the previous element
        while pos in found:
            pos -= 1
        # assign the pivot. reset i to start looping throuth the numbers
        pivot = array[pos]
        i = 0
        # loop through all numbers 
        while i < pos:
            # if i is already in the right position, go to the next number
            if array[i] > pivot:
                if i not in found:
                    # move element to the position of the pivot
                    array[pos] = array[i]
                    # move the number behind the pivot to the position of the element
                    array[i] = array[pos-1]
                    # move the pivot one space back
                    array[pos-1] = pivot
                    # update the position of the pivot
                    pos -= 1
            else:
                i+=1
        found.add(pos)

    return array

test = [21, 4, 1, 3, 9, 20, 0, 25, 6, 21, 14, 765, 2, 54]

print quicksort(test)

"""Implement a function recursivly to get the desired
Fibonacci sequence value.
Your code should have the same input/output as the 
iterative code in the instructions."""

# 0,1,1,2,3,5,8,13,21,34...

def get_fib(position):
    if position == 0:
        return 0
    prev  = 0
    next = 1
    for _ in range(position-1):
        new_number = prev+next
        prev =  next
        next = new_number
    return next

# Test cases
print get_fib(9)
print get_fib(11)
print get_fib(0)


def get_fib(position):
    if position <= 1: 
        return position
    return get_fib(position-1) + get_fib(position-2)

# Test cases
print get_fib(9)
print get_fib(11)
print get_fib(0)


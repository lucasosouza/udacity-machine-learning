# question2.py

"""
Given a string a, find the longest palindromic substring contained in a. Your function definition should look like question2(a), and return a string.
"""

"""
Solution: get all substrings in a, in order, ordered from shortest to longest. Loop through each one; if the substring is a palindrome, set it as the longest palindrome. 
"""

def question2(a):
    """ Return the longest palindromic substring in a. Considering substrings with even one letter only """

    longest_palindromic = None
    for s in substrings(a):
        if s == s[::-1]:
            longest_palindromic = s

    return longest_palindromic



def substrings(s):
    """ Helper function. Get all substrings of a string"""

    # get substrings
    substrings = list()
    for x in range(len(s)):
        for y in range(x, len(s)):
            substrings.append(s[x:y+1])

    # order descending
    substrings = sorted(substrings, key=lambda x:len(x))

    return substrings


print question2('afgfebcbeafgf')
# print ebcbe

print question2('anna is cute nen')
# print anna

print question2('')
# edge case: empty string, return None

print question2('palindrome')
# edge case: string with no palindromes, returns a single letter
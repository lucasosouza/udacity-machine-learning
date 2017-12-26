# tech_prep.py

from itertools import permutations
from collections import Counter
from time import time

"""
Question 1
Given two strings s and t, determine whether some anagram of t is a substring of s. For example: if s = "udacity" and t = "ad", then the function returns True. Your function definition should look like: question1(s, t) and return a boolean True or False.
"""

"""
Solution: get all possible anagrams of t (permutations of letters in the word), and check to see if any of them is a substring of s. If none of them is a substring, return False.
"""

def question1a(s,t):
    """ Determine whether some anagram of t is a substring of s """

    anagrams = permutations(t, len(t))
    for anagram in anagrams:
        if anagram:
            if ''.join(anagram) in s:
                return True
    return False


"""
Reviewed solution: count frequency of characters in t. For each substring in s with same length as t, check if the frequency of characters equals frequency of characters of t.
"""

def question1b(s,t):
    """ Determine whether some anagram of t is a substring of s """

    # check if they are no empty strings
    if t and s:

        # count char frequency for t
        frequency_t = Counter()
        for char in t:
            frequency_t[char] += 1

        # get number of substrings in s with length same as t
        length_t = len(t)
        n_substrings_s = len(s) - length_t + 1

        # loop through substrings in s
        for i in range(n_substrings_s):
            # define substring
            substring_s = s[i:length_t+i]
            # count char frequency for s substring
            frequency_subs = Counter()
            for char in substring_s:
                frequency_subs[char]+=1
            # compare frequency. 
            # break any time the frequency of a char does not match
            found = True
            for char, count in frequency_t.items():
                if frequency_subs[char] != count:
                    found = False
                    break
            # return True if all true
            if found:
                return True

    return False

versions = [question1a, question1b]

for question1 in versions:

    t0 = time()

    print question1('udacity','yti')
    # prints True

    print question1('udacity','y')
    # prints True

    print question1('udacity','ytiudac')
    # prints True

    print question1('udacity','ytiuudac')
    # prints False

    print question1('udacity','ytiu')
    # prints False

    print question1('udacity','')
    # edge case: prints False. no anagrams are formed

    print question1('udacity','udacitya')
    # edge case: prints False. have all letters but one extra

    print "Version: {}, Time: {:.4f}".format(question1.__name__, time()-t0)



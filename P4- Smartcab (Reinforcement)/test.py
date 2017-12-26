from collections import defaultdict
def count_words(s, n):
	words = s.split()
	words_count = defaultdict(int)
	for word in words:
		words_count[word] += 1
	return sorted(words_count.items(), key=lambda x:(-x[1],x[0]))[:n]

print count_words("cat bat mat cat bat cat", 3)
print count_words("betty bought a bit of butter but the butter was bitter", 3)

import re
import nltk

from nltk.corpus import words

eng_words = words.raw().split("\n")
# print(len(eng_words))

# pattern1 = re.compile('r..r')
# print([w for w in eng_words if pattern1.match(w)][:20])


# pattern2 = re.compile('humou?r$')
# print([w for w in eng_words if pattern2.match(w)][:20])
# ['humor', 'humour'] -> return a;; words where u may or may not be present after humo and end with r

# pattern3 = re.compile('co+l$')
# print([w for w in eng_words if pattern3.match(w)][:20])
# ['col', 'cool'] -> return all words where o comes between c and l atleast once

# pattern4 = re.compile('analy[sz]e')
# print([w for w in eng_words if pattern4.match(w)][:20])
# ['analyse', 'analyser', 'analyses', 'analyze', 'analyzer'] -> return all words where either s or z comes after analy

pattern5= re.compile('p[aeiou]+t$')
print([w for w in eng_words if pattern5.match(w)][:20])
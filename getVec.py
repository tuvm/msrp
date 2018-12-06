import io
import nltk
import gensim
import itertools
import re
import json

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model[word].tolist()

with io.open('msr_paraphrase_data.txt', 'r', encoding='utf-8') as input_file:
  data = input_file.read()
  data = data.split('\n')
  data = [nltk.word_tokenize(item.split('\t')[1]) for item in data if item != '']
  data = list(itertools.chain.from_iterable(data))
  data = [item for item in data if re.match(r"\w", item)]
  data = list(set(data))
  data = sorted(data)
  print(data)
  print(len(data))
  result = []
  for item in data:
    try:
      vector = model[item].tolist()
      result.append({item: vector})
    except:
      result.append({item: 0})
  with open('data.json', 'w') as outfile:
    json.dump(result, outfile)
import argparse
import gzip
import math
import numpy
import re
import sys
import numpy as np
from collections import defaultdict

from copy import deepcopy

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename, header_embeddings=None):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  header = None
  if header_embeddings:
      header = fileObject.readline() 

  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors, header

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName, header=None):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')  
  if header:
    outFile.write(header)
  for word, values in wordVectors.iteritems():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')      
  outFile.close()
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = defaultdict(set)
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])].update([norm_word(word) for word in words[1:]])
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters, alpha_i, sym_norm):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    print "iteration:", it+1, " over ", numIter
    for word in loopVocab:
      wordNeighbours = lexicon[word].intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      newVec = alpha_i * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      beta_i = 1
      weight = beta_i / float(numNeighbours)
      for ppWord in wordNeighbours:
        if sym_norm:
          numNeighbours_j = len(lexicon[ppWord].intersection(wvVocab))
          if numNeighbours_j > 0:
            weight = beta_i / (np.sqrt(numNeighbours) * np.sqrt(numNeighbours_j))
          else:
            print "word, ppWord, neighbours of ppword", word, ppWord, lexicon[ppWord], lexicon[ppWord].intersection(wvVocab)
          newVec += weight * newWordVecs[ppWord]
        else:
          newVec += weight * newWordVecs[ppWord]
      sum_weights = beta_i + alpha_i
      newWordVecs[word] = newVec/(sum_weights)
  return newWordVecs
  
if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--header-embeddings', action='store_true',
                      help='First line of embedding file is the count of words and dimensions')
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  parser.add_argument("--symmetric-normalized", action="store_true")
  parser.add_argument("--alpha",type=float, default=1., help="Alpha")
  parser.add_argument("--dim",type=int, default=None, help="Dimension of embeddings")
  args = parser.parse_args()
 
  if not args.input and not args.dim:
    raise ValueError("Should specify dimensions of word embeddings")

  if args.symmetric_normalized:
    print "Using symmetric normalized laplacian graph"

  if args.input:
     wordVecs, header = read_word_vecs(args.input, args.header_embeddings)
  lexicon = read_lexicon(args.lexicon)
  if not args.input:
     # build word vectors randomly initialized
     wordVecs = {w: np.random.uniform(-1, 1, args.dim) for w in lexicon.keys() if ' ' not in w and w.isalpha()}
     print wordVecs.keys()[:100]
     header = str(len(wordVecs)) + ' ' + str(args.dim) + '\n'
  numIter = int(args.numiter)
  outFileName = args.output
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  newWordVecs = retrofit(wordVecs, lexicon, numIter, args.alpha,
                         args.symmetric_normalized)
  print_word_vecs(newWordVecs, outFileName, header) 

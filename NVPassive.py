# know bugs: does not work with proper nouns that end with 'ed' that is also preceded by 'e' plus another consonant, e.g. Chinesed'
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.parse import CoreNLPParser

# Rip the -ed suffix off of the past participle.
def ed_rip(word):
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    # Tokenize and tag the word.
    token = nltk.word_tokenize(word)
    tag = pos_tagger.tag(token)[0][1]
    # Load both SnowballStemmer and WordNetLemmatizer for accurate ed stripping.
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    preprocessed = lemmatizer.lemmatize(word,'v')
    ripword = stemmer.stem(preprocessed)
    # If the stem is a proper noun, the first letter will not be capilitalized. Hence recapitalization is needed.
    if tag.startswith('NNP'):
        ripword = ripword.capitalize()
    elif word[:1].isalpha():
        ripword = ripword.capitalize()
    # Return information needed to determine NV Passive.
    riptoken = nltk.word_tokenize(ripword)
    riptag = pos_tagger.tag(riptoken)[0][1]
    if riptag.startswith('V') is True:
        NV = False
    if riptag.startswith('N') is True:
        NV = True
    return (ripword, NV)

# Determine if a word is a form of 'be' or 'get'.
def copula_test(copula: str):
    be_forms = ['be','is','are','am','was','were','been']
    get_forms = ['get','gets','got','gotten']
    if (copula in be_forms) or (copula in get_forms):
        return True
    else: return False

def passive_finder(wordnow,tagpos,words):
    if tagpos > 1:
        penultimate = words[tagpos-1]
        antipenultimate = words[tagpos-2]
        print('Analyzing:',antipenultimate,penultimate,wordnow)
        print('copula_test:',copula_test(penultimate),copula_test(antipenultimate))
        if (copula_test(penultimate) == True) or (copula_test(antipenultimate) == True):
            return True
        else:
            return False
    if tagpos == 1:
        penultimate = words[tagpos-1]
        print('Analyzing:',penultimate,wordnow)
        print('copula_test:',copula_test(penultimate))
        if (copula_test(penultimate) == True):
            return True
        else:
            return False

def NV_Passive(tagged_sentence: list) -> tuple:
    ''' Main function to detect both Passive and NVPassive using a strict one-word intermediate rule.
    Only pass in a pos_tagged sentence as an argument, or generally, a list of tuples.
    '''
    # Prepare values to record passive and N-V.
    passive = False
    NVPassive = False
    steminfo = ('not passive', False)
    # Generate two sets of lists for words and tags.
    words = [wordtagpair[0] for wordtagpair in tagged_sentence]
    tags = [wordtagpair[1] for wordtagpair in tagged_sentence]
    print(words,'\n',tags)

    # A dedicated detector for NVPassive that uses a proper noun as its verb.
    NPlist = list()
    VBNlist = list()
    for tagpos,tag in enumerate(tags):
        if tag == ('NNP' or 'NNPS'):
            NPlist.append((tagpos,tag))
        if tag == 'VBN':
            VBNlist.append((tagpos,tag))
    if len(NPlist) != 0:
        for tagpos,tag in NPlist:
            wordnow = words[tagpos]
            if passive_finder(wordnow,tagpos,words):
                if wordnow.endswith('ed'):
                    steminfo = ed_rip(wordnow)
                    if steminfo[0] == wordnow:
                        break
                    else:
                        passive = True
                        NVPassive = True
                        print('Result: This sentence is a N-V Passive.')
                        return (steminfo[0],passive,NVPassive)

    # If there is no VBN in the tags at all, both Passive and NVPassive is automatically ruled out.
    if len(VBNlist) != 0:
        for tagpos,tag in VBNlist:
            wordnow = words[tagpos]
            if passive_finder(wordnow, tagpos, words):
                passive = True
                steminfo = ed_rip(wordnow)
                if steminfo[1]:
                    NVPassive = True
                    print('Result: This sentence is a N-V Passive.')
                    break
                else:
                    print('Result: The sentence is a passive but is not an N-V Passive.')
                    break
            else:
                print('Passive voice not found.')
    else:
        print('Result: The sentence is not a passive at all.')
    return (steminfo[0], passive, NVPassive)

# Function test
def test(sent_pool, t):
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    for s in sent_pool:
        result = NV_Passive(pos_tagger.tag(word_tokenize(s)))
        print(result)
        if t == 'f':
            if result[1] == False and result[2] == False:
                print('False test passed.')
            else:
                print('▇▇False test failed.')
        elif t == 'p':
            if result[1] == True and result[2] == False:
                print('Passive test passed.')
            else: print('▇▇Passive test failed.')
        elif t == 't':
            if result[1] == True and result[2] == True:
                print('True test passed.')
            else: print('▇▇True test failed.')

false_pool = sent_tokenize(
    '''The corpus has been a great hit. Our favorited teacher had always been Fred. I have been on Twitter for a long time.'''
)
passive_pool = sent_tokenize(
    '''The corpus has been greatly applauded.
    '''
)
true_pool = sent_tokenize(
    '''The sentence is already tagged. Man, you just got Trumped so hard! He was friended on Facebook. The news was Twittered all over the place. I'm not so stupid to be Shanghaied by those idiots!
    '''
)

test(false_pool,'f')
test(passive_pool,'p')
test(true_pool,'t')
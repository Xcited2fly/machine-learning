from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
sentence=['He is the driver of all the cars','All the cars are driven by him']
def lemmatize(token,tag):
    if tag[0].lower() in ['n','v']:
        return lemmatizer.lemmatize(token,tag[0].lower())
    return token
lemmatizer=WordNetLemmatizer()
tagged_sentence=[pos_tag(word_tokenize(doc)) for doc in sentence]
print('Lemmatized: ',[[lemmatize(token,tag) for token, tag in doc] for doc in tagged_sentence])

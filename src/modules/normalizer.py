import regex
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

class Normalizer:
    """ Abstract class for normalizing string in various ways """
    
    def __init__(self):
        pass
    
    def __call__(self, string):
        raise NotImplementedError("Subclasses must implement this method")
    
    def normalize_list(self, string_list):
        return [self.normalize(string) for string in string_list]
    
    def __add__(self, other):
        return AggregateNormalizer([self, other])

class AggregateNormalizer(Normalizer):
    """ Normalizes strings by applying a list of normalizers, in order """
    
    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = normalizers
    
    def __call__(self, string):
        for normalizer in self.normalizers:
            string = normalizer(string)
        return string
    
    def __add__(self, other):
        if not isinstance(other, AggregateNormalizer):
            return AggregateNormalizer(self.normalizers + [other])
        else:
            return AggregateNormalizer(self.normalizers + other.normalizers)
    
class StringNormalizer(Normalizer):
    """ Normalizes strings by stripping whitespace & other features """
    
    def __init__(
        self,
        remove_punct=True,
        normalize_whitespace=True,
        lowercase=True,
        remove_digits=False
    ):
        super().__init__()
        self.remove_punct = remove_punct
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.remove_digits = remove_digits
    
    def __call__(self, string):
        if self.remove_punct:
            string = regex.sub(r'\p{P}+', '', string)
        if self.lowercase:
            string = string.lower()
        if self.remove_digits:
            string = regex.sub(r'\d+', '', string)
        if self.normalize_whitespace:
            string = regex.sub(r'\s+', ' ', string)
        return string
    
class StemmerNormalizer(Normalizer):
    """ Normalizes strings by stemming words """
    
    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()
        
    def __call__(self, string):
        return " ".join([self.stemmer.stem(word) for word in string.split()])
    
class TokenizerNormalizer(Normalizer):
    """ Normalizes strings by tokenizing them """
    
    def __init__(self):
        super().__init__()
        self.tokenizer = TweetTokenizer()
        
    def __call__(self, string):
        return ' '.join(self.tokenizer.tokenize(string))

class GutenbergNormalizer(Normalizer):
    """ Removes Gutenberg boilerplate text """
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, string):
        # The start can be identified by the first line that contains "START OF THE PROJECT GUTENBERG"
        start_loc = regex.search(r"(START OF THE PROJECT GUTENBERG.*?$)", string, regex.IGNORECASE)
        
        # The end location can be identified by the first line that contains "END OF THE PROJECT GUTENBERG"
        end_loc = regex.search(r"(END OF THE PROJECT GUTENBERG.*?$)", string, regex.IGNORECASE)
        
        start_loc = start_loc.end() if start_loc else 0
        end_loc = end_loc.start() if end_loc else len(string)
        
        return string[start_loc:end_loc]
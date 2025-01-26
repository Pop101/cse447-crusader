import os
from datasets import IterableDataset
import chardet
import random
import warnings
from tqdm.auto import tqdm

class FileDataloader:
    """
    Loads data from text files in a directory, returning them as a dict or Dataset with entries: \n
    - text: The text of the file \n
    - filename: The name of the file
    """
    
    def __init__(self, directory, filters=[]):
        self.directory = directory
        self.filters = filters
        
    def get_data(self):
        return IterableDataset.from_generator(lambda: iter(self))
    
    def _encoding_ambivalent_load(self, file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            raw_data = raw_data.decode(encoding)
        return raw_data
    
    def __iter__(self):
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".txt"):  # Process only .txt files
                    file_path = os.path.join(root, file)
                    text = self._encoding_ambivalent_load(file_path)
                    for filter in self.filters:
                        text = filter(text)
                    yield {"text": text, "filename": file}

class FixedLengthDataloader(FileDataloader):
    """
    Loads data from text files in a directory, returning them as a dict or Dataset with entries: \n
    - text: The text of the file \n
    - filename: The name of the file \n
    
    The text will overlap by overlap_size characters, and will be split into fixed_length chunks.
    """
    
    def __init__(self, directory, fixed_length=1000, overlap_size=100, skip_shorter_than=0, filters=[]):
        super().__init__(directory, filters)
        self.fixed_length = fixed_length
        self.overlap_size = overlap_size
        self.skip_shorter_than = skip_shorter_than
        self.filters = filters
    
    def __iter__(self):
        for data in super().__iter__():
            # Load text
            text = data['text']
            
            # Skip if shorter than skip_shorter_than
            if len(text) < self.skip_shorter_than:
                continue
            
            # Split into fixed_length chunks with overlap
            for i in range(0, len(text), self.fixed_length - self.overlap_size):
                chunk = text[i:i + self.fixed_length]
                yield {"text": chunk, "filename": data['filename']}
                
class NgramDataloader(FileDataloader):
    """
    Loads data from text files in a directory, returning them as a dict or Dataset with entries: \n
    - ngram: A tuple of words, representing the ngram \n
    \n
    Ngrams are defined by simple string split. For more advanced ngram definition, add filters from the normalizer module.\n
    <NgramDataloader(filters=[TokenizerNormalizer()])> would tokenize the string before splitting it into ngrams.\n
    \n
    Will return ngrams of lengths defined by the ngram_sizes (overwrites) and ngram_size params.
    """
    
    def __init__(self, directory, ngram_size=2, ngram_sizes=[2, 3], filters=[]):
        super().__init__(directory, filters)
        
    def __iter__(self):
        for data in super().__iter__():
            # Load text
            text = data['text']
            
            # Generate ngrams
            max_ngram = max(self.ngram_sizes)
            word_stack = []
            for word in text.split():
                word_stack.append(word)
                if len(word_stack) > max_ngram:
                    word_stack.pop(0)
                for ngram_size in self.ngram_sizes:
                    if len(word_stack) >= ngram_size:
                        yield {"ngram": tuple(word_stack[-ngram_size:]), "filename": data['filename']}

class SymlinkTestTrainSplit:
    """
    Splits all txt files in a directory into any number of test/train sets,
    creating new directories filled with symlinks to the original files.
    
    Note: to mitigate the relative cost of os.walk, this attempts to use an index.txt file at the root directory
    """
    
    def __init__(self, directory, split_dictionary):
        self.directory = directory
        
        total_weight = sum(split_dictionary.values())
        self.split_dictionary = {k: v / total_weight for k, v in split_dictionary.items()}   
        
        index_file = os.path.join(self.directory, "index.txt")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                self.index = [line.strip() for line in f if line.strip()]
        else:
            warnings.warn(f"No index.txt found in {self.directory}. Using os.walk to find files.")
            self.index = []
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.endswith(".txt"):
                        self.index.append(os.path.join(root, file))
        
        self.index = sorted(self.index)
    
    def split(self, random_state=None, shuffle=True):
        if random_state is not None:
            random.seed(random_state)
        
        idx = self.index.copy()
        if shuffle:
            random.shuffle(idx)
        
        # Calculate the number of files for each split
        split_counts = {k: int(v * len(idx)) for k, v in self.split_dictionary.items()}
        
        remaining = len(idx) - sum(split_counts.values())
        last_key = list(self.split_dictionary.keys())[-1]
        split_counts[last_key] += remaining
                
        # Create directories and distribute files
        file_assignments = dict.fromkeys(self.split_dictionary.keys(), [])
        for split_name, count in split_counts.items():
            file_assignments[split_name] = idx[:count]
            idx = idx[count:]
        file_assignments[last_key] += idx
        
        # Create symlinks in respective directories
        for split_name, files in file_assignments.items():
            split_dir = os.path.join(self.directory, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Check files against existing index.txt
            index_file = os.path.join(split_dir, "index.txt")
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    existing_files = [line.strip() for line in f if line.strip()]
                files = [f for f in files if f not in existing_files]
            
            # Symlink all files
            for file_path in tqdm(files, desc=f"Creating symlinks for {split_name}"):
                symlink_path = os.path.join(split_dir, file_path)
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    continue
                os.makedirs(os.path.dirname(symlink_path), exist_ok=True)
                os.symlink(file_path, symlink_path)
            
            # Make index.txt for this split
            with open(os.path.join(split_dir, "index.txt"), 'w') as f:
                for file in files:
                    f.write(file + '\n')
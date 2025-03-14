import random

def chunker(it, chunk_size):
    chunk = []
    for item in it:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
        
def sample_stream(it, chance:float):
    for item in it:
        if random.random() < chance:
            yield item

# Note: tools for data input are in  src/modules/datareader.py
# Note: tools for data output are in src/modules/datawriter.py
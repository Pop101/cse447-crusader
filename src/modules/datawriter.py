import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def chunker(it, chunk_size):
    chunk = []
    for item in it:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
        
def stream_to_single_parquet(iterator, output_path):
    """
    Stream iterator data to a single parquet file using pyarrow directly.
    Good for cases where you want a single file but data doesn't fit in memory.
    """
    # Get the schema from first chunk
    first_df = next(iterator)
    table = pa.Table.from_pandas(first_df)
    
    # Create ParquetWriter with the schema
    writer = pq.ParquetWriter(
        output_path,
        table.schema,
        compression='snappy'
    )
    
    # Write first chunk
    writer.write_table(table)
    
    # Process remaining chunks
    for df in iterator:
        table = pa.Table.from_pandas(df)
        writer.write_table(table)
    
    writer.close()
    
def stream_to_csv(iterator, output_path):
    """
    Stream iterator data to a single CSV file.
    Simple approach with good compatibility but larger file size.
    """
    first_chunk = True
    for df in iterator:
        if first_chunk:
            df.to_csv(output_path, index=False)
            first_chunk = False
        else:
            df.to_csv(output_path, mode='a', header=False, index=False)
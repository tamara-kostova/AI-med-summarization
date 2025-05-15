from datasets import load_dataset, concatenate_datasets
import os
import aiohttp

def load_pubmed_dataset():
    cache_path = "./data/scientific_papers_pubmed"
    
    if os.path.exists(cache_path):
        return load_from_disk(cache_path)
    
    try:
        train = load_dataset(
            "scientific_papers",
            "pubmed",
            split="train[:5000]",
            trust_remote_code=True,
            cache_dir="./data",
            download_mode="reuse_dataset_if_exists",
            storage_options={
                "client_kwargs": {
                    "timeout": aiohttp.ClientTimeout(total=7200)
                }
            }
        )
        
        validation = load_dataset(
            "scientific_papers",
            "pubmed",
            split="validation[:500]",
            trust_remote_code=True
        )
        
        dataset = concatenate_datasets([train, validation])
        dataset.save_to_disk(cache_path)
        return dataset
        
    except Exception as e:
        print(f"Loading failed: {str(e)}")
        if os.path.exists("./backup_data/pubmed"):
            return load_from_disk("./backup_data/pubmed")
        raise
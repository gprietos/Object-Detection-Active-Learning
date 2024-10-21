from pathlib import Path
import pickle


def update_cache(cache_file, current_state):
    if Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    cache.update(current_state)
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

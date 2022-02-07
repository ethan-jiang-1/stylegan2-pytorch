import os
import shutil

def clean_up_cache(root):
    home = os.path.dirname(root)
    cache = "{}/.cache/torch_extensions".format(home)
    if os.path.isdir(cache):
        print("clean up ", cache)
        shutil.rmtree(cache)
    else:
        print("no cache", cache)

def clean_up_pycache():
    pass


if __name__ == "__main__":
    root = os.getcwd()
    clean_up_cache(root)
    clean_up_pycache()

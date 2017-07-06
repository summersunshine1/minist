import os
import sys
import time

def getparentdir():
    pwd = sys.path[0]
    abs_path = os.path.abspath(pwd)
    return abs_path 
    
if __name__=="__main__":
    print(getparentdir())
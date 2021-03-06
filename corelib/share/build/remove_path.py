
## This script scans through the sire.app directory and
## finds all textual references to the full path of that
## directory. It replaces this with {[{ROOT}]}, so that
## another script can then change the path back when
## the package is unpacked

import os
import sys
import shutil

root_dir = sys.argv[1]

try:
    remove_dir = sys.argv[2]
except:
    remove_dir = root_dir

print("Scanning through %s to remove %s" % (root_dir,remove_dir))

def is_binary(filename):
    """
    Return true if the given filename appears to be binary.
    File is considered to be binary if it contains a NULL byte.
    FIXME: This approach incorrectly reports UTF-16 as binary.
    """
    with open(filename, 'rb') as f:
        for block in f:
            if b'\x00' in block:
                return True
    return False

def contains_root(filename, root):
    try:
        FILE = open(filename, "r")

        line = FILE.readline()

        while line:
            if line.find(root) != -1:
                FILE.close()
                return True

            line = FILE.readline()

        return False
    except:
        return False

def remove_root(filename, root):
    FILE = open(filename, "r")
    FILE2 = open("%s.tmpcopy" % filename, "w")

    line = FILE.readline()

    while line:
        if line.find(root) != -1:
            line = line.replace(root, "{[{ROOT}]}")

        FILE2.write(line)

        line = FILE.readline()

    FILE.close()
    FILE2.close()

    shutil.copystat(filename, "%s.tmpcopy" % filename)
    shutil.move("%s.tmpcopy" % filename, filename)    

def scanDir(root_dir, top_root_dir):
    for file in os.listdir(root_dir):
        if file.find("restore_path.py") != -1:
            # don't do anything to restore_path.py
            continue

        fullfile = "%s/%s" % (root_dir,file)

        if os.path.isdir(fullfile):
            scanDir(fullfile, top_root_dir)
            continue

        if os.path.islink(fullfile):
            continue

        if is_binary(fullfile):
            continue

        # this is a text file and not a symbolic link
        # see if it contains the text 'top_root_dir'
        if contains_root(fullfile, top_root_dir):
            remove_root(fullfile, top_root_dir)

scanDir(root_dir, remove_dir)

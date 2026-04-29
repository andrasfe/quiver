# paper/

This folder contains encrypted working material. Contents are **encrypted
at rest** with [`git-crypt`](https://github.com/AGWA/git-crypt). On
github.com they appear as binary blobs; locally they decrypt transparently
once the repo is unlocked.

## Contents

Encrypted source files and supporting material.

## To unlock

```bash
brew install git-crypt
git clone git@github.com:andrasfe/quiver.git && cd quiver
git-crypt unlock /path/to/saved/quiver-git-crypt.key
# encrypted files now decrypt automatically
```

After unlock, use the local build instructions inside the decrypted
material.

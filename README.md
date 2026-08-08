Scripts to apply patches to turbo-fieldfare-fork (which is a fork from https://github.com/drumih/turbo-fieldfare).
The purpose of the (3) patches is to download and use the QAT version of gemma-4-26b-a4b-it (q4) instead of the normal one. QAT (quantised aware training) is supposed to be a little more precise than the quantized one.

Just clone this repo locally, cd into it and execute: ./makeall.sh; it will do everything including the app bundle creation and installation on /Applications folder.

### How to run the "best" model

put the uploaded files into hw2 folder. Then run 
```
python ibm1_agreement.py -n 100000 -i 5 > alignment
python score-alignments < dice.a    
```
To recreate the uploaded alignment.

Uploaded file info
- ibm1.py: the IBM1 alignment model
- better_initailization.py: implementation of different initialization methods
- ibm1_agreement.py: implementation of joint train alignment method
- alignment: aligned output
- short_writeup.md: short writeup of the implementation
### Implemented solution

I added the French-English Alignment by Agreement paper [Liang et al, 2006](https://aclanthology.org/N06-1014.pdf).
There are three major changes

1. Joint training

Now, the model is trained jointly with both english to french and french to english alignments. The equation from section 3.2 in the paper is implemented.

2. Different Initialization

In the original model, the initialization gave a uniform distribution for all alignments. I changed it to give a uniform distribution for words that only coappear in the language pair.

I also implemented a thrid initialization which considers the number of times a word appears in the language pair and the length of each sentence into initial calculation (see code for detail).
This approac did not work well.
3. Different alignment method

Tested on different alignment method. The IBM1 was implemented with best alignment. (highest p(e |f)). However, this limits the model to only predict one english word per word.
So I tested threshold alignment, where the model predicts all english words that have $$p(e | f)p(f|e)> threshold.$$

### Experiments
All experiments were run trained on 1000 sentences and 10 iterations

#### Joint Training

| Metric     | IBM1 (F -> E) | Joint Training |
|------------|---------------|----------------|
| Precision  | 0.463245      | 0.511789       | 
| Recall     | 0.662722     | 0.692308       | 
| AER        | 0.473088      | **0.430595**   |  

Joint Training performs better than basic IBM1.

### different initialization
| Metric     | init-uniform-all | init-uniform-seen | init-ratio |
|------------|------------------|-------------------|------------|
| Precision  | 0.511789         | 0.514563          | 0.474341   |
| Recall     | 0.692308         | 0.692308          | 0.639053   |
| AER        | 0.430595         | **0.428706**      | 0.473088   |
We see that there is not substantial difference between the original uniform initialization and new initialization.
Decided to use the init-uniform-seen since AER is slightly better.
init-ratio did not perform well.

### Different alignment method

All methods use Joint training with init-uniform-all

| Metric     | max(f -> e) | threshold .75 | threshold .5 | threshold .35 | threshold .3 |
|------------|-------------|----------------|--------------|---------------|--------------|
| Precision  | 0.511789    | 0.67354       | 0.660848     | 0.623126      | 0.60161      |
| Recall     | 0.692308    | 0.482249      | 0.618343     | 0.650888      | 0.659763     |
| AER        | 0.430595    | 0.429253      | **0.358593** | 0.365217      | 0.37485      |


the max method chooses maximum p(e |f) for a word.
The threshold method aligns words if $$(p(e | f) \cdot p(f | e) )^{0.5} > threshold$$
we see that precision and recall trend follows intuition. Higher threshold results in higher precision but lower recall.
We see that .5 threshold has the best AER.

### Training on full dataset
With the result above, we trained on the full dataset with 5 iterations. We trained on following three configurations
1. Joint training with init-uniform-seen and max(f -> e)
2. Joint training with init-uniform-seen and threshold .5
3. Joint training with init-uniform-seen and threshold .35

| Model | AER      |
|-------|----------|
| 1     | .328     |
| 2     | .330     |
| 3     | **.301** |

We see that optimal threshold change when trained on full dataset. This is because there is more vocab is lower.
I have uploaded the 3rd configuration result in gradescope.
Even though threshold based alignment did better, it also seems to be sensitive, requiring adjustment of
threshold for different dataset size.

### How to run the "best" model
```
python ibm1_agreement.py -n 100000 -i 5 > alignment
python score-alignments < dice.a alignment
```



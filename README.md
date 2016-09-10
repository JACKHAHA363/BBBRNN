# Bayesian Recurrent Network Implementation

This code try to reproduce the result in this paper(https://arxiv.org/abs/1704.02798). In order to run the experienment just run:
`python main.py --BBB --cuda`
with posterior sharpening:
`python main.py --BBB --cuda --sharpen`
For the baseline result, run
`python main.py --cuda`

# Result
The baseline result is consistent with the paper, but the improvement of bayesian backprop is minor. 

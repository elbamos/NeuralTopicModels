# NeuralTopicModels

This repo contains a WIP implementation of http://nlp.cs.rpi.edu/paper/AAAI15.pdf

The NTM model is intended to work essentially as follows:

The outputs are W1 and lt. 

* W1 -- An embedding showing, approximately, the distribution of topics over each document. 
* lt -- An embedding showing, approximately, the distribution of topics over each term. 

W1 and lt are calculated as follows:

### Pre-training

`le` R^{n_terms * 300}   is created mapping each term to the sum of word2vec embeddings of grams within the term.  (Because a term may be an n > 1 gram.)

`le` is mapped to `lt` by sigmoid activation of weight matrix W2. 

W2 R^{300 * n_topics}   is pre-trained by auto-encoding `le` against itself.  

W1 R^{n_docs * n_topics}  is pre-trained so that each document's embedding is the sum of the pre-trained `lt` activations for the terms contained in the document. 

### Fine-tuning

Each example is a combination of (a) a term, (b) a document containing the term, and (c) a random document that does not contain the term.

`ld+` and `ld-` R^{n_topics} are the softmax activation of W1 for the positive and negative documents, respectively. 

`ls+` and `ls-` are calculated.  Each is a scalar representing the predicted probability that the term would appear in the positive and negative documents, respectively. 

`ls` = `lt` dotproduct ld'`

The cost is then calculated as:

c(g, d+, d-) = max(0., 0.5 - `ls-` + `ls+`)

Thus, the algorithm wants to find (a) an embedding for the documents, and (b) a weight matrix mapping the term word2vec embeddings to topics, where given any term and a document containing the term, the predicted probability that the term would appear in the document is at least 0.5 greater than the predicted probability the term would appear in a randomly chosen document that does not contain it. 

### Issues

In my testing, with a 1M document, 30000 term corpus with ~ 10M total grams, aiming for 128 topics, I found that the W1 and W2 both consistnetly converge toward 0, usually after only 1 epoch.  

* Theory:   In debugging, I observed that the calculation of `ls-` - `ls+` tends to be around 1 e-8.  Adding 0.5, I suspect that a 32-bit float would represent the number only as 0.5, losing precision.  

I suspect that this is then causing every loss to be calculated as 0.5, and confusing the gradients for W1 and W2. 

Experiments:

*  Pretraining W1 & W2 by ignoring the 0.5 separation:  I tried this cost function:

c(g, d+, d-) = mean(binary_crossentropy(`ls+`, 1), binary_crossentropy(`ls-`, 0))

Result:  Convergence toward zero. 

*  Gradient enhancement:  on the theory that the problem was underflow, I tried this cost function:

c(g, d+, d-) = max(0., 0.5 +  max(n_docs, 10 ** epoch) * (`ls-` - `ls+`))

Result: Convergence toward zero

*  Normalization:  To try to force the weights on W2 and W1 to not approach zero, I tried: 

Modifying the formula for `lt` to softmax(softplus(`le`)).  This is intended to prevent `lt` from approaching zero while encouraging greater differentiation of topics and terms. 

Enforcing a unit-norm constraint on W1, the document-topic embedding matrix.  

Result:  In testing

*  Optimization:  I experimented with `adadelta` (which I have found very effective) intead of vanilla SGD. 

Result:  W1 converged toward zero much more quickly than with vanilla SGD. 


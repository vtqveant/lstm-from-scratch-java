## LSTM From Scratch

This is an LSTM-based encoder-decoder architecture written in Java from scratch. The project has virtually no 
dependencies apart from JAMA, a concise implementation of a matrix data structure and basic matrix operations, 
and Jung library used for visualization. Computation is done on CPU.

This project was used for research and education. It demonstrates a number of important concepts of neural networks:

  * Automatic differentiation, a technique of efficient evaluation of partial derivatives at a given point. 
This is the core component of all modern deep learning frameworks such as Tensorflow and PyTorch.  
  * Computation graph, an approach of structuring complex computation in such a way that allows to implement
both evaluation of mathematical expressions during a forward pass through the graph and collection of gradients
during a backward pass. By adding a simple session-based mechanism we can adjust parameters and thus turn the 
computation graph into a full-fledged gradient descent optimization engine.
  * A Long-Short Term Memory cell, a seq2seq RNN architecture and an attention mechanism.

Finally, we create a fully working example of an RNN-based encoder-decoder architecture. 
We train on a small dataset of sequences of balanced parentheses of four different kinds &mdash; a 4Dyck language, which
is context-free. The target is to predict whether a sequence is in the language or not. 
This task models dependency structure in sentences in a natural language, and the question if RNN is capable of 
learning such languages is an important research topic in NLP.


### References:

[1] TBD

### NB: THIS IS A PORT FROM ANOTHER REPO, NOT YET FULLY FUNCTIONAL

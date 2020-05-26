# language_modelling
Generating text with one author style with LSTM, Transformer and pre-trained GPT-2

In this repository there are 4 models which can be used to generate text. Results are not great because inusfficient computational resources (even with GPU on colab notebook).

**1. LSTM**<br>
A LSTM that take as input a sequence of word encoded with Glove Word2Vec Model and output a vector on the Word2Vec space. Trained on *The Lord of The Rings*. Bad results

**2. LSTM with embeddings**<br>
A LSTM that take as input a sequence of word encoded with an index, before the LSTM layers the input goes through an embedding layer with weights taken from Glove model. The output can be seen as a probability distribution of the next index (word). Trained on *The Lord of The Rings*. Bad/Mediocre results.

**3. Transformer**<br>
A Transformed model trained on LOTR with positional encodings. Discrete results.

**4. GPT-2**<br>
An attempt to do tranfer learning on the pre trained GPT2 model from *huggingface*. The transfer learning procedure can be done on *LOTR*, *Trump tweets*, *Salvini Tweets*. Good results.

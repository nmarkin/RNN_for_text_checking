# RNN_for_text_checking

This repository demonstrates parts of the research on how RNNs can be used to check coherence of words in sentences or texts.
The main idea was to remove word from a sentence, generate predictions for replacement and check if the word removed would be in top-5, top-10 predictions. 

Different combinations of word-vectors representation, parameters of LSTM and approaches to text generation were tried to find best one possible.
This code shows only some examples of variations tested during the experiment. 

All calculations were done on text of "Republic".

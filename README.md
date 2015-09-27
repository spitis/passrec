# passrec
Basic proof of concept of a neural network recognizing the way a password is typed

Idea is to see if it can learn the particular way you type your password using a neural network. In theory, it would then differentiate between the true "owner" of the password and a false enteror who happened to steal the password. Of course, a lot more data would be necessary to get better results, and it's unclear that there is a practical function to use here (false negatives are annoying, and false positives make this meaningless... so it would have to be super accurate, but there is likely a lot of noise here even with large sample sizes).

That said, try training a password like "supreme" but think in your head "sup-re-me" while you train the neural network. My friends were unable to crack it, even though it was trained on but 15 samples, and I could reliably enter the password to get in.

Uses a simple javascript implementation of a neural network with a single hidden layer. Implementation is based off of the one taught in Andrew Ng's ML class, but rewritten from scratch in JS, using NumericJS (http://www.numericjs.com/index.php) for the matrix algebra. 

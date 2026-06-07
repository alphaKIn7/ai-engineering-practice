# ML Sprint — Tutor Notes (v2, beginner-friendly edition)

A rewrite of the original ML Sprint notes with a deliberate structure. Every concept follows the same pattern:

1. **Real-world analogy** — what this is like in everyday life
2. **What's actually happening** — the concept in plain technical language
3. **Code/math that connects** — making it concrete

This isn't dumbing it down. It's building the same depth from a foundation you can intuitively check against the world. You'll end up with the same interview-ready understanding, but with mental models that actually stick.

**How to use this:**
- One day at a time. Don't speed-run.
- Read the analogy. Then close your eyes and try to predict what the technical explanation will say. This prediction step matters more than reading.
- Type out the code yourself. Don't copy-paste.
- The 22 interview questions at the end remain the success metric.

**Prerequisites:** Phase 1 weeks 1-4 complete. Comfortable with Python, FastAPI, embeddings, basic RAG.

---

# Day 1: ML Fundamentals — Learning Without Rules

## The analogy

Imagine you're teaching a child to recognize dogs. You don't hand them a checklist: *"4 legs, fur, barks, wags tail."* That checklist would fail on a poodle, fail on a hairless cat in a costume, and miss the point entirely. Instead, you point at things. *"That's a dog. That's a dog. That's not a dog — that's a wolf, see the difference?"*

After a few hundred examples, the child can recognize dogs they've never seen before. They didn't learn rules. They learned a *pattern*. If you asked them *"why is this a dog?"* they couldn't fully tell you — they just know.

This is exactly what machine learning is. Instead of giving the computer rules, you give it examples and let it figure out the pattern.

## What's actually happening

Traditional programming: you write rules → computer follows them → output.

Machine learning: you provide examples → computer figures out rules → output.

More formally, ML is the practice of finding a **function** `f(x) → y` from examples of inputs and outputs, rather than from someone manually writing the function.

Three families of ML, differing by what you give the computer:

**Supervised learning**: you give it inputs AND the correct outputs. Like teaching the child with labeled pictures: "this is a dog, this is a cat."

**Unsupervised learning**: you give it only inputs, no labels. The computer finds patterns on its own. Like dumping a thousand photos on a table and asking the child to sort them into piles however makes sense — they'd probably end up with "things with fur" vs "things without."

**Reinforcement learning**: you give it an environment and a reward signal. Like teaching a dog tricks — treats when they sit, nothing when they don't. They figure out what behavior leads to treats.

LLMs are mostly trained via **self-supervised** learning, a clever subset of supervised: the label is generated from the input itself. The input is "The cat sat on the ___" and the label is "mat" — taken from the same source text. No human had to label anything. This is why LLMs can train on the entire internet.

## The data split

Here's a crucial piece. You don't use ALL your examples to teach the model. You hold some back.

**The analogy**: imagine a student preparing for an exam. They have:
- A textbook with practice problems (training data)
- A set of practice quizzes with solutions to check progress (validation data)
- The actual exam they haven't seen yet (test data)

If the student peeks at the actual exam while studying, their "study score" lies. They look prepared, but only because they memorized the exam, not because they learned the subject. The exam stops being a real test.

In ML, this peeking is called **overfitting**, and it's the #1 failure mode of poorly-built systems.

**What's happening technically**: you split your data into three buckets — usually 70% train, 15% validation, 15% test. You train on the training set. You check progress on validation. You touch the test set only once, at the very end, for an honest final score.

For your LLM evals (back in Phase 1 Week 6, and Phase 2 Week 5): never iterate against your full eval set. Have a "dev" subset you tune against, and a "held-out" subset you check rarely. Same principle.

## Overfitting vs underfitting — two ways to be wrong

**Analogy for overfitting**: a student who memorized every practice problem perfectly but can't solve any new problem on the exam because the wording is slightly different. They memorized; they didn't learn.

**Analogy for underfitting**: a student who didn't study enough and gets even the practice problems wrong. They never built understanding to begin with.

**What this means technically**:
- **Underfitting** = the model is too simple to capture the pattern. Wrong on training data, wrong on new data. Solution: more complex model, or more features.
- **Overfitting** = the model memorized training data noise instead of learning the real pattern. Right on training data, wrong on new data. Solution: more data, simpler model, regularization, or dropout.

The goal is **just right** — complex enough to capture the real pattern, not so complex that it memorizes noise.

## Metrics — picking the right ruler

Imagine you've built a system that flags fraudulent expense receipts. You ran it on 1000 receipts: 50 are actually fraud, 950 are legit. Your model flagged 60 as fraud. Of those: 40 were actually fraud, 20 were false alarms. It missed 10 real frauds.

You'd think: how do I summarize how good this is? With one number?

**Accuracy** = "what fraction of all predictions did I get right?"
= (40 correct frauds + 930 correct legits) / 1000 = **97%**.

Sounds great. But notice: a model that just predicts "not fraud" for everything would get **95% accuracy** — and catch zero fraud. Accuracy is misleading when classes are imbalanced.

**Precision** = "when I said fraud, was I right?"
= 40 / 60 = **67%**.
When this model flags something, it's correct 67% of the time. Useful when false alarms are expensive (you don't want to block legitimate transactions).

**Recall** = "of all the fraud out there, what fraction did I catch?"
= 40 / 50 = **80%**.
Useful when missing real fraud is expensive (you don't want fraud slipping through).

**F1** = a balanced single number combining both
= 2 × (precision × recall) / (precision + recall) = **0.73**.
The "harmonic mean" — it punishes imbalance. If precision is 0.9 but recall is 0.1, F1 ≈ 0.18, telling you the system is broken.

The interview lesson: when someone asks *"how would you evaluate this?"*, never say "accuracy" without thinking about class balance and the cost of each type of error first.

## Loss functions — telling the model it's wrong

**Analogy**: when you teach the child to recognize dogs, you correct them. *"No, that's a cat."* The correction has to convey not just "wrong" but ideally *how* wrong. "You said dog but it's clearly a fish — that's a very wrong answer" vs "you said poodle but it's a labradoodle — close, just slightly off."

**Technically**: a loss function is a number that says how wrong a model's prediction was on a given example. Training = adjust the model to minimize this number across all examples.

Two loss functions you must know:

**Mean Squared Error (MSE)** — for predicting numbers. If true value is 10 and you predicted 7, the error is 3, and squared error is 9. Squaring punishes big errors hard. A prediction off by 10 is 100x worse than off by 1.

**Cross-entropy loss** — for predicting categories or distributions. Measures how *surprised* the model is by the correct answer. If the model said "this is 95% likely to be a dog" and it is a dog, low surprise, low loss. If the model said "this is 5% likely to be a dog" and it is a dog, high surprise, high loss.

LLMs use cross-entropy. The model predicts a probability distribution over the next token; cross-entropy measures how surprised it is by the actual next token. Training = make the model less surprised over time.

## Gradient descent — finding the bottom of the valley

**Analogy**: imagine you're dropped on a foggy mountain at night with a flashlight. Your goal: reach the bottom of the valley. You can't see far. What do you do?

You shine the flashlight, see which direction is downhill, take a step that way. Look again. Step again. Repeat until the ground is flat — you've reached a valley.

This is exactly gradient descent. The "mountain" is the loss function. Lower altitude = lower loss = better model. The model's parameters are your position on the mountain. The flashlight is calculus — specifically, the **gradient**, which is the math for "which direction is steepest uphill?" You go the *opposite* direction.

**Technical version**:
1. Start with random model parameters (random position on the mountain)
2. Compute the loss (current altitude)
3. Compute the gradient (steepest uphill direction)
4. Update parameters in the opposite direction, by a small amount (one step downhill)
5. Repeat thousands or millions of times

The step size is called the **learning rate**. Too big → you overshoot the valley and bounce around. Too small → you take forever, or get stuck in a small dip that isn't the real bottom.

## Code that ties it together: linear regression in 20 lines

The simplest ML model: a straight line, `y = wx + b`. We have data points and want to find the best `w` (slope) and `b` (intercept). Gradient descent in action.

```python
import numpy as np

# Generate fake data: y = 2x + 1 + a little noise
np.random.seed(0)
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Start with random guesses
w = np.random.randn(1, 1)
b = np.random.randn(1)
learning_rate = 0.01

# Training loop
for step in range(1000):
    # Make predictions with current parameters
    pred = X @ w + b
    
    # How wrong are we? (mean squared error)
    loss = ((pred - y) ** 2).mean()
    
    # How should we adjust w and b to reduce loss?
    # (This is the gradient — calculus tells us the direction)
    grad_w = (2 * X.T @ (pred - y)) / len(X)
    grad_b = (2 * (pred - y)).mean()
    
    # Step downhill
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    if step % 100 == 0:
        print(f"step {step}: loss={loss:.4f}, w={w[0,0]:.3f}, b={b[0]:.3f}")
```

Run this. You'll watch `w` converge to ~2 and `b` to ~1. The model started with random parameters and figured out the correct ones from the data alone. That's machine learning at its most basic.

Every model in this sprint — from a 70-billion-parameter transformer to a simple classifier — is doing this at heart. Compute predictions. Measure how wrong. Take a small step toward "less wrong." Repeat.

The rest is just bigger models and cleverer architectures.

## Quiz before moving on

Try to answer these in 30-60 seconds each, out loud:

1. What's the difference between supervised, unsupervised, and reinforcement learning?
2. Why do we split data into train/validation/test?
3. What's overfitting? How do you detect it?
4. Walk me through gradient descent in plain English.
5. Why is accuracy a misleading metric for imbalanced classes?

If any felt fuzzy, re-read that section.

---

# Day 2: Practice Day

This day is shorter. Focus on actually running the linear regression code from Day 1 and modifying it. Concrete suggestions:

1. **Change the learning rate**. Try 0.001 (too small — slow convergence). Try 1.0 (too big — bounces around). Try 0.1 (sweet spot). This builds intuition for why learning rate matters.

2. **Add more dimensions**. Make X have 3 features instead of 1. Generate `y = 2*x1 + 3*x2 - x3 + 1`. Train and watch the weights converge. This is exactly how real ML scales — you don't change the algorithm, you change the dimensions.

3. **Plot the loss over time**. Use matplotlib. You should see it drop quickly, then plateau. That's the model "learning" then "converging."

4. **Break it on purpose**. Add an outlier — a single data point with `y = 1000`. Watch how it pulls the line. This is why outlier handling matters in real ML.

Doing this for 1-2 hours builds more intuition than reading three more pages.

---

# Day 3: Neural Networks — From Lines to Curves

## The analogy

Linear regression from Day 1 can fit a straight line. But the world isn't lines. Real patterns curve, twist, and branch.

**Analogy**: imagine you're trying to describe a winding mountain road on a map. A ruler (linear regression) can't do it — roads curve. What if you draw the road as many short straight segments, each at a different angle, joined together? Enough short straight segments and you can approximate any curve, any winding road, any shape.

A neural network is exactly this: many simple "straight line" pieces, stacked and connected, that together approximate any function you can imagine.

There's a famous math result called the **Universal Approximation Theorem** that proves this rigorously: with enough simple units, a neural network can approximate any continuous function to any precision you want. The mountain road analogy is the intuition; the theorem is the math.

## What's actually happening

A neural network is built from stacking two things:

**1. A linear transformation**: `y = Wx + b`. Just like linear regression. Take inputs, multiply by weights, add a bias. Outputs go into the next layer.

**2. A nonlinear activation function**: a small function applied to each output. The most common is **ReLU**: `f(x) = max(0, x)` — keeps positive numbers as-is, turns negative numbers into zero.

Why the nonlinearity? Here's a math fact: if you stack only linear transformations, the whole stack is still just one linear transformation. Two straight lines combined are still a straight line. The nonlinearity is what lets the network learn curves.

The mental picture: each layer transforms the data into a slightly more abstract representation. Early layers see raw input. Middle layers see combinations. Late layers see high-level concepts.

For an image classifier:
- Layer 1: detects edges (vertical, horizontal, diagonal)
- Layer 2: detects shapes (circles, corners)
- Layer 3: detects parts (eyes, wheels, leaves)
- Layer 4: detects objects (faces, cars, trees)

This is called **hierarchical representation learning**. It happens automatically through training — no one tells the network what to look for. It figures out useful features on its own.

For language (an LLM):
- Early layers: syntax, grammatical patterns
- Middle layers: word meanings, sentence structure
- Late layers: discourse, intent, world knowledge

## Why "deep"?

A shallow network with many neurons can theoretically learn anything (that's the Universal Approximation Theorem again). But in practice, deep networks learn complex patterns far more efficiently.

**Analogy**: imagine building a complex Lego model. You could do it from scratch every time, snapping together individual bricks (shallow approach). Or you could build small sub-assemblies first — a wheel, a door, a window — then combine them into bigger assemblies, eventually into the final model (deep approach). The deep approach is more efficient because each layer of work reuses the layer below.

Neural networks work the same way. Deep architectures let later layers reuse abstractions built by earlier ones, instead of having to learn everything from raw pixels or raw tokens.

## Code that connects it

Here's a complete 3-layer neural network, no PyTorch needed:

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Initialize random weights — these will be learned during training
# Layer 1: takes 784 inputs (e.g., 28x28 pixel image), outputs 128 features
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)

# Layer 2: 128 → 64
W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)

# Layer 3: 64 → 10 (e.g., classification into 10 categories)
W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros(10)

def forward(x):
    # x is shape (batch_size, 784) — flattened images
    h1 = relu(x @ W1 + b1)   # First hidden layer
    h2 = relu(h1 @ W2 + b2)  # Second hidden layer
    out = h2 @ W3 + b3       # Output layer (no activation — that comes from softmax later)
    return out
```

That's it. That's a neural network. You could train this to classify MNIST handwritten digits with maybe 50 more lines of code (computing softmax, cross-entropy loss, and gradient descent — same loop as Day 1's linear regression, just more parameters).

The architecture is shockingly simple. The "magic" comes from scale (more parameters), data (lots of examples), and architectural tricks (which we'll cover for transformers).

## Backpropagation — the chain rule applied to networks

**Analogy**: imagine you're managing a chain of factories. The final factory produces a defective product. To fix it, you need to figure out which factory in the chain made the original mistake. You trace backwards: check the final factory, then the one before, then the one before that. Each factory tells you "given what you sent me, here's how my output changed" — and you propagate the responsibility back.

Backpropagation is this exact idea applied to neural networks. The forward pass produces a prediction; we compute the loss. To improve, we need to know how much each weight in the network contributed to the loss. We compute this by working backward layer by layer, applying the **chain rule** of calculus.

You don't need to derive the math. You need to be able to say in an interview:

*"Backpropagation is the chain rule applied through the network's layers, working from output back to input, to compute how much each parameter contributed to the final loss. Those contributions are the gradients, and we use them to update the weights via gradient descent."*

That answer is enough. Anyone asking you to actually derive backprop math is testing a different skill set than AI engineer interviews care about.

## Why training is hard

A few classic problems and the fixes:

**Vanishing gradients**: when gradients propagate back through many layers, they can shrink toward zero. Earlier layers barely update. The fix: ReLU (instead of older activations like sigmoid), batch normalization, and residual connections (we'll see these in transformers).

**Exploding gradients**: the opposite — gradients grow huge. The fix: gradient clipping (cap them at some value).

**Local minima and saddle points**: the loss landscape is bumpy. Gradient descent can get stuck in small dips that aren't the real minimum. The fix: momentum-based optimizers like Adam, which build velocity and roll past small obstacles.

**Overfitting** (from Day 1): network memorizes training data instead of learning patterns. The fix: dropout (randomly disable some neurons during training, forcing redundancy), more data, regularization.

These problems and fixes are real, but in modern deep learning frameworks (PyTorch, TensorFlow), most are handled by default. You should know they exist; you don't need to implement the fixes yourself.

## Quiz before moving on

1. Why do neural networks need nonlinear activation functions?
2. What is hierarchical representation learning?
3. Explain backpropagation in plain English.
4. What's the vanishing gradient problem? How is it mitigated?
5. Why use ReLU instead of sigmoid?

---

# Day 4: Embeddings — Meaning as Numbers

You touched embeddings in Phase 1 Week 3 from a practical angle. Now we go deeper conceptually.

## The analogy

Imagine you're organizing books in a library. You don't shelve them randomly — you put related books near each other. Cookbooks in one section. History next to biography. Physics next to math. The *location* of a book encodes information about what it's about.

Now imagine a library so big and sophisticated that you can describe any book's location with 1500 numbers — its latitude, longitude, altitude, depth, and 1496 other dimensions. The genre dimension, the difficulty dimension, the era dimension, the mood dimension. Two books on similar topics end up at nearby coordinates in this 1500-dimensional library.

An **embedding** is exactly this: a list of numbers (a "vector") that represents an item's location in a high-dimensional space where similar things end up near each other.

## What's actually happening

When an LLM (or an embedding model) sees the word "dog," it converts it into a long list of numbers — say, 1536 of them. This list is the embedding for "dog."

Before training, those numbers are random — the model has no idea where to "shelve" the word. During training, the model adjusts the numbers so that "dog" ends up near "puppy" and far from "refrigerator." Why? Because doing so helps the model predict surrounding words correctly. Words that appear in similar contexts naturally end up with similar embeddings.

The famous demonstration:

```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```

This works because the model has implicitly learned that "king" and "queen" differ in the same way "man" and "woman" differ — by gender. No human told it. It discovered the structure of language from data alone.

This is the deep idea: **embeddings are learned representations**. They're not programmed; they emerge from training. They capture meaning, relationships, similarity — all as math.

## Similarity, measured by angle

If "dog" is at coordinates [0.2, -0.5, 0.8, ...], how do we measure how similar it is to "puppy" at coordinates [0.21, -0.49, 0.81, ...]?

**Cosine similarity** measures the angle between two vectors. Vectors pointing in similar directions → angle near 0 → cosine near 1 → very similar. Vectors pointing in opposite directions → angle near 180 degrees → cosine near -1 → opposite. Vectors at 90 degrees → cosine 0 → unrelated.

**Analogy**: imagine two arrows pointing out from the origin. If they point in roughly the same direction, they're "similar." If they point in totally different directions, they're not. Cosine similarity is the math for measuring "same direction" in any number of dimensions.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Made-up embeddings
dog = np.array([0.2, -0.5, 0.8, 0.1])
puppy = np.array([0.21, -0.49, 0.81, 0.09])
refrigerator = np.array([-0.7, 0.3, -0.2, 0.6])

print(cosine_similarity(dog, puppy))           # ~0.99 — very similar
print(cosine_similarity(dog, refrigerator))    # ~-0.5 — dissimilar
```

In your RAG system, this is exactly what's happening when you "search for similar chunks." You convert the query into an embedding, then find chunks whose embeddings have the highest cosine similarity. The vector database (Qdrant) does this efficiently at scale.

## Embeddings as the universal currency

Modern AI systems use embeddings to represent everything: text, images, audio, video. Once everything is a vector, you can:

- Search across modalities (text query → image result, because they're in the same space)
- Compare similarity of anything to anything
- Cluster items by meaning
- Do arithmetic (king - man + woman = queen)

This is why "embedding model" is now a product category. OpenAI sells embedding APIs. Hugging Face hosts thousands of embedding models. Every modern search system uses them.

## Quiz before moving on

1. In your own words, what is an embedding?
2. Why does cosine similarity work as a similarity measure?
3. How do embeddings "know" that king and queen are related?
4. Can you compare an OpenAI embedding to a Hugging Face BGE embedding? Why or why not?
5. Why are embeddings called "learned representations"?

---

# Day 5: Transformers Part 1 — Why Attention Won

This is the most important day in the sprint. Take your time.

## The analogy

Imagine you're reading a mystery novel. On page 200, the detective says: *"He couldn't have done it. Remember what she said at dinner?"*

To understand this sentence, your brain instantly reaches back through the book — what dinner? Which "he"? What did she say? You don't re-read every page; you selectively recall specific moments that matter for this sentence. Your attention is dynamically focused on relevant earlier content.

This is exactly what **attention** does in a transformer. Each word can directly "look at" any earlier word and decide how much it matters for understanding the current word.

Before transformers (around 2017), language models used **RNNs** — they read text one word at a time, carrying a "memory" forward. But this memory was a single fixed-size vector trying to summarize everything seen so far. By page 200, the dinner detail had been blurred together with hundreds of other details into one foggy summary.

Attention fixed this. Each word can selectively retrieve any earlier word, with full clarity, on demand.

## What's actually happening

The famous 2017 paper "Attention Is All You Need" introduced the transformer architecture. The mechanism:

For each word in the sequence, the model generates three vectors:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I offer?"
- **Value (V)**: "What's my actual content?"

**Analogy**: imagine you're at a library, searching for books on a topic.
- Your **query** is the topic you want.
- Each book has a **key** — its title and subject — that the librarian uses to match against your query.
- Each book has a **value** — its actual content — which is what you'll read once you find it.

To decide which books are relevant, you compare your query against every book's key. High match → that book is relevant. Then you read the values of the most relevant books, weighted by how relevant they were.

In a transformer, every word does this for every other word, all at once. Word "she" asks "what am I about?" Every other word's key is compared. Words like "Maria" or "the witness" have keys that match — high attention scores. Words like "the" have low-matching keys — low scores. Word "she" then retrieves a weighted blend of values from all words, mostly the ones with high scores.

## The math, but lightly

The attention calculation, step by step:

1. Each word has its Q, K, V vectors (learned during training).
2. To compute "how much should word A attend to word B," take the dot product of A's query with B's key: `Q_A · K_B`. High dot product = high relevance.
3. Do this for all pairs. You get a matrix of scores.
4. Apply **softmax** to each row, turning scores into probabilities that sum to 1. Now you have a clean "how much attention does A pay to each word?" distribution.
5. Each word's output = weighted sum of all values, weighted by these probabilities.

In code:

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))  # stable softmax
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    """
    Q, K, V are each shape (seq_len, d_model).
    Returns shape (seq_len, d_model).
    """
    d_k = K.shape[-1]
    
    # Step 1: compute attention scores — how much each word should look at each other word
    scores = Q @ K.T  # shape: (seq_len, seq_len)
    
    # Step 2: scale by sqrt(d_k) — prevents the softmax from saturating when d_k is large
    scores = scores / np.sqrt(d_k)
    
    # Step 3: convert scores to attention weights via softmax
    weights = softmax(scores, axis=-1)
    
    # Step 4: weighted sum of values
    output = weights @ V
    
    return output
```

That's the heart of a transformer. Everything else in the architecture supports this operation.

**Why divide by √d_k?** When the dimension is large, dot products tend to be large too, which makes softmax saturate (the highest score becomes ~1, everything else becomes ~0). Scaling by √d_k keeps the softmax in a sensible range. This is called **scaled dot-product attention**.

## Multi-head attention

One round of attention is good. Multiple rounds in parallel is better.

**Analogy**: when you read a sentence, your brain processes it in multiple ways simultaneously — syntactic structure, semantic meaning, emotional tone, factual content. You're not doing one analysis; you're doing several, in parallel.

In a transformer, instead of computing one attention pattern, you compute 8-32 of them in parallel, each with its own learned Q, K, V projections. Each "head" can specialize: one might track grammar, another semantics, another long-range dependencies. The outputs are concatenated and projected back to the model's dimension.

This is **multi-head attention**. Conceptually simple, hugely effective in practice.

## Causal masking — looking back only

For a decoder-only model (which is what every modern LLM is — GPT, Claude, Llama all decoder-only), there's one more wrinkle. The model is trained to predict the next word. So when it's looking at word position 5, it shouldn't be allowed to "cheat" by attending to words at positions 6, 7, 8 — those are the future.

**Causal masking** sets the attention scores for "future" positions to negative infinity before the softmax, which makes their attention weight zero. Each word can only attend to itself and earlier words.

## Quiz before moving on

1. Why did transformers win against RNNs/LSTMs?
2. Explain Q, K, V in your own words.
3. Why divide by √d_k?
4. What does multi-head attention give you that a single attention head doesn't?
5. What's causal masking and why is it needed?
6. Walk through what happens for each word during one attention layer.

---

# Day 6: Transformers Part 2 — The Full Architecture

Day 5 covered attention. Now we put the whole transformer together.

## The transformer block

A transformer is built from stacking identical **blocks** on top of each other. Each block has:

1. **Multi-head self-attention** — what we covered Day 5
2. **Layer normalization** — stabilizes training (technical detail; just know it's there)
3. **A feed-forward network** — a small MLP (just a 2-layer neural net) applied to each position
4. **Residual connections** — adds the input back to the output of each sub-layer

**Analogy for residual connections**: imagine you're editing a document. Instead of rewriting the entire document each time, you write a small set of edits that get added to the original. The original survives; the edits modify it. This is what residual connections do — each layer learns a "delta" to add to its input, rather than transforming from scratch. This is what allows extremely deep networks (100+ layers) to train successfully.

In pseudocode:

```python
def transformer_block(x):
    # Attention sub-layer with residual
    attention_out = multi_head_attention(layer_norm(x))
    x = x + attention_out
    
    # Feed-forward sub-layer with residual
    ffn_out = feed_forward(layer_norm(x))
    x = x + ffn_out
    
    return x
```

Stack 12-100+ of these blocks. Each one refines the representation further. Final block's output gets projected to vocabulary size and softmaxed — giving you a probability distribution over the next token.

## Positional encoding — because attention is order-blind

There's a subtle problem with attention. It computes relationships between all tokens, but it has no inherent sense of *order*. "Dog bites man" and "man bites dog" would produce identical attention patterns. That's clearly wrong.

**Analogy**: imagine you're looking at words on a page, but the words are floating in random positions. You can see the relationships between concepts, but you don't know what comes first. To restore order, someone has to label each word with its position: "1st word, 2nd word, 3rd word."

In transformers, this is **positional encoding** — adding position information to each token's embedding before the first attention layer. Two main approaches:

**Sinusoidal positional encoding** (original transformer): use sine and cosine functions of different frequencies. Mathematically elegant but rarely used in modern LLMs.

**Rotary Position Embedding (RoPE)** (modern standard): instead of adding position info, *rotate* the Q and K vectors based on position before computing attention. The rotation amount depends on position, so tokens at different positions produce different attention scores even with the same content.

You don't need the math. You need to know:

- Attention is permutation-invariant (no order awareness)
- Positional encoding adds order awareness
- Modern LLMs use RoPE
- RoPE is part of why context windows can grow to millions of tokens

## Encoder vs Decoder vs Both

Three architectures, same building blocks:

**Encoder-only** (BERT, embedding models): bidirectional attention — every token sees every other. Optimized for understanding (classification, search, embeddings). Not used for generation.

**Decoder-only** (GPT, Claude, Llama, basically all chat LLMs): causal attention — each token only sees previous ones. Trained to predict next token. Generates by sampling tokens one at a time.

**Encoder-decoder** (T5, original Transformer for translation): an encoder reads the input, a decoder generates the output, with cross-attention between them. Used for translation, summarization. Less common in modern LLMs.

Modern LLMs are almost all decoder-only. Why? Simpler architecture, scales better, can do everything via prompting (instruction → output, treating the whole thing as one continuous sequence).

## The training objective — next token prediction

This is the simplest, most counterintuitive thing about modern LLMs: they're trained on one task only.

Given a sequence of tokens, predict the next one.

That's it. Trillions of tokens, billions of parameters, one objective: predict the next token.

How does this become an AI that can write essays, answer questions, write code? Because predicting the next token *correctly* requires:

- Understanding grammar (to produce well-formed sentences)
- Understanding meaning (to produce coherent content)
- Understanding facts (to predict the right entity)
- Understanding reasoning (to predict the conclusion of an argument)
- Understanding code (to predict the right syntax and logic)

If a model can reliably predict the next token across all of internet text, it has *implicitly* learned all of those things — because they're necessary for the prediction. The simplicity is deceptive.

## Karpathy's video — required this week

Stop reading. Open Karpathy's *"Let's build GPT: from scratch, in code, spelled out"* on YouTube. Watch the whole thing. Pen and paper open.

The notes above give you the vocabulary. Karpathy gives you the muscle memory of actually building it. You need both.

When he gets to the "mathematical trick" section around the 50-minute mark, pause and try to predict what he's going to do *before* he does it. That section is the conceptual heart of attention.

## Quiz before moving on

1. Walk through what's in a single transformer block.
2. What's a residual connection and why is it important?
3. Why is positional encoding needed?
4. What's RoPE and why is it now standard?
5. Difference between encoder-only, decoder-only, and encoder-decoder. When use each?
6. Why are most modern LLMs decoder-only?
7. What's the training objective of a base LLM? Why is it powerful?

---

# Day 7: How LLMs Get Their Personality — Training Pipeline

A base LLM (just trained on next-token prediction) is weird. It doesn't follow instructions. It just continues text. If you type "What's the capital of France?" a base LLM might continue with more questions, or with completely unrelated text. It's autocomplete, not an assistant.

How do we get from there to Claude or ChatGPT?

## The three-stage analogy

Imagine teaching a brilliant but socially awkward person to be a great consultant. They have all the knowledge. But they:
- Don't know how to format answers helpfully
- Don't know when to refuse inappropriate requests
- Sometimes ramble or go off-topic
- Have no sense of which answer style suits which client

You'd train them in three stages:

**Stage 1: Read everything.** Have them read every book in the library. Now they know facts. (= **Pretraining**)

**Stage 2: Watch good consultants work.** Have them shadow experienced consultants, observing thousands of "question → ideal answer" pairs. They learn the *style* of being helpful. (= **Supervised Fine-Tuning / Instruction Tuning**)

**Stage 3: Get critiqued.** Have them give answers, and have humans rate which answers are better. Then optimize them to produce more of the kind of answers humans prefer. (= **Reinforcement Learning from Human Feedback / RLHF**)

After all three, you have a model that knows things, follows instructions, and is calibrated to human preferences. That's a modern chat LLM.

## What's actually happening

**Pretraining**: a randomly-initialized transformer is trained on next-token prediction across trillions of tokens (internet, books, code, papers). Cost: $10M-$100M+ in compute. Result: a base model that "knows a lot" but isn't useful for chat.

**Instruction Tuning (Supervised Fine-Tuning, SFT)**: take the base model. Train it further on a dataset of `(instruction, ideal response)` pairs written by humans. Same loss function (cross-entropy), much smaller dataset (maybe 10K-100K examples), much smaller compute cost. Now the model follows instructions.

**RLHF**: 
- Take the SFT model. For each prompt, generate multiple candidate responses.
- Have humans rank the candidates: A > B > C > D.
- Train a separate **reward model** to predict human rankings: given two responses, which would a human prefer?
- Use reinforcement learning to fine-tune the SFT model: generate responses, have the reward model score them, update the model to produce higher-scoring responses.

This last stage is what makes Claude feel like Claude and ChatGPT feel like ChatGPT. Each has been preference-trained differently.

**A modern variant: DPO (Direct Preference Optimization)** does the same end goal as RLHF but in one step, without an explicit reward model. Simpler and increasingly common. Worth knowing the name.

## Why LLMs hallucinate — the inconvenient truth

Pretraining objective: predict the most *probable* next token.

NOT: predict the most *true* next token.

There's no fact-checker in the training loop. The model learns to produce text that *looks like* what comes next, based on patterns in its training data. When asked about something rare or obscure, the model fills in *plausible-sounding* tokens — not necessarily true ones.

Three distinct causes of hallucination (interview answer):

1. **No grounding**: the model has no external truth source during generation. It can't look things up. It generates from internal weights.

2. **Training objective mismatch**: predicting plausible tokens isn't the same as predicting true tokens. The model is optimized for plausibility, not truth.

3. **Uncalibrated confidence**: the model has no introspection about what it knows vs doesn't. It generates with the same confidence whether it's stating a well-known fact or fabricating.

This is why RAG works — it grounds the model in retrieved real facts. This is why fine-tuning for honesty works — it teaches the model to refuse rather than confabulate. This is why citations help — they make the source verifiable.

## Sampling — how the model picks the next word

The model produces a probability distribution over the entire vocabulary (~50K tokens) for what comes next. We have to pick one. Several strategies:

**Greedy**: always pick the highest-probability token. Boring, repetitive. Often gets stuck in loops.

**Temperature sampling**: divide all probabilities by a "temperature" before normalizing. T < 1 sharpens (more deterministic). T > 1 flattens (more random). T = 0 ≈ greedy.

**Top-k sampling**: only sample from the top K most likely tokens. Hard cutoff.

**Top-p (nucleus) sampling**: sample from the smallest set of tokens whose cumulative probability exceeds P (e.g., 0.9). Adaptive cutoff — usually preferred over top-k.

Practical settings:
- Temperature 0 for deterministic tasks (classification, extraction, code generation)
- Temperature 0.7, top-p 0.9 for chat/creative tasks
- Temperature 1.0+ for explicit randomness/brainstorming

## Quiz before moving on

1. Walk through the three stages of modern LLM training.
2. What's RLHF? What does the reward model do?
3. What's DPO and why is it preferred over RLHF in many cases?
4. Why do LLMs hallucinate? Name three distinct causes.
5. What's the difference between temperature, top-k, and top-p sampling?
6. When would you use temperature 0 vs 0.7?

---

# Day 8: Context Windows and the KV Cache

## The analogy

Imagine you're a translator. Every time someone gives you a new sentence to translate, you have to re-read the entire conversation from the beginning to keep context.

That's how an LLM works *naively*: every time it generates one new token, it has to re-process the entire prompt + every token it's generated so far.

That's wasteful. What if you wrote down the key insights from each sentence as you went, and just referenced those notes for new sentences? Now adding one more sentence only costs you the work of processing that one sentence, not re-reading the whole conversation.

This is the **KV cache** — a notebook of intermediate computations the LLM keeps as it generates, so it doesn't redo work.

## What's actually happening

When a transformer processes a sequence, it computes Q, K, V vectors for every token. The K and V vectors for older tokens *don't change* as new tokens are added — they're a function of just that token and the tokens before it (in a causal model).

So: cache them. When generating token 1001, you don't recompute K and V for tokens 1-1000. You compute them only for token 1001, then attend over all 1001 of them.

Without KV cache: generating each new token is O(n²) — must reprocess everything.
With KV cache: generating each new token is O(n) — just process the new one.

For long sequences, this is a massive speedup. It's also why long-context inference is *memory-bound*: the cache itself takes memory proportional to context length × model dimension × number of layers. For long contexts, the KV cache can be many gigabytes.

## Context windows — why they're hard to scale

Self-attention has a fundamental computational property: it's O(n²) in sequence length. To attend across 1 million tokens, you have to compute 1 trillion attention scores (1M × 1M). The compute and memory both scale quadratically.

That's why context windows used to be 2K (early GPT) and growing to 4K → 8K → 32K → 200K → 1M+ was a big deal each time. Each jump required engineering tricks.

Modern approaches to scaling context:

**Flash Attention**: same math as standard attention, but cleverly reorganized memory accesses on the GPU to be much more efficient. Same result, dramatically less memory and time.

**Sliding window attention**: each token only attends to nearby tokens (e.g., the previous 1000). Loses long-range dependencies but makes very long contexts feasible.

**Sparse attention patterns**: each token only attends to a structured subset of others (e.g., local + global). Strikes a balance.

**State-space models (Mamba, etc.)**: an alternative architecture to attention with linear complexity in sequence length. Newer, gaining traction in 2025-2026, may eventually replace or supplement attention.

You don't need to implement these. You need to know they exist and why long context is fundamentally hard.

## Quiz before moving on

1. What does the KV cache actually cache?
2. Why does KV cache speed up generation?
3. Why is attention O(n²)? What does that mean for context length?
4. What's Flash Attention? What does it improve?
5. Why is long-context inference memory-bound?

---

# Day 9: Adapting Models — Fine-Tuning, LoRA, Quantization

This is one of the most practically important days. The decisions here come up constantly.

## The Fine-Tuning vs RAG vs Prompting Decision

**The analogy**: you've hired a new employee. They have the right general skills, but you need them to do a specific job.

Three approaches:

1. **Prompting**: give them clear written instructions for each task. Cheap. They adapt instantly. But they might forget some instructions, or misunderstand. (= **Prompting**)

2. **Lookup tools**: give them access to your company's wiki and documentation. They can look things up as needed. Knowledge stays current. (= **RAG**)

3. **Internal training**: send them to a multi-week internal training program. They internalize your processes, your style, your knowledge. They no longer need the wiki for routine work. Expensive, takes time, but produces a transformed employee. (= **Fine-Tuning**)

You'd default to prompting + lookup tools for most tasks. Send them to training only when prompting and lookup aren't sufficient — when you need the behavior baked in.

## What's actually happening — the decision tree

For any "I want my LLM to do X" problem, work through this in order:

**Step 1: Try prompting first.** Cheapest, fastest to iterate. Can solve more than people realize. Use system prompts, few-shot examples, structured outputs.

**Step 2: If prompting isn't enough, try RAG.** Best when the issue is "the model doesn't have the right knowledge." Knowledge stays current; no retraining needed. The model + your data, not a new model.

**Step 3: If RAG isn't enough, consider fine-tuning.** Best when:
- You need a specific style/format the model can't reliably produce via prompting
- You need to teach a complex skill that doesn't fit in a prompt
- You need to bake in domain expertise that's stable over time
- You need lower inference cost (a fine-tuned small model can beat a larger general one on a narrow task)

Most candidates skip directly to fine-tuning when they should be prompting better. The interview signal: be the candidate who articulates *why* you'd reach for each tool.

## Full fine-tuning is expensive

To fine-tune a 70-billion-parameter model in standard precision:
- Load weights: ~140 GB
- Compute gradients for all weights: another ~140 GB
- Adam optimizer state: ~280 GB (Adam keeps two more numbers per weight)
- Total: ~560 GB of GPU memory just to do one update step

That's beyond most setups. Even modest fine-tuning of smaller models requires substantial GPUs.

## LoRA — the trick

Empirically, fine-tuning doesn't change model weights randomly. It changes them in *low-rank* patterns. This insight inspired **Low-Rank Adaptation (LoRA)**.

**The analogy**: imagine you have a giant company spreadsheet (the model weights). To customize it for a new use case, you don't rewrite the whole spreadsheet. You add a small "patch" spreadsheet that adjusts specific cells. The original is untouched; only the patch is your custom work. The patch is tiny compared to the full spreadsheet.

**What's actually happening**: instead of updating the full weight matrix `W` (e.g., 4096 × 4096 = 16M parameters), LoRA learns a small additive update expressed as the product of two thin matrices: `A` (4096 × 8) and `B` (8 × 4096) — only ~65K parameters total. At inference, you use `W + AB`. 

You train only A and B. The original W stays frozen.

Result: train 0.1-1% of parameters, capture 90%+ of the benefit of full fine-tuning. Suddenly fine-tuning 70B models becomes practical on a single consumer-grade GPU.

In an interview: *"LoRA is parameter-efficient fine-tuning that exploits the empirical observation that fine-tuning updates are low-rank. Instead of updating the full weight matrix, you learn a small low-rank decomposition that's added to it. You typically train 0.1-1% of parameters and get most of the benefit, with vastly less compute and memory."*

## QLoRA — taking it further

**QLoRA** combines LoRA with **quantization**: the base model is loaded in 4-bit precision (rather than 16-bit), and only the LoRA adapters are in higher precision. This further reduces memory.

Result: people fine-tune 70B models on a single 24GB consumer GPU. Five years ago this would have been considered impossible. The combination of LoRA + quantization made fine-tuning democratized.

## Quantization — making models smaller

**The analogy**: imagine a photograph. The original is 24-bit color — millions of distinct shades. You can compress it to 8-bit (256 colors) and it still looks recognizable, just slightly less smooth. Compress to 4-bit (16 colors) and it's clearly degraded but still recognizable. Quantization is the same idea applied to model weights — reduce precision to save memory and speed up computation, accepting some quality loss.

**What's actually happening**: models are normally stored as 16-bit floating-point numbers (fp16 or bf16). Each weight takes 2 bytes. A 20B parameter model = 40 GB just for weights.

Quantization stores them in lower precision:

- **fp16** (16 bits): baseline, 40 GB for a 20B model
- **int8** (8 bits): half the size, 20 GB, minor quality loss
- **int4** (4 bits): quarter the size, 10 GB, noticeable quality loss
- Various schemes for HOW to quantize: GGUF, GPTQ, AWQ each have tradeoffs

Your **GPT OSS 20B running in LM Studio** is almost certainly quantized to 4-bit GGUF. The fp16 version (40 GB) wouldn't fit on a Mac Mini; the Q4 version (~12 GB) does.

This is genuinely a real interview asset for you. *"I run quantized models locally on Apple Silicon — usually 4-bit GGUF — which lets me prototype against real LLMs with no API costs and full privacy. There's a measurable quality drop versus fp16, but for many tasks it's acceptable, and the iteration speed is worth it for development."*

## Quiz before moving on

1. When would you fine-tune vs use RAG vs prompt-engineer?
2. Why is full fine-tuning of large models so expensive?
3. Explain LoRA in your own words.
4. Why does LoRA work? What's the underlying insight?
5. What's QLoRA and why is it important?
6. What's quantization? What's the tradeoff?
7. What's the difference between Q4, Q8, and fp16?

---

# Day 10: Mixture of Experts and the Big Picture

## Mixture of Experts (MoE)

**The analogy**: imagine a large law firm. Instead of every lawyer being a generalist, you have specialists — tax law, criminal law, intellectual property, etc. When a client comes in with a specific question, the firm routes them to the right specialist, not the whole firm. Each client interaction involves just one or two lawyers, even though the firm has hundreds.

This is what **MoE** does inside a transformer. Instead of one big feed-forward network in each transformer block, you have many smaller "expert" networks (often 8, 16, or more), and a **router** that picks 1-2 of them for each token.

## What's actually happening

In a standard transformer block, every token passes through the same feed-forward network. In an MoE transformer, the router examines each token and sends it to a small subset of experts.

The benefits:
- **Capacity**: the total parameters in the model can be huge (because you have many experts)
- **Inference cost**: only the active experts run for any given token, so per-token cost stays reasonable

Tradeoffs:
- All experts must be loaded into memory, even if only 1-2 are active per token
- Training is harder (routing instabilities, expert collapse)
- More complex deployment

GPT-4 is widely believed to be MoE (though OpenAI hasn't confirmed). Mixtral 8x7B is openly MoE. DeepSeek V3 is MoE. The trend is clearly toward MoE for frontier models.

## The full stack — top to bottom

Here's the conceptual hierarchy of a modern LLM-based system. Be able to walk up and down this in interviews:

```
Hardware (GPUs/TPUs)
   ↓
Quantized weights (Q4 GGUF, etc.) loaded into memory
   ↓
Transformer architecture (attention + FFN + possibly MoE)
   ↓
Pretrained on internet-scale data (next-token prediction)
   ↓
Instruction-tuned (SFT on examples)
   ↓
Preference-optimized (RLHF or DPO)
   ↓
Served via inference framework (vLLM, TGI, etc.)
   With KV cache, continuous batching, speculative decoding
   ↓
Exposed via API (Anthropic, OpenAI, or self-hosted)
   ↓
Your application layer:
   - Prompting strategies
   - Tool calling
   - RAG (with embedding model + vector DB + retrieval logic)
   - Agent control flow
   - Output validation
   ↓
Evals + observability + monitoring
   ↓
User
```

Being able to discuss any layer of this hierarchy fluently, and articulate how a choice at one layer affects others, is what makes you a senior candidate in interviews.

For example: *"If we use MoE models, the active params per token are smaller, so latency drops — but the memory required to host the full model is the same as a dense model with equivalent total params. So self-hosting MoE is harder than self-hosting dense, even though inference is cheaper."* That's the kind of cross-layer reasoning that signals depth.

## The 22 interview questions — your final exam

If you can crisply answer all 22 of these in under 90 seconds each, without notes, you're done with the sprint. Drill them ruthlessly.

**Fundamentals**
1. Walk me through what happens when you train a neural network.
2. What's the difference between accuracy and F1? When would you prefer F1?
3. What's overfitting? How do you detect it? How do you fix it?
4. Explain backpropagation in 30 seconds.

**Transformers**
5. Explain self-attention to a smart non-technical person.
6. Why are transformers better than RNNs?
7. What's the difference between encoder-only, decoder-only, and encoder-decoder models? When do you use each?
8. What does multi-head attention give you that single-head doesn't?
9. Why is attention O(n²) and what does that mean for long context?

**LLM specifics**
10. Walk me through how a modern LLM is trained.
11. Why do LLMs hallucinate? Name three causes.
12. What's the difference between temperature, top-k, and top-p?
13. When would you use fine-tuning vs RAG?
14. What's a KV cache and why does it matter?

**Fine-tuning & efficiency**
15. Explain LoRA. Why does it work?
16. What's QLoRA and why is it important?
17. What's quantization? What's the tradeoff?
18. What's a Mixture of Experts model?

**Systems thinking**
19. You have a RAG system with 70% accuracy. Walk me through how you'd debug it.
20. When is RAG the wrong choice?
21. How would you measure if an LLM output is "good"?
22. Your latency is 8 seconds and users are complaining. What do you investigate?

## After this sprint

You should be able to:
- Explain any concept in this doc in 60 seconds without notes
- Read a modern LLM paper (Llama 3, Mixtral, DeepSeek tech reports) and follow the architectural decisions
- Have an informed opinion when someone says "we should fine-tune" or "we need a bigger context window"
- Walk up and down the LLM stack fluently in interviews

You do **not** need to:
- Derive backpropagation math
- Memorize the exact attention formula
- Know research-level transformer variants

You're an AI engineer, not a researcher. The bar is fluent practical understanding, not theoretical depth.

## Resources, in order of priority

1. **3Blue1Brown's neural network series** (4 videos, ~1.5 hrs) — visual intuition for everything Day 3-4
2. **Karpathy's "Intro to Large Language Models"** (1 hr) — if not finished from Phase 1
3. **The Illustrated Transformer by Jay Alammar** (1 hr read) — best transformer explainer that exists
4. **Karpathy's "Let's build GPT"** (2 hrs) — the build-it-yourself transformer video. Required for Day 5-6.
5. **The Annotated Transformer (Harvard NLP)** — original paper alongside code, if you want to go deeper

Avoid:
- ML textbooks (Bishop, Murphy) — too theoretical for your goals
- Most Coursera ML courses — too broad, too academic
- "Build an AI startup in 30 days" type content — substanceless

---

That's the sprint. Take it day by day. The depth is in the engagement, not the speed. By the end you should feel a step-change in your ability to discuss LLM systems with anyone — interviewers included.

Welcome to the other side.

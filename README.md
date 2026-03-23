# Self-Attention

A minimalist, pure-NumPy implementation of the self-attention mechanism. This example demonstrates how Transformers (like GPT) resolve word ambiguity by "attending" to different contexts—ranging from physical properties to emotional idioms.

## What is Self-Attention?

Self-Attention is a mechanism that allows a sequence-to-sequence model to weigh the importance of different words in a sentence relative to a specific target word. It enables the model to create a "contextualized" representation of a word by looking at its neighbors. The core logic follows the official formula from the original Transformer research paper (_Attention Is All You Need_, 2017]

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

It is the core architectural component of modern Large Language Models (LLMs), enabling them to capture long-range dependencies and nuanced meanings that a static dictionary cannot provide.

## Multi-Contextual attention example

In this example, each word is assigned a vector that describes certain features. For this example, I created several of the following features in the form of matrices: `Electronics`, `Weight`, `Fire`, `Action`, and `Emotions` and assigned weights to them. For example the word `macbook` will have the following weights `[2.5, 0.3, 0.0, 0.0, 0.0]` where feature related to electronicts is highlighted (2.5).

In real llm, features are not "fixed." They are initially initialized with random values ​​and change as the network is trained to its final form, so the better the network is trained, the better the results will be. However, for this example, they are fixed to simply demonstrate the mechanism in action.

So lets take a look on **"light"** word which is a perfect candidate for testing attention because its meaning shifts dramatically based on its context. We take a look on four distinct scenarios:

| Sentence | Role of "Light" | Primary Feature |
| :--- | :--- | :--- |
| "Turn on the **light**" | Noun (Object) | **Electronics** |
| "**Light** luggage" | Adjective (Weight) | **Physical Weight** |
| "**Light** a cigarette" | Verb (Action) | **Fire / Heat** |
| "Makes **light** of problems" | Idiom (Emotion) | **Emotions / Sentiment** |

## Output

When running the script, you will observe how the neutral "light" vector absorbs features from its neighbors. Here are the actual results from the simulation:

#### 1. Electronics Context
Sentence: `'turn on light'`
* Attention for 'light': `{'turn': 0.393, 'on': 0.344, 'light': 0.263}`
* Contextualized vector: `[2.07, 0.05, 0.05, 1.58, 0.05]`
* High values in Electronics (2.07) and Action (1.58).

#### 2. Physical Weight Context
Sentence: `'light luggage'`
* Attention for 'light': `{'light': 0.411, 'luggage': 0.589}`
* Contextualized vector: `[0.08, 3.02, 0.08, 0.08, 0.08]`
* The Weight dimension (3.02) becomes dominant.

#### 3. Fire/Heat Context
Sentence: `'light cigarette'`
* Attention for 'light': `{'light': 0.411, 'cigarette': 0.589}`
* Contextualized vector: `[0.08, 0.08, 3.02, 0.08, 0.08]`
* The Fire/Heat dimension (3.02) becomes dominant.

#### 4. Idiomatic/Emotional Context
Sentence: `'makes light of problems'`
* Attention for 'light': `{'makes': 0.27, 'light': 0.216, 'of': 0.206, 'problems': 0.308}`
* Contextualized vector: `[0.04, 0.04, 0.04, 0.58, 2.09]`
* High value in Emotions (2.09), showing the metaphorical shift.

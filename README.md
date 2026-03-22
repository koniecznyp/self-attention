# Self-Attention: The "Light" Experiment

A minimalist, pure-NumPy implementation of the **Scaled Dot-Product Attention** mechanism. This example demonstrates how Transformers (like GPT) resolve word ambiguity by "attending" to different contexts—ranging from physical properties to emotional idioms.

## What is Self-Attention?

**Self-Attention** is a mechanism that allows a sequence-to-sequence model to weigh the importance of different words in a sentence relative to a specific target word. It enables the model to create a "contextualized" representation of a word by looking at its neighbors.

The core logic follows the official formula from the original Transformer research paper (_Attention Is All You Need_, 2017]

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

It is the core architectural component of modern Large Language Models (LLMs), enabling them to capture long-range dependencies and nuanced meanings that a static dictionary cannot provide.

## The Experiment: Multi-Contextual "Light"

The word **"light"** in English is a perfect candidate for testing attention because its meaning shifts dramatically based on its context. This project simulates four distinct scenarios:

| Sentence | Role of "Light" | Primary Feature |
| :--- | :--- | :--- |
| "Turn on the **light**" | Noun (Object) | **Electronics** |
| "**Light** luggage" | Adjective (Weight) | **Physical Weight** |
| "**Light** a cigarette" | Verb (Action) | **Fire / Heat** |
| "Makes **light** of problems" | Idiom (Emotion) | **Emotions / Sentiment** |

### Feature Dimensions
In this demo, we represent words using a 5-dimensional embedding space:
`[Electronics, Weight, Fire, Action, Emotions]`

---

## Output

When running the script, you will observe how the neutral "light" vector absorbs features from its neighbors. Here are the actual results from the simulation:

### 1. Electronics Context
**Sentence:** `'turn on light'`
* **Attention for 'light':** `{'turn': 0.393, 'on': 0.344, 'light': 0.263}`
* **Contextualized vector:** `[2.07, 0.05, 0.05, 1.58, 0.05]`
* *High values in Electronics (2.07) and Action (1.58).*

### 2. Physical Weight Context
**Sentence:** `'light luggage'`
* **Attention for 'light':** `{'light': 0.411, 'luggage': 0.589}`
* **Contextualized vector:** `[0.08, 3.02, 0.08, 0.08, 0.08]`
* *The Weight dimension (3.02) becomes dominant.*

### 3. Fire/Heat Context
**Sentence:** `'light cigarette'`
* **Attention for 'light':** `{'light': 0.411, 'cigarette': 0.589}`
* **Contextualized vector:** `[0.08, 0.08, 3.02, 0.08, 0.08]`
* *The Fire/Heat dimension (3.02) becomes dominant.*

### 4. Idiomatic/Emotional Context
**Sentence:** `'makes light of problems'`
* **Attention for 'light':** `{'makes': 0.27, 'light': 0.216, 'of': 0.206, 'problems': 0.308}`
* **Contextualized vector:** `[0.04, 0.04, 0.04, 0.58, 2.09]`
* *High value in Emotions (2.09), showing the metaphorical shift.*
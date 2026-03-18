# Theoretical Foundation: Attention Visualization and Attribution

## 1. The Physics of the Attention Mechanism
The core innovation of the Transformer architecture is the Self-Attention mechanism, which allows the model to weigh the relevance of every token in a sequence against every other token. 
The mechanism is defined by three matrices: Queries ($Q$), Keys ($K$), and Values ($V$). 

The attention calculation is:
$Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

The crucial component for interpretability is the intermediate matrix:
$P = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$

$P$ is an $N \times N$ probability distribution matrix (where $N$ is the sequence length). The value at $P_{i,j}$ physically quantifies exactly what percentage of "attention" the $i$-th token pays to the $j$-th token when determining its own contextual meaning. By visualizing this matrix, we can see exactly how the model routes information across words.

## 2. Mechanistic Interpretability and Head Semantics
A modern LLM contains dozens of layers, each with dozens of independent "Attention Heads." Mechanistic Interpretability posits that these heads are not homogeneous; they are highly specialized neural circuits that learn specific algorithmic functions.

We can classify these heads based on their physical attention patterns:
- **Induction Heads:** Look for patterns like `[A] [B] ... [A]` and attend to `[B]`, facilitating in-context learning and repetition.
- **Previous-Token Heads:** Attend almost exclusively to the token immediately preceding the current one, constructing low-level bigram syntax.
- **Global / Completion Heads:** Diffuse their attention across vast swaths of prior context to maintain long-range semantic coherence (characterized by low density on the diagonal of the attention matrix).
- **Instruction-Following Heads:** Act as rigid functional gates. They attend intensely to the localized prompt instructions (e.g., "Translate to French") and strongly prioritize their own internal query state (characterized by extreme high density on the localized diagonal of the matrix).

By calculating statistical properties like the diagonal trace of the attention matrix, we can map the "semantic signature" of a head. Using hierarchical clustering algorithms (like Ward's method based on cosine similarity), we can mathematically group hundreds of heads into distinct functional families, separating the syntactical processors from the logical reasoning circuits.

## 3. Attention Rollout for Token Attribution
A severe limitation of analyzing a single attention layer is the presence of the residual stream. If Token A attends to Token B in Layer 1, and Token C attends to Token A in Layer 2, Token C has indirectly integrated information from Token B.

**Attention Rollout** is a mathematical algorithm used to compute the global, cumulative influence of tokens across the entire depth of the network. It assumes that the connections between tokens are linear and can be multiplied sequentially backwards through the layers.

1. The raw attention matrix is aggregated across all heads in a layer to find the mean routing map.
2. The Identity matrix ($I$) is added to simulate the skip-connections of the residual stream: $A' = \frac{1}{2}(Attention + I)$
3. The matrices are recursively multiplied backward from the final layer to the first: $Rollout = A'_L \times A'_{L-1} \times ... \times A'_1$

The final row of the resulting global Rollout matrix provides a precise percentage breakdown of how much every single word in the prompt physically contributed to the generation of the final output token, allowing us to map the absolute causal pathway of the model's logic.

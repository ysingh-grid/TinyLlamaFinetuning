# Theoretical Foundation: Logit Lens and Layer-wise Ablation

## 1. The Logit Lens Hypothesis
The standard conceptual model of a Transformer treats the intermediate layers as an opaque pipeline: raw tokens enter Layer 0, undergo a series of incomprehensible high-dimensional transformations, and coherent language probabilities selectively emerge only after the final `lm_head` projection layer.

The **Logit Lens** technique hypothesizes a different structural reality: the residual stream of a transformer maintains a continually evolving, latent representation of the vocabulary space.
Instead of waiting for the final layer, the Logit Lens takes the hidden state activation $h_l$ at an intermediate layer $l$, applies the final layer normalization, and forces it through the vocabulary projection matrix (`lm_head`). 

This translates the intermediate, halfway-processed thought into an explicit list of next-token predictions. By tracking these predictions across all layers, we can mathematically map the trajectory of the model's "reasoning." 
- We can observe exactly at which depth the model resolves grammatical syntax.
- We can pinpoint the specific layer where the model accesses factual trivia to successfully predict the target entity.

## 2. Frobenius Norms and Parameter Geometry
When a model is fine-tuned using LoRA, low-rank matrices $A$ and $B$ are updated to alter the network's behavior. However, neural networks are highly elastic; not all layers contribute equally to the new learned behavior.

To theoretically quantify how "important" a specific LoRA adapter is, we measure its geometric magnitude. The update matrix is $\Delta W = B \times A \times \text{scale}$. We evaluate the severity of this update by calculating its **Frobenius Norm**:
$|| \Delta W ||_F = \sqrt{\sum_{i} \sum_{j} |\Delta W_{ij}|^2}$

The Frobenius Norm acts as a mathematical proxy for structural distortion. 
- If the norm is near zero, the adapter barely modified the layer; the layer relies almost entirely on its pre-trained baseline knowledge.
- If the norm is massive, the fine-tuning process violently restructured the routing logic of that specific layer to accomplish the new task (e.g., instructional compliance formatting).

## 3. Structural Ablation and Network Redundancy
Neural networks are famously over-parameterized. The **Ablation Hypothesis** dictates that we can systematically cripple or remove components of the network to determine their causal necessity.

If a Logit Lens or a Frobenius Norm identifies a layer as having minimal impact, we can ablate (silence) the LoRA adapter at that layer entirely by forcing its output to zero.

The theoretical goal of sequential ablation is to find the network's absolute breaking point. By iteratively silencing the least-important $10\%$, $25\%$, or $50\%$ of the fine-tuned parameters and measuring the resultant spike in Cross-Entropy Loss (Perplexity), we construct a decay curve. 
This proves mathematically how much of the fine-tuning optimization was structurally necessary for the task, versus how much was merely mathematical noise distributing randomly across the redundant tensor volume. In many cases, huge swaths of the adaptation matrices can be ablated with near-zero degradation to the final output generation.

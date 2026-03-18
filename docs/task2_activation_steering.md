# Theoretical Foundation: Activation Vector Steering

## 1. Representation Engineering (RepE)
Historically, modifying the behavior of a neural network—such as making it safer, more polite, or less biased—has relied on updating its weights via Reinforcement Learning from Human Feedback (RLHF) or Constitutional AI fine-tuning. 

Representation Engineering (RepE) proposes a radical alternative based on the manifold hypothesis: Large Language Models do not just map input text to output text; they construct a high-dimensional internal representation (a latent space) of abstract concepts. Complex behaviors like "honesty," "malice," or "refusal" are not magical emergent properties, but rather specific geometric directions within this latent space.

If a model "knows" a request is dangerous, that knowledge must be physically encoded in the activation patterns of its hidden layers before the final text is ever generated.

## 2. The Calculus of Concept Vectors
To control a model without retraining it, we first isolate the abstract concept's geometric vector. 
Let $h_l(x) \in \mathbb{R}^d$ be the activation of the residual stream at layer $l$ for an input sequence $x$. 
We construct two contrasting datasets: $S_{safe}$ (benign, helpful instructions) and $S_{unsafe}$ (malicious, harmful instructions).

By feeding both datasets through the model, we calculate the geometric "center of mass" (the mean activation) for both concepts at a specific layer:
$\mu_{safe} = \frac{1}{|S_{safe}|} \sum_{x \in S_{safe}} h_l(x)$
$\mu_{unsafe} = \frac{1}{|S_{unsafe}|} \sum_{x \in S_{unsafe}} h_l(x)$

The steering vector $v_l$ is defined as the unit-normalized difference between these means:
$v_l = \frac{\mu_{safe} - \mu_{unsafe}}{||\mu_{safe} - \mu_{unsafe}||_2}$

This vector $v_l$ represents the pure, isolated semantic direction of "Safety vs. Harm" within the model's neuro-computational space.

## 3. Inference-Time Intervention (ITI)
With the concept vector isolated, we can execute an Inference-Time Intervention. During the generation of any new text, we interrupt the forward pass of the model at the target layer $l$ and artificially inject the steering vector into the residual stream:

$\tilde{h}_l(x) = h_l(x) + c \cdot v_l$

The coefficient $c$ dictates the intensity of the intervention. 
- **$c > 0$:** Pushes the model's internal state toward the "safe" direction, increasing the likelihood of refusal or safe outputs, regardless of the user's prompt.
- **$c < 0$:** Inverts the alignment, theoretically forcing the model to become more compliant with harmful requests (a "jailbreak" vector).

This physical shift alters the trajectory of the information cascading through all subsequent layers, ultimately changing the probability distribution of the output logits without modifying a single permanent weight in the matrix.

## 4. Layer Sensitivity and Variance
Information processing in transformers is hierarchical. Early layers typically process raw syntax and lexical tokens. Middle layers synthesize complex semantic concepts and reasoning. Final layers map these concepts back onto the output vocabulary.

Consequently, activation steering is highly sensitive to depth. Injecting a semantic vector into an early syntactic layer often corrupts the grammar (yielding gibberish), while injecting it too late fails to alter the model's reasoning process. The most profound behavioral control is almost always achieved by steering within the middle layers of the network. Furthermore, evaluating the success of steering is best measured by tracking the *variance* of the output behaviors: a highly effective steering vector will crush the behavioral variance, guaranteeing a consistent, unbreakable refusal state across a wide range of adversarial inputs.

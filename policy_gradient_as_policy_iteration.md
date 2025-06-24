---
layout: default
---

<link rel="stylesheet" href="/assets/theme.css">
<script src="/assets/theme-toggle.js" defer></script>
<link rel="icon" href="/assets/favicon.png" type="image/png">

<script type="text/javascript">
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      renderActions: {
        findScript: [10, function (doc) {
          // escape Markdown italics (*) inside math
          for (const script of document.querySelectorAll('script[type^="math/tex"]')) {
            script.text = script.text.replace(/\^([^\s^_{}\\])/g, '^{$1}');
          }
        }, '']
      }
    }
  };
</script>

# Policy Gradient as Policy Iteration

In this blog, I am going to document another interpretation of Policy Gradient in terms of Policy Iteration. To keep this blog short, I’m skipping a lot of fundamentals and will cite resources for readers to refer to wherever necessary. With that, let's try to look at Policy Gradient from the lens of Policy Iteration.

## Policy Iteration

Let’s say we have a policy $\pi$ which specifies a distribution over actions $A = \{a_1, \cdots, a_k\}$ given a state $s \in S$. In other words,  
$$
\pi(a|s) \in [0, 1], \quad \sum_{a\in A(s)}\pi(a|s) = 1
$$
> Note: $S$ and $A$ are the state and action spaces respectively. $A(s)$ is the set of actions admissible in state $s$.

An RL agent in a particular state $s$ will refer to this policy $\pi$ to pick an action $a \in A(s)$. The goal of RL is to find the best policy that yields the maximum return. And if $\pi$ is already the best policy, then solving the problem is trivial — just act according to $\pi$!

An agent is usually initialized with some random policy $\pi_0$, and at each iteration $t$, we repeat the following steps:
- Evaluate the policy $\pi_t$
- Improve the policy to get $\pi_{t+1} \leftarrow \text{improved}(\pi_t)$

Let’s unpack both steps:

- **Evaluation**: This is essentially the calculation of expected return from following $\pi$. Given an initial state distribution $h(s)$, the value of policy $\pi$ can be evaluated as:
$$
\eta(\pi) = \mathbb{E}_{s \sim h(s)}\left[V_\pi(s)\right]
$$

- **Improvement**: Assuming a finite number of actions and states, we can improve the policy by choosing the action in each state that maximizes the $Q$-value under $\pi$:
$$
\pi'(s) = \arg\max_a Q_\pi(s, a)
$$

Is this guaranteed to improve the policy? Let’s analyze:
$$
\begin{align*}
\eta(\pi') &= \mathbb{E}_{s \sim h(s)}\left[V_{\pi'}(s)\right] \\
&= \mathbb{E}_{s \sim h(s)}\left[\max_a Q_\pi(s, a)\right] \\
&\geq \mathbb{E}_{s \sim h(s)}\left[ \mathbb{E}_{a \sim \pi}\left[Q_\pi(s, a)\right] \right] \\
&= \mathbb{E}_{s \sim h(s)}\left[ V_\pi(s) \right] \\
&= \eta(\pi) \\
&\boxed{\therefore \eta(\pi') \geq \eta(\pi)}
\end{align*}
$$

So policy iteration guarantees improvement over the current policy, as long as the current policy is suboptimal. Once the policy becomes optimal, the improvement step yields the same policy again. This makes intuitive sense: if there exists a better policy, the improvement step will find it; if not, we must already be at the best.

To all my dear skeptic readers, here’s a short proof:

Assume a policy $\pi$ is suboptimal but still satisfies:
$$
\forall s \in S,\quad \pi(s) = \arg\max_a Q_\pi(s, a)
$$
Then by definition of improvement:
$$
\pi'(s) = \arg\max_a Q_\pi(s, a) = \pi(s)
$$

But we also know that the optimal policy $\pi^*$ satisfies:
$$
\forall s \in S, \quad \pi^*(s) = \arg\max_a Q_{\pi^*}(s, a)
$$

This contradicts the assumption that $\pi$ is suboptimal, yet invariant to improvement. Hence, such a $\pi$ must already be the optimal policy, i.e., $\pi = \pi^*$.

So far, we’ve seen:
- Policy iteration consists of two steps: Evaluation and Improvement
- Policy iteration guarantees improvement at every step
- Once optimal, further improvement has no effect

## Policy Gradient

You can check out my blog on [Policy Gradients](policy-gradient.md) to dig into the derivation. One may wonder: *Why do we need another method if we already have a proven one?* That’s actually outside the scope of this blog, since it would require introducing a few more topics first — but here’s a rough idea.

The improvement step in policy iteration demands a `max` over the set of actions. Now imagine the set of actions is infinite — say, continuous. Take it a step further: the improvement needs to be done for *every* state, and imagine the state space is also infinite. Can you really do policy iteration in practice then?

Let’s say, somehow, you’ve managed to carve out a significant chunk of the state-action space and now want to perform policy evaluation. What happens when you encounter a state you’ve never seen before?

To deal with such issues, function approximation was introduced, which later evolved into deep RL. Policy Gradient is one such approach that proposes a different way to handle this — a school of thought in its own right.
 

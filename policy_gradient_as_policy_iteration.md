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

<script async id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

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

You can check out my blog on [Policy Gradients](policy-gradient.md) to dig into the derivation. One may wonder: *Why do we need another method if we already have a proven one?* That’s actually outside the scope of this blog, since it would require bootstrapping a few more topics — but here’s a rough idea.

The improvement step in policy iteration demands a `max` over the set of actions. Now imagine the set of actions is infinite — say, continuous. Take it a step further: the improvement needs to be done for *every* state, and imagine the state space is also infinite. Can you really do policy iteration in practice then?

Let’s say, somehow, you’ve managed to carve out a significant chunk of the state-action space and now want to perform policy evaluation. What happens when you encounter a state you’ve never seen before?

To deal with such issues, function approximation was introduced, which later evolved into deep RL. Policy Gradient aims to find the best policy which yields the maximum return by moving in the direction of the gradient of the policy. Therefore a natual question which might emerge is *Is an improvement guaranteed for moving in the direction of gradient like that of Policy Iteration ?* And the answer is - Yes !

The objective of Policy Gradient is formulated as :

$$
\max_\theta \eta(\theta) = \max_\theta \mathbb{E}_{\tau \sim \rho_\theta(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)\right]
$$

> $\eta(\theta)$ is the measure of the performance of the policy $\pi_\theta$ which is essentially the expected discounted return for trajectories sampled according to the trajectory distribution $\rho_\theta$

Therefore according to the gradient ascent update rule, the parameter is updated as 

$$
\theta' \leftarrow \theta + \alpha \nabla_\theta \eta(\theta)
$$

Policy Gradient claims that maximizing $\eta(\theta')$ is consistent with maximizing $\eta(\theta') - \eta(\theta)$ and therefore is equivalent to the objective of maximizing the expected cumulative discounted **Advantage** under the policy $\pi_{\theta}$ for states and actions sampled according to the trajectory distribution $\rho_{\theta'}$. More formally, 

$$
\max_{\theta'}\eta(\theta') \equiv \max_{\theta'}\eta(\theta') - \eta(\theta) \equiv \max_{\theta'}\mathbb{E}_{\tau \sim \rho_{\theta'}(\tau)}\left[\sum_{t=0}^\infty \gamma^t A_{\pi_{\theta}}(s_t, a_t)\right]
$$

Before we jump into the proof, let's convince ourselves first, why doing this is actually justified. Let's imagine 2 policies which is exactly similar to the other except for a state $s$ where policy $\pi_1$ takes $a_1$ whereas $\pi_2$ takes $a_2$. Both these actions take the agent to state $s'$ but they yield different reward. Specifically, $r(s, a_1) = -1, r(s, a_2) = 1$. The figure below is an illustration of the presented scenario. 

<img src="assets/2_trajectpry_w_1_diff_action.png" alt="Cumulative Cognition vs T" width="500"/>

The advantage of taking an action $a$ in state $s$ and then acting according to policy $\pi$ is formulated as :

$$
\begin{align*}
A_{\pi}(s, a) &= Q_\pi(s, a) - \mathbb{E}_{a, \sim \pi}\left[Q_\pi(s, a) \right]\\ 
&= Q_\pi(s, a)  - V_\pi(s) \\
&= \mathbb{E}_{s'\sim P(s'|s, a)}\left[r(s, a) + \gamma V_{\pi}(s')\right] - V_\pi(s)\\
&\approx r(s, a) + \gamma V_\pi(s') - V_\pi(s)
\end{align*}
$$

Therefore, the advantages of taking actions $a_1, a_2$ under the policies $\pi_1, \pi_2$ are 

$$
\begin{align*}
A_{\pi_1}(s, \pi_1(s)) &= -1 + \gamma V_{\pi_1}(s') - V_{\pi_1}(s)\\
&= -1 + \gamma V_{\pi_1}(s') - Q_{\pi_1}(s, a_1)\\
& = -1 + \gamma V_{\pi_1}(s') - r(s, a_1) - \gamma V_{\pi_1}(s') = 0
\end{align*}
$$

$$
\begin{align*}
A_{\pi_1}(s, \pi_2(s)) &= 1 + \gamma V_{\pi_1}(s') - V_{\pi_1}(s)\\
&= 1 + \gamma V_{\pi_1}(s') - Q_{\pi_1}(s, a_1)\\
& = 1 + \gamma V_{\pi_1}(s') - r(s, a_1) - \gamma V_{\pi_1}(s') = 2
\end{align*}
$$

$$
\boxed{\therefore \quad A_{\pi_1}(s, \pi_2(s)) > A_{\pi_1}(s, \pi_1(s)) }
$$

As policy $\pi_1, \pi_2$ are otherwise similar except for the above case, we have 

$$
\sum_{t=1}^\infty\mathbb{E}_{s_t, a_t \sim \rho_{\pi_1} }\left[A_{\pi_1}(s_t, a_t)\right] = \sum_{t=1}^\infty\mathbb{E}_{s_t, a_t \sim \rho_{\pi_2} }\left[A_{\pi_1}(s_t, a_t)\right] \quad : s_t \neq s
$$

Therefore following this equality, we have 

$$
\boxed{
\sum_{t=1}^\infty\mathbb{E}_{s_t, a_t \sim \rho_{\pi_1} }\left[A_{\pi_1}(s_t, a_t)\right] > \sum_{t=1}^\infty\mathbb{E}_{s_t, a_t \sim \rho_{\pi_2} }\left[A_{\pi_1}(s_t, a_t)\right]}
$$

Hence, if a policy is better than the old policy even by a smudge, we can gurantee the increase in the expected cumulative advantage. *This can easily be extended to the discounted setting, we'll show this afterwards*

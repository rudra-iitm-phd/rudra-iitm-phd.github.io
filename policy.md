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


# 🎯 Policy Gradient: *L’Equazione Elegante*

In this blog, I aim to explore the elegant derivation of the policy gradient.  
Why? Because I find the equation not just insightful, but genuinely beautiful in its simplicity and structure.  
And I’d like to share that appreciation with anyone who reads this.

So, the goal here is to explain the derivation in a way that’s digestible for a novice reader.  
That said, there might be parts where I don’t explain things well enough — and that could interrupt the flow a bit.  
For that, I ask a little patience (and effort) from you, the reader.  

Honestly, this is also my way of pushing myself to understand things better —  
to the point where I can explain them without using fancy terminology at all.  

But hey, this is just *iteration one*, and I’ll keep improving it as I go.  
So, pardon me — and thanks in advance for bearing with me. 

>These notes are based from the lectures of Sergey Levine [^1].

---

## 🔧 A Refresher: Optimization via Calculus

Folks who are familiar with optimization, or have taken a basic course in *Calculus*,  
might remember solving problems that involve maximizing or minimizing a function with respect to a variable. 

Let’s consider an arbitrary function:

$$
f(x) = 2x^2 - 20x
$$

and aim to **minimize** its value with respect to $x$. In other words:

$$
\min_x \ f(x) = \min_x \ (2x^2 - 20x)
$$

What does this mean geometrically?  
Picture $x$ on the horizontal axis and $f(x)$ on the vertical axis of a 2D graph.  
Our goal is to find the point where the graph dips to its lowest value — the **minimum**.

---

### Step 1: Take the Derivative and Set It to Zero

This uses the fact that the slope of the graph at the minimum point is zero:

$$
\begin{align*}
f'(x) &= \frac{d}{dx}(2x^2 - 20x) \\
      &= 4x - 20 \\
      &= 0 \Rightarrow x = 5
\end{align*}
$$

> *Think of a straw attached to an invisible bicycle wheel and we are interested in finding out the top or the bottom of the wheel and as easy as it seems, it's only at the top or bottom of its rotation, the straw is perfectly parallel with the ground (the slope is zero) indicating the points we're interested in*

---

### Step 2: Check the Second Derivative

To ensure it’s a **minimum**, not a maximum:

$$
f''(x) = \frac{d^2}{dx^2}(2x^2 - 20x) = 4
$$

Since $f''(x) > 0$, this confirms that $x = 5$ is a **minimum**.

And the minimum value of the function is:

$$
f(5) = 2 \times 5^2 - 20 \times 5 = 50 - 100 = -50
$$

Therefore, the function $f(x)$ reaches its minimum at $x = 5$ with a value of $-50$.

---

Who should we thank for making this easy?  
From my perspective — we should thank the function for being **differentiable**!

---

## 🚀 From Differentiable Functions to Policy Gradients

I think we've strayed far enough for a decent recap,   but let’s get back to **Policies** and their **gradients**.

Let $\pi_\theta$ be a policy parameterized by $\theta$. What we’re after is the *best* $\theta$, which we’ll call $\theta^*$ — the one that maximizes the **expected return**.

---

In slightly more precise terms: we’re trying to find a $\theta ^\ast$ such that when we execute the corresponding policy $\pi_\theta^\ast$, the trajectories it generates yield a higher expected return than trajectories from any other policy $\pi_\theta $ where $ \theta \neq \theta^\ast $.

So the goal is basically:

> Find the sweet spot in parameter space where the policy just *works better* than all others —  
> not for one lucky rollout, but *on average*, across the space of all trajectories it could produce.

---

Let’s define a trajectory:

$$
\tau = (s_1, a_1, \cdots, s_T, a_T)
$$

This is basically a full episode — a sequence of states and actions the agent experiences over time.

Given a policy $\pi_\theta$, the probability of seeing a particular trajectory $\tau$ is:

$$
\rho_\theta(\tau) = P(s_1)\prod_{t=1}^{T} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)
$$

This makes sense: we start in some state $s_1$,  
then at each step we choose an action using the policy,  
and the environment transitions us to the next state based on dynamics.

---

With that in mind, we can write the optimization problem we’re trying to solve as:

$$
\theta^* = \arg\max_{\theta} \ \mathbb{E}_{\tau \sim \rho_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right] = \arg\max_{\theta} \ \mathbb{E}_{\tau \sim \rho_\theta(\tau)} \left[ r(\tau) \right]
$$

In words:  
we want to find the policy parameters $\theta^*$ that give us the highest expected return  
across all possible trajectories $\tau$ sampled from our policy.

---

Here, $r(s_t, a_t)$ represents the expected reward for taking action $a_t$ in state $s_t$.  
More formally:

$$
r(s_t, a_t) = \mathbb{E}_{s_{t+1}}[r \mid s_t, a_t, s_{t+1}] = \int_{s_{t+1}} r \cdot p(r \mid s_t, a_t, s_{t+1}) \, dr
$$

> Note: We're assuming an **undiscounted** return here, so $\gamma = 1$.

---

## ✏️ Calculating the Gradient

Now again with the help of calculus,  
we are going to find $\theta^*$.  

Therefore, let's calculate the gradient of the policy:

$$
\begin{align*}
\nabla_\theta \mathbb{E}_{\tau \sim \rho_\theta(\tau)} \left[ \sum_{t=1}^T r(s_t, a_t) \right] &= \nabla_\theta \int \rho_\theta(\tau)\cdot r(\tau) \,d\tau & r(\tau) =  \sum_{t=1}^T r(s_t, a_t)\\
&= \int \nabla_\theta \rho_\theta(\tau)\cdot r(\tau) \,d\tau \\
&= \int \rho_\theta(\tau) \frac{\nabla_\theta \rho_\theta(\tau)}{\rho_\theta(\tau)} \cdot r(\tau) \,d\tau \\
&= \int \nabla_\theta \log \rho_\theta(\tau)\cdot r(\tau) \,d\tau \\
&= \mathbb{E}_{\tau \sim \rho_\theta(\tau)} \left[ \nabla_\theta \log \rho_\theta(\tau)\cdot r(\tau) \right]
\end{align*}
$$
Let's unroll $\rho_\theta(\tau)$ and calculate its derivative:
$$
\begin{align*}
\nabla_\theta \log \rho_\theta(\tau)&=\nabla_\theta \log \left[P(s_1)\prod_{t=1}^{T} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)\right]\\
&=\nabla_\theta \left[\log P(s_1)+\sum_{t=1}^{T} \log\pi_\theta(a_t \mid s_t) +\sum_{t=1}^{T} \log P(s_{t+1} \mid s_t, a_t)\right]\\
&=\nabla_\theta\left[\sum_{t=1}^T\log\pi_\theta(a_t \mid s_t) \right]
\end{align*}
$$

## References

[^1]: [Sergey Levine RAIL lectures from UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/).
[^2]:


---
layout: default
title : ""
---

<link rel="stylesheet" href="/assets/theme.css">
<script src="/assets/theme-toggle.js" defer></script>
<link rel="icon" href="/assets/favicon.png" type="image/png">


<input type="text" id="searchInput" placeholder="🔍 Search topics..." />

<ul id="searchList">
  <!-- JS will populate this -->
</ul>

<script defer>
  document.addEventListener("DOMContentLoaded", async () => {
    const input = document.getElementById("searchInput");
    const list = document.getElementById("searchList");

    const res = await fetch('/data/posts.json');
    const posts = await res.json();

    function render(filtered) {
      list.innerHTML = filtered.map(p =>
        `<li>
           <a href="${p.url}">📌 ${p.title}</a><br>
           <small>${p.desc}</small>
         </li>`).join('');
    }

    render(posts);

    input.addEventListener("input", () => {
      const q = input.value.toLowerCase();
      render(posts.filter(p => p.title.toLowerCase().includes(q) ||
                                p.desc.toLowerCase().includes(q)));
    });
  });
</script>



# 👋 Welcome to My Blog
Explore key concepts in reinforcement learning, machine learning, and more.  
Click below to dive into specific topics:

> **[Policy Gradients](policy-gradient.md)**  
> _Appreciating the elegant derivation of the policy gradient theorem with intuitive explanations._

---

> **[Using AI doesn't make you dumb](human_ai_loop.md)**  
_Human-AI collaboration to boost cognition._

---


## 🧠 About Me

I am a PhD student at IIT Madras currently working on Reinforcement Learning under the guidance of [Dr. Balaraman Ravindran](https://www.cse.iitm.ac.in/~ravindran/).This blog serves as a curated collection of my notes, tutorials, and some random ideas/inspirations. I keep this blog updated, so feel free to point out mistakes or collaborate at any part of the iteration.



## 🔗 Connect

- [LinkedIn → Rudra Sarkar](https://www.linkedin.com/in/rudra-sarkar-a411891a0/)



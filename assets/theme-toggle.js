// Toggle theme and save preference
(function() {
      const storedTheme = localStorage.getItem("theme");
      if (storedTheme === "dark") {
        document.documentElement.classList.add("dark-theme");
      }
    
      const button = document.createElement("button");
      button.textContent = "🌓 Theme";
      button.style.position = "fixed";
      button.style.top = "1rem";
      button.style.right = "1rem";
      button.style.zIndex = "999";
      button.onclick = () => {
        document.documentElement.classList.toggle("dark-theme");
        const isDark = document.documentElement.classList.contains("dark-theme");
        localStorage.setItem("theme", isDark ? "dark" : "light");
      };
      document.body.appendChild(button);
    })();
    
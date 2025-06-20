
(function() {
      const currentTheme = localStorage.getItem("theme");
      if (currentTheme === "dark") {
        document.documentElement.classList.add("dark-theme");
      }
    
      const button = document.createElement("button");
      button.innerHTML = document.documentElement.classList.contains("dark-theme") ? "🌞 Light" : "🌙 Dark";
      button.style.position = "fixed";
      button.style.top = "1rem";
      button.style.right = "1rem";
      button.style.zIndex = "9999";
      button.style.padding = "0.5rem 1rem";
      button.style.border = "none";
      button.style.borderRadius = "6px";
      button.style.cursor = "pointer";
      button.style.background = "#ccc";
      button.style.fontSize = "1rem";
    
      button.onclick = () => {
        document.documentElement.classList.toggle("dark-theme");
        const isDark = document.documentElement.classList.contains("dark-theme");
        button.innerHTML = isDark ? "🌞 Light" : "🌙 Dark";
        localStorage.setItem("theme", isDark ? "dark" : "light");
      };
    
      document.body.appendChild(button);
    })();
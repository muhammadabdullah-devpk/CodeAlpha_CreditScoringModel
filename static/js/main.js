/**
 * main.js — CreditAI Client-side Orchestrations
 */

document.addEventListener("DOMContentLoaded", function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Setup Theme Mode (Dark/Light)
    initThemeMode();

    // Trigger Scroll Animations for elements coming into view
    setupScrollAnimations();
});

/**
 * Dark/Light Mode Theme Settings
 */
function initThemeMode() {
    const themeToggle = document.getElementById("themeToggle");
    const themeIcon = document.getElementById("themeIcon");
    const htmlElement = document.documentElement;

    // Check saved choice or default to dark
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const currentTheme = htmlElement.getAttribute("data-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            
            htmlElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            updateThemeIcon(newTheme);
            toastNotify(`Switched to ${newTheme} mode.`);
        });
    }
}

function updateThemeIcon(theme) {
    const themeIcon = document.getElementById("themeIcon");
    if (themeIcon) {
        if (theme === "dark") {
            themeIcon.className = "fa-solid fa-sun";
        } else {
            themeIcon.className = "fa-solid fa-moon";
        }
    }
}

/**
 * Toast notification wrapper
 */
function toastNotify(message, type = "primary") {
    const toastEl = document.getElementById("mainToast");
    const bodyEl = document.getElementById("toastMessage");
    if (toastEl && bodyEl) {
        bodyEl.innerText = message;
        toastEl.className = `toast align-items-center border-0 text-white bg-${type}`;
        const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
        toast.show();
    }
}

/**
 * Intersection Observer for scroll triggers
 */
function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("animate-fade-up");
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll(".animate-on-scroll").forEach(el => {
        observer.observe(el);
    });
}

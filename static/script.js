document.addEventListener("DOMContentLoaded", function () {

    // 1. Highlight the current navigation link
    const currentPath = window.location.pathname;

    document.querySelectorAll(".nav-links a").forEach(function (link) {
        const linkPath = new URL(link.href, window.location.origin).pathname;

        if (linkPath === currentPath) {
            link.classList.add("active");
            link.setAttribute("aria-current", "page");
        }
    });


    // 2. Logout confirmation
    // Add class "logout-link" to a logout link if you want confirmation.
    document.querySelectorAll(".logout-link").forEach(function (link) {
        link.addEventListener("click", function (event) {
            const confirmed = window.confirm(
                "Are you sure you want to log out of WaterWise?"
            );

            if (!confirmed) {
                event.preventDefault();
            }
        });
    });


    // 3. Show/hide password
    document.querySelectorAll("[data-toggle-password]").forEach(
        function (button) {
            button.addEventListener("click", function () {
                const targetId = button.getAttribute("data-toggle-password");
                const passwordInput = document.getElementById(targetId);

                if (!passwordInput) {
                    return;
                }

                const isPassword =
                    passwordInput.type === "password";

                passwordInput.type = isPassword ? "text" : "password";

                button.textContent = isPassword ? "Hide" : "Show";
                button.setAttribute(
                    "aria-label",
                    isPassword ? "Hide password" : "Show password"
                );
            });
        }
    );


    // 4. Mobile navigation toggle
    const menuButton = document.getElementById("menu-toggle");
    const navigation = document.querySelector(".nav-links");

    if (menuButton && navigation) {
        menuButton.addEventListener("click", function () {
            const isOpen = navigation.classList.toggle("menu-open");

            menuButton.setAttribute("aria-expanded", String(isOpen));
        });
    }


    // 5. Add a loading state to forms when submitted
    // This does not stop normal form submission.
    document.querySelectorAll("form[data-loading-form]").forEach(
        function (form) {
            form.addEventListener("submit", function () {
                const submitButton = form.querySelector(
                    'button[type="submit"], input[type="submit"]'
                );

                if (!submitButton) {
                    return;
                }

                submitButton.dataset.originalText =
                    submitButton.textContent;

                submitButton.textContent = "Please wait...";
                submitButton.disabled = true;
            });
        }
    );


    // 6. Allow dismissible alerts
    document.querySelectorAll("[data-dismiss-alert]").forEach(
        function (button) {
            button.addEventListener("click", function () {
                const alert = button.closest(".alert");

                if (alert) {
                    alert.remove();
                }
            });
        }
    );


    // 7. Smooth scrolling for internal page links
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
        link.addEventListener("click", function (event) {
            const targetId = link.getAttribute("href");

            if (!targetId || targetId === "#") {
                return;
            }

            const target = document.querySelector(targetId);

            if (target) {
                event.preventDefault();

                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    });


    // 8. Update elements showing the current year
    document.querySelectorAll("[data-current-year]").forEach(
        function (element) {
            element.textContent = new Date().getFullYear();
        }
    );

});

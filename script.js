// script.js

// Wait until the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function() {
  const startButton = document.getElementById("startButton");

  startButton.addEventListener("click", function() {
    // Redirect to index.html in the same tab
    window.location.href = "index.html";
  });
});
async function searchImage() {
    const input = document.getElementById("imageInput");
    const resultsDiv = document.getElementById("results");

    if (!input.files.length) {
        alert("Please select an image");
        return;
    }

    resultsDiv.innerHTML = "<p>Searching...</p>";

    const formData = new FormData();
    formData.append("file", input.files[0]);

    const response = await fetch("/search", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    resultsDiv.innerHTML = "";

    data.results.forEach(item => {
        const card = document.createElement("div");
        card.className = "card";

        card.innerHTML = `
            <img src="${item.url}">
            <div class="info">
                <h3>${item.image}</h3>
                <p>Category: ${item.category}</p>
                <p>Price: ₹${item.price}</p>
                <p>${item.description}</p>
                <span>Similarity: ${(item.similarity * 100).toFixed(2)}%</span>
            </div>
        `;


        resultsDiv.appendChild(card);
    });
    
}

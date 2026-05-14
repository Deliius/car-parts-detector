const form = document.getElementById("upload-form");
const resultImage = document.getElementById("result-image");
const detectionsContainer = document.getElementById("detections");

function renderDetections(detections) {
    detectionsContainer.innerHTML = "";
    if (!detections.length) {
        detectionsContainer.textContent = "No detections found.";
        return;
    }

    const title = document.createElement("h2");
    title.textContent = "Detections";
    detectionsContainer.appendChild(title);

    const list = document.createElement("ul");
    detections.forEach((detection) => {
        const item = document.createElement("li");
        const confidence = (detection.confidence * 100).toFixed(2);
        const bbox = detection.bbox.map((value) => value.toFixed(1)).join(", ");
        item.textContent = `${detection.class_name} - ${confidence}% - bbox: [${bbox}]`;
        list.appendChild(item);
    });
    detectionsContainer.appendChild(list);
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    detectionsContainer.textContent = "Running prediction...";
    resultImage.style.display = "none";

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const prediction = await response.json();
        resultImage.src = prediction.fileout;
        resultImage.style.display = "block";
        renderDetections(prediction.detections);
    } catch (error) {
        console.error(error);
        detectionsContainer.textContent = "";
        alert("Error during prediction.");
    }
});

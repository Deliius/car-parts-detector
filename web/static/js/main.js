const form = document.getElementById("upload-form");
const resultImage = document.getElementById("result-image");
const detectionsContainer = document.getElementById("detections");
const statusMessage = document.getElementById("status");
const resultTitle = document.getElementById("result-title");
const detectionsTitle = document.getElementById("detections-title");
const previewPanel = document.getElementById("preview-panel");
const detectionsPanel = document.getElementById("detections-panel");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const confidenceInput = document.getElementById("confidence-input");

function formatBox(bbox) {
    return bbox.map((value) => value.toFixed(1)).join(", ");
}

function formatMask(mask) {
    if (!mask || !mask.length) {
        return "No mask";
    }

    return `${mask.length} points`;
}

function getConfidenceClass(confidence) {
    const percentage = confidence * 100;

    if (percentage > 75) {
        return "confidence-high";
    }

    if (percentage >= 50) {
        return "confidence-medium";
    }

    if (percentage >= 25) {
        return "confidence-low";
    }

    return "confidence-critical";
}

function renderRows(tbody, detections) {
    tbody.innerHTML = "";

    detections.forEach((detection, index) => {
        const confidenceClass = getConfidenceClass(detection.confidence);
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${index + 1}</td>
            <td><span class="class-badge">${detection.class_name}</span></td>
            <td><span class="confidence-badge ${confidenceClass}">${(detection.confidence * 100).toFixed(2)}%</span></td>
            <td>${formatBox(detection.bbox)}</td>
            <td>${formatMask(detection.mask)}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderDetections(detections) {
    detectionsContainer.innerHTML = "";

    if (!detections.length) {
        detectionsContainer.innerHTML = '<p class="empty-state">No detections found.</p>';
        return;
    }

    const classes = [...new Set(detections.map((detection) => detection.class_name))].sort();
    const table = document.createElement("table");
    table.className = "detections-table";
    table.innerHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Class<select id="class-filter" class="table-control"><option value="">All classes</option>${classes.map((className) => `<option value="${className}">${className}</option>`).join("")}</select></th>
                <th>Confidence<button id="confidence-sort" class="table-sort" type="button">Sort desc</button></th>
                <th>Box</th>
                <th>Mask</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;

    const tbody = table.querySelector("tbody");
    let selectedClass = "";
    let sortDirection = "desc";

    function updateTable() {
        const filteredDetections = detections
            .filter((detection) => !selectedClass || detection.class_name === selectedClass)
            .sort((a, b) => sortDirection === "desc" ? b.confidence - a.confidence : a.confidence - b.confidence);

        renderRows(tbody, filteredDetections);
    }

    detectionsContainer.appendChild(table);
    table.querySelector("#class-filter").addEventListener("change", (event) => {
        selectedClass = event.target.value;
        updateTable();
    });
    table.querySelector("#confidence-sort").addEventListener("click", (event) => {
        sortDirection = sortDirection === "desc" ? "asc" : "desc";
        event.target.textContent = sortDirection === "desc" ? "Sort desc" : "Sort asc";
        updateTable();
    });
    updateTable();
}

fileInput.addEventListener("change", () => {
    fileName.textContent = fileInput.files[0]?.name || "No file selected";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    const confidence = Number(confidenceInput.value) / 100;
    formData.append("file", file);
    formData.append("confidence", confidence);
    statusMessage.textContent = "Processing image...";
    detectionsContainer.innerHTML = "";
    resultImage.style.display = "none";
    previewPanel.classList.add("hidden");
    detectionsPanel.classList.add("hidden");
    resultTitle.classList.add("hidden");
    detectionsTitle.classList.add("hidden");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const prediction = await response.json();
        resultImage.src = prediction.fileout;
        resultImage.style.display = "block";
        previewPanel.classList.remove("hidden");
        detectionsPanel.classList.remove("hidden");
        resultTitle.classList.remove("hidden");
        detectionsTitle.classList.remove("hidden");
        statusMessage.textContent = prediction.message;
        renderDetections(prediction.detections);
    } catch (error) {
        console.error(error);
        statusMessage.textContent = "";
        alert("Error during prediction.");
    }
});

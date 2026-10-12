// ============================================================
// AGRIPREDICT AI - FRONTEND
// Tomato Disease Detection + Crop Yield Prediction
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

    // ============================================================
    // TOMATO DISEASE DETECTION
    // ============================================================

    const diseaseInput = document.getElementById("disease-image");
    const diseasePreview = document.getElementById("disease-preview");
    const diseaseResult = document.getElementById("disease-result");
    const diseaseName = document.getElementById("disease-name");
    const diseaseConfidence = document.getElementById("disease-confidence");

    if (diseaseInput) {

        diseaseInput.addEventListener("change", function () {

            const file = this.files && this.files[0];

            if (!file) {
                return;
            }

            console.log("Tomato image selected:", file.name);

            if (!file.type.startsWith("image/")) {
                alert("Please select a JPG, JPEG or PNG image.");
                this.value = "";
                return;
            }

            const reader = new FileReader();

            reader.onload = function (event) {

                if (diseasePreview) {
                    diseasePreview.src = event.target.result;
                    diseasePreview.classList.remove("hidden");

                    // Make absolutely sure the preview is visible
                    diseasePreview.style.display = "block";
                }
            };

            reader.onerror = function () {
                alert("Could not read the selected image.");
            };

            reader.readAsDataURL(file);
        });
    }


    // ============================================================
    // TOMATO DISEASE PREDICTION
    // ============================================================

    window.predictDisease = async function () {

        if (!diseaseInput ||
            !diseaseInput.files ||
            diseaseInput.files.length === 0) {

            alert("Please select a tomato leaf image first.");
            return;
        }

        const file = diseaseInput.files[0];

        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image.");
            return;
        }

        if (diseaseResult) {
            diseaseResult.classList.remove("hidden");
        }

        if (diseaseName) {
            diseaseName.textContent = "Analysing image...";
        }

        if (diseaseConfidence) {
            diseaseConfidence.textContent = "Please wait...";
        }

        const reader = new FileReader();

        reader.onload = async function (event) {

            try {

                const response = await fetch(
                    "http://127.0.0.1:8001/disease_predict",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            image: event.target.result
                        })
                    }
                );

                const result = await response.json();

                console.log("Disease prediction:", result);

                if (result.success) {

                    if (diseaseName) {
                        diseaseName.textContent = result.disease;
                    }

                    if (diseaseConfidence) {
                        diseaseConfidence.textContent =
                            "Confidence: " +
                            Number(result.confidence).toFixed(2) +
                            "%";
                    }

                } else {

                    if (diseaseName) {
                        diseaseName.textContent = "Prediction failed";
                    }

                    if (diseaseConfidence) {
                        diseaseConfidence.textContent =
                            result.error || "Unknown error";
                    }
                }

            } catch (error) {

                console.error("Disease API error:", error);

                if (diseaseName) {
                    diseaseName.textContent = "Connection error";
                }

                if (diseaseConfidence) {
                    diseaseConfidence.textContent =
                        "Disease API is not reachable. Make sure port 8001 is running.";
                }
            }
        };

        reader.readAsDataURL(file);
    };


    // ============================================================
    // CROP YIELD PREDICTION
    // ============================================================

    window.predictYield = async function () {

        const crop = document.getElementById("crop");
        const rainfall = document.getElementById("rainfall");
        const temperature = document.getElementById("temperature");
        const soilPh = document.getElementById("soil_ph");
        const fertilizer = document.getElementById("fertilizer");
        const area = document.getElementById("area");

        if (!crop ||
            !rainfall ||
            !temperature ||
            !soilPh ||
            !fertilizer ||
            !area) {

            alert("Yield prediction form could not be found.");
            return;
        }

        const data = {
            crop: crop.value,
            rainfall_mm: Number(rainfall.value),
            temperature_c: Number(temperature.value),
            soil_ph: Number(soilPh.value),
            fertilizer_kg: Number(fertilizer.value),
            area_hectare: Number(area.value)
        };

        console.log("Yield prediction request:", data);

        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                }
            );

            const result = await response.json();

            console.log("Yield prediction result:", result);

            if (!response.ok) {
                throw new Error(
                    result.error || "Yield prediction failed"
                );
            }

            // Find the result elements used by the page
            const resultBox =
                document.getElementById("result");

            const yieldValue =
                document.getElementById("prediction-value");

            const resultValue =
                document.getElementById("prediction-value");

            if (resultBox) {
                resultBox.classList.remove("hidden");
            }

            const predicted =
                result.predicted_yield ??
                result.yield ??
                result.prediction;

            if (yieldValue) {
                yieldValue.textContent =
                    Number(predicted).toFixed(2);
            }

            if (resultValue) {
                resultValue.textContent =
                    Number(predicted).toFixed(2);
            }

            // If the server returns a complete message
            const message =
                document.getElementById("result-message");

            if (message && result.message) {
                message.textContent = result.message;
            }

        } catch (error) {

            console.error("Yield prediction error:", error);

            alert(
                "Yield prediction failed: " +
                error.message
            );
        }
    };

});

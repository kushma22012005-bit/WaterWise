document.addEventListener("DOMContentLoaded", function () {
    loadPerformance();
    loadSummary();
    loadTrend();
    loadDataset();
    setupPredictionForm();
});

/* ---------------------------------
   1. MODEL PERFORMANCE
--------------------------------- */

async function loadPerformance() {
    const status = document.getElementById("performance-status");

    if (!document.getElementById("mae")) {
        return;
    }

    try {
        const response = await fetch("/api/performance");

        if (!response.ok) {
            throw new Error("Performance API is not available.");
        }

        const data = await response.json();

        document.getElementById("mae").textContent =
            formatNumber(data.mae);

        document.getElementById("rmse").textContent =
            formatNumber(data.rmse);

        document.getElementById("r2").textContent =
            formatNumber(data.r2);

        if (status) {
            status.textContent = "Model evaluation results loaded successfully.";
        }
    } catch (error) {
        if (status) {
            status.textContent =
                "Unable to load model metrics. Please check the FastAPI backend and /api/performance endpoint.";
        }

        console.error("Performance loading error:", error);
    }
}


/* ---------------------------------
   2. DASHBOARD SUMMARY
--------------------------------- */

async function loadSummary() {
    try {
        const response = await fetch("/api/summary");

        if (!response.ok) {
            throw new Error("Could not load summary.");
        }

        const data = await response.json();

        setText("total-records", data.total_records);
        setText(
            "average-consumption",
            formatNumber(data.average_consumption)
        );
        setText(
            "maximum-consumption",
            formatNumber(data.maximum_consumption)
        );
    } catch (error) {
        console.error("Summary loading error:", error);
    }
}


/* ---------------------------------
   3. HISTORICAL TREND
--------------------------------- */

async function loadTrend() {
    try {
        const response = await fetch("/api/trend");

        if (!response.ok) {
            throw new Error("Could not load historical trend.");
        }

        const data = await response.json();

        /*
          If your dashboard has an element with ID "trend-data",
          display the latest daily values as text.
          This does not replace a chart library if you already use one.
        */

        const trendElement = document.getElementById("trend-data");

        if (trendElement && data.dates && data.values) {
            if (data.dates.length === 0) {
                trendElement.textContent =
                    "No historical trend data is available.";
                return;
            }

            trendElement.textContent =
                "Latest available date: " +
                data.dates[data.dates.length - 1] +
                " | Daily consumption: " +
                formatNumber(data.values[data.values.length - 1]);
        }

        /*
          If your existing dashboard uses a chart, use the data
          arrays data.dates and data.values with that chart.
        */
        if (typeof window.updateConsumptionChart === "function") {
            window.updateConsumptionChart(data.dates, data.values);
        }
    } catch (error) {
        console.error("Trend loading error:", error);
    }
}


/* ---------------------------------
   4. DATASET PREVIEW
--------------------------------- */

async function loadDataset() {
    const tableBody = document.getElementById("dataset-body");

    if (!tableBody) {
        return;
    }

    try {
        const response = await fetch("/api/dataset");

        if (!response.ok) {
            throw new Error("Could not load dataset.");
        }

        const data = await response.json();

        tableBody.replaceChildren();

        data.rows.forEach(function (row) {
            const tr = document.createElement("tr");

            row.forEach(function (value) {
                const td = document.createElement("td");
                td.textContent = value == null ? "" : String(value);
                tr.appendChild(td);
            });

            tableBody.appendChild(tr);
        });
    } catch (error) {
        console.error("Dataset loading error:", error);

        tableBody.textContent = "";

        const tr = document.createElement("tr");
        const td = document.createElement("td");

        td.textContent = "Unable to load dataset records.";
        tr.appendChild(td);
        tableBody.appendChild(tr);
    }
}


/* ---------------------------------
   5. WATER CONSUMPTION PREDICTION
--------------------------------- */

function setupPredictionForm() {
    const form = document.getElementById("prediction-form");

    if (!form) {
        return;
    }

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        const status = document.getElementById("prediction-status");
        const resultBox = document.getElementById("prediction-result");
        const predictedValue = document.getElementById("predicted-value");
        const submitButton = form.querySelector(
            'button[type="submit"]'
        );

        if (status) {
            status.textContent = "Generating prediction...";
        }

        if (resultBox) {
            resultBox.hidden = true;
        }

        if (submitButton) {
            submitButton.disabled = true;
        }

        try {
            const formData = new FormData(form);

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || "Prediction could not be generated."
                );
            }

            const value =
                data.prediction ??
                data.predicted_consumption ??
                data.predicted_value;

            if (value === undefined || value === null) {
                throw new Error(
                    "The backend response did not contain a prediction value."
                );
            }

            if (predictedValue) {
                predictedValue.textContent = formatNumber(value);
            }

            if (resultBox) {
                resultBox.hidden = false;
            }

            if (status) {
                status.textContent = "Prediction generated successfully.";
            }
        } catch (error) {
            if (status) {
                status.textContent = error.message;
            }

            console.error("Prediction error:", error);
        } finally {
            if (submitButton) {
                submitButton.disabled = false;
            }
        }
    });
}


/* ---------------------------------
   6. SHARED HELPER FUNCTIONS
--------------------------------- */

function formatNumber(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toLocaleString("en-IN", {
        maximumFractionDigits: 3
    });
}

function setText(elementId, value) {
    const element = document.getElementById(elementId);

    if (element) {
        element.textContent = value ?? "—";
    }
}


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
            throw new Error("Performance API returned status " + response.status);
        }

        const data = await response.json();

        setText("mae", formatNumber(data.mae));
        setText("rmse", formatNumber(data.rmse));
        setText("r2", formatNumber(data.r2));

        if (status) {
            status.textContent = "Model evaluation results loaded successfully.";
        }
    } catch (error) {
        console.error("Performance loading error:", error);

        if (status) {
            status.textContent =
                "Unable to load model metrics. Please try again later.";
        }
    }
}

/* ---------------------------------
   2. DASHBOARD SUMMARY
--------------------------------- */

async function loadSummary() {
    const status = document.getElementById("dashboard-status");

    try {
        const response = await fetch("/api/summary");

        if (!response.ok) {
            throw new Error("Summary API returned status " + response.status);
        }

        const data = await response.json();

        setText("total-records", formatNumber(data.total_records));
        setText("average-consumption", formatNumber(data.average_consumption));
        setText("maximum-consumption", formatNumber(data.maximum_consumption));

        if (status) {
            status.textContent = "Dataset statistics loaded successfully.";
        }
    } catch (error) {
        console.error("Summary loading error:", error);

        if (status) {
            status.textContent =
                "Unable to load dataset statistics. Please refresh and try again.";
        }
    }
}

/* ---------------------------------
   3. HISTORICAL CONSUMPTION TREND
--------------------------------- */

async function loadTrend() {
    const chart = document.getElementById("consumption-chart");

    if (!chart) {
        return;
    }

    try {
        const response = await fetch("/api/trend");

        if (!response.ok) {
            throw new Error("Trend API returned status " + response.status);
        }

        const data = await response.json();

        if (
            !Array.isArray(data.dates) ||
            !Array.isArray(data.consumption) ||
            data.dates.length === 0 ||
            data.dates.length !== data.consumption.length
        ) {
            chart.textContent = "No valid historical trend data is available.";
            return;
        }

        if (typeof window.Plotly === "undefined") {
            chart.textContent =
                "The chart library could not load. Please refresh the page.";
            return;
        }

        await window.Plotly.newPlot(
            chart,
            [
                {
                    x: data.dates,
                    y: data.consumption,
                    type: "scatter",
                    mode: "lines",
                    name: "Water Consumption",
                    line: {
                        color: "#078f87",
                        width: 2
                    },
                    fill: "tozeroy",
                    fillcolor: "rgba(7, 143, 135, 0.10)",
                    hovertemplate:
                        "Date: %{x|%d %b %Y}<br>" +
                        "Consumption: %{y:,.2f}<extra></extra>"
                }
            ],
            {
                title: {
                    text: "Daily Water Consumption",
                    font: {
                        size: 18
                    }
                },
                xaxis: {
                    title: "Date",
                    type: "date",
                    gridcolor: "#e5eeee"
                },
                yaxis: {
                    title: "Consumption",
                    gridcolor: "#e5eeee",
                    zeroline: false
                },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                font: {
                    family: "Arial, sans-serif",
                    color: "#36545a"
                },
                margin: {
                    l: 70,
                    r: 25,
                    t: 60,
                    b: 65
                },
                autosize: true
            },
            {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["lasso2d", "select2d"]
            }
        );
    } catch (error) {
        console.error("Trend loading error:", error);

        chart.textContent =
            "Unable to load the historical consumption chart. Please try again.";
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
            throw new Error("Dataset API returned status " + response.status);
        }

        const data = await response.json();

        if (!Array.isArray(data.rows)) {
            throw new Error("The dataset response does not contain a rows array.");
        }

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

        tableBody.replaceChildren();

        const tr = document.createElement("tr");
        const td = document.createElement("td");

        td.textContent = "Unable to load dataset records.";
        td.colSpan = Math.max(
            1,
            document.querySelectorAll("#dataset-table thead th").length
        );

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

        const submitButton = form.querySelector('button[type="submit"]');

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
            console.error("Prediction error:", error);

            if (status) {
                status.textContent = error.message;
            }
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

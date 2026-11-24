const readyPlot = document.getElementById("{plot_id}");
readyPlot.dataset.plotReady = "true";
readyPlot.dispatchEvent(new Event("scan-refiner-ready"));

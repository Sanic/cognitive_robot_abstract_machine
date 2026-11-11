"use strict";

// Decode the collection wrappers emitted by DataclassJSONSerializer.
function decodeRecord(value) {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(decodeRecord);
  if (value.__items__) return value.__items__.map(decodeRecord);
  if (Array.isArray(value.keys) && Array.isArray(value.values)) return Object.fromEntries(value.keys.map((key, index) => [key, decodeRecord(value.values[index])]));
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decodeRecord(item)]));
}

const state = decodeRecord(JSON.parse(document.getElementById("viewer-state").textContent));
const scene = document.getElementById("scene");
const candidateSelect = document.getElementById("candidate");
const geometrySelect = document.getElementById("geometry");
const scanSelect = document.getElementById("scan");
const status = document.getElementById("status");
const candidates = state.registration.candidates;
const meshIndices = scene.data.map((trace, index) => trace.type === "mesh3d" ? index : -1).filter(index => index >= 0);
let originalRGB = Array.from(scene.data[0].marker.color);
const initialTransform = state.selected_candidate === null
  ? [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  : candidates[state.selected_candidate].scan_T_root;
const rootCoordinates = meshIndices.map(index => {
  const trace = scene.data[index];
  return [0,1,2].map(axis => Float64Array.from(trace.x, (_, point) =>
    [trace.x[point],trace.y[point],trace.z[point]].reduce((sum, value, coordinate) =>
      sum + initialTransform[coordinate][axis] * (value + state.display_origin[coordinate] - initialTransform[coordinate][3]), 0)
  ));
});
let scanIsOverview = true;

if (!candidates.length) {
  candidateSelect.add(new Option("Original coordinates", ""));
  candidateSelect.disabled = true;
} else {
  candidates.forEach((candidate, index) => candidateSelect.add(new Option(
    `${index}: ${(100 * candidate.support_fraction).toFixed(1)}% sample support · median ${(1000 * candidate.median_distance).toFixed(1)} mm`, index
  )));
  candidateSelect.value = state.selected_candidate;
}

function refreshScan() {
  const selected = candidateSelect.value === "" ? null : Number(candidateSelect.value);
  const validHeatmap = scanIsOverview && state.has_heatmap && selected === state.selected_candidate;
  scanSelect.querySelector('[value="heatmap"]').disabled = !validHeatmap;
  if (!validHeatmap && scanSelect.value === "heatmap") scanSelect.value = "rgb";
  const heatmap = scanSelect.value === "heatmap";
  Plotly.restyle(scene, {
    visible: scanSelect.value !== "none",
    "marker.color": [heatmap ? state.scan_distances : originalRGB],
    "marker.showscale": heatmap
  }, [0]);
  status.textContent = selected === null
    ? "No registration candidates. Run the align command to generate placements automatically."
    : `${selected === state.accepted_candidate ? "Accepted" : "Provisional"} candidate ${selected}. ` +
      (validHeatmap ? "Heatmap available for this placement." : "Generate a heatmap for this candidate with the view command.");
}

function refreshGeometry() {
  const visible = meshIndices.map(index => geometrySelect.value === "both" || geometrySelect.value === scene.data[index].legendgroup);
  Plotly.restyle(scene, { visible }, meshIndices);
}

candidateSelect.addEventListener("change", () => {
  const transform = candidates[Number(candidateSelect.value)].scan_T_root;
  const coordinates = rootCoordinates.map(root => [0,1,2].map(axis =>
    Float32Array.from(root[0], (_, point) => transform[axis][3] - state.display_origin[axis] +
      root.reduce((sum, values, column) => sum + transform[axis][column] * values[point], 0))
  ));
  Plotly.restyle(scene, {
    x: coordinates.map(axes => axes[0]),
    y: coordinates.map(axes => axes[1]),
    z: coordinates.map(axes => axes[2])
  }, meshIndices);
  refreshScan();
});
geometrySelect.addEventListener("change", refreshGeometry);
scanSelect.addEventListener("change", refreshScan);
document.getElementById("point-size").addEventListener("input", event => Plotly.restyle(scene, {"marker.size": Number(event.target.value)}, [0]));
document.getElementById("mesh-opacity").addEventListener("change", event => Plotly.restyle(scene, {opacity: Number(event.target.value)}, meshIndices));
document.getElementById("point-hover").addEventListener("change", event => Plotly.restyle(scene, {hoverinfo: event.target.checked ? "all" : "skip"}, [0]));
document.getElementById("rotate-view").addEventListener("click", () => Plotly.relayout(scene, {"scene.dragmode": "orbit"}));
document.querySelectorAll("[data-camera]").forEach(button => button.addEventListener("click", () => {
  const view = button.dataset.camera;
  const camera = view === "top"
    ? { eye: { x: 0, y: 0, z: 2.5 }, up: { x: 0, y: 1, z: 0 }, projection: { type: "orthographic" } }
    : view === "front"
      ? { eye: { x: 0, y: -2.5, z: 0 }, up: { x: 0, y: 0, z: 1 }, projection: { type: "orthographic" } }
      : { eye: { x: 1.25, y: 1.25, z: 1.25 }, up: { x: 0, y: 0, z: 1 }, projection: { type: "perspective" } };
  Plotly.relayout(scene, { "scene.camera": camera, "scene.dragmode": "orbit" });
}));
document.getElementById("display-origin").textContent = `Scan origin of local view: ${state.display_origin.map(value => value.toFixed(4)).join(", ")} m.`;
refreshScan();

// Regional detail is optional; self-contained pages still work without a server.
const regionStatus = document.getElementById("region-status");
const lowerInput = document.getElementById("region-lower");
const upperInput = document.getElementById("region-upper");
const loadRegionButton = document.getElementById("load-region");
const loadOverviewButton = document.getElementById("load-overview");
let indexInformation = null;
let pendingRegion = null;

scene.on("plotly_click", event => {
  const trace = scene.data[event.points[0].curveNumber];
  if (trace.type !== "mesh3d") return;
  const axes = [trace.x, trace.y, trace.z];
  const lower = axes.map((values, axis) => values.reduce((minimum, value) => Math.min(minimum, value), Infinity) + state.display_origin[axis] - 0.15);
  const upper = axes.map((values, axis) => values.reduce((maximum, value) => Math.max(maximum, value), -Infinity) + state.display_origin[axis] + 0.15);
  lowerInput.value = lower.map(value => value.toFixed(6)).join(",");
  upperInput.value = upper.map(value => value.toFixed(6)).join(",");
  regionStatus.textContent = `${trace.name} selected with 15 cm of surrounding scan context. Load the region for more detail.`;
});

async function loadRegion(lower, upper) {
  if (!indexInformation) return;
  if (pendingRegion) pendingRegion.abort();
  pendingRegion = new AbortController();
  const currentRequest = pendingRegion;
  regionStatus.textContent = "Loading indexed scan region…";
  const count = document.getElementById("point-budget").value;
  const query = new URLSearchParams({lower: lower.join(","), upper: upper.join(","), count});
  try {
    const response = await fetch(`/api/region?${query}`, {signal: currentRequest.signal});
    if (!response.ok) throw new Error(`Region request failed (${response.status}). Check bounds and point budget.`);
    const payload = await response.arrayBuffer();
    if (pendingRegion !== currentRequest) return;
    const view = new DataView(payload);
    const layout = indexInformation;
    if (payload.byteLength % layout.record_size) throw new Error("Incomplete regional evidence response.");
    const retainedCount = payload.byteLength / layout.record_size;
    const coordinates = [new Float32Array(retainedCount), new Float32Array(retainedCount), new Float32Array(retainedCount)];
    const colors = new Array(retainedCount);
    const identities = new Array(retainedCount);
    for (let point = 0; point < retainedCount; point++) {
      const offset = point * layout.record_size;
      for (let axis = 0; axis < 3; axis++) coordinates[axis][point] = view.getFloat64(offset + layout.record_offsets.points + axis * 8, true) - state.display_origin[axis];
      colors[point] = "#" + [0,1,2].map(channel => view.getUint8(offset + layout.record_offsets.colors + channel).toString(16).padStart(2,"0")).join("");
      identities[point] = `${view.getUint32(offset + layout.record_offsets.station_indices, true)}:${view.getBigUint64(offset + layout.record_offsets.source_indices, true)}`;
    }
    originalRGB = colors;
    scanIsOverview = false;
    scanSelect.value = "rgb";
    await Plotly.restyle(scene, {x:[coordinates[0]], y:[coordinates[1]], z:[coordinates[2]], customdata:[identities], "marker.color":[colors], "marker.showscale":false, visible:true}, [0]);
    refreshScan();
    regionStatus.textContent = `${retainedCount.toLocaleString()} indexed original records displayed within the selected region (budget ${Number(count).toLocaleString()}).`;
  } catch (error) {
    if (error.name !== "AbortError") regionStatus.textContent = error.message;
  }
}

loadRegionButton.addEventListener("click", () => {
  const lower = lowerInput.value.split(",").map(Number);
  const upper = upperInput.value.split(",").map(Number);
  if (lower.length !== 3 || upper.length !== 3 || ![...lower,...upper].every(Number.isFinite) || lower.some((value, axis) => value > upper[axis])) {
    regionStatus.textContent = "Enter three finite ordered coordinates for each scan bound.";
    return;
  }
  loadRegion(lower, upper);
});
loadOverviewButton.addEventListener("click", () => {
  lowerInput.value = indexInformation.index.lower.join(",");
  upperInput.value = indexInformation.index.upper.join(",");
  loadRegion(indexInformation.index.lower, indexInformation.index.upper);
});

async function discoverScanIndex() {
  if (location.protocol === "file:") {
    regionStatus.textContent = "Serve this project with the refiner's serve command to load indexed regional detail.";
    return;
  }
  try {
    const response = await fetch("/api/scan-info");
    if (!response.ok) {
      regionStatus.textContent = "Preview only: attach a completed scan index and use the refiner's serve command for regional detail.";
      return;
    }
    indexInformation = decodeRecord(await response.json());
    if (indexInformation.index.schema_version !== 1) throw new Error("Unsupported scan index format.");
    loadRegionButton.disabled = loadOverviewButton.disabled = false;
    lowerInput.value = indexInformation.index.lower.join(",");
    upperInput.value = indexInformation.index.upper.join(",");
    regionStatus.textContent = `${indexInformation.index.point_count.toLocaleString()} indexed records available. Click a body, then load its region.`;
  } catch (error) {
    regionStatus.textContent = `Regional detail unavailable: ${error.message}`;
  }
}
discoverScanIndex();


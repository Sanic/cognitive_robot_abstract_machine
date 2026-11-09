/** Link two world inspectors through a saved, auditable geometric candidate table. */
import {ExportManifest} from './model.js';
import {ComparisonData} from './comparison_model.js';

// %% Two-world review

class ComparisonInspector {
  /** Keep frame messages scoped to the two same-origin world inspectors. */
  constructor() {
    this.elements = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
    this.frames = {prediction: this.elements['prediction-frame'], ground_truth: this.elements['ground-truth-frame']};
    this.ready = new Set();
    this.selection = null;
    this.data = null;
    this.sequence = 0;
    this.highlights = {prediction: [], ground_truth: []};
    this.elements['open-comparison'].addEventListener('submit', event => {
      event.preventDefault();
      this.load(this.elements['comparison-url'].value).catch(error => this.error(error));
    });
    for (const identifier of ['review-filter', 'review-search']) this.elements[identifier].addEventListener('input', () => this.list());
    for (const identifier of ['pair-opacity', 'pair-isolate']) this.elements[identifier].addEventListener('input', () => this.displaySettings());
    this.elements['fit-pair'].addEventListener('click', () => this.highlight(true));
    this.elements['reset-pair'].addEventListener('click', () => {
      this.selection = null;
      this.elements['pair-isolate'].checked = false;
      this.elements['review-search'].value = '';
      this.elements['review-filter'].value = 'all';
      for (const side of Object.keys(this.frames)) this.send(side, {kind: 'reset-world'});
      this.displaySettings();
      this.highlight(false);
      this.details();
      this.list();
    });
    window.addEventListener('message', event => {
      if (event.origin !== window.location.origin || !this.data) return;
      const side = Object.keys(this.frames).find(side => this.frames[side].contentWindow === event.source);
      if (!side) return;
      if (event.data?.kind === 'world-ready') {
        this.ready.add(side);
        if (side === 'prediction') this.send(side, {kind: 'place-world', transform: this.data.document.ground_truth_T_prediction});
        this.displaySettings();
        this.highlight(false);
        this.status();
      }
      if (event.data?.kind === 'body-selected' && this.ready.has(side)) this.select(side, event.data.identifier, false);
    });
  }

  /** Fetch an input with its own HTTP failure context. */
  async readJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Cannot load ${url.pathname} (${response.status}).`);
    return await response.json();
  }

  /** Load both exact export manifests before replacing the displayed comparison. */
  async load(directory) {
    const sequence = ++this.sequence;
    this.elements['comparison-status'].textContent = 'Loading comparison…';
    const url = new URL(directory, window.location.href);
    const base = new URL('./', url);
    const predictionUrl = new URL('prediction/', base);
    const groundTruthUrl = new URL('ground_truth/', base);
    const [table, prediction, groundTruth] = await Promise.all([
      this.readJson(url), this.readJson(new URL('manifest.json', predictionUrl)), this.readJson(new URL('manifest.json', groundTruthUrl)),
    ]);
    if (sequence !== this.sequence) return;
    this.data = new ComparisonData(table, new ExportManifest(prediction), new ExportManifest(groundTruth));
    this.sources = {prediction: predictionUrl, ground_truth: groundTruthUrl};
    this.selection = null;
    this.ready.clear();
    for (const side of Object.keys(this.frames)) {
      const frameUrl = new URL('index.html', window.location.href);
      frameUrl.searchParams.set('embedded', '1');
      frameUrl.searchParams.set('export', this.sources[side].href);
      this.frames[side].src = frameUrl.href;
    }
    this.elements['comparison-url'].value = url.href;
    const location = new URL(window.location.href);
    location.searchParams.set('comparison', url.href);
    window.history.replaceState(null, '', location);
    const provenance = this.elements.provenance;
    provenance.replaceChildren();
    for (const [label, value] of [
      ['Prediction', this.data.document.prediction_source], ['GT', this.data.document.ground_truth_source],
      ['Prediction manifest SHA-256', this.data.document.prediction_digest], ['GT manifest SHA-256', this.data.document.ground_truth_digest],
      ['Policy', `distance limit ${this.data.document.comparison.settings.distance_limit} m; size weight ${this.data.document.comparison.settings.size_weight}; maximum cost ${this.data.document.comparison.settings.maximum_cost}; many-to-one`],
    ]) {
      const line = document.createElement('p');
      line.textContent = `${label}: ${value}`;
      provenance.append(line);
    }
    this.list();
    this.details();
    this.status();
  }

  /** Select in either direction, preserving all predictions associated with one GT. */
  select(side, identifier, frame = true, preview = null) {
    this.selection = identifier === null ? null : {side, identifier, preview};
    this.highlight(frame);
    this.details();
    this.list();
  }

  /** Resolve highlights exclusively from the saved table or an explicit candidate preview. */
  highlight(frame) {
    if (!this.data) return;
    this.highlights = {prediction: [], ground_truth: []};
    if (this.selection?.side === 'prediction') {
      const record = this.data.forPrediction(this.selection.identifier);
      this.highlights.prediction = [this.selection.identifier];
      const target = this.selection.preview ?? record?.ground_truth_identifier;
      if (target !== null && target !== undefined) this.highlights.ground_truth = [target];
    } else if (this.selection?.side === 'ground_truth') {
      this.highlights.ground_truth = [this.selection.identifier];
      this.highlights.prediction = this.data.forGroundTruth(this.selection.identifier).map(record => record.predicted_identifier);
    }
    for (const side of Object.keys(this.frames)) this.send(side, {kind: 'inspect-bodies', identifiers: this.highlights[side], frame, active: Boolean(this.selection)});
  }

  /** Send commands only once the receiving world has completed its GLB load. */
  send(side, message) {
    if (this.ready.has(side)) this.frames[side].contentWindow.postMessage(message, window.location.origin);
  }

  /** Apply a shared opacity and isolation policy without mutating either world. */
  displaySettings() {
    for (const side of Object.keys(this.frames)) this.send(side, {kind: 'display-settings', opacity: Number(this.elements['pair-opacity'].value), isolate: this.elements['pair-isolate'].checked});
  }

  /** Display counts as inspection outcomes, rather than GT accuracy scores. */
  status() {
    const records = this.data.records;
    const matched = records.filter(record => record.ground_truth_identifier !== null).length;
    this.elements['comparison-status'].classList.remove('error');
    this.elements['comparison-status'].textContent = `${this.ready.size}/2 worlds loaded · ${matched}/${records.length} provisional matches · ${this.data.document.comparison.unmatched_ground_truth.length} GT unassigned`;
  }

  /** Keep unmatched GT in its own list so unobserved rooms are not called missed objects. */
  list() {
    if (!this.data) return;
    const rows = this.data.rows(this.elements['review-filter'].value, this.elements['review-search'].value);
    this.elements['review-count'].textContent = rows.length;
    this.elements['review-objects'].replaceChildren();
    for (const row of rows) {
      const manifest = row.side === 'prediction' ? this.data.prediction : this.data.groundTruth;
      const body = manifest.bodiesById.get(row.identifier);
      const button = document.createElement('button');
      button.className = 'object-button';
      button.setAttribute('aria-pressed', String(this.selection?.side === row.side && this.selection.identifier === row.identifier));
      const name = document.createElement('span');
      name.textContent = body.name;
      const summary = document.createElement('small');
      const record = row.side === 'prediction' ? this.data.forPrediction(row.identifier) : null;
      summary.textContent = record ? record.ground_truth_identifier === null ? 'Unmatched prediction' : `${record.class_disagreement ? 'Class differs · ' : ''}${record.parent_disagreement ? 'Parent differs · ' : ''}cost ${record.candidates[0].cost.toFixed(3)}` : 'GT without an assigned prediction';
      button.append(name, summary);
      button.addEventListener('click', () => this.select(row.side, row.identifier));
      this.elements['review-objects'].append(button);
    }
    if (!rows.length) this.paragraph(this.elements['review-objects'], 'No objects match this filter.');
  }

  /** Expose the accepted match, ranked alternatives and the selected object's hierarchy. */
  details() {
    const panel = this.elements['match-details'];
    panel.replaceChildren();
    if (!this.selection) {
      this.paragraph(panel, 'Click an object in either world, or select one from the review list.');
      return;
    }
    const {side, identifier, preview} = this.selection;
    const manifest = side === 'prediction' ? this.data.prediction : this.data.groundTruth;
    const body = manifest.bodiesById.get(identifier);
    const title = document.createElement('h2');
    title.textContent = body.name;
    panel.append(title);
    this.paragraph(panel, manifest.classesFor(body).join(', ') || 'Unannotated');
    if (side === 'prediction') {
      const record = this.data.forPrediction(identifier);
      if (!record) {
        this.paragraph(panel, 'This exported object was not included in geometric matching.');
      } else {
        const target = this.data.groundTruth.bodiesById.get(record.ground_truth_identifier);
        this.paragraph(panel, target ? `Saved match: ${target.name}` : 'No accepted GT match. Alternatives below remain candidates.');
        if (preview !== null) this.paragraph(panel, 'Previewing an alternative in GT. The saved match above has not changed.');
        for (const [text, disagreement] of [['Class', record.class_disagreement], ['Parent', record.parent_disagreement]]) {
          const badge = document.createElement('span');
          badge.className = `badge${disagreement === true ? ' problem' : ''}`;
          badge.textContent = `${text}: ${disagreement === null ? 'not judged' : disagreement ? 'differs' : 'agrees'}`;
          panel.append(badge);
        }
        this.paragraph(panel, record.margin === null ? 'No competing candidate.' : `Cost margin to runner-up: ${record.margin.toFixed(4)}${record.margin === 0 ? ' — tied geometric evidence' : ''}`);
        if (target) this.paragraph(panel, `${this.data.forGroundTruth(target.id).length} prediction object(s) assigned to this GT object.`);
        const heading = document.createElement('h3');
        heading.textContent = 'Ranked candidates · lower cost is better';
        panel.append(heading);
        for (const candidate of record.candidates) {
          const other = this.data.groundTruth.bodiesById.get(candidate.ground_truth_identifier);
          const button = document.createElement('button');
          button.className = `candidate${preview === other.id ? ' preview' : ''}`;
          const name = document.createElement('span');
          name.textContent = `${other.name}${record.ground_truth_identifier === other.id ? ' · saved match' : ''}`;
          const quality = document.createElement('small');
          quality.textContent = `Cost ${candidate.cost.toFixed(4)} · distance ${candidate.distance.toFixed(3)} m · size difference ${candidate.size_difference.toFixed(3)}`;
          button.append(name, quality);
          button.addEventListener('click', () => this.select('prediction', identifier, true, other.id));
          panel.append(button);
        }
      }
    } else {
      const records = this.data.forGroundTruth(identifier);
      this.paragraph(panel, records.length ? `${records.length} assigned prediction(s):` : 'No assigned prediction. This can also mean the scan did not observe this object.');
      for (const record of records) this.objectButton(panel, 'prediction', this.data.prediction.bodiesById.get(record.predicted_identifier));
    }
    this.relations(panel, side, manifest, body);
  }

  /** Retain parent and part navigation while reviewing a correspondence. */
  relations(panel, side, manifest, body) {
    const parent = manifest.parentOf(body);
    const heading = document.createElement('h3');
    heading.textContent = 'Parent object';
    panel.append(heading);
    if (parent) this.objectButton(panel, side, parent);
    else this.paragraph(panel, 'At the world root');
    const children = manifest.childrenOf(body);
    const parts = document.createElement('h3');
    parts.textContent = `Parts / child objects (${children.length})`;
    panel.append(parts);
    for (const child of children) this.objectButton(panel, side, child);
  }

  /** Use one relationship control for GT and prediction objects. */
  objectButton(panel, side, body) {
    const button = document.createElement('button');
    button.className = 'candidate';
    button.textContent = body.name;
    button.addEventListener('click', () => this.select(side, body.id));
    panel.append(button);
  }

  /** Append plain text without interpreting exported object names as markup. */
  paragraph(panel, text) {
    const line = document.createElement('p');
    line.textContent = text;
    panel.append(line);
  }

  /** Keep input errors visible without claiming a successful load. */
  error(error) {
    this.elements['comparison-status'].textContent = error.message;
    this.elements['comparison-status'].classList.add('error');
  }
}

// %% Startup

const inspector = new ComparisonInspector();
const source = new URLSearchParams(window.location.search).get('comparison');
if (source) inspector.load(source).catch(error => inspector.error(error));

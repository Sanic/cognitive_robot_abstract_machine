/** Navigate a pipeline's objects and final relationships without a database. */
import {ExportManifest} from './model.js';
import {WorldScene} from './scene.js';

// %% World navigation

class WorldInspector {
  /** Bind controls once, preserving the current world if loading another fails. */
  constructor() {
    this.elements = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
    this.manifest = null;
    this.selected = null;
    this.hidden = new Set();
    this.comparisonHighlights = new Set();
    this.comparisonSelectionActive = false;
    this.sequence = 0;
    this.scene = new WorldScene(this.elements.viewport, identifier => this.select(identifier));
    this.elements.folder.addEventListener('change', event => {
      this.loadFolder([...event.target.files]).catch(error => this.showError(error));
      event.target.value = '';
    });
    this.elements['open-run'].addEventListener('submit', event => {
      event.preventDefault();
      this.loadUrl(this.elements['run-url'].value, true).catch(error => this.showError(error));
    });
    for (const name of ['search', 'classes', 'isolate', 'opacity']) {
      this.elements[name].addEventListener('input', () => this.update());
    }
    this.elements.fit.addEventListener('click', () => this.scene.fit(this.focusBodies()));
    this.elements.reset.addEventListener('click', () => {
      if (!this.manifest) return;
      this.hidden.clear();
      this.elements.search.value = '';
      this.elements.classes.value = '';
      this.elements.isolate.checked = false;
      this.select(null);
      this.scene.fit(this.manifest.bodies);
    });
  }

  /** Read a folder locally; only the manifest and its GLB are consumed. */
  async loadFolder(files) {
    const sequence = ++this.sequence;
    const manifests = files.filter(file => file.name === 'manifest.json');
    if (manifests.length !== 1) throw new Error('Choose one run or world_mesh folder containing exactly one manifest.json.');
    this.elements.status.textContent = 'Reading export…';
    const manifestFile = manifests[0];
    const prefix = manifestFile.webkitRelativePath.slice(0, -manifestFile.name.length);
    const payload = JSON.parse(await manifestFile.text());
    const sceneFile = files.find(file => file.webkitRelativePath === prefix + payload.scene);
    if (!sceneFile) throw new Error(`The folder does not contain ${payload.scene}.`);
    await this.open(payload, await sceneFile.arrayBuffer(), prefix.replace(/\/$/, ''), sequence);
    if (sequence !== this.sequence) return;
    const url = new URL(window.location.href);
    url.searchParams.delete('run');
    url.searchParams.delete('export');
    window.history.replaceState(null, '', url);
    this.elements['run-url'].value = '';
  }

  /** Resolve a pipeline run to its existing world_mesh export. */
  async loadUrl(directory, run) {
    const sequence = ++this.sequence;
    this.elements.status.textContent = 'Loading export…';
    const directoryUrl = new URL(directory.endsWith('/') ? directory : directory + '/', window.location.href);
    const base = run ? new URL('world_mesh/', directoryUrl) : directoryUrl;
    const manifestResponse = await fetch(new URL('manifest.json', base));
    if (!manifestResponse.ok) throw new Error(`Cannot load manifest (${manifestResponse.status}). Check the run URL and HTTP server root.`);
    const payload = await manifestResponse.json();
    new ExportManifest(payload);
    const sceneResponse = await fetch(new URL(payload.scene, base));
    if (!sceneResponse.ok) throw new Error(`Cannot load scene (${sceneResponse.status}).`);
    await this.open(payload, await sceneResponse.arrayBuffer(), decodeURI(directoryUrl.pathname.split('/').filter(Boolean).pop()), sequence);
    if (sequence !== this.sequence) return;
    const url = new URL(window.location.href);
    url.searchParams.delete(run ? 'export' : 'run');
    url.searchParams.set(run ? 'run' : 'export', directoryUrl.href);
    window.history.replaceState(null, '', url);
    this.elements['run-url'].value = run ? directoryUrl.href : '';
  }

  /** Commit a fully parsed scene, discarding loads superseded by a newer request. */
  async open(payload, bytes, name, sequence) {
    if (sequence !== this.sequence) return;
    this.elements.status.textContent = 'Preparing selectable objects…';
    const manifest = new ExportManifest(payload);
    const loaded = await this.scene.read(bytes, manifest);
    if (sequence !== this.sequence) {
      this.scene.dispose(loaded.model);
      return;
    }
    this.scene.display(loaded);
    this.manifest = manifest;
    this.selected = null;
    this.hidden.clear();
    this.comparisonHighlights.clear();
    this.elements.search.value = '';
    this.elements.isolate.checked = false;
    this.elements.classes.replaceChildren(new Option('All classes', ''), ...manifest.labels.map(label => new Option(label, label)));
    this.elements['scene-name'].textContent = `${name} · ${manifest.bodies.filter(body => body.parent_id !== null).length} objects · ${manifest.labels.length} classes`;
    this.details();
    this.update();
    this.scene.fit(manifest.bodies);
    if (new URLSearchParams(window.location.search).has('embedded')) {
      window.parent.postMessage({kind: 'world-ready'}, window.location.origin);
    }
  }

  /** Report file and network problems without removing an already loaded world. */
  showError(error) {
    this.elements.status.textContent = error.message;
    this.elements.status.classList.add('error');
  }

  /** Use the same object set for camera framing and isolation. */
  focusBodies() {
    if (!this.manifest) return [];
    if (this.comparisonSelectionActive && !this.selected && !this.comparisonHighlights.size) return [];
    if (this.comparisonHighlights.size) return this.manifest.bodies.filter(body => this.comparisonHighlights.has(body.id));
    return this.selected ? this.manifest.assemblyOf(this.selected)
      : this.manifest.findBodies(this.elements.classes.value, this.elements.search.value);
  }

  /** Reveal an object reached from the list or a relationship link. */
  select(identifier, frame = false, notify = true) {
    if (!this.manifest) return;
    this.comparisonHighlights.clear();
    this.comparisonSelectionActive = false;
    this.selected = this.manifest.bodiesById.get(identifier) ?? null;
    if (this.selected) for (const body of this.manifest.assemblyOf(this.selected)) this.hidden.delete(body.id);
    this.details();
    this.update();
    if (frame) this.scene.fit(this.focusBodies());
    if (notify && new URLSearchParams(window.location.search).has('embedded')) {
      window.parent.postMessage({kind: 'body-selected', identifier: this.selected?.id ?? null}, window.location.origin);
    }
  }

  /** Keep filtering, selected parts, and manually hidden geometry independent. */
  update() {
    if (!this.manifest) return;
    const matches = this.manifest.findBodies(this.elements.classes.value, this.elements.search.value);
    const matching = new Set(matches.map(body => body.id));
    const parts = new Set(this.selected ? this.manifest.assemblyOf(this.selected).map(body => body.id) : []);
    const parent = this.selected ? this.manifest.parentOf(this.selected) : null;
    const filtering = Boolean(this.elements.classes.value || this.elements.search.value.trim());
    const states = new Map();
    for (const body of this.manifest.bodies) {
      const selected = body === this.selected;
      const related = parts.has(body.id);
      const compared = this.comparisonHighlights.has(body.id);
      const emphasized = selected || related || compared || body === parent || (filtering && matching.has(body.id));
      const isolated = this.comparisonHighlights.size ? compared : this.selected ? related : this.comparisonSelectionActive ? false : matching.has(body.id);
      const labelIndex = this.manifest.labels.indexOf(this.manifest.classesFor(body)[0]);
      const classColor = labelIndex < 0 ? '#7e92a8' : `hsl(${labelIndex * 137.508 % 360}, 55%, 63%)`;
      states.set(body.id, {
        visible: !this.hidden.has(body.id) && (!this.elements.isolate.checked || isolated),
        emphasized,
        color: selected ? '#ff45df' : related || compared ? '#00e5ff' : body === parent ? '#ffb74b' : classColor,
        opacity: emphasized || (!this.selected && !filtering) ? 1 : Number(this.elements.opacity.value),
      });
    }
    this.scene.paint(states);
    this.elements['selected-label'].textContent = this.selected?.name ?? (this.elements.classes.value || 'Whole world');
    this.elements['object-count'].textContent = matches.length;
    this.elements.status.classList.remove('error');
    this.elements.status.textContent = `${this.scene.meshesByBody.size} objects with geometry · ${this.hidden.size} hidden`;
    this.elements.objects.replaceChildren();
    for (const body of matches) {
      const button = document.createElement('button');
      button.className = 'object-button';
      button.setAttribute('aria-pressed', String(body === this.selected));
      const title = document.createElement('span');
      title.textContent = body.name;
      const subtitle = document.createElement('small');
      subtitle.textContent = `${this.manifest.classesFor(body).join(', ') || 'Unannotated'}${this.scene.meshesByBody.has(body.id) ? '' : ' · no geometry'}${this.hidden.has(body.id) ? ' · hidden' : ''}`;
      button.append(title, subtitle);
      button.addEventListener('click', () => this.select(body.id, true));
      this.elements.objects.append(button);
    }
    if (!matches.length) {
      const empty = document.createElement('p');
      empty.className = 'empty';
      empty.textContent = 'No objects match this search and class.';
      this.elements.objects.append(empty);
    }
    this.elements.fit.disabled = !this.focusBodies().length;
  }

  /** Expose identity and navigable relationships without inferred context labels. */
  details() {
    const panel = this.elements.details;
    panel.replaceChildren();
    panel.hidden = !this.selected;
    if (!this.selected) return;
    const body = this.selected;
    const close = document.createElement('button');
    close.className = 'close';
    close.textContent = '×';
    close.setAttribute('aria-label', 'Clear object selection');
    close.addEventListener('click', () => this.select(null));
    const heading = document.createElement('h2');
    heading.textContent = body.name;
    const classes = document.createElement('p');
    classes.textContent = this.manifest.classesFor(body).join(', ') || 'No semantic class assigned';
    const geometry = document.createElement('p');
    geometry.className = 'muted';
    geometry.textContent = `${body.face_count.toLocaleString()} faces${this.scene.meshesByBody.has(body.id) ? '' : ' · no geometry'}`;
    panel.append(close, heading, classes, geometry);
    const actions = document.createElement('div');
    actions.className = 'actions';
    for (const [title, action] of [
      ['Fit object', () => this.scene.fit([body])],
      ['Fit assembly', () => this.scene.fit(this.manifest.assemblyOf(body))],
      ['Hide object', () => { this.hidden.add(body.id); this.select(null); }],
    ]) {
      const button = document.createElement('button');
      button.textContent = title;
      button.addEventListener('click', action);
      actions.append(button);
    }
    panel.append(actions);
    this.relationships('Parent object', this.manifest.parentOf(body) ? [this.manifest.parentOf(body)] : []);
    this.relationships('Parts / child objects', this.manifest.childrenOf(body));
    const ancestors = this.manifest.ancestorsOf(body).slice(1);
    if (ancestors.length) this.relationships('Enclosing objects', ancestors);
    const technical = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Object identifiers';
    const identity = document.createElement('dl');
    for (const [title, value] of [['Export ID', body.id], ['World ID', body.source_id]]) {
      const term = document.createElement('dt');
      term.textContent = title;
      const definition = document.createElement('dd');
      definition.textContent = value;
      identity.append(term, definition);
    }
    technical.append(summary, identity);
    panel.append(technical);
  }

  /** Follow real exported parent/child edges instead of parsing class names. */
  relationships(title, bodies) {
    const heading = document.createElement('h3');
    heading.textContent = `${title} (${bodies.length})`;
    this.elements.details.append(heading);
    for (const body of bodies) {
      const button = document.createElement('button');
      button.className = 'relation';
      const name = document.createElement('span');
      name.textContent = body.name;
      const classes = document.createElement('small');
      classes.textContent = this.manifest.classesFor(body).join(', ') || 'Unannotated';
      button.append(name, classes);
      button.addEventListener('click', () => this.select(body.id, true));
      this.elements.details.append(button);
    }
    if (!bodies.length) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = title === 'Parent object' ? 'At the world root' : 'None';
      this.elements.details.append(empty);
    }
  }
}

// %% Startup

const inspector = new WorldInspector();
const parameters = new URLSearchParams(window.location.search);
if (parameters.has('embedded')) {
  document.body.classList.add('embedded');
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.origin !== window.location.origin || !inspector.manifest) return;
    if (event.data?.kind === 'inspect-bodies' && Array.isArray(event.data.identifiers)) {
      const identifiers = event.data.identifiers.filter(identifier => inspector.manifest.bodiesById.has(identifier));
      inspector.select(identifiers.length === 1 ? identifiers[0] : null, false, false);
      inspector.comparisonSelectionActive = event.data.active !== false;
      if (identifiers.length !== 1) inspector.comparisonHighlights = new Set(identifiers);
      for (const identifier of identifiers) inspector.hidden.delete(identifier);
      inspector.update();
      if (event.data.frame && identifiers.length) inspector.scene.fit(identifiers.map(identifier => inspector.manifest.bodiesById.get(identifier)));
    }
    if (event.data?.kind === 'display-settings') {
      inspector.elements.opacity.value = String(Math.max(0, Math.min(1, Number(event.data.opacity))));
      inspector.elements.isolate.checked = Boolean(event.data.isolate);
      inspector.update();
    }
    if (event.data?.kind === 'place-world' && Array.isArray(event.data.transform) &&
        event.data.transform.length === 4 && event.data.transform.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite))) {
      inspector.scene.place(event.data.transform);
      inspector.scene.fit(inspector.manifest.bodies);
    }
    if (event.data?.kind === 'reset-world') inspector.elements.reset.click();
  });
}
const directory = parameters.get('run') ?? parameters.get('export');
if (directory) inspector.loadUrl(directory, parameters.has('run')).catch(error => inspector.showError(error));

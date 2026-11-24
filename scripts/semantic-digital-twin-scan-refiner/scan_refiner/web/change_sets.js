"use strict";

// %% Agent proposals in the existing inspection scene
class ChangeSetPanel {
  constructor(workspace) {
    this.workspace = workspace;
    this.controller = workspace.controller;
    this.tools = workspace.tools;
    this.list = null;
    this.review = null;
    this.busy = false;
    this.active = false;
    this.proposed = false;
    this.status = document.getElementById('change-set-status');
    this.picker = document.getElementById('change-set-picker');
    this.picker.onchange = () => this.open(this.picker.value);
    document.getElementById('refresh-change-sets').onclick = () => this.refresh();
    document.getElementById('close-change-set').onclick = () => this.open('');
    document.getElementById('change-set-original').onclick = () => this.show(false);
    document.getElementById('change-set-proposed').onclick = () => this.show(true);
    document.getElementById('accept-change-set').onclick = () => this.decide(true);
    document.getElementById('reject-change-set').onclick = () => this.decide(false);
    document.getElementById('load-change-set-evidence').onclick = () => {
      const query = this.review.proposal.evidence.query;
      return this.controller.loadRegion(query.lower, query.upper, query.frame_T_scan);
    };
    document.getElementById('change-set-heatmaps').onclick = () => this.heatmaps();
  }

  get current() {
    return this.review?.status === 'pending' &&
      this.review.proposal.request.expected_revision === this.tools?.viewRevision;
  }

  get heatmapJob() {
    const identifier = this.review?.proposal.identifier;
    if (!identifier) return null;
    const candidate = document.getElementById('candidate').value;
    return this.workspace.jobs?.records.find(job => job.request.kind === 'change-set-heatmap' &&
      job.request.change_set_id === identifier && String(job.request.candidate_index) === candidate);
  }

  get displayedHeatmaps() {
    return this.controller.proposalHeatmap?.job.identifier === this.heatmapJob?.identifier &&
      !!this.controller.proposalHeatmap;
  }

  controls() {
    const busy = this.busy || this.tools?.applying || this.tools?.loading ||
      this.tools?.changingHistory || this.workspace.refiner?.busy;
    const available = !!this.tools?.viewRevision;
    document.getElementById('refresh-change-sets').disabled = !available || !!busy;
    this.picker.disabled = !!busy || !this.list?.reviews.length;
    for (const id of ['accept-change-set', 'reject-change-set']) {
      document.getElementById(id).disabled = !!busy || !this.current || !this.list?.can_decide;
    }
    document.getElementById('close-change-set').disabled = !!busy;
    const evidence = this.review?.proposal.evidence;
    const load = document.getElementById('load-change-set-evidence');
    load.hidden = !evidence || !this.controller.indexInformation;
    load.disabled = !!busy || !this.current;
    const heatmaps = document.getElementById('change-set-heatmaps');
    const summary = document.getElementById('change-set-heatmap-status');
    const job = this.heatmapJob;
    const running = ['queued', 'running'].includes(job?.status);
    const candidate = document.getElementById('candidate').value;
    const correctPlacement = candidate !== '' && (evidence?.query.candidate_index == null ||
      String(evidence.query.candidate_index) === candidate);
    heatmaps.hidden = summary.hidden = !evidence;
    heatmaps.disabled = !!busy || !this.current || !correctPlacement || running;
    heatmaps.textContent = running ? 'Computing paired heatmaps…' :
      job?.status === 'succeeded' ? 'Show paired heatmaps' : 'Compute paired heatmaps';
    summary.textContent = this.heatmapError || (!correctPlacement ? 'Select the linked placement in Align.' :
      running ? job.message : this.displayedHeatmaps ? this.heatmapSummary :
      job?.status === 'succeeded' ? 'Ready to inspect in 3D.' :
      job?.status === 'failed' ? 'Heatmap calculation failed. See Jobs for details.' :
      'Compare both versions against the same scan points and visual surfaces.');
    for (const [id, proposed] of [['change-set-original', false], ['change-set-proposed', true]]) {
      const button = document.getElementById(id);
      button.disabled = !!busy || !this.current;
      button.setAttribute('aria-pressed', String(this.active && this.proposed === proposed));
    }
  }

  async request(operation, type, fields = {}) {
    const response = await fetch(`/api/agent/${operation}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(InspectionTools.encode({
        __json_type__: `scan_refiner.agent.models.${type}`, ...fields,
      })),
    });
    if (!response.headers.get('Content-Type')?.startsWith('application/json')) {
      throw Error('This viewer needs a server that supports change sets.');
    }
    const value = InspectionController.decode(await response.json());
    if (!response.ok) throw Error(value.message);
    return value;
  }

  async refresh() {
    if (!this.tools?.viewRevision || this.busy || this.tools.applying) return;
    this.busy = true;
    this.controls();
    try {
      const selected = this.picker.value;
      this.list = await this.request('list_change_sets', 'ProjectQuery');
      for (const item of this.list.reviews) {
        item.status = item.status.name.toLowerCase();
      }
      this.picker.replaceChildren(new Option('Choose a proposal', ''));
      for (const {proposal, status} of this.list.reviews) {
        const label = proposal.request.reasoning.explanation.trim().slice(0, 62);
        this.picker.add(new Option(`${status} · ${label}`, proposal.identifier));
      }
      this.picker.value = selected;
      if (selected && selected !== this.picker.value) await this.close();
      if (this.review) {
        this.review = this.list.reviews.find(item => item.proposal.identifier === selected) || null;
        if (this.active && !this.current) await this.close();
      }
      this.status.textContent = this.review ? this.description() :
        this.list.reviews.length ? 'Choose an agent proposal to review.' : 'Submitted change sets appear here.';
    } catch (error) {
      this.status.textContent = error.message;
    } finally {
      this.busy = false;
      this.controls();
    }
  }

  description() {
    if (this.review.status === 'undone') return 'This change set was undone. Use Redo to restore it.';
    if (this.review.status !== 'pending') return `Review status: ${this.review.status}.`;
    if (!this.current) return 'This proposal belongs to a different displayed revision. Refresh the project before reviewing it.';
    const guidance = 'Compare both versions from any angle.';
    return this.list.can_decide ? guidance : `${guidance} Applying requires a server started in refine mode.`;
  }

  async open(identifier) {
    if (this.busy) return;
    await this.close();
    this.picker.value = identifier;
    this.review = this.list?.reviews.find(item => item.proposal.identifier === identifier) || null;
    if (!this.review) {
      this.status.textContent = 'Choose an agent proposal to review.';
      return;
    }
    const {edits, reasoning} = this.review.proposal.request;
    const report = document.getElementById('change-set-reasoning');
    report.replaceChildren();
    for (const line of [reasoning.explanation, ...reasoning.assumptions]) {
      const row = document.createElement('p');
      row.textContent = line;
      report.append(row);
    }
    const bodies = document.getElementById('change-set-bodies');
    bodies.replaceChildren();
    for (const edit of edits) {
      const row = document.createElement('li');
      row.textContent = `${edit.body_name} · ΔXYZ ${edit.translation.map(value => (value * 1000).toFixed(2)).join(', ')} mm`;
      bodies.append(row);
    }
    document.getElementById('change-set-review').hidden = false;
    this.status.textContent = this.description();
    if (this.current) await this.show(false);
    this.controls();
  }

  async show(proposed) {
    if (this.busy || !this.current) return;
    this.busy = true;
    this.controls();
    const activity = this.controller.activity?.begin('Switching proposal geometry', 'Updating the affected bodies…');
    try {
      await this.controller.activity?.paint();
      if (!this.active) {
        await this.workspace.refiner?.clear();
        this.active = true;
        this.previousScan = document.getElementById('scan').value;
        await this.controller.refreshScan();
      }
      this.proposed = proposed;
      this.controller.changeSetOffsets = proposed ?
        new Map(this.review.proposal.request.edits.map(edit => [edit.body_name, edit.translation])) : null;
      await this.controller.orientFrame(this.controller.frameMode, this.controller.frameYaw, true);
      if (this.controller.proposalHeatmap) await this.controller.refreshScan();
    } catch (error) {
      this.status.textContent = error.message;
    } finally {
      this.busy = false;
      this.workspace.synchronize();
      this.controller.activity?.end(activity);
    }
  }

  async close() {
    document.getElementById('change-set-review').hidden = true;
    this.review = null;
    this.heatmapError = '';
    this.heatmapSummary = '';
    this.picker.value = '';
    if (!this.active) return;
    this.active = false;
    this.controller.proposalHeatmap = null;
    const moved = !!this.controller.changeSetOffsets;
    this.controller.changeSetOffsets = null;
    if (moved) await this.controller.orientFrame(this.controller.frameMode, this.controller.frameYaw, true);
    const scan = document.getElementById('scan');
    if (this.previousScan === 'heatmap' && scan.value === 'rgb') scan.value = 'heatmap';
    await this.controller.refreshScan();
    this.workspace.synchronize();
  }

  async heatmaps() {
    if (this.busy || !this.current || !this.review.proposal.evidence) return;
    const job = this.heatmapJob;
    this.busy = true;
    this.heatmapError = '';
    this.controls();
    try {
      if (job?.status === 'succeeded') {
        await this.showHeatmaps(job);
      } else if (!['queued', 'running'].includes(job?.status)) {
        await this.request('compute_change_set_heatmap', 'ChangeSetHeatmapQuery', {
          identifier: this.review.proposal.identifier,
          expected_revision: this.tools.viewRevision,
          candidate_index: Number(document.getElementById('candidate').value),
          maximum_points: Number(document.getElementById('point-budget').value),
        });
        await this.workspace.jobs.refresh();
      }
    } catch (error) {
      this.heatmapError = error.message;
    } finally {
      this.busy = false;
      this.controls();
    }
  }

  async showHeatmaps(job) {
    const activity = this.controller.activity?.begin('Loading paired heatmaps', 'Drawing the shared scan evidence…');
    try {
      await this.controller.activity?.paint();
      const ends = ['points', 'distances', 'proposed-distances', 'report'];
      const responses = await Promise.all(ends.map(end => fetch(`/api/jobs/${job.identifier}/${end}`)));
      if (responses.some(response => !response.ok) || !this.current) {
        throw Error('This comparison is unfinished or belongs to an earlier proposal.');
      }
      const [points, original, proposed] = await Promise.all(responses.slice(0, 3).map(response => response.arrayBuffer()));
      const report = InspectionController.decode(await responses[3].json());
      const source = this.controller.decodeCloud(points);
      const pair = {job, original: new Float32Array(original), proposed: new Float32Array(proposed)};
      if (source.identities.length !== pair.original.length || pair.original.length !== pair.proposed.length) {
        throw Error('The scan and distance arrays do not match.');
      }
      source.proposalComparison = pair;
      this.controller.currentCloud = source;
      this.tools.restoredPreview = false;
      const {lower, upper, maximum_count, frame_T_scan} = job.request;
      document.getElementById('scan').value = 'heatmap';
      await this.tools.regionLoaded(lower, upper, maximum_count, frame_T_scan);
      const median = entry => (entry.median_distance * 1000).toFixed(1);
      const near = entry => (entry.within_tolerance * 100).toFixed(1);
      this.heatmapSummary = `${source.identities.length.toLocaleString()} shared points · median ${median(report.original)} → ${median(report.proposed)} mm · ${near(report.original)}% → ${near(report.proposed)}% within ${(report.tolerance * 1000).toFixed(0)} mm.`;
    } finally {
      this.controller.activity?.end(activity);
    }
  }

  async decide(accept) {
    if (this.busy || !this.current || !this.list.can_decide) return;
    const {proposal} = this.review;
    this.busy = true;
    this.tools.changingHistory = true;
    this.tools.historyButtons();
    this.controls();
    const activity = this.controller.activity?.begin(
      accept ? 'Applying the change set' : 'Rejecting the change set', 'Saving the review decision…');
    let message = '';
    try {
      await this.controller.activity?.paint();
      await this.tools.pending;
      await this.request('decide_change_set', 'ChangeSetDecisionQuery', {
        identifier: proposal.identifier,
        expected_revision: proposal.request.expected_revision,
        accept,
      });
      await this.close();
      if (accept) await this.tools.refreshProject();
      message = accept ? 'Change set applied. Undo restores all affected bodies.' : 'Change set rejected.';
    } catch (error) {
      this.status.textContent = error.message;
    } finally {
      this.tools.changingHistory = false;
      this.busy = false;
      this.tools.historyButtons();
      this.workspace.synchronize();
      this.controller.activity?.end(activity);
    }
    if (message) {
      await this.refresh();
      this.status.textContent = message;
      this.workspace.notice(message);
    }
  }
}

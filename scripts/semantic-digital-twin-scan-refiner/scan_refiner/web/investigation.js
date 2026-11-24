"use strict";

// %% Clipboard handoff to the existing agent conversation
class InvestigationHandoff {
  constructor(workspace) {
    this.workspace = workspace;
    this.controller = workspace.controller;
    this.button = document.getElementById('copy-investigation');
    this.manual = document.getElementById('investigation-context');
    this.button.onclick = () => this.copy();
  }

  get ready() {
    const tools = this.controller.tools, view = tools?.inspection?.view;
    return !!(view && tools.viewRevision && this.controller.frame &&
      (view.selected_body || view.selection_lower.length === 3) &&
      !tools.applying && !tools.loading && !tools.changingHistory &&
      !tools.isPreview?.() && !this.workspace.refiner?.active);
  }

  update() {
    this.button.disabled = !this.ready;
  }

  context() {
    const controller = this.controller, tools = controller.tools;
    const view = tools.inspection.view, record = controller.state.project_record;
    const value = document.getElementById('candidate').value;
    const candidate = value === '' ? null : Number(value);
    const assembly = this.workspace.assemblies?.current;
    const bounds = view.selection_lower.length === 3 ? controller.readBounds() : null;
    if ((view.selection_lower.length === 3 || assembly) && !bounds) return null;
    return {
      format: 'semdt-investigation-v1',
      project: {
        url: location.origin,
        model_source: record.model_source,
        world_sha256: record.world_sha256,
        scan_source: record.scan_source,
        scan_sha256: record.scan_sha256,
      },
      selected_body: view.selected_body || null,
      ...(assembly ? {selected_assembly: {
        identifier: assembly.identifier,
        label: assembly.label,
        origin: assembly.origin || (assembly.root_name ? 'semantic' : 'saved'),
        ...(assembly.origin === 'kinematic' ? {root_name: assembly.root_name} : {}),
        body_names: [...assembly.body_names],
      }} : {}),
      region_query: {
        expected_revision: tools.viewRevision,
        candidate_index: candidate,
        frame_T_scan: controller.frame.matrix,
        ...(bounds || {body_name: view.selected_body}),
      },
      registration: {
        accepted_candidate: controller.state.accepted_candidate,
        scan_T_root: candidate === null ? null : controller.placement(),
      },
      inspection: {
        frame: controller.frameMode,
        yaw_degrees: controller.frameYaw,
        geometry_mode: document.getElementById('geometry').value,
        comparison_role: document.getElementById('comparison-role').value,
        displayed_comparison_job: view.comparison_job || null,
      },
    };
  }

  copy() {
    if (!this.ready) return;
    const context = this.context();
    if (!context) {
      this.workspace.notice('Enter valid crop bounds before copying.');
      return;
    }
    const payload = JSON.stringify(context, null, 2);
    this.manual.hidden = true;
    const manualCopy = () => {
      this.manual.value = payload;
      this.manual.hidden = false;
      this.manual.focus();
      this.manual.select();
      this.workspace.notice('Copy the selected context and paste it into your Codex conversation.');
    };
    if (!navigator.clipboard?.writeText) {
      manualCopy();
      return;
    }
    return navigator.clipboard.writeText(payload).then(
      () => this.workspace.notice('Context copied. Paste it into your Codex conversation with your request.'),
      manualCopy,
    );
  }
}

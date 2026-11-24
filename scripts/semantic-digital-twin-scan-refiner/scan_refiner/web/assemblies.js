"use strict";

// %% Explicit assembly selection and project-local groups
class AssemblyPanel {
  constructor(workspace) {
    this.workspace = workspace;
    this.tools = workspace.tools;
    this.controller = workspace.controller;
    this.expanded = new Set();
    this.draft = new Set();
    this.savedKey = null;
    this.name = document.getElementById('assembly-name');
    this.search = document.getElementById('assembly-search');
    this.members = document.getElementById('assembly-members');
    this.save = document.getElementById('save-assembly');
    this.remove = document.getElementById('remove-assembly');
    this.name.oninput = () => this.updateEditor();
    this.search.oninput = () => this.drawMembers();
    this.save.onclick = () => this.saveAssembly();
    this.remove.onclick = () => this.removeAssembly();
    document.getElementById('assembly-builder').ontoggle = event => {
      if (event.currentTarget.open) this.drawMembers();
    };
  }

  get saved() {
    return this.tools?.inspection?.saved_assemblies || [];
  }

  get groups() {
    return [...(this.controller.state.assemblies || []), ...this.saved];
  }

  get current() {
    const identifier = this.tools?.inspection?.view.selected_assembly;
    if (!identifier) return null;
    return this.workspace.worldTree?.scope(identifier) ??
      this.groups.find(group => group.identifier === identifier);
  }

  get busy() {
    return !this.tools?.inspection || this.tools.applying || this.tools.loading || this.tools.changingHistory;
  }

  draw(list, query) {
    const groups = this.groups.filter(group =>
      [group.label,group.root_name,...group.body_names].some(value => (value||'').toLowerCase().includes(query)));
    if (!groups.length) return;
    const heading = document.createElement('div');
    heading.className = 'assembly-heading';
    heading.textContent = 'Groups';
    list.append(heading);
    for (const group of groups) {
      const item = document.createElement('div');
      item.className = 'assembly-entry';
      const buttons = document.createElement('div');
      buttons.className = 'assembly-row';
      const select = document.createElement('button');
      select.className = 'body-item assembly-item';
      select.dataset.assembly = group.identifier;
      select.title = group.root_name || 'Saved selection';
      select.textContent = group.label;
      select.onclick = () => this.select(group);
      select.disabled = !!this.busy;
      select.classList.toggle('active',group.identifier===this.current?.identifier);
      const toggle = document.createElement('button');
      const open = this.expanded.has(group.identifier);
      toggle.className = 'assembly-expand quiet';
      toggle.textContent = `${open?'▾':'▸'} ${group.body_names.length}`;
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', `${open?'Hide':'Show'} members of ${group.label}`);
      toggle.onclick = () => {
        if (open) this.expanded.delete(group.identifier);
        else this.expanded.add(group.identifier);
        this.workspace.drawObjects();
        this.workspace.synchronize();
      };
      buttons.append(select,toggle);
      item.append(buttons);
      if (open) {
        const members = document.createElement('div');
        members.className = 'assembly-children';
        for (const name of group.body_names) {
          const body = this.workspace.objects.get(name);
          if (body) members.append(this.workspace.bodyButton(body));
        }
        item.append(members);
      }
      list.append(item);
    }
  }

  update() {
    const savedKey = JSON.stringify(this.saved);
    if (savedKey !== this.savedKey) {
      this.savedKey = savedKey;
      this.workspace.drawObjects();
    }
    this.updateSelection();
    this.updateEditor();
  }

  updateSelection() {
    const selected = this.tools?.inspection?.view.selected_assembly;
    document.querySelectorAll('[data-assembly]').forEach(button => {
      button.disabled = !!this.busy;
      button.classList.toggle('active',button.dataset.assembly===selected);
    });
  }

  async selectRegion(group) {
    await this.controller.plotUpdates.pending;
    const indices = group.body_names.flatMap(name => this.workspace.objects.get(name)?.indices || []);
    const bounds = InspectionController.meshBounds(this.controller.scene,indices,this.controller.frameOrigin,.15);
    if (!bounds) return null;
    const view = this.tools.inspection.view;
    view.selected_assembly = group.identifier;
    view.selected_body = '';
    await this.tools.regionSelected(bounds.lower,bounds.upper,`Select ${group.label}`);
    document.getElementById('region-status').textContent =
      `${group.label} · ${group.body_names.length} bodies with 15 cm of scan context.`;
    return bounds;
  }

  async select(group) {
    if (this.busy) return;
    await this.workspace.changeSets?.close();
    await this.workspace.refiner?.clear();
    return this.tools.change(`Focus ${group.label}`,async()=>{
      const bounds = await this.selectRegion(group);
      if (!bounds) return;
      this.tools.inspection.view.focus_enabled = true;
      await this.controller.focusSelection(bounds.lower,bounds.upper);
    });
  }

  drawMembers() {
    const query = this.search.value.trim().toLowerCase();
    this.members.replaceChildren();
    for (const body of this.workspace.objects.values()) {
      if (!body.name.toLowerCase().includes(query)) continue;
      const label = document.createElement('label');
      label.className = 'check';
      label.title = body.name;
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = body.name;
      checkbox.checked = this.draft.has(body.name);
      checkbox.onchange = () => {
        if (checkbox.checked) this.draft.add(body.name);
        else this.draft.delete(body.name);
        this.updateEditor();
      };
      label.append(checkbox,document.createTextNode(this.workspace.pretty(body.name)));
      this.members.append(label);
    }
    this.updateEditor();
  }

  updateEditor() {
    const selected = this.tools?.inspection?.view.selected_assembly;
    const activeSaved = this.saved.some(group => group.identifier===selected);
    this.remove.hidden = !activeSaved;
    this.remove.disabled = !!this.busy;
    this.save.disabled = !!this.busy || this.draft.size<2 || !this.name.value.trim();
    document.getElementById('assembly-member-count').textContent =
      `${this.draft.size} selected · choose at least two bodies`;
  }

  async saveAssembly() {
    if (this.save.disabled) return;
    const group = {
      __json_type__: 'scan_refiner.assemblies.AssemblyMembership',
      identifier: `saved:${crypto.randomUUID()}`,
      label: this.name.value.trim(),
      body_names: [...this.draft].sort(),
      root_name: '',
    };
    await this.workspace.changeSets?.close();
    await this.workspace.refiner?.clear();
    if (!await this.tools.change(`Save ${group.label} assembly`,async()=>{
      this.tools.inspection.saved_assemblies = [...this.saved,group];
      const bounds = await this.selectRegion(group);
      if (!bounds) return;
      this.tools.inspection.view.focus_enabled = true;
      await this.controller.focusSelection(bounds.lower,bounds.upper);
    })) return;
    this.draft.clear();
    this.name.value = '';
    this.search.value = '';
    this.drawMembers();
    document.getElementById('assembly-builder').open = false;
  }

  async removeAssembly() {
    const group = this.current;
    if (this.remove.disabled || !group) return;
    await this.tools.change(`Remove ${group.label} assembly`,async()=>{
      this.tools.inspection.saved_assemblies = this.saved.filter(item=>item.identifier!==group.identifier);
      this.tools.inspection.view.selected_assembly = '';
    });
  }
}

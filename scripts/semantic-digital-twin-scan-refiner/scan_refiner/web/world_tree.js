"use strict";

// %% Kinematic outliner
class WorldTree {
  constructor(workspace, records) {
    this.workspace = workspace;
    this.nodes = new Map(records.map(record => [record.name, {
      name: record.name,
      parent: record.parent_name,
      body: workspace.objects.get(record.name),
      children: [],
    }]));
    this.roots = [];
    for (const node of this.nodes.values()) {
      if (node.parent === null) this.roots.push(node);
      else this.nodes.get(node.parent).children.push(node);
    }
    this.expanded = new Set(this.roots.map(node => node.name));
    this.focusable = new Set();
    for (const name of workspace.objects.keys()) {
      let node = this.nodes.get(name);
      while (node && !this.focusable.has(node.name)) {
        this.focusable.add(node.name);
        node = this.nodes.get(node.parent);
      }
    }
  }

  scope(identifier) {
    const prefix = 'kinematic:';
    if (!identifier?.startsWith(prefix)) return null;
    const node = this.nodes.get(identifier.slice(prefix.length));
    if (!node || !this.focusable.has(node.name)) return null;
    const members = [];
    const pending = [node];
    while (pending.length) {
      const member = pending.pop();
      if (member.body) members.push(member.name);
      pending.push(...member.children);
    }
    return {
      identifier,
      label: this.workspace.pretty(node.name),
      root_name: node.name,
      origin: 'kinematic',
      body_names: members.sort(),
    };
  }

  draw(list, query) {
    if (!this.roots.length) return false;
    const heading = document.createElement('div');
    heading.className = 'assembly-heading';
    heading.textContent = 'World tree';
    list.append(heading);

    const matches = new Set();
    if (query) {
      for (const node of this.nodes.values()) {
        if (!node.name.toLowerCase().includes(query) &&
            !this.workspace.pretty(node.name).toLowerCase().includes(query)) continue;
        let ancestor = node;
        while (ancestor && !matches.has(ancestor.name)) {
          matches.add(ancestor.name);
          ancestor = this.nodes.get(ancestor.parent);
        }
      }
    }
    for (const node of this.roots) this.drawNode(list, node, query ? matches : null);
    return true;
  }

  drawNode(list, node, matches) {
    if (matches && !matches.has(node.name)) return;
    if (!node.children.length && node.body) {
      list.append(this.workspace.bodyButton(node.body, {focus: true}));
      return;
    }

    const item = document.createElement('div');
    const row = document.createElement('div');
    row.className = 'assembly-row';
    const open = this.expanded.has(node.name);
    if (node.children.length) {
      if (matches) {
        const marker = document.createElement('span');
        marker.className = 'tree-disclosure';
        marker.textContent = '▾';
        marker.setAttribute('aria-hidden', 'true');
        row.append(marker);
      } else {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'tree-disclosure quiet';
        toggle.dataset.treeToggle = node.name;
        toggle.textContent = open ? '▾' : '▸';
        toggle.setAttribute('aria-expanded', String(open));
        toggle.setAttribute('aria-label', `${open ? 'Collapse' : 'Expand'} ${node.name}`);
        toggle.onclick = () => {
          const restoreFocus = document.activeElement === toggle;
          if (open) this.expanded.delete(node.name);
          else this.expanded.add(node.name);
          this.workspace.drawObjects();
          if (restoreFocus) {
            document.querySelector(`[data-tree-toggle="${CSS.escape(node.name)}"]`)
              ?.focus({preventScroll: true});
          }
        };
        row.append(toggle);
      }
    }

    const identifier = `kinematic:${node.name}`;
    const label = this.workspace.pretty(node.name);
    if (this.focusable.has(node.name)) {
      const select = document.createElement('button');
      select.type = 'button';
      select.className = 'body-item assembly-item';
      select.textContent = label;
      select.title = `Focus branch ${node.name}`;
      select.dataset.assembly = identifier;
      select.onclick = () => this.workspace.assemblies.select(this.scope(identifier));
      row.append(select);
    } else {
      const frame = document.createElement('span');
      frame.className = 'body-item assembly-item tree-frame';
      frame.textContent = label;
      frame.title = node.name;
      row.append(frame);
    }
    if (node.body) {
      const body = this.workspace.bodyButton(node.body, {focus: true});
      body.classList.add('tree-body');
      body.replaceChildren(document.createTextNode('▧'));
      body.title = `Focus only ${node.name}`;
      body.setAttribute('aria-label', body.title);
      row.append(body);
    }
    item.append(row);
    if (open || matches) {
      const children = document.createElement('div');
      children.className = 'assembly-children';
      for (const child of node.children) this.drawNode(children, child, matches);
      if (children.childElementCount) item.append(children);
    }
    list.append(item);
  }
}

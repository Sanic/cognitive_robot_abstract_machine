/** Object identity and the final exported hierarchy, independent of rendering. */

// %% Exported world

export class ExportManifest {
  /** Reject ambiguous identities and broken references before displaying a world. */
  constructor(document) {
    if (document.schema_version !== 2 || !Array.isArray(document.bodies) ||
        !Array.isArray(document.semantic_annotations) || typeof document.scene !== 'string') {
      throw new Error('Expected a WorldMeshExporter schema version 2 manifest.');
    }
    this.scene = document.scene;
    this.bodies = document.bodies;
    this.annotations = new Map(document.semantic_annotations.map(annotation => [annotation.id, annotation]));
    this.bodiesById = new Map(this.bodies.map(body => [body.id, body]));
    if (this.bodiesById.size !== this.bodies.length || this.annotations.size !== document.semantic_annotations.length) {
      throw new Error('Body and annotation identifiers must be unique.');
    }
    this.childrenById = new Map(this.bodies.map(body => [body.id, []]));
    for (const body of this.bodies) {
      for (const identifier of [...body.direct_semantic_annotation_ids, ...body.context_semantic_annotation_ids]) {
        if (!this.annotations.has(identifier)) throw new Error(`Unknown annotation ${identifier} on ${body.name}.`);
      }
      if (body.parent_id !== null) {
        if (!this.bodiesById.has(body.parent_id)) throw new Error(`Unknown parent of ${body.name}.`);
        this.childrenById.get(body.parent_id).push(body);
      }
    }
    for (const body of this.bodies) {
      const seen = new Set([body.id]);
      for (let parent = this.bodiesById.get(body.parent_id); parent; parent = this.bodiesById.get(parent.parent_id)) {
        if (seen.has(parent.id)) throw new Error(`Hierarchy cycle at ${body.name}.`);
        seen.add(parent.id);
      }
    }
    this.labels = [...new Set(this.bodies.flatMap(body => this.classesFor(body)))].sort();
  }

  /** Return only the body's own classes, never classes belonging to enclosing objects. */
  classesFor(body) {
    return [...new Set(body.direct_semantic_annotation_ids.map(identifier => this.annotations.get(identifier).type_name))];
  }

  /** Exclude the synthetic world root from parent navigation. */
  parentOf(body) {
    const parent = this.bodiesById.get(body.parent_id);
    return parent && parent.parent_id !== null ? parent : null;
  }

  /** Keep immediate children separate from more distant descendants. */
  childrenOf(body) {
    return this.childrenById.get(body.id);
  }

  /** Follow the nearest parent first, stopping before the synthetic world root. */
  ancestorsOf(body) {
    const ancestors = [];
    for (let parent = this.parentOf(body); parent; parent = this.parentOf(parent)) ancestors.push(parent);
    return ancestors;
  }

  /** Include an object and all of its parts for framing and isolation. */
  assemblyOf(body) {
    const assembly = [body];
    for (let index = 0; index < assembly.length; index++) assembly.push(...this.childrenOf(assembly[index]));
    return assembly;
  }

  /** Search object names and their own classes, including objects without geometry. */
  findBodies(label, query) {
    const text = query.trim().toLocaleLowerCase();
    return this.bodies.filter(body => body.parent_id !== null &&
      (!label || this.classesFor(body).includes(label)) &&
      [body.name, ...this.classesFor(body)].some(value => value.toLocaleLowerCase().includes(text)));
  }
}

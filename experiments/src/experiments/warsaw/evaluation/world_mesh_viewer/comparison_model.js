/** Navigate a saved comparison; the browser never invents replacement matches. */

// %% Saved correspondence and disagreement filters

export class ComparisonData {
  /** Resolve each table identifier against the export to which it belongs. */
  constructor(payload, prediction, groundTruth) {
    this.document = ComparisonData.decode(payload);
    this.prediction = prediction;
    this.groundTruth = groundTruth;
    this.records = this.document.comparison.bodies;
    this.byPrediction = new Map();
    this.byGroundTruth = new Map();
    for (const record of this.records) {
      if (!prediction.bodiesById.has(record.predicted_identifier)) throw new Error(`Unknown prediction ${record.predicted_identifier} in match table.`);
      if (this.byPrediction.has(record.predicted_identifier)) throw new Error('Duplicate prediction in match table.');
      this.byPrediction.set(record.predicted_identifier, record);
      if (record.ground_truth_identifier === null) continue;
      if (!groundTruth.bodiesById.has(record.ground_truth_identifier)) throw new Error(`Unknown GT ${record.ground_truth_identifier} in match table.`);
      if (!this.byGroundTruth.has(record.ground_truth_identifier)) this.byGroundTruth.set(record.ground_truth_identifier, []);
      this.byGroundTruth.get(record.ground_truth_identifier).push(record);
    }
  }

  /** Read the repository serializer's collection envelopes as browser arrays. */
  static decode(value) {
    if (Array.isArray(value)) return value.map(item => ComparisonData.decode(item));
    if (value && typeof value === 'object') {
      if (value.__collection_type__ && Array.isArray(value.__items__)) return value.__items__.map(item => ComparisonData.decode(item));
      return Object.fromEntries(Object.entries(value).filter(([key]) => key !== '__json_type__').map(([key, item]) => [key, ComparisonData.decode(item)]));
    }
    return value;
  }

  /** Return a prediction's recorded association, including an explicitly unmatched one. */
  forPrediction(identifier) {
    return this.byPrediction.get(identifier);
  }

  /** Retain every prediction assigned to a GT object instead of selecting one arbitrarily. */
  forGroundTruth(identifier) {
    return this.byGroundTruth.get(identifier) ?? [];
  }

  /** Keep unmatched GT separate from predictions and filter only recorded disagreements. */
  rows(filter, query) {
    const text = query.trim().toLocaleLowerCase();
    if (filter === 'unmatched-ground-truth') {
      return this.document.comparison.unmatched_ground_truth.map(identifier => ({side: 'ground_truth', identifier})).filter(row => this.matchesText(row, text));
    }
    return this.records.filter(record => {
      if (filter === 'unmatched-prediction') return record.ground_truth_identifier === null;
      if (filter === 'class') return record.class_disagreement === true;
      if (filter === 'parent') return record.parent_disagreement === true;
      if (filter === 'ties') return record.margin === 0;
      if (filter === 'many-to-one') return record.ground_truth_identifier !== null && this.forGroundTruth(record.ground_truth_identifier).length > 1;
      return true;
    }).map(record => ({side: 'prediction', identifier: record.predicted_identifier})).filter(row => this.matchesText(row, text));
  }

  /** Search explicit names and own classes rather than parsing display-name suffixes. */
  matchesText(row, query) {
    const manifest = row.side === 'prediction' ? this.prediction : this.groundTruth;
    const body = manifest.bodiesById.get(row.identifier);
    return body && [body.name, ...manifest.classesFor(body)].some(value => value.toLocaleLowerCase().includes(query));
  }
}

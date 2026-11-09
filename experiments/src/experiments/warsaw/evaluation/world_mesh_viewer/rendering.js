/** Draw label matches after the surroundings so their colours remain visible. */

// %% Background and highlight passes

export class SceneRenderer {
  /** Separate ghost geometry and highlighted geometry by camera layer. */
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.renderer.autoClear = false;
  }

  /** Preserve highlights' mutual depth while ignoring background occlusion. */
  render() {
    this.renderer.clear();
    this.camera.layers.set(0);
    this.renderer.render(this.scene, this.camera);
    this.renderer.clearDepth();
    const background = this.scene.background;
    this.scene.background = null;
    this.camera.layers.set(1);
    this.renderer.render(this.scene, this.camera);
    this.scene.background = background;
    this.camera.layers.enableAll();
  }
}

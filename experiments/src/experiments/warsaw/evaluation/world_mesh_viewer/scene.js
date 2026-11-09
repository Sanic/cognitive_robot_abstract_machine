/** Reuse the label inspector's picking, framing, and two-pass highlights. */
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {GLTFLoader} from './vendor/GLTFLoader.js';
import {SceneRenderer} from './rendering.js';

// %% Interactive geometry

export class WorldScene {
  /** Keep geometry and GPU resources separate from semantic navigation. */
  constructor(viewport, onSelect) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101720');
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    viewport.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.camera.position.set(8, -10, 8);
    this.sceneRenderer = new SceneRenderer(this.renderer, this.scene, this.camera);
    const grid = new THREE.GridHelper(30, 60, 0x3b5267, 0x243242);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);
    this.meshesByBody = new Map();
    this.raycaster = new THREE.Raycaster();
    this.raycaster.layers.enableAll();
    this.pointer = new THREE.Vector2();
    this.renderer.domElement.addEventListener('pointerdown', event => {
      this.pointerStart = {x: event.clientX, y: event.clientY};
    });
    this.renderer.domElement.addEventListener('pointerup', event => {
      if (event.button !== 0 || !this.pointerStart ||
          Math.hypot(event.clientX - this.pointerStart.x, event.clientY - this.pointerStart.y) > 5) return;
      const rectangle = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set((event.clientX - rectangle.left) / rectangle.width * 2 - 1,
        -(event.clientY - rectangle.top) / rectangle.height * 2 + 1);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const visible = [...this.meshesByBody.values()].flat().filter(mesh => mesh.visible && mesh.material.opacity > 0);
      const hits = this.raycaster.intersectObjects(visible, false);
      // Prefer the same foreground pass the user sees through ghosted walls.
      const hit = hits.find(hit => hit.object.layers.isEnabled(1)) ?? hits[0];
      onSelect(hit ? hit.object.userData.bodyId : null);
    });
    new ResizeObserver(() => {
      const {width, height} = viewport.getBoundingClientRect();
      if (!height || !width) return;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }).observe(viewport);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.sceneRenderer.render();
    });
  }

  /** Parse the scene while retaining the exact exported names for body matching. */
  async read(bytes, manifest) {
    const gltf = await new GLTFLoader().parseAsync(bytes, '');
    const bodiesByName = new Map(manifest.bodies.map(body => [body.node_name, body]));
    const meshesByBody = new Map();
    gltf.scene.traverse(object => {
      if (!object.isMesh) return;
      let body;
      for (let ancestor = object; ancestor && !body; ancestor = ancestor.parent) {
        const nodeIndex = gltf.parser.associations.get(ancestor)?.nodes;
        const name = nodeIndex === undefined ? ancestor.name : gltf.parser.json.nodes[nodeIndex].name;
        body = bodiesByName.get(name);
      }
      if (!body) throw new Error(`Scene mesh ${object.name} has no matching manifest body.`);
      object.userData.originalMaterial = object.material;
      object.material = new THREE.MeshBasicMaterial({side: THREE.DoubleSide, toneMapped: false});
      object.userData.bodyId = body.id;
      if (!meshesByBody.has(body.id)) meshesByBody.set(body.id, []);
      meshesByBody.get(body.id).push(object);
    });
    // Detach meshes without changing world poses so hiding a parent cannot hide its parts.
    const model = new THREE.Group();
    gltf.scene.updateMatrixWorld(true);
    for (const mesh of [...meshesByBody.values()].flat()) {
      model.attach(mesh);
      mesh.geometry.computeBoundingBox();
      const outline = new THREE.Box3Helper(mesh.geometry.boundingBox, '#142332');
      outline.material.toneMapped = false;
      outline.layers.set(1);
      mesh.add(outline);
      mesh.userData.outline = outline;
    }
    return {model, meshesByBody};
  }

  /** Replace a loaded model only when the new export is complete. */
  display(loaded) {
    if (this.model) {
      this.scene.remove(this.model);
      this.dispose(this.model);
    }
    this.model = loaded.model;
    this.meshesByBody = loaded.meshesByBody;
    this.scene.add(this.model);
    this.model.updateMatrixWorld(true);
  }

  /** Place the exported world in an explicitly supplied comparison frame. */
  place(transform) {
    this.model.matrix.set(...transform.flat());
    this.model.matrixAutoUpdate = false;
    this.model.updateMatrixWorld(true);
  }

  /** Release both display materials and the original GLB resources. */
  dispose(model) {
    const materials = new Set();
    const textures = new Set();
    model.traverse(object => {
      if (!object.isMesh && !object.isLineSegments) return;
      object.geometry.dispose();
      for (const material of [object.material, object.userData.originalMaterial].flat().filter(Boolean)) {
        materials.add(material);
        for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
      }
    });
    materials.forEach(material => material.dispose());
    textures.forEach(texture => texture.dispose());
  }

  /** Apply visibility and emphasis per object, independently of scene-graph parenting. */
  paint(states) {
    for (const [identifier, meshes] of this.meshesByBody) {
      const state = states.get(identifier);
      for (const mesh of meshes) {
        mesh.visible = state.visible;
        mesh.layers.set(state.emphasized ? 1 : 0);
        mesh.userData.outline.visible = state.emphasized;
        mesh.material.color.set(state.color);
        mesh.material.opacity = state.opacity;
        mesh.material.transparent = state.opacity < 1;
        mesh.material.depthWrite = !mesh.material.transparent;
        mesh.material.needsUpdate = true;
      }
    }
  }

  /** Frame world-space bounds without moving any object. */
  fit(bodies) {
    const bounds = new THREE.Box3();
    for (const body of bodies) {
      for (const mesh of this.meshesByBody.get(body.id) ?? []) bounds.expandByObject(mesh);
    }
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 0.05);
    const angle = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * Math.min(1, this.camera.aspect));
    const distance = radius / Math.sin(angle) * 1.2;
    const direction = this.camera.position.clone().sub(this.controls.target);
    if (direction.lengthSq() < 0.01) direction.set(1, -1.4, 1.1);
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction.normalize(), distance);
    this.camera.near = Math.max(distance / 10000, 0.001);
    this.camera.far = Math.max(distance * 100, 1000);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }
}

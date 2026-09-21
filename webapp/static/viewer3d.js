import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

function colorForHeight(height, target, offset) {
  const value = Math.max(0, Math.min(1, (height + 2.5) / 4.5));
  const color = new THREE.Color().setHSL((174 + value * 42) / 360, 0.75, 0.65);
  target[offset] = color.r;
  target[offset + 1] = color.g;
  target[offset + 2] = color.b;
}

function disposeGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose?.();
    child.material?.dispose?.();
  }
}

export class LidarViewer3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050c0b);
    this.camera = new THREE.PerspectiveCamera(52, 1, 0.05, 400);
    this.camera.position.set(10, 8, 9);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.replaceChildren(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -1.2, -35);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxDistance = 180;
    this.controls.minDistance = 2;
    this.controls.update();

    this.pointGeometry = new THREE.BufferGeometry();
    this.pointMaterial = new THREE.PointsMaterial({ size: 0.13, vertexColors: true, sizeAttenuation: true });
    this.pointCloud = new THREE.Points(this.pointGeometry, this.pointMaterial);
    this.scene.add(this.pointCloud);

    this.corridorGroup = new THREE.Group();
    this.boxGroup = new THREE.Group();
    this.scene.add(this.corridorGroup, this.boxGroup);

    const axes = new THREE.AxesHelper(3);
    axes.position.set(0, -2.3, 0);
    this.scene.add(axes);
    this.grid = new THREE.GridHelper(180, 36, 0x31534a, 0x17332c);
    this.grid.position.set(0, -2.3, -45);
    this.scene.add(this.grid);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.animate = this.animate.bind(this);
    this.animationFrame = requestAnimationFrame(this.animate);
  }

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.animationFrame = requestAnimationFrame(this.animate);
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  updateFrame(frame) {
    const source = frame.points;
    const count = source instanceof Float32Array ? source.length / 4 : source.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      const point = source instanceof Float32Array
        ? [source[index * 4], source[index * 4 + 1], source[index * 4 + 2]]
        : source[index];
      const offset = index * 3;
      positions[offset] = point[0];
      positions[offset + 1] = point[2];
      positions[offset + 2] = -point[1];
      colorForHeight(point[2], colors, offset);
    }
    this.pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.pointGeometry.computeBoundingSphere();

    disposeGroup(this.corridorGroup);
    const half = frame.corridor.width_m / 2;
    const profile = frame.path_profile.filter(item => item.forward_m <= frame.corridor.max_forward_m);
    for (const offset of [-half, half]) {
      const vertices = [];
      for (const item of profile) {
        const floor = frame.floor_profile[Math.min(frame.floor_profile.length - 1, Math.floor(item.forward_m / 2))];
        vertices.push(new THREE.Vector3(item.center_x_m + offset, floor?.z_m ?? frame.corridor.floor_z_m, -item.forward_m));
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
      this.corridorGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x57e6a1 })));
    }
    for (const side of ["left_rail_x_m", "right_rail_x_m"]) {
      const vertices = profile.filter(item => item.rail_confident && item[side] != null).map(item => {
        const floor = frame.floor_profile[Math.min(frame.floor_profile.length - 1, Math.floor(item.forward_m / 2))];
        return new THREE.Vector3(item[side], (floor?.z_m ?? frame.corridor.floor_z_m) + 0.04, -item.forward_m);
      });
      if (vertices.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
        this.corridorGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x69b7ff })));
      }
    }

    disposeGroup(this.boxGroup);
    for (const object of frame.objects) {
      const box = new THREE.Box3(
        new THREE.Vector3(object.min_m.x, object.min_m.z, -object.max_m.forward),
        new THREE.Vector3(object.max_m.x, object.max_m.z, -object.min_m.forward)
      );
      this.boxGroup.add(new THREE.Box3Helper(box, object.track_confirmed ? 0xff6272 : 0xffbd59));
    }
  }
}

"""Exercise actual viewer geometry helpers without a browser/WebGL dependency."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_winner_label_geometry_and_falcon_range():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is optional outside frontend development')
    root = Path(__file__).resolve().parents[1]
    module = root / 'web/static/js/winner_view_geometry.js'
    # Data URL keeps the test independent of Node's package.json module mode.
    source = """
      import assert from 'node:assert/strict';
      import {readFileSync} from 'node:fs';
      const api = await import('data:text/javascript;base64,' + Buffer.from(readFileSync(MODULE)).toString('base64'));
      const pairs = [25,50,75,100,125,150,175,200,225,250,275,300,325,350].map(s => ({
        source_s_m:s,left_xyz:[1.8,s,2],right_xyz:[.2,s,2]}));
      assert.equal(api.pairClosestToDistance(pairs,327).source_s_m,325);
      const selected=api.axisDistanceLabels(pairs,350);
      assert(selected.length<=9);
      assert.equal(selected.at(-1).source_s_m,350);
      const position=api.sideAxisLabelPosition(pairs[0],[-1.05,1.05]);
      assert(Math.abs(position[0]+.8)<1e-12);
      assert.equal(position[1],25);assert.equal(position[2],2.45);
      // Sloped track coordinates retain the side normal rather than pinning
      // labels to a hard-coded X axis or forgetting the rail plane height.
      const curve=api.sideAxisLabelPosition({left_xyz:[2,26,3],right_xyz:[0,24,3]},[-1,1]);
      assert(curve[0]<0 && curve[1]<24 && curve[2]===3.45);
      assert.equal(api.sceneRange({analysis_range_m:350,track:{s:[3.5,349]}}),349);
      assert.equal(api.sceneRange({analysis_range_m:350,track:{s:[3.5,351]}}),350);
      assert.equal(api.sceneRange(null),200);
    """.replace('MODULE', json.dumps(str(module)))
    completed = subprocess.run([node, '--input-type=module', '-e', source], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_scene_reuses_points_boxes_and_textures_and_clears_old_bag():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is optional outside frontend development')
    root = Path(__file__).resolve().parents[1]
    source = r"""
      import assert from 'node:assert/strict';
      import {readFileSync} from 'node:fs';
      import {pathToFileURL} from 'node:url';
      const root=ROOT;
      const url=p=>pathToFileURL(root+'/'+p).href;
      const THREE=await import(url('web/static/vendor/three.module.js'));
      let code=readFileSync(root+'/web/static/js/scene.js','utf8')
        .replace("from 'three'",`from '${url('web/static/vendor/three.module.js')}'`)
        .replace(/import \{ OrbitControls \}[^;]+;/,'const OrbitControls=null;')
        .replace("from '/static/js/winner_view_geometry.js'",`from '${url('web/static/js/winner_view_geometry.js')}'`);
      const {Scene3D}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
      globalThis.document={createElement:()=>({getContext:()=>({clearRect(){},measureText:()=>({width:100}),
        beginPath(){},roundRect(){},fill(){},fillText(){}})})};
      const view=Object.create(Scene3D.prototype);
      view.points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial());
      view.hot=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial());
      view.layers={points:true,envelope:true,path:true,learned:true,markers:true,boxes:true};
      view._unitBox=new THREE.BoxGeometry(1,1,1);view._unitEdges=new THREE.EdgesGeometry(view._unitBox);
      view._boxes=new Map();view._labels=new Map();
      for(const name of ['envGroup','pathGroup','wallGroup','markGroup','boxGroup'])view[name]=new THREE.Group();
      view.frame={status:'OBSTACLE',analysis_range_m:350,envelope:{width_m:2.1,height_m:3},
        track:{s:[3,100,200,350],c:[0,0,0,0],z:[0,0,0,0],observed_until:320},
        obstacles:[{track_id:7,distance_m:80,s_m:80,lateral_m:0,box:{s:[80,81],d:[-.3,.3],h:[.2,1.7]}}]};
      const points={n:3,xyz:new Float32Array([0,10,0,0,80,1,1,80,1]),
        cls:new Uint8Array([0,3,4]),inten:new Uint8Array([10,20,30])};
      view._fillPoints(points);
      const position=view.points.geometry.getAttribute('position');
      view._fillPoints({...points,n:2});
      assert.equal(position,view.points.geometry.getAttribute('position'));
      assert.equal(view.points.geometry.drawRange.count,2);
      assert.equal(view.hot.geometry.drawRange.count,1);
      view._rebuild();
      const box=view._boxes.get('track:7').group;
      const texture=view._labels.get('box:track:7').material.map;
      view.frame.obstacles[0].distance_m=79.9;
      view._rebuild();
      assert.equal(box,view._boxes.get('track:7').group);
      assert.equal(texture,view._labels.get('box:track:7').material.map);
      assert.equal(view._labels.get('box:track:7').userData.labelText,'79.9 м · №7');
      view.canvas={clientWidth:1200,clientHeight:800};view.camMode='cab';
      view.camera=new THREE.PerspectiveCamera(55,1.5,.1,1000);
      view.camera.position.set(-2,2,0);view.camera.lookAt(100,1,0);
      // Two labels at exactly the same projected location: actual obstacle
      // wins over the optional station/observed-end label.
      const obstacle=view._labels.get('box:track:7');
      view._labels.get('observed-end').position.copy(obstacle.position);
      view._updateLabels();
      assert.equal(obstacle.visible,true);
      assert.equal(view._labels.get('observed-end').visible,false);
      assert([...view._labels].filter(([key,s])=>key.startsWith('distance:')&&s.visible).length<=2);
      obstacle.position.set(-100,0,0);view._updateLabels();
      assert.equal(obstacle.visible,false,'Behind-camera label must not obscure visible scene');
      // Retained and uncertain geometry must never extend a green clearance
      // beyond the fresh, reliable part of the current measurement.
      view.frame.status='CLEAR';
      view.frame.track.observed_until=123;
      view.frame.track.current_observed_until=80;
      view.frame.track.uncertain_from=60;
      view._rebuild();
      const surfaces=()=>view.envGroup.children.filter(o=>o.isMesh).map(o=>({
        color:o.material.color.getHex(),
        stations:Array.from(o.geometry.getAttribute('position').array).filter((_,i)=>i%3===0)
      }));
      let parts=surfaces();
      assert.deepEqual(parts.map(p=>[p.color,Math.min(...p.stations),Math.max(...p.stations)]),[
        [0x22c55e,3,60],[0xf59e0b,60,80],[0xf59e0b,80,123],[0x64748b,123,138]
      ]);
      assert.equal(view._labels.get('observed-end').userData.labelText,'Удержание 123 м');
      assert.equal(view._labels.get('fresh-end').userData.labelText,'Измерено 80 м');
      assert(view.envGroup.children.some(o=>o.isLine&&o.material.isLineDashedMaterial&&o.material.color.getHex()===0xf59e0b));
      view.frame.status='OBSTACLE';view._rebuild();
      assert(surfaces().filter(p=>p.color!==0x64748b).every(p=>p.color===0xef4444));
      view.frame.status='CLEAR';view.frame.track.observed_until=0;
      view.frame.track.current_observed_until=0;view.frame.track.uncertain_from=null;
      view._rebuild();
      assert(surfaces().every(p=>p.color===0x64748b),'No green one-metre placeholder for absent observations');
      view.clear();
      assert.equal(view.frame,null);assert.equal(view._labels.size,0);assert.equal(view._boxes.size,0);
      assert.equal(view.points.geometry.drawRange.count,0);assert.equal(view.boxGroup.children.length,0);
    """.replace('ROOT', json.dumps(str(root)))
    completed = subprocess.run([node, '--input-type=module', '-e', source], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr

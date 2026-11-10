// Adapted from Sakhalin Ai, web/stage_4_cpu_player_axis_labels.js,
// revision 3cfd31843b911f278c08f49e53a09b1a5efa1d92, MIT, Copyright (c) 2026 saic.
// Original license: /static/vendor/sakhalin_ai.LICENSE (also in gabarit/vendor).
// Changes: explicit bounds instead of global result; return label data instead
// of creating DOM/Three objects. Coordinates remain (left, forward, up).
export function pairClosestToDistance(pairs,distance){
  let best=null,bestDelta=Infinity;
  for(const pair of pairs){const delta=Math.abs(Number(pair.source_s_m)-distance);if(delta<bestDelta){best=pair;bestDelta=delta}}
  return best;
}

export function sideAxisLabelPosition(pair,bounds=[],extraOffset=.75,extraHeight=.45){
  const center=pair.left_xyz.map((value,index)=>(value+pair.right_xyz[index])*.5);
  const side=[pair.right_xyz[0]-center[0],pair.right_xyz[1]-center[1],0];
  const length=Math.hypot(side[0],side[1])||1;
  const halfWidth=Math.max(Math.abs(Number(bounds[0])||0),Math.abs(Number(bounds[1])||0),length);
  const offset=halfWidth+extraOffset;
  return[center[0]+side[0]/length*offset,center[1]+side[1]/length*offset,center[2]+extraHeight];
}

export function axisDistanceLabels(pairs,observedEnd){
  const labelled=new Set(),stride=Math.max(1,Math.ceil(pairs.length/8));
  const labels=[];
  const envelopeEndPair=pairs.at(-1);
  for(let i=0;i<pairs.length;i+=stride){
    const pair=pairs[i];
    if(pair===envelopeEndPair)continue;
    labelled.add(pair); labels.push(pair);
  }
  const endPair=Number.isFinite(observedEnd)?pairClosestToDistance(pairs,observedEnd):envelopeEndPair;
  if(endPair&&!labelled.has(endPair))labels.push(endPair);
  return labels;
}

export function sceneRange(frame){
  const last=frame?.track?.s?.at(-1);
  const requested=Number(frame?.analysis_range_m);
  return Math.max(30,Math.min(Number.isFinite(requested)?requested:200,Number.isFinite(last)?last:200));
}

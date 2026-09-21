const assert=require('assert').strict;
const fs=require('fs');
const vm=require('vm');
const source=fs.readFileSync('static/app.js','utf8');
const nodes=new Map(),pending=[];
const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,disabled:false,readOnly:false,textContent:'',close(){}});return nodes.get(id)};
const provider={id:1,provider_type:'custom',model_settings:{a:{temperature_enabled:true,temperature:0.8,top_p_enabled:true,top_p:0.9},b:{temperature_enabled:true,temperature:0.6,top_p_enabled:true,top_p:0.9}}};
let model='a';
const context=vm.createContext({console,JSON,Number,Set,setTimeout,clearTimeout,
  $:node,selectedModel:()=>model,selectedProvider:()=>provider,syncCustomThinkingFields:()=>{},syncCustomToolFields:()=>{},
  api:(url,options)=>new Promise((resolve,reject)=>pending.push({url,options,resolve,reject})),
  isAdmin:()=>true,state:{providers:[provider]},updateProviderUi:()=>{},toast:()=>{}});
vm.runInContext(source.slice(source.indexOf('function customSettings('),source.indexOf('function syncCustomThinkingFields(')),context);
vm.runInContext(source.slice(source.indexOf('async function saveCustomSettings('),source.indexOf('async function loadUsers(')),context);
const run=code=>vm.runInContext(code,context);
(async()=>{
  let opening=run('fillCustomSettings()');
  pending[0].resolve({parameters:{model:'a',temperature:0.8}});await opening;
  assert.equal(node('#customAdvancedEnabled').checked,false);
  assert.equal(node('#customRequestOverrides').readOnly,true);
  assert.equal(JSON.parse(node('#customRequestOverrides').value).temperature,0.8);

  node('#customTemperature').value='0.4';
  const old=run('refreshCustomPreview()');
  node('#customTemperature').value='0.3';
  const current=run('refreshCustomPreview()');
  assert.equal(pending[2].options.body.temperature,0.3);
  pending[2].resolve({parameters:{model:'a',temperature:0.3}});await current;
  pending[1].resolve({parameters:{model:'a',temperature:0.4}});await old;
  assert.equal(JSON.parse(node('#customRequestOverrides').value).temperature,0.3);

  // A stale draft must never replace the preview generated from the current
  // form when advanced mode is enabled.
  run("customEditor.draft='{\"temperature\":0.99,\"stale\":true}'");
  node('#customAdvancedEnabled').checked=true;await run('toggleAdvancedSettings()');
  assert.equal(node('#customRequestOverrides').readOnly,false);
  assert.equal(JSON.parse(node('#customRequestOverrides').value).temperature,0.3);
  assert.equal(JSON.parse(node('#customRequestOverrides').value).stale,undefined);
  node('#customRequestOverrides').value='{"temperature":0.15,"vendor":true}';
  run("scheduleCustomPreview({target:{id:'customTemperature'}})");
  assert.equal(pending.length,3);
  node('#customTemperature').value='0.2';
  node('#customAdvancedEnabled').checked=false;
  const disabled=run('toggleAdvancedSettings()');
  assert.equal(pending[3].options.body.temperature,0.2);
  pending[3].resolve({parameters:{model:'a',temperature:0.2}});await disabled;
  assert.equal(node('#customRequestOverrides').readOnly,true);
  node('#customAdvancedEnabled').checked=true;await run('toggleAdvancedSettings()');
  assert.equal(JSON.parse(node('#customRequestOverrides').value).temperature,0.2);
  assert.equal(JSON.parse(node('#customRequestOverrides').value).vendor,undefined);

  node('#customRequestOverrides').value='{broken';
  await run('saveCustomSettings({preventDefault(){}})');
  assert.equal(pending.length,4);
  assert.match(node('#customStatus').textContent,/JSON/);
  node('#customRequestOverrides').value='{"temperature":0.15}';
  const saved=run('saveCustomSettings({preventDefault(){}})');
  assert.equal(pending[4].options.body.advanced_enabled,true);
  assert.equal(pending[4].options.body.advanced_request.temperature,0.15);
  pending[4].resolve(provider);await saved;

  const late=run('refreshCustomPreview()');await late;
  model='b';opening=run('fillCustomSettings()');
  pending[5].resolve({parameters:{model:'b',temperature:0.6}});await opening;
  assert.equal(node('#customAdvancedEnabled').checked,false);
  assert.equal(JSON.parse(node('#customRequestOverrides').value).model,'b');
  console.log('PASS: read-only preview, stale responses, current-form toggle handoff, validation and model isolation');
})().catch(error=>{console.error(error);process.exitCode=1});

const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const state = {me:null, providers:[], conversation:null, messages:[], job:null, page:1, pages:1, poll:null};

async function api(path, options={}) {
  const headers = {...(options.headers || {})};
  if (options.body && typeof options.body !== 'string') {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, {...options, headers});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function toast(message) {
  const node = $('#toast'); node.textContent = message; node.classList.add('show');
  clearTimeout(node.timer); node.timer = setTimeout(() => node.classList.remove('show'), 2300);
}

function escapeHtml(value='') { return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function safeUrl(value='') { return /^https?:\/\//i.test(value) ? escapeHtml(value) : '#'; }
function inline(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function markdown(raw='') {
  const codes = [];
  raw = raw.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const index = codes.length;
    const label = escapeHtml(lang.trim() || 'code');
    codes.push(`<div class="code-wrap"><div class="code-label"><span>${label}</span><button class="code-download" data-code="${index}">⇩ 下载</button></div><pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre></div>`);
    return `\n@@CODE${index}@@\n`;
  });
  const lines = escapeHtml(raw).split('\n');
  const out = []; let i = 0; let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^@@CODE\d+@@$/.test(line.trim())) { closeList(); out.push(line.trim()); i++; continue; }
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i+1])) {
      closeList(); const headers = line.replace(/^\||\|$/g,'').split('|'); i += 2; const rows=[];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(lines[i].replace(/^\||\|$/g,'').split('|')); i++; }
      out.push(`<div class="table-wrap"><table><thead><tr>${headers.map(x=>`<th>${inline(x.trim())}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${inline(x.trim())}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`); continue;
    }
    let m;
    if ((m=line.match(/^(#{1,3})\s+(.+)/))) { closeList(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); }
    else if ((m=line.match(/^\s*[-*]\s+(.+)/))) { if(list!=='ul'){closeList();list='ul';out.push('<ul>')} out.push(`<li>${inline(m[1])}</li>`); }
    else if ((m=line.match(/^\s*\d+\.\s+(.+)/))) { if(list!=='ol'){closeList();list='ol';out.push('<ol>')} out.push(`<li>${inline(m[1])}</li>`); }
    else if ((m=line.match(/^&gt;\s*(.*)/))) { closeList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); }
    else if (!line.trim()) { closeList(); }
    else { closeList(); out.push(`<p>${inline(line)}</p>`); }
    i++;
  }
  closeList();
  let html = out.join('');
  codes.forEach((code, index) => { html = html.replace(`@@CODE${index}@@`, code); });
  return html;
}

function usageHtml(usage={}) {
  if (!Object.keys(usage).length) return '';
  const input = usage.input_tokens == null ? 0 : usage.input_tokens, output = usage.output_tokens == null ? 0 : usage.output_tokens;
  const cached = usage.input_tokens_details && usage.input_tokens_details.cached_tokens != null ? usage.input_tokens_details.cached_tokens : 0;
  const reasoning = usage.output_tokens_details && usage.output_tokens_details.reasoning_tokens != null ? usage.output_tokens_details.reasoning_tokens : 0;
  return `<div class="usage"><span>输入 ${input.toLocaleString()}</span><span>缓存命中 ${cached.toLocaleString()}</span><span>输出 ${output.toLocaleString()}</span><span>推理 ${reasoning.toLocaleString()}</span><span>合计 ${(input+output).toLocaleString()}</span></div>`;
}

function traceHtml(meta={}, active=false) {
  const reasoning = meta.reasoning || '';
  const searches = meta.searches || [];
  const sources = meta.sources || [];
  if (!reasoning && !searches.length && !active && !Object.keys(meta.usage || {}).length) return '';
  const status = active ? '进行中' : (meta.stopped ? '已停止' : '已完成');
  const searchHtml = searches.map((s,i) => {
    const label=s.action==='open_page'?'读取网页':'联网搜索';
    const detail=s.url?`<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.url)}</a>`:escapeHtml(Array.isArray(s.query)?s.query.filter(x=>!String(x).startsWith('ws_call_id=')).join('；'):(s.query||'DeepSeek 未返回查询词'));
    return `<details class="search-step"><summary>${label} ${i+1} · ${escapeHtml(s.status || 'completed')}</summary><div class="search-detail">${detail}</div></details>`;
  }).join('');
  const sourceHtml = sources.length ? `<div class="sources">${sources.map(s => `<a class="source-chip" href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title || s.url)}</a>`).join('')}</div>` : '';
  return `<details class="trace" open><summary>思考与联网 · ${status}${searches.length ? ` · ${searches.length} 次搜索` : ''}</summary><div class="trace-body">${reasoning ? `<div class="reasoning-text">${escapeHtml(reasoning)}</div>` : active ? '<div class="typing"><i></i><i></i><i></i></div>' : ''}${searchHtml}${sourceHtml}${usageHtml(meta.usage || {})}</div></details>`;
}

function messageHtml(message, index, live=false) {
  const assistant = message.role === 'assistant';
  const meta = message.meta || {};
  const content = message.content || '';
  return `<article class="message ${assistant ? 'assistant' : 'user'}" data-index="${index}">
    <div class="message-icon">${assistant ? 'DS' : escapeHtml(((state.me && state.me.username) || 'U')[0].toUpperCase())}</div>
    <div class="message-body"><div class="message-head"><strong>${assistant ? 'DeepSeek' : escapeHtml((state.me && state.me.username) || '你')}</strong><div class="message-actions"><button data-action="copy">复制</button><button data-action="retry">重新回答</button></div></div>
    ${assistant ? traceHtml(meta, live) : ''}<div class="message-content">${assistant ? (content ? markdown(content) : '<div class="typing"><i></i><i></i><i></i></div>') : `<p>${escapeHtml(content).replace(/\n/g,'<br>')}</p>`}</div>
    ${meta.error ? `<p class="job-error">${escapeHtml(meta.error)}</p>` : ''}</div></article>`;
}

function renderMessages() {
  const scroll = $('#chatScroll'); const oldTop = scroll.scrollTop;
  const items = [...state.messages];
  if (state.job && ['queued','running','failed','stopped'].includes(state.job.status)) {
    items.push({role:'assistant', content:state.job.answer || '', meta:{reasoning:state.job.reasoning,searches:state.job.searches,sources:state.job.sources,usage:state.job.usage,error:state.job.error,stopped:state.job.status==='stopped'}, live:['queued','running'].includes(state.job.status)});
  }
  $('#welcome').classList.toggle('hidden', items.length > 0);
  $('#messages').innerHTML = items.map((m,i)=>messageHtml(m,i,!!m.live)).join('');
  scroll.scrollTop = oldTop;
  wireMessageActions(items);
}

function wireMessageActions(items) {
  $$('.message').forEach(node => {
    const index = Number(node.dataset.index), msg = items[index];
    $('[data-action="copy"]', node).onclick = async () => { await navigator.clipboard.writeText(msg.content || ''); toast('已复制'); };
    $('[data-action="retry"]', node).onclick = () => {
      let prompt = msg.role === 'user' ? msg.content : '';
      if (!prompt) for (let i=index-1;i>=0;i--) if(items[i].role==='user'){prompt=items[i].content;break}
      if (prompt) submitPrompt(prompt);
    };
    $$('.code-download', node).forEach(button => button.onclick = () => {
      const codeNode = $('pre code', button.closest('.code-wrap'));
      const langNode = $('span', button.parentElement);
      const code = codeNode ? codeNode.textContent : '';
      const lang = langNode ? langNode.textContent : 'txt';
      const blob = new Blob([code], {type:'text/plain;charset=utf-8'}), a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download=`code.${lang === 'code' ? 'txt' : lang}`; a.click(); URL.revokeObjectURL(a.href);
    });
  });
}

async function loadHistory(page=1) {
  const data = await api(`/api/conversations?page=${page}`); state.page=data.page; state.pages=data.pages;
  $('#historyCount').textContent=data.total;
  $('#history').innerHTML=data.items.map(c=>`<button class="history-item ${state.conversation&&state.conversation.id===c.id?'active':''}" data-id="${c.id}"><span>${escapeHtml(c.title)}</span><b class="history-delete" title="删除">×</b></button>`).join('') || '<p class="muted" style="padding:10px;font-size:12px">还没有对话</p>';
  $$('.history-item').forEach(button => {
    button.onclick=e=>{if(e.target.classList.contains('history-delete'))return;openConversation(button.dataset.id);closeSidebar()};
    $('.history-delete',button).onclick=async e=>{e.stopPropagation();if(!confirm('永久删除这个对话？'))return;try{await api(`/api/conversations/${button.dataset.id}`,{method:'DELETE'});if(state.conversation&&state.conversation.id===button.dataset.id)newConversation();await loadHistory(state.page)}catch(err){toast(err.message)}};
  });
  $('#pagination').innerHTML = state.pages>1 ? `<button id="prevPage" ${state.page<=1?'disabled':''}>‹</button><button>${state.page}/${state.pages}</button><button id="nextPage" ${state.page>=state.pages?'disabled':''}>›</button>` : '';
  if($('#prevPage'))$('#prevPage').onclick=()=>loadHistory(state.page-1); if($('#nextPage'))$('#nextPage').onclick=()=>loadHistory(state.page+1);
}

async function openConversation(id) {
  stopPolling();
  try {
    const data=await api(`/api/conversations/${id}`); state.conversation=data.conversation; state.messages=data.messages; state.job=data.active_job;
    $('#conversationTitle').textContent=state.conversation.title; renderMessages(); await loadHistory(state.page);
    $('#chatScroll').scrollTop=$('#chatScroll').scrollHeight;
    if(state.job)startPolling(state.job.id);
  } catch(err){toast(err.message)}
}

function newConversation(){stopPolling();state.conversation=null;state.messages=[];state.job=null;$('#conversationTitle').textContent='新对话';renderMessages();loadHistory(1)}
function selectedProvider(){return state.providers.find(x=>String(x.id)===$('#providerSelect').value)}

async function submitPrompt(value) {
  const content=(value == null ? $('#prompt').value : value).trim(); if(!content)return;
  const provider=selectedProvider(); if(!provider){$('#providerModal').showModal();toast('请先添加 DeepSeek API');return}
  if(state.job && ['queued','running'].includes(state.job.status)){toast('请先停止当前回答');return}
  $('#prompt').value='';resizePrompt();
  state.messages.push({role:'user',content,meta:{}});state.job={status:'queued',answer:'',reasoning:'',searches:[],sources:[],usage:{}};renderMessages();
  $('#chatScroll').scrollTop=$('#chatScroll').scrollHeight;setRunning(true);
  try{
    const data=await api('/api/chat',{method:'POST',body:{conversation_id:(state.conversation&&state.conversation.id)||null,content,provider_id:provider.id,model:'deepseek-v4-flash',effort:$('#effort').value}});
    if(!state.conversation)state.conversation={id:data.conversation_id,title:content.slice(0,36)};
    state.job.id=data.job_id;$('#conversationTitle').textContent=state.conversation.title;loadHistory(1);startPolling(data.job_id);
  }catch(err){state.messages.pop();state.job=null;setRunning(false);renderMessages();toast(err.message)}
}

function setRunning(on){$('#stopButton').classList.toggle('hidden',!on);$('#sendButton').disabled=on;$('#providerSelect').disabled=on}
function stopPolling(){if(state.poll)clearTimeout(state.poll);state.poll=null}
function startPolling(id){stopPolling();setRunning(true);const tick=async()=>{try{const job=await api(`/api/jobs/${id}`);state.job=job;renderMessages();if(['completed','failed','stopped'].includes(job.status)){stopPolling();setRunning(false);if(job.status==='completed')await openConversation(job.conversation_id);else renderMessages();return}}catch(err){toast(err.message);setRunning(false);return}state.poll=setTimeout(tick,700)};tick()}

async function loadProviders(){state.providers=await api('/api/providers');$('#providerSelect').innerHTML=state.providers.length?state.providers.map(p=>`<option value="${p.id}">${escapeHtml(p.name)} · ${escapeHtml(p.model)}</option>`).join(''):'<option value="">请先添加 API</option>';renderProviderList()}
function renderProviderList(){$('#providerList').innerHTML=state.providers.map(p=>`<div class="list-item"><div class="list-item-main"><strong>${escapeHtml(p.name)}</strong><small>${escapeHtml(p.api_key_masked)} · ${escapeHtml(p.model)}</small></div><button class="danger-btn" data-id="${p.id}">删除</button></div>`).join('')||'<p class="muted">尚未添加 API。</p>';$$('.danger-btn',$('#providerList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除这个 API 配置？'))return;try{await api(`/api/providers/${b.dataset.id}`,{method:'DELETE'});await loadProviders()}catch(err){toast(err.message)}})}

async function testProvider(){const button=$('#testProvider');button.disabled=true;$('#providerStatus').textContent='正在连接 DeepSeek…';try{const body=providerFormData();const result=await api('/api/providers/test',{method:'POST',body});const select=$('#providerModel');select.innerHTML=(result.models.length?result.models:['deepseek-v4-flash']).map(m=>`<option value="${escapeHtml(m)}" ${m==='deepseek-v4-flash'?'selected':''}>${escapeHtml(m)}</option>`).join('');$('#providerStatus').textContent=result.native_search_models.length?'连接成功，检测到 V4 Flash 原生搜索模型。':'API 可用，但模型列表中没有 deepseek-v4-flash。'}catch(err){$('#providerStatus').textContent=err.message}finally{button.disabled=false}}
function providerFormData(){return{name:$('#providerName').value||'DeepSeek',api_key:$('#providerKey').value,base_url:$('#providerBase').value,model:$('#manualModel').value.trim()||$('#providerModel').value||'deepseek-v4-flash'}}

async function loadUsers(){const users=await api('/api/users');$('#userList').innerHTML=users.map(u=>`<div class="list-item"><span class="avatar">${escapeHtml(u.username[0].toUpperCase())}</span><div class="list-item-main"><strong>${escapeHtml(u.username)}</strong><small>${u.is_admin?'管理员':'普通账号'}</small></div><div class="item-actions"><button class="soft-btn" data-password-user="${u.id}" data-username="${escapeHtml(u.username)}">改密码</button>${u.id!==state.me.id?`<button class="danger-btn" data-user="${u.id}">删除</button>`:''}</div></div>`).join('');$$('[data-user]',$('#userList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除账号及其全部独立数据？'))return;try{await api(`/api/users/${b.dataset.user}`,{method:'DELETE'});loadUsers()}catch(err){toast(err.message)}});$$('[data-password-user]',$('#userList')).forEach(b=>b.onclick=()=>{$('#passwordUserId').value=b.dataset.passwordUser;$('#passwordTarget').textContent=`正在修改：${b.dataset.username}`;$('#changePassword').value='';$('#changePasswordAgain').value='';$('#passwordError').textContent='';$('#passwordModal').showModal()})}

function resizePrompt(){const p=$('#prompt');p.style.height='auto';p.style.height=Math.min(p.scrollHeight,180)+'px'}
function openSidebar(){$('#sidebar').classList.add('open')}function closeSidebar(){$('#sidebar').classList.remove('open')}

async function boot(){try{state.me=await api('/api/me');$('#loginView').classList.add('hidden');$('#appView').classList.remove('hidden');$('#accountName').textContent=state.me.username;$('#accountRole').textContent=state.me.is_admin?'管理员':'用户';$('#avatar').textContent=state.me.username[0].toUpperCase();$('#usersButton').classList.toggle('hidden',!state.me.is_admin);await Promise.all([loadProviders(),loadHistory(1)]);renderMessages()}catch{$('#loginView').classList.remove('hidden');$('#appView').classList.add('hidden')}}

$('#loginForm').onsubmit=async e=>{e.preventDefault();$('#loginError').textContent='';try{await api('/api/login',{method:'POST',body:{username:$('#loginUser').value,password:$('#loginPass').value}});await boot()}catch(err){$('#loginError').textContent=err.message}};
$('#logout').onclick=async()=>{await api('/api/logout',{method:'POST'});location.reload()};
$('#newChat').onclick=()=>{newConversation();closeSidebar()};$('#composer').onsubmit=e=>{e.preventDefault();submitPrompt()};
$('#prompt').oninput=resizePrompt;$('#prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitPrompt()}};
$('#stopButton').onclick=async()=>{if(state.job&&state.job.id){await api(`/api/jobs/${state.job.id}/stop`,{method:'POST'});toast('正在停止')}};
$('#providerButton').onclick=()=>{$('#providerModal').showModal();renderProviderList()};$('#usersButton').onclick=()=>{loadUsers();$('#usersModal').showModal()};
$$('.close-modal').forEach(b=>b.onclick=()=>b.closest('dialog').close());$$('dialog').forEach(d=>d.onclick=e=>{if(e.target===d)d.close()});
$('#testProvider').onclick=testProvider;$('#providerForm').onsubmit=async e=>{e.preventDefault();try{const body=providerFormData();if(body.model!=='deepseek-v4-flash')throw new Error('原生联网搜索目前必须选择 deepseek-v4-flash');await api('/api/providers',{method:'POST',body});e.target.reset();$('#providerBase').value='https://api.deepseek.com';$('#providerStatus').textContent='';await loadProviders();toast('API 已保存')}catch(err){$('#providerStatus').textContent=err.message}};
$('#userForm').onsubmit=async e=>{e.preventDefault();try{await api('/api/users',{method:'POST',body:{username:$('#newUsername').value,password:$('#newPassword').value,is_admin:$('#newAdmin').checked}});e.target.reset();await loadUsers();toast('账号已新增')}catch(err){toast(err.message)}};
$('#passwordForm').onsubmit=async e=>{e.preventDefault();const password=$('#changePassword').value;if(password!==$('#changePasswordAgain').value){$('#passwordError').textContent='两次输入的密码不一致';return}try{await api(`/api/users/${$('#passwordUserId').value}/password`,{method:'PUT',body:{password}});$('#passwordModal').close();toast('密码已修改')}catch(err){$('#passwordError').textContent=err.message}};
$('#openSidebar').onclick=openSidebar;$('#closeSidebar').onclick=closeSidebar;$('#scrim').onclick=closeSidebar;
$$('[data-prompt]').forEach(b=>b.onclick=()=>submitPrompt(b.dataset.prompt));
$('#chatScroll').onscroll=()=>{const s=$('#chatScroll');$('#jumpBottom').classList.toggle('hidden',s.scrollHeight-s.scrollTop-s.clientHeight<220)};$('#jumpBottom').onclick=()=>$('#chatScroll').scrollTo({top:$('#chatScroll').scrollHeight,behavior:'smooth'});
boot();

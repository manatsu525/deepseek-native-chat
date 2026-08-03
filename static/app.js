const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const state = {me:null, providers:[], conversation:null, messages:[], job:null, page:1, pages:1, poll:null};
const detailState = new Map();

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
const CODE_EXTENSIONS = {
  python:'py', py:'py', javascript:'js', js:'js', typescript:'ts', ts:'ts',
  jsx:'jsx', tsx:'tsx', bash:'sh', shell:'sh', sh:'sh', zsh:'sh',
  html:'html', css:'css', json:'json', yaml:'yml', yml:'yml', xml:'xml',
  sql:'sql', java:'java', kotlin:'kt', c:'c', cpp:'cpp', 'c++':'cpp',
  csharp:'cs', 'c#':'cs', go:'go', rust:'rs', ruby:'rb', php:'php',
  swift:'swift', markdown:'md', md:'md', dockerfile:'Dockerfile'
};

function inline(value='') {
  const codeTokens = [], linkTokens = [];
  let text = escapeHtml(value).replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `\u0000CODE${codeTokens.length}\u0000`;
    codeTokens.push(`<code>${code}</code>`);
    return token;
  });
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
    const decoded = url.replace(/&amp;/g, '&');
    if (!/^https?:\/\//i.test(decoded)) return `${label} (${url})`;
    const token = `\u0000LINK${linkTokens.length}\u0000`;
    linkTokens.push(`<a href="${escapeHtml(decoded)}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    return token;
  });
  text = text.replace(/https?:\/\/[^\s<]+/gi, rawUrl => {
    const match = rawUrl.match(/^(.*?)([.,!?;:，。！？；：)\]}]+)$/);
    const url = match ? match[1] : rawUrl;
    const suffix = match ? match[2] : '';
    const token = `\u0000LINK${linkTokens.length}\u0000`;
    const decodedUrl = url.replace(/&amp;/g, '&');
    linkTokens.push(`<a href="${escapeHtml(decodedUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(decodedUrl)}</a>`);
    return token + suffix;
  });
  text = text
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
    .replace(/~~([^~\n]+)~~/g, '<del>$1</del>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  text = text.replace(/\u0000LINK(\d+)\u0000/g, (_, index) => linkTokens[Number(index)] || '');
  return text.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeTokens[Number(index)] || '');
}

function splitTableRow(line) {
  let value = String(line).trim();
  if (value.startsWith('|')) value = value.slice(1);
  if (value.endsWith('|')) value = value.slice(0, -1);
  return value.split('|').map(cell => cell.trim());
}

function isTableDivider(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function calloutInfo(lines) {
  if (!lines.length) return null;
  const marker = lines[0].match(/^\s*\[!(NOTE|INFO|TIP|SUCCESS|IMPORTANT|WARNING|CAUTION|DANGER|ERROR)\]\s*(.*)$/i);
  const shorthand = lines[0].match(/^\s*(⚠️?|❗|‼️|风险提示|风险|警告|注意|重要提示)\s*[:：]?\s*(.*)$/i);
  const bold = lines[0].match(/^\s*\*\*(风险提示|风险|警告|注意|重要提示|warning|caution|danger|error)\s*[:：]?\*\*\s*(.*)$/i);
  const found = marker || shorthand || bold;
  if (!found) return null;
  const rawType = String(found[1]).toLowerCase();
  const type = /warning|caution|⚠|风险|警告|注意/.test(rawType) ? 'warning' :
    /danger|error|❗|‼/.test(rawType) ? 'danger' :
    /important/.test(rawType) ? 'important' : /tip|success/.test(rawType) ? 'tip' :
    'note';
  const labels = {note:['说明','ℹ'], tip:['提示','✓'], important:['重要','◆'], warning:['注意','⚠'], danger:['风险','!']};
  return {type, label:labels[type][0], icon:labels[type][1], body:[found[2], ...lines.slice(1)].filter(line => line.trim())};
}

function renderCodeBlock(language, code) {
  const rawLanguage = String(language || '').trim().split(/\s+/)[0];
  const safeLanguage = rawLanguage.toLowerCase().replace(/[^a-z0-9_+#.-]/g, '');
  const extension = CODE_EXTENSIONS[safeLanguage] || safeLanguage || 'txt';
  const label = escapeHtml(rawLanguage || 'code');
  const languageClass = safeLanguage ? ` class="language-${escapeHtml(safeLanguage)}"` : '';
  return `<div class="code-wrap code-block" data-language="${escapeHtml(safeLanguage)}" data-extension="${escapeHtml(extension)}"><div class="code-label"><span>${label}</span><button class="code-download" type="button" title="下载代码" aria-label="下载代码"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0 5-5m-5 5-5-5M5 21h14"></path></svg></button></div><pre><code${languageClass}>${escapeHtml(String(code).replace(/\n$/, ''))}</code></pre></div>`;
}

function renderCallout(lines) {
  const info = calloutInfo(lines);
  if (!info) return `<blockquote>${lines.map(line => inline(line)).join('<br>')}</blockquote>`;
  const body = info.body.map(line => inline(line)).join('<br>');
  return `<aside class="md-callout md-callout-${info.type}"><div class="md-callout-title"><span aria-hidden="true">${info.icon}</span>${info.label}</div><div class="md-callout-body">${body || '<span class="muted"> </span>'}</div></aside>`;
}

function markdown(raw='') {
  const lines = String(raw || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [], paragraph = [];
  let listType = '';
  const flushParagraph = () => {
    if (!paragraph.length) return;
    out.push(`<p>${paragraph.map(line => inline(line)).join('<br>')}</p>`);
    paragraph.length = 0;
  };
  const closeList = () => { if (listType) out.push(`</${listType}>`); listType = ''; };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i], trimmed = line.trim();
    if (/^```/.test(trimmed)) {
      flushParagraph(); closeList();
      const language = trimmed.slice(3).trim().split(/\s+/)[0] || '';
      const code = []; i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) { code.push(lines[i]); i++; }
      out.push(renderCodeBlock(language, code.join('\n'))); continue;
    }
    if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      flushParagraph(); closeList();
      const headers = splitTableRow(line), divider = splitTableRow(lines[i + 1]);
      const align = divider.map(cell => cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : cell.startsWith(':') ? 'left' : '');
      i += 2; const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(splitTableRow(lines[i])); i++; }
      i--;
      let table = '<div class="table-wrap"><table><thead><tr>';
      table += headers.map((cell, index) => `<th${align[index] ? ` style="text-align:${align[index]}"` : ''}>${inline(cell)}</th>`).join('');
      table += '</tr></thead><tbody>';
      rows.forEach(row => { table += `<tr>${headers.map((_, index) => `<td${align[index] ? ` style="text-align:${align[index]}"` : ''}>${inline(row[index] || '')}</td>`).join('')}</tr>`; });
      out.push(`${table}</tbody></table></div>`); continue;
    }
    if (!trimmed) { flushParagraph(); closeList(); continue; }
    let match = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (match) { flushParagraph(); closeList(); const level=match[1].length; out.push(`<h${level}>${inline(match[2])}</h${level}>`); continue; }
    if (/^([-*_])(?:\s*\1){2,}$/.test(trimmed)) { flushParagraph(); closeList(); out.push('<hr>'); continue; }
    if (calloutInfo([trimmed])) { flushParagraph(); closeList(); out.push(renderCallout([trimmed])); continue; }
    if (/^\s*>/.test(line)) {
      flushParagraph(); closeList(); const quote=[];
      while (i < lines.length && /^\s*>/.test(lines[i])) { quote.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
      i--; out.push(renderCallout(quote)); continue;
    }
    const unordered = trimmed.match(/^[-*+]\s+(.+)$/), ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph(); const nextType = unordered ? 'ul' : 'ol';
      if (listType !== nextType) { closeList(); listType=nextType; out.push(`<${listType}>`); }
      out.push(`<li>${inline((unordered || ordered)[1])}</li>`); continue;
    }
    closeList(); paragraph.push(line);
  }
  flushParagraph(); closeList();
  return out.join('');
}

function usageHtml(usage={}) {
  if (!Object.keys(usage).length) return '';
  const input = usage.input_tokens == null ? 0 : usage.input_tokens, output = usage.output_tokens == null ? 0 : usage.output_tokens;
  const cached = usage.input_tokens_details && usage.input_tokens_details.cached_tokens != null ? usage.input_tokens_details.cached_tokens : 0;
  const reasoning = usage.output_tokens_details && usage.output_tokens_details.reasoning_tokens != null ? usage.output_tokens_details.reasoning_tokens : 0;
  return `<div class="usage"><span>输入 ${input.toLocaleString()}</span><span>缓存命中 ${cached.toLocaleString()}</span><span>输出 ${output.toLocaleString()}</span><span>推理 ${reasoning.toLocaleString()}</span><span>合计 ${(input+output).toLocaleString()}</span></div>`;
}

function traceHtml(meta={}, active=false, detailKey='trace') {
  const reasoning = meta.reasoning || '';
  const searches = meta.searches || [];
  const sources = [];
  const seenSources = new Set();
  [...(meta.sources || []), ...searches.filter(item => item.action === 'open_page' && item.url).map(item => ({url:item.url,title:item.url}))].forEach(source => {
    const url = String(source.url || '');
    if (!/^https?:\/\//i.test(url) || seenSources.has(url)) return;
    seenSources.add(url); sources.push({url, title:source.title || url});
  });
  if (!reasoning && !searches.length && !active && !Object.keys(meta.usage || {}).length) return '';
  const status = active ? '进行中' : (meta.stopped ? '已停止' : '已完成');
  const searchHtml = searches.map((s,i) => {
    const label=s.action==='open_page'?'读取网页':'联网搜索';
    const searchKey=`${detailKey}-search-${s.id || i}`;
    const searchOpen=detailState.get(searchKey) ? ' open' : '';
    const detail=s.url?`<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.url)}</a>`:escapeHtml(Array.isArray(s.query)?s.query.filter(x=>!String(x).startsWith('ws_call_id=')).join('；'):(s.query||'DeepSeek 未返回查询词'));
    return `<details class="search-step" data-detail-key="${escapeHtml(searchKey)}"${searchOpen}><summary>${label} ${i+1} · ${escapeHtml(s.status || 'completed')}</summary><div class="search-detail">${detail}</div></details>`;
  }).join('');
  const sourceHtml = sources.length ? `<div class="sources"><span class="sources-label">来源</span>${sources.map(s => `<a class="source-chip" href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.url)}">${escapeHtml(s.title || s.url)}</a>`).join('')}</div>` : '';
  const traceOpen=detailState.get(detailKey) ? ' open' : '';
  return `<details class="trace" data-detail-key="${escapeHtml(detailKey)}"${traceOpen}><summary>思考与联网 · ${status}${searches.length ? ` · ${searches.length} 次搜索` : ''}</summary><div class="trace-body">${reasoning ? `<div class="reasoning-text">${escapeHtml(reasoning)}</div>` : active ? '<div class="typing"><i></i><i></i><i></i></div>' : ''}${searchHtml}${sourceHtml}${usageHtml(meta.usage || {})}</div></details>`;
}

function messageHtml(message, index, live=false) {
  const assistant = message.role === 'assistant';
  const meta = message.meta || {};
  const content = message.content || '';
  const detailKey = `trace-${meta.job_id || `message-${index}`}`;
  return `<article class="message ${assistant ? 'assistant' : 'user'}${live ? ' live-message' : ''}" data-index="${index}">
    <div class="message-icon">${assistant ? 'DS' : escapeHtml(((state.me && state.me.username) || 'U')[0].toUpperCase())}</div>
    <div class="message-body"><div class="message-head"><strong>${assistant ? 'DeepSeek' : escapeHtml((state.me && state.me.username) || '你')}</strong></div>
    ${assistant ? traceHtml(meta, live, detailKey) : ''}<div class="message-content">${assistant ? (content ? markdown(content) : '<div class="typing"><i></i><i></i><i></i></div>') : `<p>${escapeHtml(content).replace(/\n/g,'<br>')}</p>`}</div><div class="message-actions"><button type="button" data-action="copy">复制</button><button type="button" data-action="retry">重新回答</button></div>
    ${meta.error ? `<p class="job-error">${escapeHtml(meta.error)}</p>` : ''}</div></article>`;
}

function renderMessages() {
  const scroll = $('#chatScroll'); const oldTop = scroll.scrollTop;
  $$('details[data-detail-key]').forEach(detail => detailState.set(detail.dataset.detailKey, detail.open));
  const items = [...state.messages];
  if (state.job && ['queued','running','failed','stopped'].includes(state.job.status)) {
    items.push({role:'assistant', content:state.job.answer || '', meta:{job_id:state.job.id,reasoning:state.job.reasoning,searches:state.job.searches,sources:state.job.sources,usage:state.job.usage,error:state.job.error,stopped:state.job.status==='stopped'}, live:['queued','running'].includes(state.job.status)});
  }
  $('#welcome').classList.toggle('hidden', items.length > 0);
  $('#messages').innerHTML = items.map((m,i)=>messageHtml(m,i,!!m.live)).join('');
  scroll.scrollTop = oldTop;
  $$('details[data-detail-key]').forEach(detail => detail.addEventListener('toggle', () => detailState.set(detail.dataset.detailKey, detail.open)));
  wireMessageActions(items);
}

function replaceJobMessage(job, liveState) {
  const current = $('#messages .live-message');
  const live = {
    role: 'assistant',
    content: job.answer || '',
    meta: {
      job_id: job.id,
      reasoning: job.reasoning,
      searches: job.searches,
      sources: job.sources,
      usage: job.usage,
      error: job.error,
      stopped: job.status === 'stopped',
    },
  };
  if (!current) {
    if (liveState) renderMessages();
    return liveState ? null : live;
  }
  $$('details[data-detail-key]').forEach(detail => detailState.set(detail.dataset.detailKey, detail.open));
  const fragment = document.createRange().createContextualFragment(messageHtml(live, state.messages.length, liveState));
  current.replaceWith(fragment.firstElementChild);
  $$('details[data-detail-key]').forEach(detail => detail.addEventListener('toggle', () => detailState.set(detail.dataset.detailKey, detail.open)));
  wireMessageActions([...state.messages, live]);
  return live;
}

function updateLiveMessage() {
  if (!state.job || !['queued','running'].includes(state.job.status)) {
    renderMessages();
    return;
  }
  replaceJobMessage(state.job, true);
}

function finalizeLiveMessage(job) {
  const hadLiveNode = Boolean($('#messages .live-message'));
  const assistant = replaceJobMessage(job, false);
  if (assistant) {
    state.messages.push(assistant);
    state.job = null;
    if (!hadLiveNode) renderMessages();
  } else {
    state.job = null;
    renderMessages();
  }
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
      const wrap = button.closest('.code-wrap');
      const codeNode = $('pre code', wrap);
      const langNode = $('span', button.parentElement);
      const code = codeNode ? codeNode.textContent : '';
      const lang = wrap && wrap.dataset.extension ? wrap.dataset.extension : (langNode ? langNode.textContent : 'txt');
      const blob = new Blob([code], {type:'text/plain;charset=utf-8'}), a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download=`code.${lang === 'code' ? 'txt' : lang}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 0);
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
function startPolling(id){stopPolling();setRunning(true);const tick=async()=>{try{const job=await api(`/api/jobs/${id}`);state.job=job;if(job.status==='completed'){stopPolling();setRunning(false);finalizeLiveMessage(job);await loadHistory(1);return}if(['failed','stopped'].includes(job.status)){stopPolling();setRunning(false);renderMessages();return}updateLiveMessage()}catch(err){toast(err.message);setRunning(false);return}state.poll=setTimeout(tick,700)};tick()}

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

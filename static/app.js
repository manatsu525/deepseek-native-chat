const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const state = {me:null, providers:[], conversation:null, messages:[], job:null, page:1, pages:1, poll:null};
const detailState = new Map();

function storageKey(name) { return `deepseek-native-chat.${name}.${state.me ? state.me.id : 'guest'}`; }
function storedValue(name) {
  try { return localStorage.getItem(storageKey(name)); } catch { return null; }
}
function storeValue(name, value) {
  try {
    if (value == null || value === '') localStorage.removeItem(storageKey(name));
    else localStorage.setItem(storageKey(name), String(value));
  } catch {}
}

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
  const web = usage.web_search_usage || {};
  const webUsage = web.tool_usage == null ? '' : `<span>联网 ${Number(web.tool_usage).toLocaleString()} 次</span><span>网页 ${Number(web.page_usage || 0).toLocaleString()} 篇</span>`;
  return `<div class="usage"><span>输入 ${input.toLocaleString()}</span><span>缓存命中 ${cached.toLocaleString()}</span><span>输出 ${output.toLocaleString()}</span><span>推理 ${reasoning.toLocaleString()}</span><span>合计 ${(input+output).toLocaleString()}</span>${webUsage}</div>`;
}

function traceHtml(meta={}, active=false, detailKey='trace') {
  const reasoning = meta.reasoning || '';
  const searches = meta.searches || [];
  const sources = [];
  const seenSources = new Set();
  [...(meta.sources || []), ...searches.filter(item => item.action === 'open_page' && item.url).map(item => ({url:item.url,title:item.url}))].forEach(source => {
    const url = String(source.url || '');
    if (!/^https?:\/\//i.test(url) || seenSources.has(url)) return;
    seenSources.add(url); sources.push(source);
  });
  if (!reasoning && !searches.length && !active && !Object.keys(meta.usage || {}).length) return '';
  const status = active ? '进行中' : (meta.stopped ? '已停止' : '已完成');
  const searchHtml = searches.map((s,i) => {
    const label=s.action==='open_page'?'读取网页':'联网搜索';
    const searchKey=`${detailKey}-search-${s.id || i}`;
    const searchOpen=detailState.get(searchKey) ? ' open' : '';
    const detail=s.url?`<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.url)}</a>`:escapeHtml(Array.isArray(s.query)?s.query.filter(x=>!String(x).startsWith('ws_call_id=')).join('；'):(s.query||'DeepSeek 未返回查询词'));
    const error=s.error?`<div class="search-error">${escapeHtml(s.error)}</div>`:'';
    return `<details class="search-step" data-detail-key="${escapeHtml(searchKey)}"${searchOpen}><summary>${label} ${i+1} · ${escapeHtml(s.status || 'completed')}</summary><div class="search-detail">${detail}${error}</div></details>`;
  }).join('');
  const sourceHtml = sources.length ? `<div class="sources"><span class="sources-label">来源</span>${sources.map(s => { const logo=s.logo_url&&safeUrl(s.logo_url)!=='#'?`<img src="${safeUrl(s.logo_url)}" alt="" loading="lazy">`:''; return `<a class="source-chip" href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.summary || s.url)}">${logo}<span>${escapeHtml(s.title || s.url)}</span></a>`; }).join('')}</div>` : '';
  const traceOpen=detailState.get(detailKey) ? ' open' : '';
  return `<details class="trace" data-detail-key="${escapeHtml(detailKey)}"${traceOpen}><summary>思考与联网 · ${status}${searches.length ? ` · ${searches.length} 次搜索` : ''}</summary><div class="trace-body">${reasoning ? `<div class="reasoning-text">${escapeHtml(reasoning)}</div>` : active ? '<div class="typing"><i></i><i></i><i></i></div>' : ''}${searchHtml}${sourceHtml}${usageHtml(meta.usage || {})}</div></details>`;
}

function messageHtml(message, index, live=false) {
  const assistant = message.role === 'assistant';
  const meta = message.meta || {};
  const content = message.content || '';
  const detailKey = `trace-${meta.job_id || `message-${index}`}`;
  const assistantName = meta.provider_type === 'mimo' ? 'MiMo' : 'DeepSeek';
  return `<article class="message ${assistant ? 'assistant' : 'user'}${live ? ' live-message' : ''}" data-index="${index}">
    <div class="message-icon">${assistant ? (meta.provider_type === 'mimo' ? 'MM' : 'DS') : escapeHtml(((state.me && state.me.username) || 'U')[0].toUpperCase())}</div>
    <div class="message-body"><div class="message-head"><strong>${assistant ? assistantName : escapeHtml((state.me && state.me.username) || '你')}</strong></div>
    ${assistant ? traceHtml(meta, live, detailKey) : ''}<div class="message-content">${assistant ? (content ? markdown(content) : live ? '<div class="typing"><i></i><i></i><i></i></div>' : '') : `<p>${escapeHtml(content).replace(/\n/g,'<br>')}</p>`}</div><div class="message-actions"><button type="button" data-action="copy">复制</button><button type="button" data-action="retry">重新回答</button></div>
    ${meta.error ? `<p class="job-error">${escapeHtml(meta.error)}</p>` : ''}</div></article>`;
}

function renderMessages() {
  const scroll = $('#chatScroll'); const oldTop = scroll.scrollTop;
  $$('details[data-detail-key]').forEach(detail => detailState.set(detail.dataset.detailKey, detail.open));
  const items = [...state.messages];
  if (state.job && ['queued','running','failed','stopped'].includes(state.job.status)) {
    items.push({role:'assistant', content:state.job.answer || '', meta:{job_id:state.job.id,provider_id:state.job.provider_id,provider_type:state.job.provider_type,model:state.job.model,reasoning:state.job.reasoning,searches:state.job.searches,sources:state.job.sources,usage:state.job.usage,error:state.job.error,stopped:state.job.status==='stopped'}, live:['queued','running'].includes(state.job.status)});
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
      provider_id: job.provider_id,
      provider_type: job.provider_type || (selectedProvider() && selectedProvider().provider_type),
      model: job.model,
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
    restoreProviderForConversation(data);
    storeValue('active-conversation', state.conversation.id);
    $('#conversationTitle').textContent=state.conversation.title; renderMessages(); await loadHistory(state.page);
    if(state.job)startPolling(state.job.id);
    return true;
  } catch(err){
    if (err.message === '对话不存在') storeValue('active-conversation', null);
    toast(err.message);
  }
  return false;
}

function newConversation(){stopPolling();state.conversation=null;state.messages=[];state.job=null;storeValue('active-conversation', null);$('#conversationTitle').textContent='新对话';renderMessages();loadHistory(1)}

async function submitPrompt(value) {
  const content=(value == null ? $('#prompt').value : value).trim(); if(!content)return;
  const provider=selectedProvider(); if(!provider){$('#providerModal').showModal();toast('请先添加 API 配置');return}
  if(state.job && ['queued','running'].includes(state.job.status)){toast('请先停止当前回答');return}
  $('#prompt').value='';resizePrompt();
  state.messages.push({role:'user',content,meta:{}});state.job={status:'queued',provider_type:provider.provider_type,provider_id:provider.id,answer:'',reasoning:'',searches:[],sources:[],usage:{}};renderMessages();
  $('#chatScroll').scrollTop=$('#chatScroll').scrollHeight;setRunning(true);
  try{
    const data=await api('/api/chat',{method:'POST',body:{conversation_id:(state.conversation&&state.conversation.id)||null,content,provider_id:provider.id,model:provider.model,effort:$('#effort').value}});
    if(!state.conversation)state.conversation={id:data.conversation_id,title:content.slice(0,36)};
    storeValue('active-conversation', state.conversation.id);
    state.job.id=data.job_id;$('#conversationTitle').textContent=state.conversation.title;loadHistory(1);startPolling(data.job_id);
  }catch(err){state.messages.pop();state.job=null;setRunning(false);renderMessages();toast(err.message)}
}

function setRunning(on){$('#stopButton').classList.toggle('hidden',!on);$('#sendButton').disabled=on;$('#providerSelect').disabled=on}
function stopPolling(){if(state.poll)clearTimeout(state.poll);state.poll=null}
function startPolling(id){stopPolling();setRunning(true);const tick=async()=>{try{const job=await api(`/api/jobs/${id}`);if(job.provider_id)selectProvider(job.provider_id,false);state.job=job;if(job.status==='completed'){stopPolling();setRunning(false);finalizeLiveMessage(job);await loadHistory(1);return}if(['failed','stopped'].includes(job.status)){stopPolling();setRunning(false);renderMessages();return}updateLiveMessage()}catch(err){toast(err.message);setRunning(false);return}state.poll=setTimeout(tick,700)};tick()}

function providerLabel(provider){return provider.provider_type==='mimo'?'MiMo':'DeepSeek'}
function selectedProvider(){return state.providers.find(x=>String(x.id)===$('#providerSelect').value)}
function selectProvider(providerId, persist=true) {
  const provider=state.providers.find(x=>String(x.id)===String(providerId));
  if(!provider) return null;
  const changed=$('#providerSelect').value!==String(provider.id);
  $('#providerSelect').value=String(provider.id);
  if(persist) storeValue('active-provider', provider.id);
  if(changed) updateProviderUi();
  return provider;
}
function providerForConversation(data) {
  const job=data.active_job;
  if(job && job.provider_id) return job.provider_id;
  for(const message of [...(data.messages || [])].reverse()) {
    const meta=message.meta || {};
    if(meta.provider_id) return meta.provider_id;
    if(meta.provider_type || meta.model) {
      const match=state.providers.find(provider =>
        (!meta.provider_type || provider.provider_type===meta.provider_type) &&
        (!meta.model || provider.model===meta.model)
      );
      if(match) return match.id;
    }
  }
  return storedValue('active-provider');
}
function restoreProviderForConversation(data) {
  const providerId=providerForConversation(data);
  if(providerId) selectProvider(providerId);
}
function updateProviderUi(){
  const provider=selectedProvider(), mimo=provider&&provider.provider_type==='mimo';
  $('#mimoSettingsButton').classList.toggle('hidden',!mimo);
  $('#effort').disabled=!!mimo;
  $('#effort').title=mimo?'MiMo 的思考开关在“MiMo 联网”设置中配置':'控制 DeepSeek 模型推理投入';
  $('#nativePill').textContent=mimo?'● MiMo Web':'● Native Web';
  $('#welcomeOrbitMark').textContent=mimo?'MM':'DS';
  $('#welcomeEyebrow').textContent=mimo?'XIAOMI MIMO V2.5': 'DEEPSEEK V4 FLASH';
  $('#welcomeTitle').textContent=mimo?'使用 MiMo 原生联网':'问点需要查证的问题';
  $('#welcomeDescription').textContent=mimo?'模型会通过 MiMo 服务端联网工具获取实时公开信息。':'模型会在 DeepSeek 服务端自行判断是否搜索，并在需要时多轮检索。';
}

async function loadProviders(){
  state.providers=await api('/api/providers');
  $('#providerSelect').innerHTML=state.providers.length?state.providers.map(p=>`<option value="${p.id}">${escapeHtml(p.name)} · ${providerLabel(p)} · ${escapeHtml(p.model)}</option>`).join(''):'<option value="">请先添加 API</option>';
  const preferred=storedValue('active-provider');
  if(state.providers.length) selectProvider(state.providers.some(p=>String(p.id)===String(preferred)) ? preferred : state.providers[0].id);
  renderProviderList(); updateProviderUi();
}
function renderProviderList(){
  $('#providerList').innerHTML=state.providers.map(p=>`<div class="list-item"><div class="list-item-main"><strong>${escapeHtml(p.name)} <span class="provider-kind">${providerLabel(p)}</span></strong><small>${escapeHtml(p.api_key_masked)} · ${escapeHtml(p.model)}</small></div><button class="danger-btn" data-id="${p.id}">删除</button></div>`).join('')||'<p class="muted">尚未添加 API。</p>';
  $$('.danger-btn',$('#providerList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除这个 API 配置？'))return;try{await api(`/api/providers/${b.dataset.id}`,{method:'DELETE'});await loadProviders()}catch(err){toast(err.message)}})
}

function providerType(){return $('#providerType').value||'deepseek'}
function syncProviderForm(){
  const mimo=providerType()==='mimo';
  const base=$('#providerBase'), model=$('#providerModel');
  base.value=mimo?'https://api.xiaomimimo.com/v1':'https://api.deepseek.com';
  model.innerHTML=mimo?'<option value="mimo-v2.5-pro">mimo-v2.5-pro</option><option value="mimo-v2.5">mimo-v2.5</option>':'<option value="deepseek-v4-flash">deepseek-v4-flash</option>';
  $('#providerName').placeholder=mimo?'例如：我的 MiMo':'例如：我的 DeepSeek';
  $('#providerStatus').textContent='';
}
async function testProvider(){
  const button=$('#testProvider');button.disabled=true;$('#providerStatus').textContent=`正在连接 ${providerType()==='mimo'?'MiMo':'DeepSeek'}…`;
  try{
    const body=providerFormData();const result=await api('/api/providers/test',{method:'POST',body});const select=$('#providerModel');
    const fallback=providerType()==='mimo'?['mimo-v2.5-pro','mimo-v2.5']:['deepseek-v4-flash'];
    // The provider may list TTS/ASR models too; this chat only accepts the
    // text models that support the native web-search workflow.
    const advertised=result.models || [];
    const supportedModels=(result.supported_models||[]).filter(m=>advertised.includes(m));
    const models=supportedModels.length?supportedModels:fallback;
    select.innerHTML=models.map(m=>`<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('');
    const supported=(result.supported_models||[]).filter(m=>models.includes(m));
    $('#providerStatus').textContent=supported.length?`连接成功，可用联网模型：${supported.join('、')}`:`API 可用，但没有检测到当前服务商支持的联网模型。`;
  }catch(err){$('#providerStatus').textContent=err.message}finally{button.disabled=false}
}
function providerFormData(){return{name:$('#providerName').value||providerLabel({provider_type:providerType()}),api_key:$('#providerKey').value,provider_type:providerType(),base_url:$('#providerBase').value,model:$('#manualModel').value.trim()||$('#providerModel').value|| (providerType()==='mimo'?'mimo-v2.5-pro':'deepseek-v4-flash')}}

function mimoSettings(provider){return {...{max_keyword:3,limit:5,force_search:false,country:'',region:'',city:'',thinking:'enabled',max_completion_tokens:8192,temperature:1,top_p:.95},...(provider&&provider.settings||{})}}
function fillMimoSettings(){
  const config=mimoSettings(selectedProvider());
  $('#mimoMaxKeyword').value=config.max_keyword;$('#mimoLimit').value=config.limit;$('#mimoForceSearch').checked=!!config.force_search;
  $('#mimoThinking').value=config.thinking;$('#mimoMaxCompletion').value=config.max_completion_tokens;$('#mimoTemperature').value=config.temperature;$('#mimoTopP').value=config.top_p;
  $('#mimoCountry').value=config.country||'';$('#mimoRegion').value=config.region||'';$('#mimoCity').value=config.city||'';syncMimoThinkingFields();
}
function syncMimoThinkingFields(){const disabled=$('#mimoThinking').value==='enabled';$('#mimoTemperature').disabled=disabled;$('#mimoTopP').disabled=disabled;$('#mimoSamplingNote').textContent=disabled?'深度思考开启时，MiMo 会强制使用 temperature=1.0、top_p=0.95。':'关闭思考后可自定义 temperature 和 top_p。'}
function openMimoSettings(){if(!selectedProvider()||selectedProvider().provider_type!=='mimo'){toast('请先选择 MiMo API');return}fillMimoSettings();$('#mimoModal').showModal()}
async function saveMimoSettings(event){
  event.preventDefault(); const provider=selectedProvider(); if(!provider)return;
  const body={max_keyword:Number($('#mimoMaxKeyword').value),limit:Number($('#mimoLimit').value),force_search:$('#mimoForceSearch').checked,thinking:$('#mimoThinking').value,max_completion_tokens:Number($('#mimoMaxCompletion').value),temperature:Number($('#mimoTemperature').value),top_p:Number($('#mimoTopP').value),country:$('#mimoCountry').value.trim(),region:$('#mimoRegion').value.trim(),city:$('#mimoCity').value.trim()};
  try{const updated=await api(`/api/providers/${provider.id}/settings`,{method:'PUT',body});const index=state.providers.findIndex(x=>x.id===provider.id);if(index>=0)state.providers[index]=updated;$('#mimoModal').close();updateProviderUi();toast('MiMo 参数已保存')}catch(err){$('#mimoStatus').textContent=err.message}
}

async function loadUsers(){const users=await api('/api/users');$('#userList').innerHTML=users.map(u=>`<div class="list-item"><span class="avatar">${escapeHtml(u.username[0].toUpperCase())}</span><div class="list-item-main"><strong>${escapeHtml(u.username)}</strong><small>${u.is_admin?'管理员':'普通账号'}</small></div><div class="item-actions"><button class="soft-btn" data-password-user="${u.id}" data-username="${escapeHtml(u.username)}">改密码</button>${u.id!==state.me.id?`<button class="danger-btn" data-user="${u.id}">删除</button>`:''}</div></div>`).join('');$$('[data-user]',$('#userList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除账号及其全部独立数据？'))return;try{await api(`/api/users/${b.dataset.user}`,{method:'DELETE'});loadUsers()}catch(err){toast(err.message)}});$$('[data-password-user]',$('#userList')).forEach(b=>b.onclick=()=>{$('#passwordUserId').value=b.dataset.passwordUser;$('#passwordTarget').textContent=`正在修改：${b.dataset.username}`;$('#changePassword').value='';$('#changePasswordAgain').value='';$('#passwordError').textContent='';$('#passwordModal').showModal()})}

function resizePrompt(){const p=$('#prompt');p.style.height='auto';p.style.height=Math.min(p.scrollHeight,180)+'px'}
function openSidebar(){$('#sidebar').classList.add('open')}function closeSidebar(){$('#sidebar').classList.remove('open')}

async function boot(){try{state.me=await api('/api/me');$('#loginView').classList.add('hidden');$('#appView').classList.remove('hidden');$('#accountName').textContent=state.me.username;$('#accountRole').textContent=state.me.is_admin?'管理员':'用户';$('#avatar').textContent=state.me.username[0].toUpperCase();$('#usersButton').classList.toggle('hidden',!state.me.is_admin);await Promise.all([loadProviders(),loadHistory(1)]);const activeId=storedValue('active-conversation');const restored=activeId?await openConversation(activeId):false;if(!restored)renderMessages()}catch{$('#loginView').classList.remove('hidden');$('#appView').classList.add('hidden')}}

$('#loginForm').onsubmit=async e=>{e.preventDefault();$('#loginError').textContent='';try{await api('/api/login',{method:'POST',body:{username:$('#loginUser').value,password:$('#loginPass').value}});await boot()}catch(err){$('#loginError').textContent=err.message}};
$('#logout').onclick=async()=>{await api('/api/logout',{method:'POST'});location.reload()};
$('#newChat').onclick=()=>{newConversation();closeSidebar()};$('#composer').onsubmit=e=>{e.preventDefault();submitPrompt()};
// Enter always inserts a newline. Sending is explicit via the send button.
$('#prompt').oninput=resizePrompt;
$('#stopButton').onclick=async()=>{if(state.job&&state.job.id){await api(`/api/jobs/${state.job.id}/stop`,{method:'POST'});toast('正在停止')}};
$('#providerButton').onclick=()=>{$('#providerModal').showModal();renderProviderList()};$('#usersButton').onclick=()=>{loadUsers();$('#usersModal').showModal()};
$('#mimoSettingsButton').onclick=openMimoSettings;$('#providerSelect').onchange=()=>{storeValue('active-provider',$('#providerSelect').value);updateProviderUi()};$('#providerType').onchange=syncProviderForm;$('#mimoThinking').onchange=syncMimoThinkingFields;
$$('.close-modal').forEach(b=>b.onclick=()=>b.closest('dialog').close());$$('dialog').forEach(d=>d.onclick=e=>{if(e.target===d)d.close()});
$('#testProvider').onclick=testProvider;$('#providerForm').onsubmit=async e=>{e.preventDefault();try{const body=providerFormData();await api('/api/providers',{method:'POST',body});e.target.reset();$('#providerType').value='deepseek';syncProviderForm();$('#providerKey').value='';$('#manualModel').value='';$('#providerStatus').textContent='';await loadProviders();toast('API 已保存')}catch(err){$('#providerStatus').textContent=err.message}};
$('#mimoForm').onsubmit=saveMimoSettings;
$('#userForm').onsubmit=async e=>{e.preventDefault();try{await api('/api/users',{method:'POST',body:{username:$('#newUsername').value,password:$('#newPassword').value,is_admin:$('#newAdmin').checked}});e.target.reset();await loadUsers();toast('账号已新增')}catch(err){toast(err.message)}};
$('#passwordForm').onsubmit=async e=>{e.preventDefault();const password=$('#changePassword').value;if(password!==$('#changePasswordAgain').value){$('#passwordError').textContent='两次输入的密码不一致';return}try{await api(`/api/users/${$('#passwordUserId').value}/password`,{method:'PUT',body:{password}});$('#passwordModal').close();toast('密码已修改')}catch(err){$('#passwordError').textContent=err.message}};
$('#openSidebar').onclick=openSidebar;$('#closeSidebar').onclick=closeSidebar;$('#scrim').onclick=closeSidebar;
$$('[data-prompt]').forEach(b=>b.onclick=()=>submitPrompt(b.dataset.prompt));
$('#chatScroll').onscroll=()=>{const s=$('#chatScroll');$('#jumpBottom').classList.toggle('hidden',s.scrollHeight-s.scrollTop-s.clientHeight<220)};$('#jumpBottom').onclick=()=>$('#chatScroll').scrollTo({top:$('#chatScroll').scrollHeight,behavior:'smooth'});
syncProviderForm();
boot();

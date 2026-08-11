const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const state = {me:null, providers:[], conversation:null, messages:[], job:null, page:1, pages:1, poll:null, latestConversationId:null, editingProviderId:null, pendingAttachments:[], attachmentDraftId:null, uploadingAttachments:false,retryingAnswer:false};
const detailState = new Map();
const nestedScrollState = new Map();
const MAX_ATTACHMENT_FILES = 10;
const MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024;

function rememberNestedScroll(root=document) {
  $$('[data-scroll-key]', root).forEach(node => nestedScrollState.set(node.dataset.scrollKey, {top:node.scrollTop, left:node.scrollLeft}));
}

function restoreNestedScroll(root=document) {
  $$('[data-scroll-key]', root).forEach(node => {
    const position = nestedScrollState.get(node.dataset.scrollKey);
    if (!position) return;
    node.scrollTop = position.top;
    node.scrollLeft = position.left;
  });
}

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
function browserTimezone(){try{return Intl.DateTimeFormat().resolvedOptions().timeZone||'UTC'}catch{return 'UTC'}}
function newAttachmentDraftId(){return globalThis.crypto&&globalThis.crypto.randomUUID?globalThis.crypto.randomUUID().replace(/-/g,''):`${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`.padEnd(32,'0').slice(0,32)}
function currentAttachmentDraftId(){
  if(state.conversation&&state.conversation.id)return state.conversation.id;
  if(state.attachmentDraftId)return state.attachmentDraftId;
  const saved=storedValue('attachment-draft');
  state.attachmentDraftId=/^[a-f0-9]{32}$/.test(saved||'')?saved:newAttachmentDraftId();
  storeValue('attachment-draft',state.attachmentDraftId);
  return state.attachmentDraftId;
}
function formatBytes(value){const bytes=Number(value)||0;if(bytes<1024)return `${bytes}B`;if(bytes<1024*1024)return `${(bytes/1024).toFixed(1)}KB`;return `${(bytes/1024/1024).toFixed(1)}MB`}

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

function setAttachmentStatus(message='', bad=false){const node=$('#attachmentStatus');node.textContent=message;node.classList.toggle('bad',bad)}
function attachmentIcon(item){return item&&item.kind==='image'?'▧':'▤'}
function attachmentChips(items=[], removable=false){
  return items.map(item=>`<span class="attachment-chip" title="${escapeHtml(item.name)}"><span aria-hidden="true">${attachmentIcon(item)}</span><span class="attachment-name">${escapeHtml(item.name)}</span><span>${escapeHtml(formatBytes(item.size))}</span>${removable?`<button class="attachment-remove" type="button" data-attachment-remove="${escapeHtml(item.id)}" title="移除附件" aria-label="移除附件">×</button>`:''}</span>`).join('');
}
function renderPendingAttachments(){
  const tray=$('#attachmentTray');tray.innerHTML=attachmentChips(state.pendingAttachments,true);
  $$('[data-attachment-remove]',tray).forEach(button=>button.onclick=async()=>{
    if(state.uploadingAttachments||state.job&&['queued','running'].includes(state.job.status))return;
    const id=button.dataset.attachmentRemove;
    try{await api(`/api/attachments/${id}`,{method:'DELETE'})}catch(err){toast(err.message);return}
    state.pendingAttachments=state.pendingAttachments.filter(item=>item.id!==id);renderPendingAttachments();setAttachmentStatus('');
  });
}
async function loadPendingAttachments(){
  const draftId=currentAttachmentDraftId();state.pendingAttachments=[];renderPendingAttachments();setAttachmentStatus('');
  try{const result=await api(`/api/attachments?draft_id=${encodeURIComponent(draftId)}`);state.pendingAttachments=result.data||[];renderPendingAttachments()}catch(err){setAttachmentStatus(err.message,true)}
}
async function discardPendingAttachments(){
  const items=[...state.pendingAttachments];state.pendingAttachments=[];renderPendingAttachments();setAttachmentStatus('');
  await Promise.allSettled(items.map(item=>api(`/api/attachments/${item.id}`,{method:'DELETE'})));
}
async function uploadOneAttachment(file){
  const query=new URLSearchParams({draft_id:currentAttachmentDraftId(),filename:file.name||'attachment'});
  const response=await fetch(`/api/attachments?${query}`,{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});
  const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.detail||`附件上传失败 (${response.status})`);return data;
}
async function selectAttachments(files){
  const list=[...files];if(!list.length)return;
  if(state.retryingAnswer){toast('正在重新回答');return}
  if(state.pendingAttachments.length+list.length>MAX_ATTACHMENT_FILES){setAttachmentStatus('一次消息最多上传 10 个附件。',true);return}
  const currentBytes=state.pendingAttachments.reduce((sum,item)=>sum+(Number(item.size)||0),0);
  const newBytes=list.reduce((sum,item)=>sum+(Number(item.size)||0),0);
  if(list.some(file=>file.size>MAX_ATTACHMENT_BYTES)||currentBytes+newBytes>MAX_ATTACHMENT_BYTES){setAttachmentStatus('一次消息的附件总量不能超过 50MB。',true);return}
  state.uploadingAttachments=true;setRunning(Boolean(state.job&&['queued','running'].includes(state.job.status)));
  try{
    for(let index=0;index<list.length;index++){
      setAttachmentStatus(`正在处理 ${index+1}/${list.length}：${list[index].name}`);
      const uploaded=await uploadOneAttachment(list[index]);state.pendingAttachments.push(uploaded);renderPendingAttachments();
    }
    setAttachmentStatus('附件已就绪：图片会压缩发送，文档只提取有限文字。');
  }catch(err){setAttachmentStatus(err.message,true)}finally{state.uploadingAttachments=false;setRunning(Boolean(state.job&&['queued','running'].includes(state.job.status)))}
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

function renderMathMarkup(tex, displayMode=false) {
  const source = String(tex || '').trim();
  if (!source) return '';
  if (!window.katex || typeof window.katex.renderToString !== 'function') {
    return `<code class="math-error">${escapeHtml(source)}</code>`;
  }
  try {
    return window.katex.renderToString(source, {
      displayMode,
      throwOnError: false,
      strict: false,
      trust: false,
      maxExpand: 1000,
      maxSize: 20,
      output: 'htmlAndMathml',
    });
  } catch {
    return `<code class="math-error">${escapeHtml(source)}</code>`;
  }
}

function inline(value='') {
  const codeTokens = [], mathTokens = [], linkTokens = [];
  let source = String(value).replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `\u0000CODE${codeTokens.length}\u0000`;
    codeTokens.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });
  const keepInlineMath = tex => {
    const token = `\u0000MATH${mathTokens.length}\u0000`;
    mathTokens.push(`<span class="math-inline">${renderMathMarkup(tex, false)}</span>`);
    return token;
  };
  source = source.replace(/\\\((.+?)\\\)/g, (_, tex) => keepInlineMath(tex));
  source = source.replace(/\$(?!\$)([^$\n]+?)\$/g, (match, tex) => (
    /^\s|\s$/.test(tex) ? match : keepInlineMath(tex)
  ));
  let text = escapeHtml(source);
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
  text = text.replace(/\u0000MATH(\d+)\u0000/g, (_, index) => mathTokens[Number(index)] || '');
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
  // Models occasionally emit two dashes for a narrow/blank header column.
  // Accept that common near-GFM form while still requiring every cell to be
  // made exclusively from an optional alignment marker and 2+ dashes.
  return cells.length > 0 && cells.every(cell => /^:?-{2,}:?$/.test(cell));
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
  return `<div class="code-wrap code-block" data-language="${escapeHtml(safeLanguage)}" data-extension="${escapeHtml(extension)}"><div class="code-label"><span>${label}</span><span class="code-actions"><button class="code-tool code-copy" type="button" title="复制代码" aria-label="复制代码"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"></rect><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"></path></svg></button><button class="code-tool code-download" type="button" title="下载代码" aria-label="下载代码"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v11m0 0 4-4m-4 4-4-4M5 18v2h14v-2"></path></svg></button></span></div><pre><code${languageClass}>${escapeHtml(String(code).replace(/\n$/, ''))}</code></pre></div>`;
}

function renderCallout(lines) {
  const info = calloutInfo(lines);
  if (!info) return `<blockquote>${lines.map(line => inline(line)).join('<br>')}</blockquote>`;
  const body = info.body.map(line => inline(line)).join('<br>');
  return `<aside class="md-callout md-callout-${info.type}"><div class="md-callout-title"><span aria-hidden="true">${info.icon}</span>${info.label}</div><div class="md-callout-body">${body || '<span class="muted"> </span>'}</div></aside>`;
}

function markdown(raw='', scrollKeyPrefix='message') {
  const lines = String(raw || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [], paragraph = [];
  let listType = '';
  let tableIndex = 0, mathIndex = 0;
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
    const mathOpen = trimmed.startsWith('$$') ? '$$' : trimmed.startsWith('\\[') ? '\\[' : '';
    if (mathOpen) {
      const mathClose = mathOpen === '$$' ? '$$' : '\\]';
      let end = -1;
      const first = trimmed.slice(mathOpen.length);
      if (first.endsWith(mathClose)) end = i;
      else {
        for (let cursor = i + 1; cursor < lines.length; cursor++) {
          if (lines[cursor].trim().endsWith(mathClose)) { end = cursor; break; }
        }
      }
      if (end >= 0) {
        flushParagraph(); closeList();
        const expressionLines = lines.slice(i, end + 1);
        expressionLines[0] = expressionLines[0].trim().slice(mathOpen.length);
        expressionLines[expressionLines.length - 1] = expressionLines[expressionLines.length - 1].trim().slice(0, -mathClose.length);
        const mathScrollKey = `${scrollKeyPrefix}-math-${mathIndex++}`;
        out.push(`<div class="math-display" data-scroll-key="${escapeHtml(mathScrollKey)}">${renderMathMarkup(expressionLines.join('\n'), true)}</div>`);
        i = end;
        continue;
      }
    }
    if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      flushParagraph(); closeList();
      const headers = splitTableRow(line), divider = splitTableRow(lines[i + 1]);
      const align = divider.map(cell => cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : cell.startsWith(':') ? 'left' : '');
      i += 2; const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(splitTableRow(lines[i])); i++; }
      i--;
      const tableScrollKey = `${scrollKeyPrefix}-table-${tableIndex++}`;
      let table = `<div class="table-wrap" data-scroll-key="${escapeHtml(tableScrollKey)}"><table><thead><tr>`;
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
  if (!reasoning && !searches.length && !active) return '';
  const status = active ? '进行中' : (meta.stopped ? '已停止' : '已完成');
  const searchHtml = searches.map((s,i) => {
    const label=s.action==='open_page'?'读取网页':'联网搜索';
    const searchKey=`${detailKey}-search-${s.id || i}`;
    const searchOpen=detailState.get(searchKey) ? ' open' : '';
    const detail=s.url?`<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.url)}</a>`:escapeHtml(Array.isArray(s.query)?s.query.filter(x=>!String(x).startsWith('ws_call_id=')).join('；'):(s.query||'DeepSeek 未返回查询词'));
    const error=s.error?`<div class="search-error">${escapeHtml(s.error)}</div>`:'';
    const statusLabels={running:'进行中',searching:'搜索中',completed:'已完成',failed:'失败',rejected:'已拒绝',skipped:'已跳过'};
    return `<details class="search-step" data-detail-key="${escapeHtml(searchKey)}"${searchOpen}><summary>${label} ${i+1} · ${escapeHtml(statusLabels[s.status] || s.status || '已完成')}</summary><div class="search-detail">${detail}${error}</div></details>`;
  }).join('');
  const traceOpen=detailState.get(detailKey) ? ' open' : '';
  const searchCount=searches.filter(item=>item.action!=='open_page').length;
  const readCount=searches.filter(item=>item.action==='open_page'&&item.status==='completed').length;
  const activity=`${searchCount ? ` · ${searchCount} 次搜索` : ''}${readCount ? ` · ${readCount} 次读取` : ''}`;
  return `<details class="trace" data-detail-key="${escapeHtml(detailKey)}"${traceOpen}><summary>思考与联网 · ${status}${activity}</summary><div class="trace-body">${reasoning ? `<div class="reasoning-text" data-scroll-key="${escapeHtml(detailKey)}-reasoning">${escapeHtml(reasoning)}</div>` : active ? '<div class="typing"><i></i><i></i><i></i></div>' : ''}${searchHtml}</div></details>`;
}

function sourcesHtml(meta={}, detailKey='trace') {
  const searches = meta.searches || [];
  const sources = [];
  const seenSources = new Set();
  [...(meta.sources || []), ...searches.filter(item => item.action === 'open_page' && item.status === 'completed' && item.url).map(item => ({url:item.url,title:item.url}))].forEach(source => {
    const url = String(source.url || '');
    if (!/^https?:\/\//i.test(url) || seenSources.has(url)) return;
    seenSources.add(url); sources.push(source);
  });
  if (!sources.length) return '';
  const sourceChips = sources.map(s => { const logo=s.logo_url&&safeUrl(s.logo_url)!=='#'?`<img src="${safeUrl(s.logo_url)}" alt="" loading="lazy">`:''; return `<a class="source-chip" href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.summary || s.url)}">${logo}<span>${escapeHtml(s.title || s.url)}</span></a>`; }).join('');
  const sourceKey = `${detailKey}-sources`;
  const sourceOpen = detailState.get(sourceKey) ? ' open' : '';
  return sources.length > 5
    ? `<details class="source-list" data-detail-key="${escapeHtml(sourceKey)}"${sourceOpen}><summary>来源 · ${sources.length} 条</summary><div class="sources">${sourceChips}</div></details>`
    : `<div class="sources"><span class="sources-label">来源</span>${sourceChips}</div>`;
}

function messageHtml(message, index, live=false, retryable=false) {
  const assistant = message.role === 'assistant';
  const meta = message.meta || {};
  const content = message.content || '';
  const detailKey = meta.trace_key || `trace-${meta.job_id || `message-${index}`}`;
  const messageAttachments = Array.isArray(meta.attachments) && meta.attachments.length ? `<div class="message-attachments">${attachmentChips(meta.attachments)}</div>` : '';
  const custom = normalizeProviderType(meta.provider_type)==='custom';
  const assistantName = custom ? 'Custom' : 'DeepSeek';
  return `<article class="message ${assistant ? 'assistant' : 'user'}${live ? ' live-message' : ''}" data-index="${index}">
    <div class="message-icon">${assistant ? (custom ? 'CU' : 'DS') : escapeHtml(((state.me && state.me.username) || 'U')[0].toUpperCase())}</div>
    <div class="message-body"><div class="message-head"><strong>${assistant ? assistantName : escapeHtml((state.me && state.me.username) || '你')}</strong></div>
    ${assistant ? traceHtml(meta, live, detailKey) : ''}${messageAttachments}<div class="message-content">${assistant ? (content ? markdown(content, detailKey) : live ? '<div class="typing"><i></i><i></i><i></i></div>' : '') : `<p>${escapeHtml(content).replace(/\n/g,'<br>')}</p>`}</div>${assistant ? sourcesHtml(meta,detailKey) : ''}${assistant && !live ? usageHtml(meta.usage || {}) : ''}<div class="message-actions"><button type="button" data-action="copy">复制</button>${retryable ? `<button type="button" data-action="retry" ${state.retryingAnswer ? 'disabled' : ''}>重新回答</button>` : ''}</div>
    ${meta.error ? `<p class="job-error">${escapeHtml(meta.error)}</p>` : ''}</div></article>`;
}

function renderMessages() {
  const scroll = $('#chatScroll'); const oldTop = scroll.scrollTop;
  $$('details[data-detail-key]').forEach(detail => detailState.set(detail.dataset.detailKey, detail.open));
  rememberNestedScroll($('#messages'));
  const items = [...state.messages];
  if (state.job && ['queued','running','failed','stopped'].includes(state.job.status)) {
    items.push({role:'assistant', content:state.job.answer || '', meta:{job_id:state.job.id,trace_key:state.job.trace_key,provider_id:state.job.provider_id,provider_type:state.job.provider_type,model:state.job.model,reasoning:state.job.reasoning,searches:state.job.searches,sources:state.job.sources,usage:state.job.usage,error:state.job.error,stopped:state.job.status==='stopped'}, live:['queued','running'].includes(state.job.status)});
  }
  const retryableIndex = !state.job && state.messages.length && state.messages[state.messages.length-1].role === 'assistant' ? state.messages.length-1 : -1;
  $('#welcome').classList.toggle('hidden', items.length > 0);
  $('#messages').innerHTML = items.map((m,i)=>messageHtml(m,i,!!m.live,i===retryableIndex)).join('');
  restoreNestedScroll($('#messages'));
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
      trace_key: job.trace_key,
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
  const chatScroll = $('#chatScroll');
  const oldTop = chatScroll.scrollTop;
  $$('details[data-detail-key]').forEach(detail => detailState.set(detail.dataset.detailKey, detail.open));
  rememberNestedScroll(current);
  const fragment = document.createRange().createContextualFragment(messageHtml(live, state.messages.length, liveState,!liveState));
  const replacement = fragment.firstElementChild;
  current.replaceWith(replacement);
  restoreNestedScroll(replacement);
  chatScroll.scrollTop = oldTop;
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
    const retryButton = $('[data-action="retry"]', node);
    if(retryButton)retryButton.onclick=retryLatestAnswer;
    $$('.code-download', node).forEach(button => button.onclick = () => {
      const wrap = button.closest('.code-wrap');
      const codeNode = $('pre code', wrap);
      const code = codeNode ? codeNode.textContent : '';
      const lang = wrap && wrap.dataset.extension ? wrap.dataset.extension : 'txt';
      const blob = new Blob([code], {type:'text/plain;charset=utf-8'}), a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download=`code.${lang === 'code' ? 'txt' : lang}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 0);
    });
    $$('.code-copy', node).forEach(button => button.onclick = async () => {
      const codeNode = $('pre code', button.closest('.code-wrap'));
      await navigator.clipboard.writeText(codeNode ? codeNode.textContent : '');
      toast('代码已复制');
    });
  });
}

async function retryLatestAnswer(){
  if(state.retryingAnswer||!state.conversation||!state.conversation.id)return;
  if(state.uploadingAttachments){toast('请等待附件处理完成');return}
  if(state.job&&['queued','running'].includes(state.job.status)){toast('请先停止当前回答');return}
  const last=state.messages[state.messages.length-1];
  if(!last||last.role!=='assistant')return;
  const provider=selectedProvider();if(!provider){$('#providerModal').showModal();toast('请先添加 API 配置');return}
  const model=selectedModel();if(!model){$('#providerModal').showModal();toast('请先选择模型');return}
  const conversationId=state.conversation.id;
  state.retryingAnswer=true;setRunning(false);renderMessages();
  try{
    const data=await api(`/api/conversations/${encodeURIComponent(conversationId)}/retry`,{method:'POST',body:{provider_id:provider.id,model,effort:$('#effort').value,timezone:browserTimezone()}});
    if(!state.conversation||String(state.conversation.id)!==String(conversationId)){
      toast('已在原对话中开始重新回答');loadHistory(1);return;
    }
    state.messages.pop();
    state.job={id:data.job_id,status:'queued',trace_key:`trace-${data.job_id}`,provider_type:provider.provider_type,provider_id:provider.id,model,answer:'',reasoning:'',searches:[],sources:[],usage:{}};
    state.retryingAnswer=false;renderMessages();startPolling(data.job_id);loadHistory(1);
  }catch(err){toast(err.message)}finally{
    if(state.retryingAnswer){state.retryingAnswer=false;setRunning(false);renderMessages()}
  }
}

async function loadHistory(page=1) {
  const data = await api(`/api/conversations?page=${page}`); state.page=data.page; state.pages=data.pages;
  if(page===1) state.latestConversationId=data.items[0] ? data.items[0].id : null;
  $('#historyCount').textContent=data.total;
  $('#clearHistory').disabled=!data.total;
  $('#history').innerHTML=data.items.map(c=>`<button class="history-item ${state.conversation&&state.conversation.id===c.id?'active':''}" data-id="${c.id}"><span>${escapeHtml(c.title)}</span><b class="history-delete" title="删除">×</b></button>`).join('') || '<p class="muted" style="padding:10px;font-size:12px">还没有对话</p>';
  $$('.history-item').forEach(button => {
    button.onclick=e=>{if(e.target.classList.contains('history-delete'))return;openConversation(button.dataset.id);closeSidebar()};
    $('.history-delete',button).onclick=async e=>{e.stopPropagation();if(state.retryingAnswer){toast('正在重新回答');return}if(!confirm('永久删除这个对话？'))return;try{await api(`/api/conversations/${button.dataset.id}`,{method:'DELETE'});if(state.conversation&&state.conversation.id===button.dataset.id)newConversation();await loadHistory(state.page)}catch(err){toast(err.message)}};
  });
  $('#pagination').innerHTML = state.pages>1 ? `<button id="prevPage" ${state.page<=1?'disabled':''}>‹</button><button>${state.page}/${state.pages}</button><button id="nextPage" ${state.page>=state.pages?'disabled':''}>›</button>` : '';
  if($('#prevPage'))$('#prevPage').onclick=()=>loadHistory(state.page-1); if($('#nextPage'))$('#nextPage').onclick=()=>loadHistory(state.page+1);
}

async function clearAllHistory() {
  if(state.retryingAnswer){toast('正在重新回答');return}
  if(state.job && ['queued','running'].includes(state.job.status)){toast('请先停止当前回答');return}
  if(!confirm('永久删除当前账号的全部聊天记录？\n\n对话、消息、思考、搜索记录和任务统计都会删除，并立即压缩数据库。此操作不可恢复。'))return;
  const button=$('#clearHistory');button.disabled=true;
  try{
    const result=await api('/api/conversations',{method:'DELETE'});
    stopPolling();state.conversation=null;state.messages=[];state.job=null;state.latestConversationId=null;state.page=1;state.pages=1;
    state.pendingAttachments=[];state.attachmentDraftId=newAttachmentDraftId();detailState.clear();nestedScrollState.clear();storeValue('active-conversation',null);storeValue('attachment-draft',state.attachmentDraftId);renderPendingAttachments();setAttachmentStatus('');
    $('#conversationTitle').textContent='新对话';renderMessages();await loadHistory(1);closeSidebar();
    toast(`已删除 ${result.deleted} 个对话并释放本地空间`);
  }catch(err){toast(err.message);await loadHistory(state.page)}
}

async function openConversation(id) {
  if(state.retryingAnswer){toast('正在重新回答');return false}
  stopPolling();
  try {
    if(!state.conversation&&state.pendingAttachments.length)await discardPendingAttachments();
    const data=await api(`/api/conversations/${id}`); state.conversation=data.conversation; state.messages=data.messages; state.job=data.active_job;
    state.attachmentDraftId=null;state.pendingAttachments=[];renderPendingAttachments();setAttachmentStatus('');
    restoreProviderForConversation(data);
    storeValue('active-conversation', state.conversation.id);
    $('#conversationTitle').textContent=state.conversation.title; renderMessages(); await Promise.all([loadHistory(state.page),loadPendingAttachments()]);
    if(state.job)startPolling(state.job.id);
    return true;
  } catch(err){
    if (err.message === '对话不存在') storeValue('active-conversation', null);
    toast(err.message);
  }
  return false;
}

async function newConversation(){
  if(state.retryingAnswer){toast('正在重新回答');return}
  if(!state.conversation&&state.pendingAttachments.length)await discardPendingAttachments();
  stopPolling();state.conversation=null;state.messages=[];state.job=null;state.pendingAttachments=[];state.attachmentDraftId=newAttachmentDraftId();
  storeValue('active-conversation','__new__');storeValue('attachment-draft',state.attachmentDraftId);$('#conversationTitle').textContent='新对话';renderPendingAttachments();setAttachmentStatus('');renderMessages();loadHistory(1);
}

async function submitPrompt(value) {
  const prompt=$('#prompt'),fromComposer=value==null,originalPrompt=prompt.value;
  const rawContent=(fromComposer ? originalPrompt : value).trim();
  if(!rawContent&&!state.pendingAttachments.length)return;
  if(state.retryingAnswer){toast('正在重新回答');return}
  if(state.uploadingAttachments){toast('请等待附件处理完成');return}
  const content=rawContent||'请分析这些附件。';
  const provider=selectedProvider(); if(!provider){$('#providerModal').showModal();toast('请先添加 API 配置');return}
  const model=selectedModel(); if(!model){$('#providerModal').showModal();toast('请先选择模型');return}
  if(state.job && ['queued','running'].includes(state.job.status)){toast('请先停止当前回答');return}
  const pendingSnapshot=state.pendingAttachments.map(item=>({...item}));
  const attachmentIds=pendingSnapshot.map(item=>item.id);
  if(fromComposer){prompt.value='';resizePrompt()}
  const traceKey=`trace-live-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
  state.messages.push({role:'user',content,meta:{attachments:pendingSnapshot}});state.job={status:'queued',trace_key:traceKey,provider_type:provider.provider_type,provider_id:provider.id,model,answer:'',reasoning:'',searches:[],sources:[],usage:{}};renderMessages();
  $('#chatScroll').scrollTop=$('#chatScroll').scrollHeight;setRunning(true);
  try{
    const data=await api('/api/chat',{method:'POST',body:{conversation_id:(state.conversation&&state.conversation.id)||null,content,attachment_ids:attachmentIds,provider_id:provider.id,model,effort:$('#effort').value,timezone:browserTimezone()}});
    if(!state.conversation)state.conversation={id:data.conversation_id,title:content.slice(0,36)};
    state.pendingAttachments=[];state.attachmentDraftId=null;storeValue('attachment-draft',null);renderPendingAttachments();setAttachmentStatus('');
    storeValue('active-conversation', state.conversation.id);
    state.job.id=data.job_id;$('#conversationTitle').textContent=state.conversation.title;loadHistory(1);startPolling(data.job_id);
  }catch(err){state.messages.pop();state.job=null;if(fromComposer){const newerDraft=prompt.value;prompt.value=newerDraft?`${originalPrompt}\n\n${newerDraft}`:originalPrompt;resizePrompt()}setRunning(false);renderMessages();toast(err.message)}
}

function setRunning(on){const locked=on||state.uploadingAttachments||state.retryingAnswer;$('#stopButton').classList.toggle('hidden',!on);$('#sendButton').disabled=locked;$('#attachButton').disabled=locked;$('#providerSelect').disabled=on||state.retryingAnswer}
function stopPolling(){if(state.poll)clearTimeout(state.poll);state.poll=null}
function startPolling(id){
  stopPolling();setRunning(true);
  const traceKey=(state.job&&state.job.trace_key)||`trace-${id}`,conversationId=state.conversation&&state.conversation.id;
  const current=()=>state.job&&String(state.job.id)===String(id)&&state.conversation&&String(state.conversation.id)===String(conversationId);
  const tick=async()=>{
    try{
      const job=await api(`/api/jobs/${id}`);if(!current())return;
      job.trace_key=traceKey;if(job.provider_id)selectProvider(job.provider_id,false,job.model);state.job=job;
      if(job.status==='completed'){stopPolling();setRunning(false);finalizeLiveMessage(job);return}
      if(['failed','stopped'].includes(job.status)){stopPolling();setRunning(false);renderMessages();return}
      updateLiveMessage();
    }catch(err){if(current()){toast(err.message);setRunning(false)}return}
    if(current())state.poll=setTimeout(tick,700);
  };
  tick();
}

function normalizeProviderType(value){return value==='mimo'?'custom':(value||'deepseek')}
function isMimoModel(value){return String(value||'').toLowerCase().startsWith('mimo-')}
function providerLabel(provider){return normalizeProviderType(provider.provider_type)==='custom'?'Custom':'DeepSeek'}
function providerModels(provider){const models=Array.isArray(provider&&provider.models)&&provider.models.length?provider.models:[provider&&provider.model];return [...new Set(models.filter(Boolean).map(String))]}
function selectedOption(){return $('#providerSelect').selectedOptions[0]||null}
function selectedProvider(){const option=selectedOption();const id=option&&option.dataset.providerId||$('#providerSelect').value;return state.providers.find(x=>String(x.id)===String(id))}
function selectedModel(){const option=selectedOption();const provider=selectedProvider();return option&&option.dataset.model||providerModels(provider)[0]||''}
function selectProvider(providerId, persist=true, model='') {
  const provider=state.providers.find(x=>String(x.id)===String(providerId));
  if(!provider) return null;
  const option=[...$('#providerSelect').options].find(item=>String(item.dataset.providerId)===String(provider.id)&&(!model||item.dataset.model===String(model)))||[...$('#providerSelect').options].find(item=>String(item.dataset.providerId)===String(provider.id));
  if(!option)return null;
  const changed=$('#providerSelect').value!==option.value;
  $('#providerSelect').value=option.value;
  if(persist){storeValue('active-provider', provider.id);storeValue('active-model', option.dataset.model||'')}
  if(changed) updateProviderUi();
  return provider;
}
function providerForConversation(data) {
  const job=data.active_job;
  if(job && job.provider_id) return {id:job.provider_id,model:job.model||''};
  for(const message of [...(data.messages || [])].reverse()) {
    const meta=message.meta || {};
    if(meta.provider_id) return {id:meta.provider_id,model:meta.model||''};
    if(meta.provider_type || meta.model) {
      const match=state.providers.find(provider =>
        (!meta.provider_type || normalizeProviderType(provider.provider_type)===normalizeProviderType(meta.provider_type)) &&
        (!meta.model || providerModels(provider).includes(meta.model))
      );
      if(match) return {id:match.id,model:meta.model||''};
    }
  }
  const stored=storedValue('active-provider');
  return stored?{id:stored,model:''}:null;
}
function restoreProviderForConversation(data) {
  const target=providerForConversation(data);
  if(target) selectProvider(target.id,true,target.model);
}
const customWebToolInfo={
  parallel:{label:'Parallel',description:'Parallel Search MCP 负责搜索和网页抓取。'},
  keenable:{label:'Keenable',description:'Keenable MCP 负责搜索，并以 live 模式抓取网页正文。'},
  tavily:{label:'Tavily',description:'Tavily Keyless 负责 Search 和 Extract，请求会发送 X-Tavily-Access-Mode: keyless。'},
  firecrawl:{label:'Firecrawl',description:'Firecrawl Keyless MCP 只使用 Search 和 Scrape。'},
  you:{label:'You.com + Jina',description:'You.com Free MCP 负责搜索，现有 Jina Reader 负责网页正文抓取。'},
  legacy:{label:'DDG + Jina',description:'本机 DuckDuckGo Lite/HTML 负责搜索，现有 Jina Reader 负责网页正文抓取。'}
};
function selectedWebToolInfo(provider=selectedProvider()){const key=customSettings(provider).web_tool_backend;return customWebToolInfo[key]||customWebToolInfo.parallel}
function updateProviderUi(){
  const provider=selectedProvider(), custom=provider&&normalizeProviderType(provider.provider_type)==='custom', customEffort=custom&&customSettings(provider).reasoning_effort_enabled;
  const webInfo=custom?selectedWebToolInfo(provider):null;
  $('#customSettingsButton').classList.toggle('hidden',!custom);
  $('#effort').disabled=!!custom&&!customEffort;
  $('#effort').title=custom?(customEffort?'控制发送给 Custom 模型的 reasoning_effort':'Custom 参数中已关闭 reasoning_effort'):'控制 DeepSeek 模型推理投入';
  $('#nativePill').textContent=custom?`● ${webInfo.label}`:'● Native Web';
  $('#welcomeOrbitMark').textContent=custom?'CU':'DS';
  $('#welcomeEyebrow').textContent=custom?'CUSTOM · OPENAI CHAT': 'DEEPSEEK V4 FLASH';
  $('#welcomeTitle').textContent=custom?'使用 Custom 本地联网':'问点需要查证的问题';
  $('#welcomeDescription').textContent=custom?`模型通过标准 Chat Completions 调用 ${webInfo.label} 搜索与读取真实来源。`:'模型会在 DeepSeek 服务端自行判断是否搜索，并在需要时多轮检索。';
}

async function loadProviders(){
  state.providers=await api('/api/providers');
  const options=state.providers.flatMap(p=>providerModels(p).map((model,index)=>`<option value="${p.id}:${index}" data-provider-id="${p.id}" data-model="${escapeHtml(model)}">${escapeHtml(p.name)} · ${providerLabel(p)} · ${escapeHtml(model)}</option>`));
  $('#providerSelect').innerHTML=options.length?options.join(''):'<option value="">请先添加 API</option>';
  const preferred=storedValue('active-provider');
  const preferredModel=storedValue('active-model')||'';
  if(state.providers.length) selectProvider(state.providers.some(p=>String(p.id)===String(preferred)) ? preferred : state.providers[0].id,true,preferredModel);
  renderProviderList(); updateProviderUi();
}
function renderProviderList(){
  $('#providerList').innerHTML=state.providers.map(p=>`<div class="list-item"><div class="list-item-main"><strong>${escapeHtml(p.name)} <span class="provider-kind">${providerLabel(p)}</span></strong><small>${escapeHtml(p.api_key_masked)} · ${escapeHtml(providerModels(p).join('、'))}</small></div><div class="item-actions"><button class="soft-btn" data-provider-edit="${p.id}">编辑模型</button><button class="danger-btn" data-provider-delete="${p.id}">删除</button></div></div>`).join('')||'<p class="muted">尚未添加 API。</p>';
  $$('[data-provider-edit]',$('#providerList')).forEach(b=>b.onclick=()=>editProviderModels(b.dataset.providerEdit));
  $$('[data-provider-delete]',$('#providerList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除这个 API 配置？'))return;try{await api(`/api/providers/${b.dataset.providerDelete}`,{method:'DELETE'});if(String(state.editingProviderId)===String(b.dataset.providerDelete))resetProviderEditor();await loadProviders()}catch(err){toast(err.message)}})
}

function providerType(){return $('#providerType').value||'deepseek'}
function manualModelList(){return $('#manualModel').value.split(/[\n,]+/).map(item=>item.trim()).filter(Boolean).filter((item,index,array)=>array.indexOf(item)===index)}
function renderCustomModels(models=[], selected=[]){
  const list=$('#customModelList'), selectedSet=new Set(selected), values=[...new Set((models||[]).map(String).filter(Boolean))];
  list.innerHTML=values.length?values.map(model=>`<label class="custom-model-option" data-model="${escapeHtml(model.toLowerCase())}"><input type="checkbox" value="${escapeHtml(model)}" ${selectedSet.has(model)?'checked':''}><span title="${escapeHtml(model)}">${escapeHtml(model)}</span></label>`).join(''):'<p class="custom-model-empty">测试 API 后显示可勾选模型。</p>';
  $('#customModelCount').textContent=values.length?`共 ${values.length} 个，可勾选保存`:'请先测试 API';
  $('#customModelsPanel').classList.toggle('hidden',providerType()!=='custom'||!values.length);
}
function syncProviderForm(){
  const custom=providerType()==='custom';
  const base=$('#providerBase'), model=$('#providerModel');
  base.value=custom?'https://api.openai.com/v1':'https://api.deepseek.com';
  model.innerHTML='<option value="deepseek-v4-flash">deepseek-v4-flash</option>';
  $('#providerModelField').classList.toggle('hidden',custom);
  $('#manualModelField').classList.toggle('hidden',!custom);
  $('#customModelsPanel').classList.toggle('hidden',!custom||!$('#customModelList input'));
  $('#providerName').placeholder=custom?'例如：我的 Custom API':'例如：我的 DeepSeek';
  if(!custom)renderCustomModels([],[]);
  $('#providerStatus').textContent='';
}
function resetProviderEditor(){
  state.editingProviderId=null;
  $('#providerForm').reset();
  $('#providerName').disabled=false;$('#providerType').disabled=false;$('#providerKey').disabled=false;$('#providerKey').required=true;$('#providerBase').disabled=false;
  $('#providerKey').placeholder='输入对应服务商 API Key';
  $('#providerModalTitle').textContent='模型 API';$('#providerSubmit').textContent='保存 API';$('#cancelProviderEdit').classList.add('hidden');
  $('#providerType').value='deepseek';syncProviderForm();$('#providerKey').value='';$('#manualModel').value='';renderCustomModels([],[]);$('#providerStatus').textContent='';
}
function editProviderModels(providerId){
  const provider=state.providers.find(item=>String(item.id)===String(providerId));if(!provider)return;
  resetProviderEditor();state.editingProviderId=provider.id;
  $('#providerType').value=normalizeProviderType(provider.provider_type);syncProviderForm();
  $('#providerName').value=provider.name;$('#providerBase').value=provider.base_url;$('#providerKey').value='';
  $('#providerName').disabled=true;$('#providerType').disabled=true;$('#providerKey').disabled=true;$('#providerKey').required=false;$('#providerBase').disabled=true;
  $('#providerKey').placeholder=`沿用已保存密钥 ${provider.api_key_masked}`;
  $('#providerModalTitle').textContent='编辑 API 模型';$('#providerSubmit').textContent='保存模型';$('#cancelProviderEdit').classList.remove('hidden');
  if(normalizeProviderType(provider.provider_type)==='custom'){
    const models=providerModels(provider);renderCustomModels(models,models);$('#manualModel').value='';
  }else{$('#providerModel').value=provider.model||'deepseek-v4-flash'}
  $('#providerStatus').textContent='连接信息和密钥保持不变；可重新测试模型列表、调整勾选或补充手填模型。';
  $('#providerForm').scrollIntoView({behavior:'smooth',block:'start'});
}
async function testProvider(){
  const button=$('#testProvider');button.disabled=true;$('#providerStatus').textContent=`正在连接 ${providerType()==='custom'?'Custom':'DeepSeek'}…`;
  try{
    const editing=state.providers.find(item=>String(item.id)===String(state.editingProviderId));
    const before=checkedCustomModels();
    const body=providerFormData();
    const result=editing
      ? await api(`/api/providers/${editing.id}/test`,{method:'POST',body:{manual_models:manualModelList()}})
      : await api('/api/providers/test',{method:'POST',body});
    if(providerType()==='custom'){
      const selected=[...new Set([...before,...manualModelList()])];
      const available=[...new Set([...(result.models||[]),...(editing?providerModels(editing):[]),...manualModelList()])];
      renderCustomModels(available,selected.length?selected:(editing?providerModels(editing):available.slice(0,1)));
      $('#providerStatus').textContent=`连接成功，读取 ${(result.models||[]).length} 个模型${(result.manual_tested||[]).length?`，手填模型验证 ${result.manual_tested.length} 个`:''}${result.models_warning?'（/models 不可用，已使用手填模型验证）':''}。请勾选后保存。`;
    }else{
      $('#providerModel').value='deepseek-v4-flash';
      $('#providerStatus').textContent='连接成功，可用模型：deepseek-v4-flash';
    }
  }catch(err){$('#providerStatus').textContent=err.message}finally{button.disabled=false}
}
function checkedCustomModels(){return $$('#customModelList input[type="checkbox"]:checked').map(input=>input.value)}
function providerFormData(){const custom=providerType()==='custom';const manual=custom?manualModelList():[];const selected=custom?[...new Set([...checkedCustomModels(),...manual])]:['deepseek-v4-flash'];return{name:$('#providerName').value||providerLabel({provider_type:providerType()}),api_key:$('#providerKey').value,provider_type:providerType(),base_url:$('#providerBase').value,model:selected[0]||$('#providerModel').value||'deepseek-v4-flash',selected_models:selected,manual_models:manual}}

function customSettings(provider){return {...{thinking:'enabled',reasoning_effort_enabled:true,include_reasoning_enabled:false,dsml_fallback_enabled:false,max_completion_tokens:65536,temperature:1,top_p:.95,web_tool_backend:'parallel'},...(provider&&provider.settings||{})}}
function fillCustomSettings(){
  const config=customSettings(selectedProvider());
  $('#customThinking').value=config.thinking;$('#customReasoningEffortEnabled').checked=config.reasoning_effort_enabled;$('#customIncludeReasoningEnabled').checked=config.include_reasoning_enabled;$('#customDsmlFallbackEnabled').checked=config.dsml_fallback_enabled;$('#customMaxCompletion').value=config.max_completion_tokens;$('#customTemperature').value=config.temperature;$('#customTopP').value=config.top_p;$('#customWebToolBackend').value=config.web_tool_backend;
  syncCustomThinkingFields();syncCustomToolFields();
}
function syncCustomThinkingFields(){const mimo=isMimoModel(selectedModel()),thinking=$('#customThinking').value==='enabled',effort=$('#customReasoningEffortEnabled').checked,includeReasoning=$('#customIncludeReasoningEnabled').checked;$('#customTemperature').disabled=mimo&&thinking;$('#customTopP').disabled=mimo&&thinking;$('#customSamplingNote').textContent=`thinking 将${thinking?'开启':'关闭'}；reasoning_effort 将${effort?'按顶部 High/Max 发送':'不发送'}；include_reasoning 将${includeReasoning?'发送 true':'不发送'}。已知模型使用官方字段，其他 Custom 使用通用顶层字段；接口不支持时可在这里关闭。`}
function syncCustomToolFields(){const info=customWebToolInfo[$('#customWebToolBackend').value]||customWebToolInfo.parallel;$('#customToolNote').textContent=`${info.description} 不需要 API Key；不会自动切换、并发调用或回退到其他方案。`}
function openCustomSettings(){if(!selectedProvider()||normalizeProviderType(selectedProvider().provider_type)!=='custom'){toast('请先选择 Custom API');return}fillCustomSettings();$('#customModal').showModal()}
async function saveCustomSettings(event){
  event.preventDefault(); const provider=selectedProvider(); if(!provider)return;
  const body={thinking:$('#customThinking').value,reasoning_effort_enabled:$('#customReasoningEffortEnabled').checked,include_reasoning_enabled:$('#customIncludeReasoningEnabled').checked,dsml_fallback_enabled:$('#customDsmlFallbackEnabled').checked,max_completion_tokens:Number($('#customMaxCompletion').value),temperature:Number($('#customTemperature').value),top_p:Number($('#customTopP').value),web_tool_backend:$('#customWebToolBackend').value};
  try{const updated=await api(`/api/providers/${provider.id}/settings`,{method:'PUT',body});const index=state.providers.findIndex(x=>x.id===provider.id);if(index>=0)state.providers[index]=updated;$('#customModal').close();updateProviderUi();toast('Custom 参数已保存')}catch(err){$('#customStatus').textContent=err.message}
}

async function loadUsers(){const users=await api('/api/users');$('#userList').innerHTML=users.map(u=>`<div class="list-item"><span class="avatar">${escapeHtml(u.username[0].toUpperCase())}</span><div class="list-item-main"><strong>${escapeHtml(u.username)}</strong><small>${u.is_admin?'管理员':'普通账号'}</small></div><div class="item-actions"><button class="soft-btn" data-password-user="${u.id}" data-username="${escapeHtml(u.username)}">改密码</button>${u.id!==state.me.id?`<button class="danger-btn" data-user="${u.id}">删除</button>`:''}</div></div>`).join('');$$('[data-user]',$('#userList')).forEach(b=>b.onclick=async()=>{if(!confirm('删除账号及其全部独立数据？'))return;try{await api(`/api/users/${b.dataset.user}`,{method:'DELETE'});loadUsers()}catch(err){toast(err.message)}});$$('[data-password-user]',$('#userList')).forEach(b=>b.onclick=()=>{$('#passwordUserId').value=b.dataset.passwordUser;$('#passwordTarget').textContent=`正在修改：${b.dataset.username}`;$('#changePassword').value='';$('#changePasswordAgain').value='';$('#passwordError').textContent='';$('#passwordModal').showModal()})}

function resizePrompt(){const p=$('#prompt');p.style.height='auto';p.style.height=Math.min(p.scrollHeight,180)+'px'}
function openSidebar(){$('#sidebar').classList.add('open')}function closeSidebar(){$('#sidebar').classList.remove('open')}

async function boot(){try{state.me=await api('/api/me');$('#loginView').classList.add('hidden');$('#appView').classList.remove('hidden');$('#accountName').textContent=state.me.username;$('#accountRole').textContent=state.me.is_admin?'管理员':'用户';$('#avatar').textContent=state.me.username[0].toUpperCase();$('#usersButton').classList.toggle('hidden',!state.me.is_admin);await Promise.all([loadProviders(),loadHistory(1)]);const stored=storedValue('active-conversation');const activeId=stored==='__new__'?null:(stored||state.latestConversationId);const restored=activeId?await openConversation(activeId):false;if(!restored){renderMessages();await loadPendingAttachments()}}catch{$('#loginView').classList.remove('hidden');$('#appView').classList.add('hidden')}}

$('#loginForm').onsubmit=async e=>{e.preventDefault();$('#loginError').textContent='';try{await api('/api/login',{method:'POST',body:{username:$('#loginUser').value,password:$('#loginPass').value}});await boot()}catch(err){$('#loginError').textContent=err.message}};
$('#logout').onclick=async()=>{await api('/api/logout',{method:'POST'});location.reload()};
$('#newChat').onclick=()=>{newConversation();closeSidebar()};$('#composer').onsubmit=e=>{e.preventDefault();submitPrompt()};
$('#clearHistory').onclick=clearAllHistory;
// Enter always inserts a newline. Sending is explicit via the send button.
$('#prompt').oninput=resizePrompt;
$('#attachButton').onclick=()=>{if(!state.uploadingAttachments&&!state.retryingAnswer&&!(state.job&&['queued','running'].includes(state.job.status)))$('#fileInput').click()};
$('#fileInput').onchange=async event=>{const files=[...(event.target.files||[])];event.target.value='';await selectAttachments(files)};
$('#stopButton').onclick=async()=>{if(state.job&&state.job.id){await api(`/api/jobs/${state.job.id}/stop`,{method:'POST'});toast('正在停止')}};
$('#providerButton').onclick=()=>{resetProviderEditor();$('#providerModal').showModal();renderProviderList()};$('#usersButton').onclick=()=>{loadUsers();$('#usersModal').showModal()};
$('#customSettingsButton').onclick=openCustomSettings;$('#providerSelect').onchange=()=>{const provider=selectedProvider();if(provider){storeValue('active-provider',provider.id);storeValue('active-model',selectedModel())}updateProviderUi()};$('#providerType').onchange=syncProviderForm;$('#customThinking').onchange=syncCustomThinkingFields;$('#customReasoningEffortEnabled').onchange=syncCustomThinkingFields;$('#customIncludeReasoningEnabled').onchange=syncCustomThinkingFields;$('#customWebToolBackend').onchange=syncCustomToolFields;$('#customModelFilter').oninput=e=>{const value=e.target.value.trim().toLowerCase();$$('.custom-model-option',$('#customModelList')).forEach(item=>item.dataset.hidden=value&&!item.dataset.model.includes(value)?'1':'0')};
$$('.close-modal').forEach(b=>b.onclick=()=>b.closest('dialog').close());$$('dialog').forEach(d=>d.onclick=e=>{if(e.target===d)d.close()});
$('#testProvider').onclick=testProvider;$('#cancelProviderEdit').onclick=resetProviderEditor;$('#providerForm').onsubmit=async e=>{e.preventDefault();try{const body=providerFormData();if(state.editingProviderId){await api(`/api/providers/${state.editingProviderId}/models`,{method:'PUT',body:{model:body.model,selected_models:body.selected_models,manual_models:body.manual_models}});resetProviderEditor();await loadProviders();toast('模型列表已更新')}else{await api('/api/providers',{method:'POST',body});resetProviderEditor();await loadProviders();toast('API 已保存')}}catch(err){$('#providerStatus').textContent=err.message}};
$('#customForm').onsubmit=saveCustomSettings;
$('#userForm').onsubmit=async e=>{e.preventDefault();try{await api('/api/users',{method:'POST',body:{username:$('#newUsername').value,password:$('#newPassword').value,is_admin:$('#newAdmin').checked}});e.target.reset();await loadUsers();toast('账号已新增')}catch(err){toast(err.message)}};
$('#passwordForm').onsubmit=async e=>{e.preventDefault();const password=$('#changePassword').value;if(password!==$('#changePasswordAgain').value){$('#passwordError').textContent='两次输入的密码不一致';return}try{await api(`/api/users/${$('#passwordUserId').value}/password`,{method:'PUT',body:{password}});$('#passwordModal').close();toast('密码已修改')}catch(err){$('#passwordError').textContent=err.message}};
$('#openSidebar').onclick=openSidebar;$('#closeSidebar').onclick=closeSidebar;$('#scrim').onclick=closeSidebar;
$$('[data-prompt]').forEach(b=>b.onclick=()=>submitPrompt(b.dataset.prompt));
$('#chatScroll').onscroll=()=>{const s=$('#chatScroll');$('#jumpBottom').classList.toggle('hidden',s.scrollHeight-s.scrollTop-s.clientHeight<220)};$('#jumpBottom').onclick=()=>$('#chatScroll').scrollTo({top:$('#chatScroll').scrollHeight,behavior:'smooth'});
syncProviderForm();
boot();

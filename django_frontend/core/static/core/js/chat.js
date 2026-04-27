// FinanceHQ — Chat tab
// FASTAPI_URL is injected by scripts.html as a global const

const Chat = (() => {

  // ── State ────────────────────────────────────────────────────────
  let sessionId   = null;
  let pollTimer   = null;
  let isStreaming = false;

  const POLL_INTERVAL_MS = 3000;

  // ── DOM helpers ──────────────────────────────────────────────────
  const $  = id => document.getElementById(id);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls)  e.className = cls;
    if (html) e.innerHTML = html;
    return e;
  };

  // ── Session header (panel-header in left column) ─────────────────
  function showSessionBar(id) {
    $('session-bar').classList.remove('hidden');
    $('session-id-display').classList.remove('hidden');
    $('session-id-display').textContent = id.slice(0, 8) + '…';
  }

  // Maps raw pipeline status → { dotClass, label } for the header dot+label
  const STATUS_HEADER = {
    READY:      { dot: 'bg-green-400',            label: 'Ready'         },
    FAILED:     { dot: 'bg-red-400',              label: 'Failed'        },
    PROCESSING: { dot: 'bg-yellow-400 pulse-dot', label: 'Processing…'  },
    EXTRACTING: { dot: 'bg-yellow-400 pulse-dot', label: 'Extracting…'  },
    CLEANING:   { dot: 'bg-yellow-400 pulse-dot', label: 'Cleaning…'    },
    CHUNKING:   { dot: 'bg-yellow-400 pulse-dot', label: 'Chunking…'    },
    INDEXING:   { dot: 'bg-yellow-400 pulse-dot', label: 'Indexing…'    },
  };

  function setSessionStatus(status) {
    const s     = STATUS_HEADER[status] || { dot: 'bg-gray-400 pulse-dot', label: 'Waiting…' };
    const dot   = $('session-status-dot');
    const label = $('session-status-label');
    dot.className   = 'w-2 h-2 rounded-full inline-block flex-shrink-0 ' + s.dot;
    label.textContent = s.label;
  }

  // ── Drop zone ────────────────────────────────────────────────────
  function onDragOver(e) {
    e.preventDefault();
    $('drop-zone').style.borderColor = 'var(--cyan)';
    $('drop-zone').style.background  = 'var(--cyan-dim)';
  }

  function onDragLeave() {
    $('drop-zone').style.borderColor = 'var(--border)';
    $('drop-zone').style.background  = 'var(--bg-card)';
  }

  function onDrop(e) {
    e.preventDefault();
    onDragLeave();
    const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    if (files.length) handleFiles(files);
  }

  function onFileSelect(e) {
    const files = Array.from(e.target.files);
    if (files.length) handleFiles(files);
    e.target.value = '';
  }

  // ── File handling ────────────────────────────────────────────────
  async function handleFiles(files) {
    if (!sessionId) {
      sessionId = await createSession();
      if (!sessionId) return;
      showSessionBar(sessionId);
      setSessionStatus('PROCESSING');
      showPipelineProgress();
    }

    for (const file of files) {
      await uploadFile(file);
    }

    startPolling();
  }

  async function createSession() {
    try {
      const res  = await fetch(`${FASTAPI_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'Demo Session' }),
      });
      const data = await res.json();
      return data.session_id;
    } catch (err) {
      showError('Could not reach the API. Is FastAPI running?');
      return null;
    }
  }

  async function uploadFile(file) {
    const docCard = addDocCard(file.name, 'UPLOADING');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res  = await fetch(`${FASTAPI_URL}/sessions/${sessionId}/documents`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        updateDocCard(docCard, 'FAILED', data.detail || 'Upload failed');
        return;
      }
      docCard.dataset.docId = data.doc_id;
      updateDocCard(docCard, 'PENDING');
    } catch (err) {
      updateDocCard(docCard, 'FAILED', 'Network error');
    }
  }

  // ── Doc cards ────────────────────────────────────────────────────
  function addDocCard(filename, status) {
    const card = el('div', 'doc-card rounded-xl px-4 py-3 flex items-center gap-3');
    card.style.cssText = 'background:var(--bg-card2); border:1px solid var(--border);';

    const icon = el('div', 'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 cyan-dim');
    icon.innerHTML = '<i class="fas fa-file-pdf cyan-text text-sm"></i>';

    const info = el('div', 'flex-1 min-w-0');
    info.innerHTML = `
      <p class="text-sm font-medium text-white truncate">${escHtml(filename)}</p>
      <p class="doc-status text-xs mt-0.5"></p>`;

    card.append(icon, info);
    $('doc-list').appendChild(card);
    updateDocCard(card, status);
    return card;
  }

  // Feedback Density: specific stage labels instead of generic "Extracting…"
  const STATUS_STYLE = {
    UPLOADING:  { color: 'var(--cyan)',   label: 'Uploading to S3…'            },
    PENDING:    { color: 'var(--muted2)', label: 'Queued'                       },
    EXTRACTING: { color: '#FBBF24',       label: 'Running AWS Textract…'        },
    CLEANING:   { color: '#FBBF24',       label: 'Cleaning text…'               },
    CHUNKING:   { color: '#FBBF24',       label: 'Splitting into chunks…'       },
    INDEXING:   { color: '#FBBF24',       label: 'Building FAISS index…'        },
    READY:      { color: '#34D399',       label: 'Ready'                        },
    FAILED:     { color: '#F87171',       label: 'Failed'                       },
  };

  function updateDocCard(card, status, errorMsg) {
    const s = STATUS_STYLE[status] || STATUS_STYLE.PENDING;
    const p = card.querySelector('.doc-status');
    p.style.color = s.color;
    p.textContent = errorMsg ? `Failed: ${errorMsg}` : s.label;
    card.style.borderColor =
      status === 'READY'  ? 'rgba(52,211,153,0.3)'  :
      status === 'FAILED' ? 'rgba(248,113,113,0.3)' :
      'var(--border)';
  }

  // ── Pipeline progress block (Feedback Density) ───────────────────
  // Shown in chat area while documents are being processed
  const PIPELINE_STEPS = [
    { key: 'UPLOADING',  label: 'Upload to S3'           },
    { key: 'EXTRACTING', label: 'AWS Textract OCR'        },
    { key: 'CHUNKING',   label: 'Chunk & embed text'      },
    { key: 'INDEXING',   label: 'Build FAISS index'       },
  ];

  const STEP_ORDER = ['UPLOADING', 'EXTRACTING', 'CHUNKING', 'INDEXING', 'READY'];

  function showPipelineProgress() {
    $('empty-state')?.remove();
    const thread = $('chat-messages');

    const block = el('div', 'flex flex-col items-center justify-center h-full gap-6 py-12');
    block.id = 'pipeline-progress';

    const stepsHtml = PIPELINE_STEPS.map((s, i) => `
      <div class="pipeline-step flex items-center gap-3 text-xs" data-key="${s.key}">
        <span class="step-circle w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold transition-all"
              style="background:var(--bg-card2); color:var(--muted); border:1px solid var(--border);">${i + 1}</span>
        <span class="step-label transition-colors" style="color:var(--muted2);">${s.label}</span>
      </div>`).join('');

    block.innerHTML = `
      <div class="text-center">
        <div class="w-12 h-12 rounded-2xl cyan-dim flex items-center justify-center mx-auto mb-4">
          <i class="fas fa-cog fa-spin cyan-text text-xl"></i>
        </div>
        <p class="font-semibold text-white mb-1">Processing document</p>
        <p id="pipeline-stage-label" class="text-xs" style="color:var(--muted);">Starting pipeline…</p>
      </div>
      <div class="flex flex-col gap-3 w-full max-w-xs">${stepsHtml}</div>`;

    thread.appendChild(block);
    updatePipelineProgress('UPLOADING');
  }

  function updatePipelineProgress(currentStatus) {
    const block = $('pipeline-progress');
    if (!block) return;

    const stageLabel = $('pipeline-stage-label');
    if (stageLabel && STATUS_STYLE[currentStatus]) {
      stageLabel.textContent = STATUS_STYLE[currentStatus].label;
    }

    const currentIdx = STEP_ORDER.indexOf(currentStatus);

    block.querySelectorAll('.pipeline-step').forEach(step => {
      const stepIdx  = STEP_ORDER.indexOf(step.dataset.key);
      const circle   = step.querySelector('.step-circle');
      const label    = step.querySelector('.step-label');

      if (stepIdx < currentIdx) {
        // Completed
        circle.style.background   = 'rgba(52,211,153,0.15)';
        circle.style.color        = '#34D399';
        circle.style.borderColor  = 'rgba(52,211,153,0.3)';
        circle.innerHTML          = '<i class="fas fa-check text-xs"></i>';
        label.style.color         = '#34D399';
      } else if (stepIdx === currentIdx) {
        // Active
        circle.style.background   = 'var(--cyan-dim)';
        circle.style.color        = 'var(--cyan)';
        circle.style.borderColor  = 'var(--cyan-glow)';
        circle.textContent        = String(stepIdx + 1);
        label.style.color         = 'var(--cyan)';
      } else {
        // Pending
        circle.style.background   = 'var(--bg-card2)';
        circle.style.color        = 'var(--muted)';
        circle.style.borderColor  = 'var(--border)';
        circle.textContent        = String(stepIdx + 1);
        label.style.color         = 'var(--muted2)';
      }
    });
  }

  // ── Polling ──────────────────────────────────────────────────────
  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, POLL_INTERVAL_MS);
  }

  async function pollStatus() {
    if (!sessionId) return;
    try {
      const res  = await fetch(`${FASTAPI_URL}/sessions/${sessionId}/status`);
      const data = await res.json();

      setSessionStatus(data.status);
      updatePipelineProgress(data.status);

      (data.documents || []).forEach(doc => {
        $('doc-list').querySelectorAll('.doc-card').forEach(card => {
          if (card.dataset.docId === doc.doc_id) {
            updateDocCard(card, doc.status, doc.error);
          }
        });
      });

      if (data.status === 'READY') {
        clearInterval(pollTimer);
        pollTimer = null;
        unlockInput();
      } else if (data.status === 'FAILED') {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (_) { /* network hiccup — keep polling */ }
  }

  // ── Input ────────────────────────────────────────────────────────
  // Progressive Disclosure: reveal tips + ready hint only when session is READY
  function unlockInput() {
    $('pipeline-progress')?.remove();

    // Reveal ready-hint above input bar
    const hint = $('ready-hint');
    if (hint) hint.classList.remove('hidden');

    // Reveal tips panel in left column
    const tips = $('tips-panel');
    if (tips) tips.classList.remove('hidden');

    const input = $('question-input');
    const btn   = $('send-btn');
    input.disabled    = false;
    input.placeholder = 'Ask anything about your document…';
    btn.disabled      = false;
    btn.style.background = 'var(--cyan)';
    btn.style.color      = '#0A0F1E';
  }

  function fillQuestion(text) {
    const input = $('question-input');
    input.value = text;
    input.focus();
    autoResize(input);
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ── Send & Stream ─────────────────────────────────────────────────
  async function sendMessage() {
    if (isStreaming) return;
    const input    = $('question-input');
    const question = input.value.trim();
    if (!question || !sessionId) return;

    input.value = '';
    autoResize(input);

    appendUserBubble(question);
    const assistantBubble = appendAssistantBubble();

    isStreaming = true;
    setInputLock(true);

    try {
      const res = await fetch(`${FASTAPI_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, question, top_k: 5 }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        finishAssistantBubble(assistantBubble, null, err.detail || `Error ${res.status}`);
        return;
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = '';
      let   answer  = '';
      let   meta    = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = JSON.parse(line.slice(6));

          if (payload.done) {
            meta = payload;
          } else if (payload.token !== undefined) {
            answer += payload.token;
            streamToken(assistantBubble, payload.token);
          } else if (payload.error) {
            finishAssistantBubble(assistantBubble, null, payload.error);
            return;
          }
        }
      }

      finishAssistantBubble(assistantBubble, meta, null);

    } catch (err) {
      finishAssistantBubble(assistantBubble, null, 'Network error — is FastAPI running?');
    } finally {
      isStreaming = false;
      setInputLock(false);
    }
  }

  function setInputLock(locked) {
    const input = $('question-input');
    const btn   = $('send-btn');
    input.disabled = locked;
    btn.disabled   = locked;
    if (locked) {
      btn.style.background = 'var(--border)';
      btn.style.color      = 'var(--muted)';
    } else {
      btn.style.background = 'var(--cyan)';
      btn.style.color      = '#0A0F1E';
    }
  }

  // ── Message bubbles ──────────────────────────────────────────────
  function appendUserBubble(text) {
    const thread = $('chat-messages');
    const wrap   = el('div', 'flex justify-end');
    const bubble = el('div', 'max-w-lg px-4 py-3 rounded-2xl text-sm leading-relaxed rounded-br-sm');
    bubble.style.cssText = 'background:var(--cyan-dim); border:1px solid var(--cyan-glow); color:var(--text);';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    thread.appendChild(wrap);
    scrollToBottom();
  }

  function appendAssistantBubble() {
    const thread = $('chat-messages');
    const wrap   = el('div', 'flex flex-col gap-2');

    const bubble = el('div', 'max-w-2xl px-4 py-3 rounded-2xl text-sm leading-relaxed rounded-bl-sm assistant-bubble');
    bubble.style.cssText = 'background:var(--bg-card2); border:1px solid var(--border); color:var(--text); white-space:pre-wrap;';

    const cursor = el('span', 'typing-cursor inline-block w-2 h-4 ml-0.5 cyan-bg rounded-sm');
    cursor.style.animation = 'pulseDot 0.8s infinite';
    bubble.appendChild(cursor);

    wrap.appendChild(bubble);
    thread.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function streamToken(wrap, token) {
    const bubble = wrap.querySelector('.assistant-bubble');
    const cursor = bubble.querySelector('.typing-cursor');
    bubble.insertBefore(document.createTextNode(token), cursor);
    scrollToBottom();
  }

  function finishAssistantBubble(wrap, meta, errorMsg) {
    const bubble = wrap.querySelector('.assistant-bubble');
    bubble.querySelector('.typing-cursor')?.remove();

    if (errorMsg) {
      bubble.style.borderColor = 'rgba(248,113,113,0.4)';
      bubble.style.color       = '#F87171';
      bubble.textContent       = '⚠ ' + errorMsg;
      return;
    }

    if (meta) {
      const chip = el('div', 'flex flex-wrap gap-2 text-xs');
      chip.innerHTML = `
        <span class="px-2 py-1 rounded-md" style="background:var(--bg-card2); color:var(--cyan); border:1px solid var(--border);">
          <i class="fas fa-tag mr-1"></i>${escHtml(meta.intent)}
        </span>
        <span class="px-2 py-1 rounded-md" style="background:var(--bg-card2); color:var(--muted2); border:1px solid var(--border);">
          <i class="fas fa-code-branch mr-1"></i>${escHtml(meta.variant)}
        </span>
        <span class="px-2 py-1 rounded-md" style="background:var(--bg-card2); color:var(--muted2); border:1px solid var(--border);">
          <i class="fas fa-clock mr-1"></i>${meta.latency_ms.toFixed(0)}ms
        </span>
        <span class="px-2 py-1 rounded-md" style="background:var(--bg-card2); color:var(--muted2); border:1px solid var(--border);">
          <i class="fas fa-layer-group mr-1"></i>${meta.chunks_used} chunks
        </span>`;
      wrap.appendChild(chip);
    }

    scrollToBottom();
  }

  function scrollToBottom() {
    const thread = $('chat-messages');
    thread.scrollTop = thread.scrollHeight;
  }

  // ── Reset ────────────────────────────────────────────────────────
  function resetSession() {
    if (pollTimer) clearInterval(pollTimer);
    sessionId   = null;
    pollTimer   = null;
    isStreaming = false;

    // Hide progressive-disclosure elements
    $('session-bar').classList.add('hidden');
    $('session-id-display').classList.add('hidden');
    $('tips-panel').classList.add('hidden');
    const hint = $('ready-hint');
    if (hint) hint.classList.add('hidden');

    $('doc-list').innerHTML = '';

    // Restore empty state with 3-step guide (matching chat.html)
    $('chat-messages').innerHTML = `
      <div id="empty-state" class="flex flex-col items-center justify-center h-full gap-8 py-12">
        <div class="text-center">
          <div class="w-14 h-14 rounded-2xl cyan-dim flex items-center justify-center mx-auto mb-4">
            <i class="fas fa-file-invoice-dollar cyan-text text-xl"></i>
          </div>
          <p class="font-semibold text-white mb-1">No document loaded</p>
          <p class="text-xs" style="color:var(--muted);">Upload a PDF loan document on the left to get started.</p>
        </div>
        <div class="flex items-center gap-3 text-xs" style="color:var(--muted2);">
          <div class="flex items-center gap-2">
            <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                  style="background:var(--cyan-dim); color:var(--cyan); border:1px solid var(--cyan-glow);">1</span>
            Upload PDF
          </div>
          <i class="fas fa-chevron-right" style="color:var(--border);"></i>
          <div class="flex items-center gap-2">
            <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                  style="background:var(--bg-card2); color:var(--muted); border:1px solid var(--border);">2</span>
            Processing
          </div>
          <i class="fas fa-chevron-right" style="color:var(--border);"></i>
          <div class="flex items-center gap-2">
            <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                  style="background:var(--bg-card2); color:var(--muted); border:1px solid var(--border);">3</span>
            Ask anything
          </div>
        </div>
      </div>`;

    const input = $('question-input');
    input.disabled    = true;
    input.placeholder = 'Upload a document to unlock the chat…';
    input.value       = '';
    const btn = $('send-btn');
    btn.disabled         = true;
    btn.style.background = 'var(--border)';
    btn.style.color      = 'var(--muted)';

    setSessionStatus('NO_SESSION');
    $('session-status-label').textContent = 'No session';
    $('session-status-dot').className     = 'w-2 h-2 rounded-full bg-gray-600 inline-block flex-shrink-0';
  }

  // ── Utilities ────────────────────────────────────────────────────
  function showError(msg) {
    $('empty-state')?.remove();
    const notice = el('div', 'text-center py-8 text-sm');
    notice.style.color = '#F87171';
    notice.textContent = '⚠ ' + msg;
    $('chat-messages').appendChild(notice);
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ── Public API ───────────────────────────────────────────────────
  return { onDragOver, onDragLeave, onDrop, onFileSelect, sendMessage,
           fillQuestion, autoResize, onKeyDown, resetSession };

})();

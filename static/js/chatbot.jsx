/**
 * chatbot.jsx - React 18 ChatGPT-like Interface
 * Connects directly to Flask Chatbot API and Google Gemini Generative AI.
 */

const { useState, useEffect, useRef, useCallback } = React;

function ChatbotApp() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentTitle, setCurrentTitle] = useState("New Conversation");
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [studentContext, setStudentContext] = useState(null);
  const [showContextModal, setShowContextModal] = useState(false);
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [keySaveMsg, setKeySaveMsg] = useState("");
  const [copiedId, setCopiedId] = useState(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Suggested Prompts
  const quickPrompts = [
    {
      icon: "bi-graph-up-arrow",
      title: "Analyze My Performance",
      desc: "Review my attendance, internal marks, and exam score forecast.",
      prompt: "Can you analyze my current academic performance, explain my predicted final exam marks, and highlight any weak areas or risks?"
    },
    {
      icon: "bi-calendar-week",
      title: "Generate Study Timetable",
      desc: "Create an hour-by-hour weekly plan targeting my lowest subjects.",
      prompt: "Please generate a personalized daily and weekly study timetable based on my actual coursework and attendance records."
    },
    {
      icon: "bi-code-slash",
      title: "Explain QuickSort Algorithm",
      desc: "Provide clean Python & Java code with line-by-line breakdown.",
      prompt: "Explain the QuickSort algorithm step-by-step. Provide working code in both Python and Java, and explain the code line by line with time complexity analysis."
    },
    {
      icon: "bi-database-fill-gear",
      title: "DBMS ACID & Indexing",
      desc: "Top technical interview questions with clear examples.",
      prompt: "Explain ACID properties in DBMS and how B-Tree database indexing works with real-world examples as if preparing for a technical software interview."
    }
  ];

  const quickPills = [
    "Why are my predicted marks at this score?",
    "How to bring my attendance above 75%?",
    "Explain Deadlocks in Operating Systems",
    "Binary Search Tree in C++ with code",
    "Explain this project's ML model architecture"
  ];

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Adjust textarea height dynamically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [inputMessage]);

  // Fetch status, conversations, and student context on initial mount
  useEffect(() => {
    // 1. Fetch system status
    fetch('/api/chat/status')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setSystemStatus(data);
        }
      })
      .catch(err => console.error("Error fetching chat status:", err));

    // 2. Fetch student context snapshot
    fetch('/api/chat/student-context')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setStudentContext(data.context);
        }
      })
      .catch(err => console.error("Error fetching context:", err));

    // 3. Fetch conversations
    loadConversations(true);
  }, []);

  // Handle URL query parameters if launched with a pre-set prompt (e.g. from Dashboard)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const initialPrompt = params.get('q') || params.get('prompt');
    if (initialPrompt) {
      setInputMessage(initialPrompt);
      // Clean query parameter from URL without reloading
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  // Load conversations list
  const loadConversations = (selectFirst = false) => {
    fetch('/api/chat/conversations')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setConversations(data.conversations || []);
          if (selectFirst && data.conversations && data.conversations.length > 0) {
            loadMessages(data.conversations[0].id, data.conversations[0].title);
          }
        }
      })
      .catch(err => console.error("Error loading conversations:", err));
  };

  // Load messages for a specific conversation
  const loadMessages = (convId, title) => {
    setCurrentConversationId(convId);
    if (title) setCurrentTitle(title);

    fetch(`/api/chat/conversations/${convId}/messages`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setMessages(data.messages || []);
          if (data.conversation?.title) {
            setCurrentTitle(data.conversation.title);
          }
        }
      })
      .catch(err => console.error("Error loading messages:", err));
  };

  // Start a new chat
  const handleNewChat = () => {
    fetch('/api/chat/conversations/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Conversation' })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setCurrentConversationId(data.conversation.id);
          setCurrentTitle(data.conversation.title);
          setMessages([]);
          loadConversations();
          if (window.innerWidth < 768) {
            setIsSidebarOpen(false);
          }
        }
      })
      .catch(err => console.error("Error creating new chat:", err));
  };

  // Delete conversation
  const handleDeleteConversation = (convId, e) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this chat conversation?")) return;

    fetch(`/api/chat/conversations/${convId}`, { method: 'DELETE' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          if (currentConversationId === convId) {
            setCurrentConversationId(null);
            setCurrentTitle("New Conversation");
            setMessages([]);
          }
          loadConversations(true);
        }
      })
      .catch(err => console.error("Error deleting conversation:", err));
  };

  // Clear current conversation messages
  const handleClearChat = () => {
    if (!currentConversationId) return;
    if (!confirm("Clear all messages in this conversation?")) return;

    fetch(`/api/chat/conversations/${currentConversationId}/clear`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setMessages([]);
          loadConversations();
        }
      })
      .catch(err => console.error("Error clearing chat:", err));
  };

  // Send message
  const handleSendMessage = (overrideText = null) => {
    const textToSend = overrideText || inputMessage;
    if (!textToSend.trim() || isLoading) return;

    const userMessageObj = {
      id: `temp_${Date.now()}`,
      role: 'user',
      content: textToSend.trim(),
      created_at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // Optimistically append user message
    setMessages(prev => [...prev, userMessageObj]);
    setInputMessage("");
    setIsLoading(true);

    fetch('/api/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: currentConversationId,
        message: textToSend.trim()
      })
    })
      .then(async res => {
        if (!res.ok) {
          const text = await res.text();
          let errData;
          try { errData = JSON.parse(text); } catch(e) { errData = null; }
          if (errData && errData.error) throw new Error(errData.error);
          if (res.status === 401 || res.status === 302 || text.includes('Sign In')) {
            throw new Error("Your session has timed out. Please refresh the page or sign in again.");
          }
          throw new Error(`Server returned HTTP ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        setIsLoading(false);
        if (data.conversation_id && data.conversation_id !== currentConversationId) {
          setCurrentConversationId(data.conversation_id);
        }
        if (data.title) {
          setCurrentTitle(data.title);
        }

        const aiMessageObj = {
          id: `ai_${Date.now()}`,
          role: 'model',
          content: data.message || "I have processed your query.",
          created_at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          metadata: { model: data.model, success: data.success }
        };

        setMessages(prev => [...prev, aiMessageObj]);
        loadConversations();
      })
      .catch(err => {
        setIsLoading(false);
        console.error("Error sending message:", err);
        const errorMsgObj = {
          id: `err_${Date.now()}`,
          role: 'model',
          content: `### ⚠️ Notice\n\n${err.message || 'Unable to process your request. Please try again.'}`,
          created_at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages(prev => [...prev, errorMsgObj]);
      });
  };

  // Copy response text
  const handleCopy = (content, id) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  // Save Gemini API Key directly from UI
  const handleSaveApiKey = () => {
    if (!apiKeyInput.trim()) return;
    fetch('/api/chat/set-api-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKeyInput.trim() })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setKeySaveMsg("✅ Gemini API Key saved and activated successfully!");
          fetch('/api/chat/status')
            .then(res => res.json())
            .then(st => { if (st.success) setSystemStatus(st); });
          setTimeout(() => {
            setShowKeyModal(false);
            setKeySaveMsg("");
          }, 1500);
        } else {
          setKeySaveMsg(`❌ ${data.error || 'Failed to save key'}`);
        }
      })
      .catch(err => setKeySaveMsg(`❌ Error: ${err.message}`));
  };

  // Render markdown with syntax highlighting
  const renderMarkdown = (content) => {
    try {
      if (window.marked) {
        return { __html: window.marked.parse(content || '') };
      }
      return { __html: content };
    } catch (e) {
      return { __html: content };
    }
  };

  // Run Highlight.js on code blocks after render
  useEffect(() => {
    if (window.hljs) {
      document.querySelectorAll('.ai-bubble pre code').forEach((block) => {
        window.hljs.highlightElement(block);

        // Add copy button to code block if not present
        const pre = block.parentElement;
        if (pre && !pre.querySelector('.code-copy-btn')) {
          const wrapper = document.createElement('div');
          wrapper.className = 'code-block-wrapper';
          pre.parentNode.insertBefore(wrapper, pre);

          const header = document.createElement('div');
          header.className = 'code-block-header';

          // Detect language
          const langClass = Array.from(block.classList).find(c => c.startsWith('language-'));
          const lang = langClass ? langClass.replace('language-', '').toUpperCase() : 'CODE';

          header.innerHTML = `<span>${lang}</span>`;

          const copyBtn = document.createElement('button');
          copyBtn.className = 'code-copy-btn';
          copyBtn.innerHTML = '<i class="bi bi-clipboard"></i> Copy Code';
          copyBtn.onclick = () => {
            navigator.clipboard.writeText(block.innerText);
            copyBtn.innerHTML = '<i class="bi bi-check2"></i> Copied!';
            setTimeout(() => {
              copyBtn.innerHTML = '<i class="bi bi-clipboard"></i> Copy Code';
            }, 2000);
          };

          header.appendChild(copyBtn);
          wrapper.appendChild(header);
          wrapper.appendChild(pre);
        }
      });
    }
  }, [messages, isLoading]);

  return (
    <div className="chatbot-container">
      {/* --- SIDEBAR --- */}
      <aside className={`chatbot-sidebar ${!isSidebarOpen ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <div className="d-flex align-items-center justify-content-between mb-3">
            <div className="d-flex align-items-center gap-2">
              <span className="ai-avatar" style={{ width: 28, height: 28, fontSize: '0.85rem' }}>
                <i class="bi bi-stars"></i>
              </span>
              <span className="fw-bold text-white small">EduPredict Copilot</span>
            </div>
            <button
              className="btn btn-sm text-secondary d-md-none p-1"
              onClick={() => setIsSidebarOpen(false)}
            >
              <i className="bi bi-x-lg"></i>
            </button>
          </div>
          <button className="new-chat-btn" onClick={handleNewChat}>
            <i className="bi bi-plus-lg"></i>
            <span>New Chat</span>
          </button>
        </div>

        <div className="sidebar-threads">
          <div className="thread-category-label">Recent Conversations</div>
          {conversations.length === 0 ? (
            <div className="text-muted small px-3 py-2">No conversations yet.</div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                className={`thread-item ${currentConversationId === conv.id ? 'active' : ''}`}
                onClick={() => loadMessages(conv.id, conv.title)}
              >
                <div className="thread-info">
                  <i className="bi bi-chat-left-text"></i>
                  <span className="thread-title">{conv.title}</span>
                </div>
                <div className="thread-actions">
                  <button
                    className="thread-action-btn"
                    title="Delete Chat"
                    onClick={(e) => handleDeleteConversation(conv.id, e)}
                  >
                    <i className="bi bi-trash"></i>
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className="storage-status-pill mb-2">
            <span className={`status-dot ${systemStatus?.storage?.is_mongodb ? '' : 'sqlite'}`}></span>
            <span className="text-truncate">
              Storage: {systemStatus?.storage?.is_mongodb ? 'MongoDB' : 'SQLite Local'}
            </span>
          </div>
          <div className="d-flex align-items-center justify-content-between">
            <div className="overflow-hidden">
              <div className="text-truncate text-white small fw-bold">
                {systemStatus?.student?.name || 'Student'}
              </div>
              <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                {systemStatus?.student?.student_id || 'ID'}
              </div>
            </div>
            <a href="/dashboard" className="btn btn-sm btn-outline-light py-1 px-2" title="Dashboard">
              <i className="bi bi-grid-1x2"></i>
            </a>
          </div>
        </div>
      </aside>

      {/* --- MAIN CHAT CANVAS --- */}
      <main className="chatbot-main">
        {/* Chat Top Navbar */}
        <header className="chat-navbar">
          <div className="d-flex align-items-center gap-2">
            <button
              className="btn btn-sm btn-outline-secondary p-1 px-2"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              title="Toggle Sidebar"
            >
              <i className="bi bi-layout-sidebar"></i>
            </button>
            <h2 className="chat-navbar-title">{currentTitle}</h2>
          </div>

          <div className="d-flex align-items-center gap-2">
            <span className="model-badge d-none d-sm-inline-flex">
              <i className="bi bi-cpu-fill text-primary"></i>
              {systemStatus?.model || "EduPredict AI"}
            </span>
            <button
              className={`btn btn-sm ${systemStatus?.configured ? 'btn-outline-success' : 'btn-primary-soft'}`}
              onClick={() => setShowKeyModal(true)}
              title="Configure Google Gemini API Key"
            >
              <i className="bi bi-key-fill me-1"></i>
              <span className="d-none d-md-inline">{systemStatus?.configured ? "Gemini Key Active" : "Add Gemini Key"}</span>
            </button>
            <button
              className="btn btn-sm btn-outline-soft"
              onClick={() => setShowContextModal(true)}
              title="View Live Academic Context Grounding"
            >
              <i className="bi bi-file-earmark-person me-1"></i> Context
            </button>
            {messages.length > 0 && (
              <button
                className="btn btn-sm btn-outline-danger"
                onClick={handleClearChat}
                title="Clear current messages"
              >
                <i className="bi bi-trash"></i>
              </button>
            )}
          </div>
        </header>

        {/* Message Stream */}
        <div className="messages-container">
          {messages.length === 0 ? (
            /* Welcome Hero View */
            <div className="welcome-hero">
              <div className="welcome-avatar">
                <i className="bi bi-stars"></i>
              </div>
              <h1 className="welcome-title">
                Hello, {systemStatus?.student?.name ? systemStatus.student.name.split(' ')[0] : 'Student'}! 👋
              </h1>
              <p className="welcome-subtitle">
                I am your <strong>EduPredict AI Study Copilot</strong>, powered by Google Gemini.
                Ask me any question — from technical engineering and programming to personalized exam forecasts and study plans!
              </p>

              <div className="prompt-suggestions-grid">
                {quickPrompts.map((p, idx) => (
                  <div
                    key={idx}
                    className="prompt-card"
                    onClick={() => handleSendMessage(p.prompt)}
                  >
                    <div className="prompt-card-icon">
                      <i className={`bi ${p.icon}`}></i>
                    </div>
                    <div className="prompt-card-title">{p.title}</div>
                    <div className="prompt-card-desc">{p.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Message Bubbles */
            messages.map((msg, index) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={msg.id || index}
                  className={`message-row ${isUser ? 'user-row' : 'ai-row'}`}
                >
                  <div className={`message-avatar ${isUser ? 'user-avatar' : 'ai-avatar'}`}>
                    <i className={`bi ${isUser ? 'bi-person-fill' : 'bi-stars'}`}></i>
                  </div>

                  <div className="message-content-wrapper">
                    <div className="message-meta">
                      <span>{isUser ? 'You' : 'EduPredict Copilot'}</span>
                      <span>•</span>
                      <span>{msg.created_at || 'Just now'}</span>
                    </div>

                    <div className={`message-bubble ${isUser ? 'user-bubble' : 'ai-bubble'}`}>
                      {isUser ? (
                        <p>{msg.content}</p>
                      ) : (
                        <div dangerouslySetInnerHTML={renderMarkdown(msg.content)} />
                      )}
                    </div>

                    {!isUser && (
                      <div className="message-actions-bar">
                        <button
                          className="msg-action-btn"
                          onClick={() => handleCopy(msg.content, msg.id || index)}
                          title="Copy full response"
                        >
                          <i className={`bi ${copiedId === (msg.id || index) ? 'bi-check2 text-success' : 'bi-clipboard'}`}></i>
                          <span>{copiedId === (msg.id || index) ? 'Copied' : 'Copy'}</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Typing Indicator */}
          {isLoading && (
            <div className="message-row ai-row">
              <div className="message-avatar ai-avatar">
                <i className="bi bi-stars"></i>
              </div>
              <div className="message-content-wrapper">
                <div className="message-meta">
                  <span>EduPredict Copilot</span>
                  <span>•</span>
                  <span>Generating response...</span>
                </div>
                <div className="typing-bubble">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* --- BOTTOM INPUT COMPOSER --- */}
        <div className="chat-composer-container">
          {/* Quick Action Suggestion Pills */}
          <div className="composer-suggestions">
            {quickPills.map((pill, idx) => (
              <button
                key={idx}
                type="button"
                className="suggestion-pill"
                onClick={() => handleSendMessage(pill)}
              >
                {pill}
              </button>
            ))}
          </div>

          {/* Chat Composer Box */}
          <div className="composer-box">
            <textarea
              ref={textareaRef}
              className="composer-textarea"
              placeholder="Ask anything... (e.g. Explain QuickSort, analyze my attendance, or write Java code)"
              value={inputMessage}
              rows="1"
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <button
              type="button"
              className="composer-send-btn"
              disabled={!inputMessage.trim() || isLoading}
              onClick={() => handleSendMessage()}
              title="Send Message (Enter)"
            >
              <i className="bi bi-arrow-up"></i>
            </button>
          </div>

          <div className="composer-disclaimer">
            EduPredict AI connects Google Gemini with your live college records. Answers can be general or personalized.
          </div>
        </div>
      </main>

      {/* --- API KEY CONFIGURATION MODAL --- */}
      {showKeyModal && (
        <div className="context-modal-backdrop" onClick={() => setShowKeyModal(false)}>
          <div className="context-modal" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
            <div className="context-modal-header">
              <h5 className="modal-title fw-bold m-0 d-flex align-items-center gap-2">
                <i className="bi bi-key-fill text-primary"></i>
                Google Gemini API Key
              </h5>
              <button
                type="button"
                className="btn-close"
                onClick={() => setShowKeyModal(false)}
              ></button>
            </div>
            <div className="context-modal-body">
              <div className="alert alert-info py-2 small mb-3">
                <i className="bi bi-info-circle me-1"></i>
                The AI Bot Assistant is <strong>currently active with the built-in AI engine</strong>.
                Adding your Google Gemini API key enables live cloud <strong>Gemini 1.5 Flash</strong> reasoning.
              </div>

              <div className="mb-3">
                <label className="form-label fw-semibold small">Paste Google Gemini API Key</label>
                <input
                  type="password"
                  className="form-control"
                  placeholder="AIzaSy..."
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                />
                <div className="form-text small mt-1">
                  Need a key? Get one 100% free at{" "}
                  <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="fw-semibold">
                    Google AI Studio <i className="bi bi-box-arrow-up-right small"></i>
                  </a>
                </div>
              </div>

              {keySaveMsg && (
                <div className={`alert ${keySaveMsg.startsWith('✅') ? 'alert-success' : 'alert-danger'} py-2 small`}>
                  {keySaveMsg}
                </div>
              )}
            </div>
            <div className="p-3 border-top d-flex justify-content-between align-items-center bg-light">
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => setShowKeyModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-sm btn-primary px-3"
                disabled={!apiKeyInput.trim()}
                onClick={handleSaveApiKey}
              >
                <i className="bi bi-check-lg me-1"></i> Save & Activate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- STUDENT CONTEXT MODAL --- */}
      {showContextModal && (
        <div className="context-modal-backdrop" onClick={() => setShowContextModal(false)}>
          <div className="context-modal" onClick={(e) => e.stopPropagation()}>
            <div className="context-modal-header">
              <h5 className="modal-title fw-bold m-0 d-flex align-items-center gap-2">
                <i className="bi bi-shield-check text-success"></i>
                AI Academic Grounding Snapshot
              </h5>
              <button
                type="button"
                className="btn-close"
                onClick={() => setShowContextModal(false)}
              ></button>
            </div>
            <div className="context-modal-body">
              <p className="text-muted small mb-3">
                This verified student data is securely provided to Google Gemini as context when you ask academic questions:
              </p>
              <div className="bg-light p-3 rounded border mb-3">
                <div className="row g-2 small">
                  <div className="col-6"><strong>Student:</strong> {studentContext?.student?.name}</div>
                  <div className="col-6"><strong>ID:</strong> {studentContext?.student?.student_id}</div>
                  <div className="col-6"><strong>Department:</strong> {studentContext?.student?.department}</div>
                  <div className="col-6"><strong>Semester:</strong> {studentContext?.student?.semester}</div>
                  <div className="col-6">
                    <strong>Attendance:</strong> {studentContext?.aggregates?.overall_attendance_pct}%
                  </div>
                  <div className="col-6">
                    <strong>Avg Internals:</strong> {studentContext?.aggregates?.avg_internal}%
                  </div>
                  <div className="col-6">
                    <strong>Predicted Marks:</strong> {studentContext?.aggregates?.predicted_marks || 'N/A'}/100
                  </div>
                  <div className="col-6">
                    <strong>Performance:</strong> <span className="badge bg-primary">{studentContext?.aggregates?.category}</span>
                  </div>
                </div>
              </div>

              <h6 className="fw-bold mb-2">Subject Breakdown</h6>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead className="table-light">
                    <tr>
                      <th>Subject</th>
                      <th>Attendance</th>
                      <th>Internal (75)</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {studentContext?.attendance?.map((att, i) => (
                      <tr key={i}>
                        <td>{att.subject_name}</td>
                        <td>{att.attended}/{att.total} ({att.percentage}%)</td>
                        <td>{studentContext?.marks?.[i]?.total_internal || 'N/A'}</td>
                        <td>
                          <span className={`badge bg-${att.badge}`}>{att.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="p-3 border-top text-end bg-light">
              <button
                className="btn btn-sm btn-primary"
                onClick={() => setShowContextModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Mount the React Application
const rootElement = document.getElementById('chatbot-app-root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<ChatbotApp />);
}

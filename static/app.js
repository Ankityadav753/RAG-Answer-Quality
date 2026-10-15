/**
 * DAI-009: RAG Answer Quality Browser Interface
 * Client-side Controller & Dynamic Document Management Handler
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // DOM Elements - Query Interface
  // =========================================================================
  const form = document.getElementById("rag-query-form");
  const questionInput = document.getElementById("question-input");
  const retrieverSelect = document.getElementById("retriever-select");
  const topKSelect = document.getElementById("top-k-select");
  const askButton = document.getElementById("ask-button");
  const btnSpinner = document.getElementById("btn-spinner");
  const clearButton = document.getElementById("clear-button");
  const sampleChips = document.querySelectorAll(".chip");

  const systemStatusText = document.getElementById("system-status-text");
  const modeText = document.getElementById("mode-text");
  const chunksCountText = document.getElementById("chunks-count-text");
  const docsCountText = document.getElementById("docs-count-text");
  const tabDocsBadge = document.getElementById("tab-docs-badge");

  const errorContainer = document.getElementById("error-container");
  const errorMessage = document.getElementById("error-message");
  const errorDismiss = document.getElementById("error-dismiss");

  const loadingState = document.getElementById("loading-state");
  const resultContainer = document.getElementById("result-container");
  const answerCard = document.getElementById("answer-card");
  const abstentionBadge = document.getElementById("abstention-badge");
  const abstentionText = document.getElementById("abstention-text");
  const providerNameText = document.getElementById("provider-name-text");
  const latencyValue = document.getElementById("latency-value");
  const retrieverNameValue = document.getElementById("retriever-name-value");
  const answerText = document.getElementById("answer-text");
  const sourceDocsList = document.getElementById("source-docs-list");
  const sourceChunksList = document.getElementById("source-chunks-list");
  const chunksRetrievedCount = document.getElementById("chunks-retrieved-count");
  const chunksList = document.getElementById("chunks-list");

  // =========================================================================
  // DOM Elements - Document Management & Tabs
  // =========================================================================
  const tabQueryBtn = document.getElementById("tab-query-btn");
  const tabDocsBtn = document.getElementById("tab-docs-btn");
  const panelQuery = document.getElementById("panel-query");
  const panelDocs = document.getElementById("panel-docs");

  const uploadForm = document.getElementById("upload-form");
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const fileSelectedBar = document.getElementById("file-selected-bar");
  const selectedFilename = document.getElementById("selected-filename");
  const selectedFilesize = document.getElementById("selected-filesize");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const overwriteCheck = document.getElementById("overwrite-check");
  const uploadSubmitBtn = document.getElementById("upload-submit-btn");
  const uploadSpinner = document.getElementById("upload-spinner");

  const uploadAlert = document.getElementById("upload-alert");
  const uploadAlertIcon = document.getElementById("upload-alert-icon");
  const uploadAlertTitle = document.getElementById("upload-alert-title");
  const uploadAlertMessage = document.getElementById("upload-alert-message");
  const uploadAlertDismiss = document.getElementById("upload-alert-dismiss");

  const docSearchInput = document.getElementById("doc-search-input");
  const refreshDocsBtn = document.getElementById("refresh-docs-btn");
  const docsTableBody = document.getElementById("docs-table-body");
  const docsEmptyState = document.getElementById("docs-empty-state");

  const deleteModal = document.getElementById("delete-modal");
  const modalDocId = document.getElementById("modal-doc-id");
  const modalDocTitle = document.getElementById("modal-doc-title");
  const modalDocChunks = document.getElementById("modal-doc-chunks");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalCancelBtn = document.getElementById("modal-cancel-btn");
  const modalConfirmDeleteBtn = document.getElementById("modal-confirm-delete-btn");
  const modalSpinner = document.getElementById("modal-spinner");

  let currentPendingDeleteId = null;
  let cachedDocumentsList = [];
  let selectedFileObject = null;

  // =========================================================================
  // Tab Navigation Handling
  // =========================================================================
  function switchTab(targetTab) {
    if (targetTab === "panel-query") {
      tabQueryBtn.classList.add("active");
      tabQueryBtn.setAttribute("aria-selected", "true");
      tabDocsBtn.classList.remove("active");
      tabDocsBtn.setAttribute("aria-selected", "false");

      panelQuery.classList.remove("hidden");
      panelDocs.classList.add("hidden");
    } else {
      tabDocsBtn.classList.add("active");
      tabDocsBtn.setAttribute("aria-selected", "true");
      tabQueryBtn.classList.remove("active");
      tabQueryBtn.setAttribute("aria-selected", "false");

      panelDocs.classList.remove("hidden");
      panelQuery.classList.add("hidden");
      loadDocumentsList();
    }
  }

  if (tabQueryBtn && tabDocsBtn) {
    tabQueryBtn.addEventListener("click", () => switchTab("panel-query"));
    tabDocsBtn.addEventListener("click", () => switchTab("panel-docs"));
  }

  // =========================================================================
  // System Status & Counts
  // =========================================================================
  async function loadSystemStatus() {
    try {
      const resp = await fetch("/api/info");
      if (resp.ok) {
        const data = await resp.json();
        if (modeText) {
          modeText.textContent = data.is_offline ? "Offline Deterministic" : `Online: ${data.llm_provider}`;
        }
        if (chunksCountText) {
          chunksCountText.textContent = `${data.chunks_indexed || 0} Chunks`;
        }
        if (docsCountText) {
          docsCountText.textContent = `${data.documents_count || 0} Documents`;
        }
        if (tabDocsBadge) {
          tabDocsBadge.textContent = `${data.documents_count || 0}`;
        }
        if (systemStatusText) {
          systemStatusText.textContent = "System Ready";
        }
      }
    } catch (err) {
      console.warn("Could not load /api/info metadata:", err);
      if (systemStatusText) systemStatusText.textContent = "Offline Mode";
    }
  }

  loadSystemStatus();

  // =========================================================================
  // Query Error Helpers
  // =========================================================================
  function showError(msg) {
    if (errorMessage && errorContainer) {
      errorMessage.textContent = msg;
      errorContainer.classList.remove("hidden");
      errorContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function hideError() {
    if (errorContainer) {
      errorContainer.classList.add("hidden");
    }
  }

  if (errorDismiss) {
    errorDismiss.addEventListener("click", hideError);
  }

  // =========================================================================
  // Sample Questions Handling
  // =========================================================================
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query && questionInput) {
        questionInput.value = query;
        questionInput.focus();
        hideError();
      }
    });
  });

  // Clear query form
  if (clearButton) {
    clearButton.addEventListener("click", () => {
      if (questionInput) questionInput.value = "";
      hideError();
      if (resultContainer) resultContainer.classList.add("hidden");
      if (loadingState) loadingState.classList.add("hidden");
      if (questionInput) questionInput.focus();
    });
  }

  // =========================================================================
  // Query Submission Handler
  // =========================================================================
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();

      const question = questionInput.value.trim();
      if (!question) {
        showError("Please enter a non-empty question to query the knowledge base.");
        return;
      }

      const retriever = retrieverSelect.value;
      const topK = parseInt(topKSelect.value, 10) || 3;

      // Loading state
      if (resultContainer) resultContainer.classList.add("hidden");
      if (loadingState) loadingState.classList.remove("hidden");
      if (askButton) askButton.disabled = true;
      if (btnSpinner) btnSpinner.classList.remove("hidden");

      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: question,
            retriever: retriever,
            top_k: topK,
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const detail = errData.detail || (errData.errors ? errData.errors.join(", ") : "Query processing failed.");
          showError(`Query Error (${response.status}): ${detail}`);
          return;
        }

        const data = await response.json();
        renderResult(data);

      } catch (networkErr) {
        showError(`Network connection error: ${networkErr.message}. Ensure backend is running.`);
      } finally {
        if (loadingState) loadingState.classList.add("hidden");
        if (askButton) askButton.disabled = false;
        if (btnSpinner) btnSpinner.classList.add("hidden");
      }
    });
  }

  // =========================================================================
  // Query Result Rendering
  // =========================================================================
  function renderResult(data) {
    if (!resultContainer) return;

    if (answerText) answerText.textContent = data.answer || "No response generated.";
    if (latencyValue) latencyValue.textContent = Number(data.latency_ms || 0).toFixed(2);
    if (retrieverNameValue) retrieverNameValue.textContent = data.retriever_name || "Lexical";
    if (providerNameText) {
      providerNameText.textContent = data.is_mock ? "Deterministic Mock Engine" : (data.provider || "Online Model");
    }

    // Abstention badge styling
    if (abstentionBadge && abstentionText) {
      if (data.is_abstention) {
        abstentionBadge.className = "badge badge-warning";
        abstentionBadge.querySelector(".badge-icon").textContent = "⚠";
        abstentionText.textContent = "Abstention Triggered (No Evidence)";
        if (answerCard) answerCard.classList.add("card-abstention");
      } else {
        abstentionBadge.className = "badge badge-success";
        abstentionBadge.querySelector(".badge-icon").textContent = "✓";
        abstentionText.textContent = "Evidence-Grounded Answer";
        if (answerCard) answerCard.classList.remove("card-abstention");
      }
    }

    // Render citations
    renderCitationTags(sourceDocsList, data.source_doc_ids, "tag-doc");
    renderCitationTags(sourceChunksList, data.source_chunk_ids, "tag-chunk");

    // Render retrieved chunks
    const chunks = data.retrieved_chunks || [];
    if (chunksRetrievedCount) {
      chunksRetrievedCount.textContent = `${chunks.length} Chunk${chunks.length === 1 ? "" : "s"}`;
    }

    if (chunksList) {
      chunksList.innerHTML = "";
      if (chunks.length === 0) {
        chunksList.innerHTML = `<div class="empty-chunks">No context chunks met the retrieval threshold.</div>`;
      } else {
        chunks.forEach((chunk) => {
          const isCited = (data.source_chunk_ids || []).includes(chunk.chunk_id);
          const chunkCard = document.createElement("div");
          chunkCard.className = `chunk-item ${isCited ? "chunk-cited" : ""}`;

          chunkCard.innerHTML = `
            <div class="chunk-meta-header">
              <div class="chunk-meta-left">
                <span class="chunk-rank-badge">Rank #${chunk.rank}</span>
                <span class="chunk-doc-id">${escapeHtml(chunk.document_id)}</span>
                <span class="chunk-id-tag">${escapeHtml(chunk.chunk_id)}</span>
              </div>
              <div class="chunk-meta-right">
                <span class="chunk-score">Score: ${Number(chunk.score).toFixed(4)}</span>
                ${isCited ? '<span class="badge badge-success-sm">Cited in Answer</span>' : ""}
              </div>
            </div>
            <div class="chunk-content-body">
              <p>${escapeHtml(chunk.content)}</p>
            </div>
          `;
          chunksList.appendChild(chunkCard);
        });
      }
    }

    resultContainer.classList.remove("hidden");
    resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderCitationTags(container, items, tagClass) {
    if (!container) return;
    container.innerHTML = "";

    if (!items || items.length === 0) {
      const emptySpan = document.createElement("span");
      emptySpan.className = "tag tag-empty";
      emptySpan.textContent = "None (Refused)";
      container.appendChild(emptySpan);
      return;
    }

    items.forEach((item) => {
      const tag = document.createElement("span");
      tag.className = `tag ${tagClass}`;
      tag.textContent = item;
      container.appendChild(tag);
    });
  }

  // =========================================================================
  // Document Upload Handling
  // =========================================================================
  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function setSelectedFile(file) {
    selectedFileObject = file;
    if (file) {
      if (selectedFilename) selectedFilename.textContent = file.name;
      if (selectedFilesize) selectedFilesize.textContent = `(${formatBytes(file.size)})`;
      if (fileSelectedBar) fileSelectedBar.classList.remove("hidden");
      if (uploadSubmitBtn) uploadSubmitBtn.disabled = false;
    } else {
      if (fileInput) fileInput.value = "";
      if (fileSelectedBar) fileSelectedBar.classList.add("hidden");
      if (uploadSubmitBtn) uploadSubmitBtn.disabled = true;
    }
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        setSelectedFile(e.target.files[0]);
      }
    });
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", () => {
      setSelectedFile(null);
      hideUploadAlert();
    });
  }

  // Drag and drop handling
  if (dropzone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        setSelectedFile(e.dataTransfer.files[0]);
      }
    });
  }

  function showUploadAlert(type, title, message) {
    if (!uploadAlert) return;
    uploadAlert.className = `alert alert-${type}`;
    if (uploadAlertTitle) uploadAlertTitle.textContent = title;
    if (uploadAlertMessage) uploadAlertMessage.innerHTML = message;
    if (uploadAlertIcon) {
      uploadAlertIcon.innerHTML = type === "success" 
        ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>'
        : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
    }
    uploadAlert.classList.remove("hidden");
  }

  function hideUploadAlert() {
    if (uploadAlert) uploadAlert.classList.add("hidden");
  }

  if (uploadAlertDismiss) {
    uploadAlertDismiss.addEventListener("click", hideUploadAlert);
  }

  // Upload Form Submit
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideUploadAlert();

      if (!selectedFileObject) {
        showUploadAlert("error", "No File Selected", "Please select a file to upload.");
        return;
      }

      const overwrite = overwriteCheck ? overwriteCheck.checked : false;
      const formData = new FormData();
      formData.append("file", selectedFileObject);

      if (uploadSubmitBtn) uploadSubmitBtn.disabled = true;
      if (uploadSpinner) uploadSpinner.classList.remove("hidden");

      try {
        const resp = await fetch(`/api/documents/upload?overwrite=${overwrite}`, {
          method: "POST",
          body: formData,
        });

        const data = await resp.json().catch(() => ({}));

        if (!resp.ok) {
          const detail = data.detail || "Document upload failed.";
          showUploadAlert("error", "Upload Failed", detail);
          return;
        }

        // Success
        const successHtml = `
          Document <strong>${escapeHtml(data.filename)}</strong> ingested as 
          <code class="font-mono">${escapeHtml(data.document_id)}</code> (${data.chunks_created} chunks).
          <br/><a href="#" id="query-uploaded-link" style="color: inherit; text-decoration: underline; font-weight: 600; margin-top: 4px; display: inline-block;">
            → Click here to ask a question about this document
          </a>
        `;
        showUploadAlert("success", "Document Ingested Successfully", successHtml);

        const queryLink = document.getElementById("query-uploaded-link");
        if (queryLink) {
          queryLink.addEventListener("click", (evt) => {
            evt.preventDefault();
            switchTab("panel-query");
            if (questionInput) {
              questionInput.value = `Tell me about the contents of ${data.filename}`;
              questionInput.focus();
            }
          });
        }

        // Reset file selection
        setSelectedFile(null);

        // Reload document library and header status
        loadDocumentsList();
        loadSystemStatus();

      } catch (err) {
        showUploadAlert("error", "Network Error", `Failed to upload document: ${err.message}`);
      } finally {
        if (uploadSubmitBtn) uploadSubmitBtn.disabled = false;
        if (uploadSpinner) uploadSpinner.classList.add("hidden");
      }
    });
  }

  // =========================================================================
  // Document Library & Management Table
  // =========================================================================
  async function loadDocumentsList() {
    if (!docsTableBody) return;

    try {
      const resp = await fetch("/api/documents");
      if (resp.ok) {
        const data = await resp.json();
        cachedDocumentsList = data.documents || [];
        renderDocumentsTable(cachedDocumentsList);

        if (docsCountText) {
          docsCountText.textContent = `${data.total || cachedDocumentsList.length} Documents`;
        }
        if (tabDocsBadge) {
          tabDocsBadge.textContent = `${data.total || cachedDocumentsList.length}`;
        }
      }
    } catch (err) {
      console.warn("Could not load /api/documents:", err);
    }
  }

  function renderDocumentsTable(documents) {
    if (!docsTableBody) return;
    docsTableBody.innerHTML = "";

    const searchTerm = (docSearchInput ? docSearchInput.value : "").trim().toLowerCase();
    const filtered = documents.filter((doc) => {
      if (!searchTerm) return true;
      return (
        doc.id.toLowerCase().includes(searchTerm) ||
        doc.title.toLowerCase().includes(searchTerm) ||
        (doc.source && doc.source.toLowerCase().includes(searchTerm))
      );
    });

    if (filtered.length === 0) {
      if (docsEmptyState) docsEmptyState.classList.remove("hidden");
      return;
    }

    if (docsEmptyState) docsEmptyState.classList.add("hidden");

    filtered.forEach((doc) => {
      const tr = document.createElement("tr");

      const typeBadge = doc.is_sample
        ? '<span class="badge badge-purple">Core Baseline</span>'
        : '<span class="badge badge-success-sm">User Upload</span>';

      const actionHtml = doc.is_sample
        ? '<span class="badge badge-protected" title="Core benchmark baseline document is protected from deletion">Protected</span>'
        : `<button type="button" class="btn btn-danger btn-sm doc-delete-btn" 
             data-id="${escapeHtml(doc.id)}" 
             data-title="${escapeHtml(doc.title)}" 
             data-chunks="${doc.chunk_count}">
             Delete
           </button>`;

      const formattedDate = doc.created_at ? doc.created_at.replace("T", " ").substring(0, 19) : "N/A";

      tr.innerHTML = `
        <td class="font-mono font-medium">${escapeHtml(doc.id)}</td>
        <td><strong>${escapeHtml(doc.title)}</strong></td>
        <td>${typeBadge}</td>
        <td><span class="badge badge-outline">${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}</span></td>
        <td class="text-muted">${escapeHtml(formattedDate)}</td>
        <td class="text-right">${actionHtml}</td>
      `;

      docsTableBody.appendChild(tr);
    });

    // Attach click handlers to delete buttons
    document.querySelectorAll(".doc-delete-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const docId = btn.getAttribute("data-id");
        const docTitle = btn.getAttribute("data-title");
        const docChunks = btn.getAttribute("data-chunks");
        openDeleteModal(docId, docTitle, docChunks);
      });
    });
  }

  // Filter input event
  if (docSearchInput) {
    docSearchInput.addEventListener("input", () => {
      renderDocumentsTable(cachedDocumentsList);
    });
  }

  // Refresh button event
  if (refreshDocsBtn) {
    refreshDocsBtn.addEventListener("click", () => {
      loadDocumentsList();
      loadSystemStatus();
    });
  }

  // =========================================================================
  // Delete Confirmation Modal Handling
  // =========================================================================
  function openDeleteModal(docId, title, chunks) {
    currentPendingDeleteId = docId;
    if (modalDocId) modalDocId.textContent = docId;
    if (modalDocTitle) modalDocTitle.textContent = title;
    if (modalDocChunks) modalDocChunks.textContent = `${chunks} chunk(s)`;
    if (deleteModal) deleteModal.classList.remove("hidden");
  }

  function closeDeleteModal() {
    currentPendingDeleteId = null;
    if (deleteModal) deleteModal.classList.add("hidden");
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeDeleteModal);
  if (modalCancelBtn) modalCancelBtn.addEventListener("click", closeDeleteModal);

  if (modalConfirmDeleteBtn) {
    modalConfirmDeleteBtn.addEventListener("click", async () => {
      if (!currentPendingDeleteId) return;

      if (modalConfirmDeleteBtn) modalConfirmDeleteBtn.disabled = true;
      if (modalSpinner) modalSpinner.classList.remove("hidden");

      try {
        const resp = await fetch(`/api/documents/${encodeURIComponent(currentPendingDeleteId)}`, {
          method: "DELETE",
        });

        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          alert(`Deletion failed: ${errData.detail || "Unknown error"}`);
          return;
        }

        const data = await resp.json();
        closeDeleteModal();
        showUploadAlert("success", "Document Deleted", `Document <strong>${escapeHtml(data.document_id)}</strong> and all ${data.deleted_chunks} associated chunks were removed.`);

        // Refresh list and status counters
        loadDocumentsList();
        loadSystemStatus();

      } catch (err) {
        alert(`Error communicating with server: ${err.message}`);
      } finally {
        if (modalConfirmDeleteBtn) modalConfirmDeleteBtn.disabled = false;
        if (modalSpinner) modalSpinner.classList.add("hidden");
      }
    });
  }

  // Helper: Escape HTML
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

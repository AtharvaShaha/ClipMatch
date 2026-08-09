/**
 * ClipMatch - Frontend Application
 * Video Source Detection System
 */

// API Configuration
const API_BASE = 'http://localhost:5000/api';

// State
const state = {
    currentPage: 'match',
    selectedFile: null,
    isProcessing: false,
    clipQuality: 'original', // 'original' or 'edited'
    references: [],
    history: []
};

// DOM Elements
const elements = {
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    pages: document.querySelectorAll('.page'),
    
    // Match Page
    uploadZone: document.getElementById('uploadZone'),
    clipInput: document.getElementById('clipInput'),
    filePreview: document.getElementById('filePreview'),
    fileName: document.getElementById('fileName'),
    fileMeta: document.getElementById('fileMeta'),
    removeFile: document.getElementById('removeFile'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    qualityOriginal: document.getElementById('qualityOriginal'),
    qualityEdited: document.getElementById('qualityEdited'),
    processing: document.getElementById('processing'),
    processingStatus: document.getElementById('processingStatus'),
    progressBar: document.getElementById('progressBar'),
    results: document.getElementById('results'),
    processingTime: document.getElementById('processingTime'),
    matchResult: document.getElementById('matchResult'),
    meterFill: document.getElementById('meterFill'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceLabel: document.getElementById('confidenceLabel'),
    matchedVideo: document.getElementById('matchedVideo'),
    timestampRange: document.getElementById('timestampRange'),
    similarityScore: document.getElementById('similarityScore'),
    noMatch: document.getElementById('noMatch'),
    resetBtn: document.getElementById('resetBtn'),
    
    // Library Page
    libraryStats: document.getElementById('libraryStats'),
    statVideos: document.getElementById('statVideos'),
    statFrames: document.getElementById('statFrames'),
    statDuration: document.getElementById('statDuration'),
    uploadReference: document.getElementById('uploadReference'),
    referenceInput: document.getElementById('referenceInput'),
    indexAllBtn: document.getElementById('indexAllBtn'),
    libraryList: document.getElementById('libraryList'),
    emptyLibrary: document.getElementById('emptyLibrary'),
    
    // History Page
    historyList: document.getElementById('historyList'),
    emptyHistory: document.getElementById('emptyHistory'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer'),
    
    // Modal
    uploadModal: document.getElementById('uploadModal'),
    modalClose: document.getElementById('modalClose'),
    indexFileName: document.getElementById('indexFileName'),
    indexStatus: document.getElementById('indexStatus'),
    indexBarFill: document.getElementById('indexBarFill'),
    
    // Processing extras
    elapsedTimer: document.getElementById('elapsedTimer'),
    longWaitMsg: document.getElementById('longWaitMsg'),
    cancelBtn: document.getElementById('cancelBtn'),
    
    // Evidence panel
    evidencePanel: document.getElementById('evidencePanel'),
    evidenceToggle: document.getElementById('evidenceToggle'),
    evidenceBody: document.getElementById('evidenceBody'),
    evHashRatio: document.getElementById('evHashRatio'),
    evColorRatio: document.getElementById('evColorRatio'),
    evAvgDist: document.getElementById('evAvgDist'),
    evMinDist: document.getElementById('evMinDist'),
    evFrames: document.getElementById('evFrames'),
    evMethod: document.getElementById('evMethod'),
    evidenceNccGrid: document.getElementById('evidenceNccGrid'),
    evNcc: document.getElementById('evNcc'),
    evSsim: document.getElementById('evSsim'),
    evidenceCandidates: document.getElementById('evidenceCandidates'),
    evCandidates: document.getElementById('evCandidates'),
    evidenceTiming: document.getElementById('evidenceTiming'),
    timingBars: document.getElementById('timingBars'),
    pipelineStages: document.getElementById('pipelineStages')
};

// ============================================================
// Navigation
// ============================================================

function initNavigation() {
    elements.navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const page = link.dataset.page;
            
            // Handle HOME specially - navigate to home route
            if (page === 'home') {
                e.preventDefault();
                window.location.href = '/'; // Navigate to home route
                return;
            }
            
            e.preventDefault();
            navigateTo(page);
        });
    });
}

function navigateTo(page) {
    // Update nav links
    elements.navLinks.forEach(link => {
        link.classList.toggle('active', link.dataset.page === page);
    });
    
    // Update pages
    elements.pages.forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });
    
    state.currentPage = page;
    
    // Load data for page
    if (page === 'library') {
        loadLibrary();
    } else if (page === 'history') {
        loadHistory();
    }
}

// ============================================================
// Match Page
// ============================================================

function initMatchPage() {
    // Upload zone click
    elements.uploadZone.addEventListener('click', () => {
        elements.clipInput.click();
    });
    
    // File input change
    elements.clipInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.add('dragover');
    });
    
    elements.uploadZone.addEventListener('dragleave', () => {
        elements.uploadZone.classList.remove('dragover');
    });
    
    elements.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Remove file
    elements.removeFile.addEventListener('click', resetMatchPage);
    
    // Analyze button
    elements.analyzeBtn.addEventListener('click', analyzeClip);
    
    // Quality selector buttons
    elements.qualityOriginal.addEventListener('click', () => {
        state.clipQuality = 'original';
        updateQualityUI();
    });
    
    elements.qualityEdited.addEventListener('click', () => {
        state.clipQuality = 'edited';
        updateQualityUI();
    });
    
    // Reset button
    elements.resetBtn.addEventListener('click', resetMatchPage);
    
    // Cancel button
    elements.cancelBtn.addEventListener('click', () => {
        if (_analyzeAbortController) {
            _analyzeAbortController.abort();
        }
    });
    
    // Evidence panel toggle
    elements.evidenceToggle.addEventListener('click', () => {
        const body = elements.evidenceBody;
        const icon = document.querySelector('.evidence-toggle-icon');
        body.classList.toggle('hidden');
        icon.textContent = body.classList.contains('hidden') ? '▸' : '▾';
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/x-matroska', 'video/quicktime', 'video/webm'];
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'flv'];
    
    if (!validExts.includes(ext)) {
        showToast('error', 'Unsupported file format. Please use MP4, AVI, MKV, MOV, or WEBM.');
        return;
    }
    
    state.selectedFile = file;
    
    // Update UI
    elements.fileName.textContent = file.name;
    elements.fileMeta.textContent = formatFileSize(file.size);
    
    elements.uploadZone.classList.add('hidden');
    elements.filePreview.classList.remove('hidden');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function updateQualityUI() {
    // Update visual state of quality buttons
    const buttons = document.querySelectorAll('.quality-btn');
    buttons.forEach(btn => {
        const quality = btn.getAttribute('data-quality');
        if (quality === state.clipQuality) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// Tracking variables for active analysis
let _analyzeAbortController = null;
let _elapsedInterval = null;

async function analyzeClip() {
    if (!state.selectedFile || state.isProcessing) return;
    
    state.isProcessing = true;
    _analyzeAbortController = new AbortController();
    
    // Show processing
    elements.filePreview.classList.add('hidden');
    elements.processing.classList.remove('hidden');
    elements.longWaitMsg.classList.add('hidden');
    elements.elapsedTimer.textContent = '0.0s elapsed';
     // Live elapsed timer + time-based stage messages + pipeline stage visualization
    const startedAt = performance.now();
    let progress = 0;
    const stageOrder = ['upload', 'extract', 'hash', 'match', 'verify', 'score'];
    
    // Reset pipeline stages
    document.querySelectorAll('.pipeline-stage').forEach(s => {
        s.classList.remove('active', 'completed');
    });
    document.querySelectorAll('.pipeline-connector').forEach(c => {
        c.classList.remove('completed');
    });
    const firstStage = document.querySelector('.pipeline-stage[data-stage="upload"]');
    if (firstStage) firstStage.classList.add('active');
    
    function activatePipelineStage(stageName) {
        const stageIdx = stageOrder.indexOf(stageName);
        if (stageIdx < 0) return;
        document.querySelectorAll('.pipeline-stage').forEach((s, i) => {
            const si = stageOrder.indexOf(s.dataset.stage);
            if (si < stageIdx) {
                s.classList.remove('active');
                s.classList.add('completed');
            } else if (si === stageIdx) {
                s.classList.add('active');
                s.classList.remove('completed');
            } else {
                s.classList.remove('active', 'completed');
            }
        });
        // Mark connectors
        document.querySelectorAll('.pipeline-connector').forEach((c, i) => {
            if (i < stageIdx) {
                c.classList.add('completed');
            } else {
                c.classList.remove('completed');
            }
        });
    }
    
    _elapsedInterval = setInterval(() => {
        const elapsed = (performance.now() - startedAt) / 1000;
        elements.elapsedTimer.textContent = elapsed.toFixed(1) + 's elapsed';
        
        // Time-based stage messages + pipeline stage visualization
        if (elapsed < 3) {
            elements.processingStatus.textContent = 'Uploading clip to server...';
            activatePipelineStage('upload');
            progress = Math.min(10, progress + 2);
        } else if (elapsed < 6) {
            elements.processingStatus.textContent = 'Extracting frames from clip...';
            activatePipelineStage('extract');
            progress = Math.min(25, progress + 1.5);
        } else if (elapsed < 10) {
            elements.processingStatus.textContent = 'Computing perceptual hashes...';
            activatePipelineStage('hash');
            progress = Math.min(40, progress + 1.5);
        } else if (elapsed < 25) {
            if (state.clipQuality === 'edited') {
                elements.processingStatus.textContent = 'Deep matching in progress...';
                activatePipelineStage('verify');
            } else {
                elements.processingStatus.textContent = 'Comparing against reference library...';
                activatePipelineStage('match');
            }
            progress = Math.min(75, progress + 0.6);
        } else {
            elements.processingStatus.textContent = 'Scoring candidates & ranking results...';
            activatePipelineStage('score');
            progress = Math.min(92, progress + 0.2);
        }
        elements.progressBar.style.width = progress + '%';
    }, 200);
    
    try {
        const formData = new FormData();
        formData.append('clip', state.selectedFile);
        formData.append('clip_quality', state.clipQuality);
        
        const response = await fetch(`${API_BASE}/match`, {
            method: 'POST',
            body: formData,
            signal: _analyzeAbortController.signal
        });
        
        const data = await response.json();
        
        clearInterval(_elapsedInterval);
        _elapsedInterval = null;
        elements.progressBar.style.width = '100%';
        
        await new Promise(resolve => setTimeout(resolve, 400));
        
        displayResults(data);
        
    } catch (error) {
        clearInterval(_elapsedInterval);
        _elapsedInterval = null;
        if (error.name === 'AbortError') {
            showToast('info', 'Analysis cancelled.');
        } else {
            console.error('Analysis error:', error);
            showToast('error', 'Analysis failed. Please try again.');
        }
        resetMatchPage();
    }
    
    // Cleanup
    finally {
        state.isProcessing = false;
        _analyzeAbortController = null;
    }
}

function displayResults(data) {
    elements.processing.classList.add('hidden');
    elements.results.classList.remove('hidden');
    
    if (!data.success) {
        // Handle API error
        elements.matchResult.classList.add('hidden');
        elements.noMatch.classList.remove('hidden');
        
        if (data.error) {
            console.error('API Error:', data.error);
            // Show specific error message
            if (data.error.includes('No reference videos')) {
                showToast('error', 'No reference videos indexed. Upload videos to the library first.');
            } else if (data.error.includes('too')) {
                showToast('error', 'File size or duration issue: ' + data.error);
            } else {
                showToast('error', data.error);
            }
        }
        return;
    }
    
    elements.processingTime.textContent = `Processed in ${data.processing_time?.toFixed(2) || '?'}s`;
    
    if (data.best_match) {
        const match = data.best_match;
        
        elements.matchResult.classList.remove('hidden');
        elements.noMatch.classList.add('hidden');
        
        // Animate confidence meter
        animateConfidence(match.confidence);
        
        // Update details
        elements.matchedVideo.textContent = match.video_title || match.video_filename;
        elements.timestampRange.textContent = match.timestamp_formatted || '—';
        elements.similarityScore.textContent = (match.avg_similarity || 0).toFixed(1) + '%';
        
        // Populate evidence panel
        const matchRatio = match.match_ratio ?? match.hash_match_ratio;
        elements.evHashRatio.textContent = matchRatio != null ? (matchRatio * 100).toFixed(1) + '%' : '—';
        elements.evColorRatio.textContent = match.color_match_ratio != null ? (match.color_match_ratio * 100).toFixed(1) + '%' : '—';
        elements.evAvgDist.textContent = match.avg_distance != null ? match.avg_distance + ' bits' : '—';
        elements.evMinDist.textContent = match.min_distance != null ? match.min_distance + ' bits' : '—';
        elements.evFrames.textContent = (data.query?.frames_analyzed || match.clip_frames_analyzed || '—');
        // Show user-friendly method name (no technical jargon)
        const method = match.verification_method;
        if (method === 'NCC+SSIM') {
            elements.evMethod.textContent = 'Deep Pixel Analysis';
        } else {
            elements.evMethod.textContent = 'Perceptual Hashing';
        }
        
        // Pixel & structural match evidence (show only when available)
        if (match.ncc_score != null || match.ssim_score != null) {
            elements.evidenceNccGrid.classList.remove('hidden');
            elements.evNcc.textContent = match.ncc_score != null ? (match.ncc_score * 100).toFixed(1) + '%' : '—';
            elements.evSsim.textContent = match.ssim_score != null ? (match.ssim_score * 100).toFixed(1) + '%' : '—';
            if (match.ncc_score != null) {
                elements.evNcc.className = 'evidence-value ncc-value ' + (match.ncc_score >= 0.8 ? 'score-high' : match.ncc_score >= 0.5 ? 'score-mid' : 'score-low');
            }
            if (match.ssim_score != null) {
                elements.evSsim.className = 'evidence-value ssim-value ' + (match.ssim_score >= 0.8 ? 'score-high' : match.ssim_score >= 0.5 ? 'score-mid' : 'score-low');
            }
        } else {
            elements.evidenceNccGrid.classList.add('hidden');
        }
        
        // Candidate stats
        const stages = data.pipeline_stages;
        if (stages && (stages.candidates_accepted != null || stages.candidates_rejected != null)) {
            const accepted = stages.candidates_accepted ?? 0;
            const rejected = stages.candidates_rejected ?? 0;
            const searched = stages.references_searched ?? 0;
            elements.evCandidates.textContent = `${accepted} accepted / ${rejected} rejected (of ${searched} refs)`;
            elements.evidenceCandidates.classList.remove('hidden');
        } else {
            elements.evidenceCandidates.classList.add('hidden');
        }
        
        // Populate timing bars
        if (stages) {
            elements.evidenceTiming.classList.remove('hidden');
            const total = stages.total_sec || 1;
            elements.timingBars.innerHTML = '';
            
            // Map stage keys to readable labels
            const stageLabels = {
                'frame_extraction_sec': 'Frame Extraction',
                'feature_extraction_sec': 'Feature Analysis',
                'hash_precompute_sec': 'Signature Prep',
                'db_retrieval_sec': 'Library Loading',
                'matching_sec': 'Comparison',
                'ncc_verification_sec': 'Deep Verification',
                'confidence_calculation_sec': 'Scoring',
            };
            
            const timingEntries = Object.entries(stages)
                .filter(([key, val]) => key.endsWith('_sec') && key !== 'total_sec' && val != null && stageLabels[key])
                .map(([key, val]) => ({ label: stageLabels[key], sec: val }));
            
            timingEntries.forEach(entry => {
                const pct = Math.max(2, (entry.sec / total) * 100);
                const bar = document.createElement('div');
                bar.className = 'timing-bar-row';
                bar.innerHTML = `
                    <span class="timing-label">${entry.label}</span>
                    <div class="timing-track">
                        <div class="timing-fill" style="width:${pct}%"></div>
                    </div>
                    <span class="timing-value">${entry.sec.toFixed(2)}s</span>
                `;
                elements.timingBars.appendChild(bar);
            });
        } else {
            elements.evidenceTiming.classList.add('hidden');
        }
        
        // Reset evidence panel to collapsed state
        elements.evidenceBody.classList.add('hidden');
        const toggleIcon = document.querySelector('.evidence-toggle-icon');
        if (toggleIcon) toggleIcon.textContent = '▸';
        
    } else {
        elements.matchResult.classList.add('hidden');
        elements.noMatch.classList.remove('hidden');
    }
}

function animateConfidence(value) {
    const duration = 1500;
    const start = 0;
    const startTime = performance.now();
    
    // Determine color class
    let colorClass = 'weak';
    if (value >= 75) colorClass = 'very-strong';
    else if (value >= 50) colorClass = 'strong';
    else if (value >= 25) colorClass = 'possible';
    else if (value >= 10) colorClass = 'weak';
    else colorClass = 'unlikely';
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentValue = Math.round(start + (value - start) * easeOut);
        
        elements.confidenceValue.textContent = currentValue;
        
        // Update meter fill (arc from 0 to 251 stroke-dashoffset)
        const dashOffset = 251 - (251 * (currentValue / 100));
        elements.meterFill.style.strokeDashoffset = dashOffset;
        elements.meterFill.className = 'meter-fill ' + colorClass;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            // Set final label
            let label = 'UNLIKELY MATCH';
            if (value >= 75) label = 'VERY STRONG MATCH';
            else if (value >= 50) label = 'STRONG MATCH';
            else if (value >= 25) label = 'POSSIBLE MATCH';
            else if (value >= 10) label = 'WEAK MATCH';
            
            elements.confidenceLabel.textContent = label;
            elements.confidenceLabel.className = 'meter-status confidence-' + colorClass;
        }
    }
    
    requestAnimationFrame(animate);
}

function resetMatchPage() {
    state.selectedFile = null;
    state.isProcessing = false;
    
    // Abort any in-flight request
    if (_analyzeAbortController) {
        _analyzeAbortController.abort();
        _analyzeAbortController = null;
    }
    if (_elapsedInterval) {
        clearInterval(_elapsedInterval);
        _elapsedInterval = null;
    }
    
    elements.clipInput.value = '';
    elements.uploadZone.classList.remove('hidden');
    elements.filePreview.classList.add('hidden');
    elements.processing.classList.add('hidden');
    elements.results.classList.add('hidden');
    elements.progressBar.style.width = '0%';
    elements.meterFill.style.strokeDashoffset = 251;
    elements.confidenceValue.textContent = '0';
    elements.confidenceLabel.textContent = 'ANALYZING';
    elements.confidenceLabel.className = 'meter-status';
    
    // Reset new elements
    elements.elapsedTimer.textContent = '0.0s elapsed';
    elements.longWaitMsg.classList.add('hidden');
    elements.evidenceBody.classList.add('hidden');
}

// ============================================================
// Library Page
// ============================================================

function initLibraryPage() {
    // Upload reference
    elements.uploadReference.addEventListener('click', () => {
        elements.referenceInput.click();
    });
    
    elements.referenceInput.addEventListener('change', handleReferenceUpload);
    
    // Index all button
    elements.indexAllBtn.addEventListener('click', indexDirectory);
    
    // Modal close
    elements.modalClose.addEventListener('click', closeModal);
    document.querySelector('.modal-overlay')?.addEventListener('click', closeModal);
}

async function loadLibrary() {
    try {
        const response = await fetch(`${API_BASE}/references`);
        const data = await response.json();
        
        if (data.success) {
            state.references = data.videos;
            
            // Update stats
            elements.statVideos.textContent = data.stats.indexed_videos;
            elements.statFrames.textContent = formatNumber(data.stats.total_frames);
            elements.statDuration.textContent = data.stats.total_duration_formatted;
            
            renderLibrary();
        }
    } catch (error) {
        showToast('error', 'Failed to load library');
    }
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function renderLibrary() {
    if (state.references.length === 0) {
        elements.emptyLibrary.classList.remove('hidden');
        return;
    }
    
    elements.emptyLibrary.classList.add('hidden');
    
    // Remove old cards
    document.querySelectorAll('.video-card').forEach(el => el.remove());
    
    state.references.forEach(video => {
        const card = createVideoCard(video);
        elements.libraryList.appendChild(card);
    });
}

function createVideoCard(video) {
    const card = document.createElement('div');
    card.className = 'video-card';
    card.innerHTML = `
        <div class="video-icon">🎬</div>
        <div class="video-info">
            <div class="video-title">${escapeHtml(video.title || video.filename)}</div>
            <div class="video-meta">
                <span>⏱ ${video.duration_formatted}</span>
                <span>📐 ${video.resolution}</span>
                <span>🖼 ${formatNumber(video.frame_count)} frames</span>
            </div>
        </div>
        <span class="video-status ${video.status}">${video.status.toUpperCase()}</span>
        <div class="video-actions">
            <button class="btn-video-action delete" data-id="${video.id}" title="Remove">🗑</button>
        </div>
    `;
    
    // Delete button
    card.querySelector('.delete').addEventListener('click', () => deleteVideo(video.id));
    
    return card;
}

async function handleReferenceUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'flv'];
    
    if (!validExts.includes(ext)) {
        showToast('error', 'Unsupported file format');
        return;
    }
    
    // Show modal
    elements.uploadModal.classList.remove('hidden');
    elements.indexFileName.textContent = file.name;
    elements.indexStatus.textContent = 'Uploading and indexing...';
    elements.indexBarFill.style.width = '0%';
    
    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) progress = 90;
        elements.indexBarFill.style.width = progress + '%';
    }, 500);
    
    try {
        const formData = new FormData();
        formData.append('video', file);
        
        const response = await fetch(`${API_BASE}/references`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        elements.indexBarFill.style.width = '100%';
        
        if (data.success) {
            elements.indexStatus.textContent = `Indexed ${data.frames_indexed} frames`;
            showToast('success', 'Video indexed successfully');
            
            setTimeout(() => {
                closeModal();
                loadLibrary();
            }, 1500);
        } else {
            elements.indexStatus.textContent = 'Error: ' + data.error;
            showToast('error', data.error);
        }
        
    } catch (error) {
        clearInterval(progressInterval);
        elements.indexStatus.textContent = 'Upload failed';
        showToast('error', 'Failed to upload video');
    }
    
    elements.referenceInput.value = '';
}

async function indexDirectory() {
    showToast('info', 'Indexing all videos in references directory...');
    
    try {
        const response = await fetch(`${API_BASE}/references/index-directory`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            const indexed = data.indexed?.length || 0;
            const skipped = data.skipped?.length || 0;
            const failed = data.failed?.length || 0;
            
            showToast('success', `Indexed: ${indexed}, Skipped: ${skipped}, Failed: ${failed}`);
            loadLibrary();
        } else {
            showToast('error', data.error);
        }
    } catch (error) {
        showToast('error', 'Failed to index directory');
    }
}

async function deleteVideo(id) {
    if (!confirm('Remove this video from the index?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/references/${id}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('success', 'Video removed');
            loadLibrary();
        } else {
            showToast('error', data.error);
        }
    } catch (error) {
        showToast('error', 'Failed to remove video');
    }
}

function closeModal() {
    elements.uploadModal.classList.add('hidden');
}

// ============================================================
// History Page
// ============================================================

async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/match/history`);
        const data = await response.json();
        
        if (data.success) {
            state.history = data.history;
            renderHistory();
        }
    } catch (error) {
        showToast('error', 'Failed to load history');
    }
}

function renderHistory() {
    if (state.history.length === 0) {
        elements.emptyHistory.classList.remove('hidden');
        return;
    }
    
    elements.emptyHistory.classList.add('hidden');
    
    // Remove old items
    document.querySelectorAll('.history-item').forEach(el => el.remove());
    
    state.history.forEach(item => {
        const el = createHistoryItem(item);
        elements.historyList.appendChild(el);
    });
}

function createHistoryItem(item) {
    const confidence = item.confidence_score;
    let colorClass = 'weak';
    if (confidence >= 80) colorClass = 'very-strong';
    else if (confidence >= 60) colorClass = 'strong';
    else if (confidence >= 30) colorClass = 'possible';
    
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
        <div class="history-confidence ${colorClass}">
            <span class="confidence-number">${Math.round(confidence)}</span>
            <span class="confidence-label-small">CCS</span>
        </div>
        <div class="history-details">
            <div class="history-query">${escapeHtml(item.query_filename)}</div>
            <div class="history-match">${item.matched_video ? 'Matched: ' + escapeHtml(item.matched_video.title || item.matched_video.filename) : 'No match found'}</div>
            <div class="history-meta">
                ${item.timestamp_range || ''} • ${formatDate(item.queried_at)}
            </div>
        </div>
    `;
    
    return div;
}

function formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ============================================================
// Toast Notifications
// ============================================================

function showToast(type, message) {
    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type]}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
        <button class="toast-close">✕</button>
    `;
    
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

// ============================================================
// Utilities
// ============================================================

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// Initialize
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initMatchPage();
    initLibraryPage();
    
    // Check API health
    fetch(`${API_BASE}/health`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'healthy') {
                console.log('ClipMatch API connected');
            }
        })
        .catch(() => {
            showToast('error', 'Cannot connect to ClipMatch API. Make sure the server is running.');
        });
});

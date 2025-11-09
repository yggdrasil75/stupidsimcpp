let streamInterval = null;
let isStreaming = false;
let frameCount = 0;
let lastFrameTime = 0;
let currentFPS = 0;

const canvas = document.getElementById('streamCanvas');
const ctx = canvas.getContext('2d');
const fpsCounter = document.getElementById('fpsCounter');
const frameCounter = document.getElementById('frameCounter');

function toggleStream() {
    const button = document.getElementById('streamBtn');
    
    if (isStreaming) {
        stopStream();
        button.textContent = 'Start Stream';
        button.classList.remove('danger');
        updateStatus('Stream stopped');
    } else {
        startStream();
        button.textContent = 'Stop Stream';
        button.classList.add('danger');
        updateStatus('Stream started');
    }
}

function startStream() {
    isStreaming = true;
    frameCount = 0;
    lastFrameTime = performance.now();
    
    // Start the stream loop
    streamInterval = setInterval(fetchFrame, 1000 / 30); // Start with 30 FPS
}

function stopStream() {
    isStreaming = false;
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#333';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Stream Stopped', canvas.width / 2, canvas.height / 2);
}

async function fetchFrame() {
    if (!isStreaming) return;
    
    try {
        const response = await fetch('/api/live-stream');
        const data = await response.json();
        
        if (data.frame_available) {
            // In a real implementation, you'd decode the frame data and draw it
            // For now, we'll simulate by generating a pattern based on time
            drawSimulatedFrame();
            updateFrameCounter();
        }
    } catch (error) {
        console.error('Error fetching frame:', error);
        updateStatus('Error fetching frame', 'error');
    }
}

function updateFrameCounter() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    
    if (elapsed >= 1000) {
        currentFPS = Math.round((frameCount * 1000) / elapsed);
        fpsCounter.textContent = `${currentFPS} FPS`;
        frameCounter.textContent = `Frame: ${frameCount}`;
        frameCount = 0;
        lastFrameTime = now;
    }
}

function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusEl.textContent = '';
        statusEl.className = 'status';
    }, 5000);
}

function showStats() {
    fetch('/api/timing-stats')
        .then(response => response.json())
        .then(data => {
            displayStats(data);
            document.getElementById('statsPanel').style.display = 'block';
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
            updateStatus('Error loading stats', 'error');
        });
}

function hideStats() {
    document.getElementById('statsPanel').style.display = 'none';
}

function displayStats(stats) {
    const statsContent = document.getElementById('statsContent');
    
    if (stats.length === 0) {
        statsContent.innerHTML = '<p>No timing data available.</p>';
        return;
    }
    
    let html = '<table class="stats-table">';
    html += '<tr><th>Function</th><th>Calls</th><th>Total (s)</th><th>Avg (s)</th><th>Min (s)</th><th>Max (s)</th><th>Median (s)</th><th>P90 (s)</th><th>P95 (s)</th><th>P99 (s)</th></tr>';
    
    stats.forEach(stat => {
        html += `<tr>
            <td>${stat.function}</td>
            <td>${stat.call_count}</td>
            <td>${parseFloat(stat.total_time).toFixed(6)}</td>
            <td>${parseFloat(stat.avg_time).toFixed(6)}</td>
            <td>${parseFloat(stat.min_time).toFixed(6)}</td>
            <td>${parseFloat(stat.max_time).toFixed(6)}</td>
            <td>${parseFloat(stat.median_time).toFixed(6)}</td>
            <td>${parseFloat(stat.p90_time).toFixed(6)}</td>
            <td>${parseFloat(stat.p95_time).toFixed(6)}</td>
            <td>${parseFloat(stat.p99_time).toFixed(6)}</td>
        </tr>`;
    });
    
    html += '</table>';
    statsContent.innerHTML = html;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Setup sliders
    setupSlider('fps', 'fpsValue', 30);
    setupSlider('quality', 'qualityValue', 85);
    
    // Show stopped state
    stopStream();
});
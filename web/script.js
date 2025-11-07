let autoRefreshInterval = null;
let currentMode = 'gradient';

function refreshImage() {
    const img = document.getElementById('displayImage');
    
    // Check if server has finished encoding
    fetch('/api/image-ready')
        .then(response => response.json())
        .then(data => {
            if (data.ready) {
                performSmoothImageSwap();
            } else {
                // Try again in 1 second
                setTimeout(refreshImage, 1000);
            }
        })
        .catch(error => {
            console.error('Error checking image status:', error);
            updateStatus('Error checking image status', 'error');
        });
}

function performSmoothImageSwap() {
    const img = document.getElementById('displayImage');
    const newImage = new Image();
    
    newImage.onload = function() {
        // Crossfade transition
        img.style.opacity = '0';
        setTimeout(() => {
            img.src = './output/display.jxl';
            img.style.opacity = '1';
            updateStatus('Image refreshed at ' + new Date().toLocaleTimeString());
        }, 300);
    };
    
    newImage.onerror = function() {
        updateStatus('Error loading image', 'error');
    };
    
    // Load with cache busting only when we know it's ready
    newImage.src = './output/display.jxl?' + new Date().getTime();
}

function refreshTerrain() {
    fetch('/api/refresh-terrain', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                refreshImage();
            } else {
                updateStatus('Error refreshing terrain', 'error');
            }
        })
        .catch(error => {
            console.error('Error refreshing terrain:', error);
            updateStatus('Error refreshing terrain', 'error');
        });
}

function toggleAutoRefresh() {
    const button = document.getElementById('autoRefreshBtn');
    
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        button.textContent = 'Start Auto-Refresh';
        button.classList.remove('danger');
        updateStatus('Auto-refresh stopped');
    } else {
        // Faster refresh for terrain (5 seconds)
        autoRefreshInterval = setInterval(currentMode === 'terrain' ? refreshTerrain : refreshImage, 100);
        button.textContent = 'Stop Auto-Refresh';
        button.classList.add('danger');
        updateStatus('Auto-refresh started');
    }
}

function switchMode() {
    fetch('/api/switch-mode', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                currentMode = data.mode;
                updateModeDisplay();
                refreshImage();
                updateStatus('Switched to ' + data.mode + ' mode');
                
                // Restart auto-refresh if active
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = setInterval(currentMode === 'terrain' ? refreshTerrain : refreshImage, 5000);
                }
            } else {
                updateStatus('Error switching mode', 'error');
            }
        })
        .catch(error => {
            console.error('Error switching mode:', error);
            updateStatus('Error switching mode', 'error');
        });
}

function updateModeDisplay() {
    const modeDisplay = document.getElementById('modeDisplay');
    const switchBtn = document.getElementById('switchModeBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    
    if (modeDisplay) {
        modeDisplay.textContent = 'Current Mode: ' + currentMode;
    }
    
    if (switchBtn) {
        switchBtn.textContent = 'Switch to ' + (currentMode === 'gradient' ? 'Terrain' : 'Gradient');
    }
    
    if (refreshBtn) {
        if (currentMode === 'terrain') {
            refreshBtn.textContent = 'Refresh Terrain';
            refreshBtn.onclick = refreshTerrain;
        } else {
            refreshBtn.textContent = 'Refresh Image';
            refreshBtn.onclick = refreshImage;
        }
    }
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

function clearStats() {
    fetch('/api/clear-stats', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            updateStatus('Statistics cleared');
            hideStats();
        })
        .catch(error => {
            console.error('Error clearing stats:', error);
            updateStatus('Error clearing stats', 'error');
        });
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're in all mode and get current mode
    fetch('/api/current-mode')
        .then(response => response.json())
        .then(data => {
            if (data.mode) {
                currentMode = data.mode;
                updateModeDisplay();
            }
            refreshImage();
        })
        .catch(error => {
            // If endpoint doesn't exist, we're not in all mode
            console.log('Not in all mode, using default');
            refreshImage();
        });
});

function toggleParameters() {
    const panel = document.getElementById('parameterPanel');
    const btn = document.getElementById('paramsBtn');
    
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.textContent = 'Hide Parameters';
        loadCurrentParameters();
    } else {
        panel.style.display = 'none';
        btn.textContent = 'Show Parameters';
    }
}

function loadCurrentParameters() {
    // This would ideally fetch current parameters from the server
    // For now, we'll just initialize the sliders with their current values
    setupSlider('scale', 'scaleValue', 4.0);
    setupSlider('octaves', 'octavesValue', 4);
    setupSlider('persistence', 'persistenceValue', 0.5);
    setupSlider('lacunarity', 'lacunarityValue', 2.0);
    setupSlider('elevation', 'elevationValue', 1.0);
    setupSlider('waterLevel', 'waterLevelValue', 0.3);
}

function setupSlider(sliderId, valueId, defaultValue) {
    const slider = document.getElementById(sliderId);
    const value = document.getElementById(valueId);
    
    slider.value = defaultValue;
    value.textContent = defaultValue;
    
    slider.addEventListener('input', function() {
        value.textContent = this.value;
    });
}

function applyParameters() {
    const params = {
        scale: parseFloat(document.getElementById('scale').value),
        octaves: parseInt(document.getElementById('octaves').value),
        persistence: parseFloat(document.getElementById('persistence').value),
        lacunarity: parseFloat(document.getElementById('lacunarity').value),
        elevation: parseFloat(document.getElementById('elevation').value),
        waterLevel: parseFloat(document.getElementById('waterLevel').value),
        seed: parseInt(document.getElementById('seed').value)
    };
    
    console.log("Sending parameters:", params);  // Debug log
    
    fetch('/api/set-parameters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        console.log("Server response:", data);  // Debug log
        if (data.status === 'success') {
            updateStatus('Parameters applied successfully');
            if (currentMode === 'terrain') {
                refreshTerrain();
            }
        } else {
            updateStatus('Error applying parameters', 'error');
        }
    })
    .catch(error => {
        console.error('Error applying parameters:', error);
        updateStatus('Error applying parameters', 'error');
    });
}

function resetParameters() {
    document.getElementById('scale').value = 4.0;
    document.getElementById('scaleValue').textContent = '4.0';
    
    document.getElementById('octaves').value = 4;
    document.getElementById('octavesValue').textContent = '4';
    
    document.getElementById('persistence').value = 0.5;
    document.getElementById('persistenceValue').textContent = '0.5';
    
    document.getElementById('lacunarity').value = 2.0;
    document.getElementById('lacunarityValue').textContent = '2.0';
    
    document.getElementById('elevation').value = 1.0;
    document.getElementById('elevationValue').textContent = '1.0';
    
    document.getElementById('waterLevel').value = 0.3;
    document.getElementById('waterLevelValue').textContent = '0.3';
    
    document.getElementById('seed').value = 42;
    
    updateStatus('Parameters reset to defaults');
}

function randomizeSeed() {
    const newSeed = Math.floor(Math.random() * 1000000);
    document.getElementById('seed').value = newSeed;
    updateStatus('Random seed generated: ' + newSeed);
}
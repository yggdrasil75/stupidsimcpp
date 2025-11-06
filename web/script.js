let autoRefreshInterval = null;

function refreshImage() {
    const img = document.getElementById('gradientImage');
    img.classList.add('loading');
    
    const timestamp = new Date().getTime();
    //img.src = 'gradient.jxl?' + timestamp;
    img.src = './output/gradient.jxl';
    
    img.onload = function() {
        img.classList.remove('loading');
        updateStatus('Image refreshed at ' + new Date().toLocaleTimeString());
    };
    
    img.onerror = function() {
        img.classList.remove('loading');
        updateStatus('Error loading image', 'error');
    };
}

function toggleAutoRefresh() {
    const button = document.getElementById('autoRefreshBtn');
    
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        button.textContent = 'Start Auto-Refresh (30s)';
        button.classList.remove('danger');
        updateStatus('Auto-refresh stopped');
    } else {
        autoRefreshInterval = setInterval(refreshImage, 30000);
        button.textContent = 'Stop Auto-Refresh';
        button.classList.add('danger');
        updateStatus('Auto-refresh started (30s interval)');
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

// Auto-refresh when page loads
document.addEventListener('DOMContentLoaded', function() {
    refreshImage();
});
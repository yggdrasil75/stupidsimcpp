let autoRefreshInterval = null;

function refreshImage() {
    const img = document.getElementById('gradientImage');
    img.classList.add('loading');
    
    const timestamp = new Date().getTime();
    img.src = 'gradient.jxl?' + timestamp;
    
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


// Auto-refresh when page loads
document.addEventListener('DOMContentLoaded', function() {
    refreshImage();
});
// ============================================
// STADIUM OVERRIDE - Load from localStorage BEFORE any initialization
// ============================================
console.log('🏟️ Checking for custom stadium in localStorage...');

// Check if custom stadium exists
if (localStorage.getItem('customStadium')) {
    try {
        var customStadiumJSON = localStorage.getItem('customStadium');
        window.customStadiumOverride = JSON.parse(customStadiumJSON);
        console.log(`✅ Custom stadium loaded: "${window.customStadiumOverride.name}"`);
        console.log('Stadium properties:', {
            dimensions: `${window.customStadiumOverride.width}x${window.customStadiumOverride.height}`,
            traits: window.customStadiumOverride.traits ? Object.keys(window.customStadiumOverride.traits).length : 0,
            vertices: window.customStadiumOverride.vertexes ? window.customStadiumOverride.vertexes.length : 0,
            segments: window.customStadiumOverride.segments ? window.customStadiumOverride.segments.length : 0,
            planes: window.customStadiumOverride.planes ? window.customStadiumOverride.planes.length : 0,
            discs: window.customStadiumOverride.discs ? window.customStadiumOverride.discs.length : 0,
            goals: window.customStadiumOverride.goals ? window.customStadiumOverride.goals.length : 0,
            ballPhysics: !!window.customStadiumOverride.ballPhysics,
            playerPhysics: !!window.customStadiumOverride.playerPhysics
        });
    } catch (e) {
        console.error('❌ Failed to parse custom stadium:', e);
        window.customStadiumOverride = null;
    }
} else {
    console.log('ℹ️ No custom stadium found - using Classic');
    window.customStadiumOverride = null;
}

// This global variable will be checked by script.js to override default stadium initialization

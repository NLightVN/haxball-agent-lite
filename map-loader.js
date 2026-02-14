// Custom Stadium Map Loader for Haxball Agent Lite
// Loads .hbs (Haxball Stadium) files and applies physics automatically

console.log('🗺️ Map Loader module loading...');

/**
 * Load custom map from file
 * @param {File} file - The .hbs file to load
 */
function loadCustomMap(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        try {
            const mapData = JSON.parse(e.target.result);
            console.log(`📥 Parsing map: ${mapData.name || 'Unnamed'}`);
            validateAndApplyMap(mapData);
        } catch (error) {
            console.error('❌ Invalid HBS file:', error);
            alert('Error loading map. Please check file format.\n\n' + error.message);
        }
    };
    reader.onerror = function () {
        console.error('❌ Error reading file');
        alert('Failed to read file. Please try again.');
    };
    reader.readAsText(file);
}

/**
 * Validate and apply map data to the game
 * @param {Object} mapData - Parsed HBS map data
 */
function validateAndApplyMap(mapData) {
    // Validate required fields
    if (!mapData.name) {
        throw new Error('Map must have a "name" field');
    }
    if (!mapData.width || !mapData.height) {
        throw new Error('Map must have "width" and "height" fields');
    }
    if (!mapData.vertexes || !Array.isArray(mapData.vertexes)) {
        throw new Error('Map must have "vertexes" array');
    }
    if (!mapData.segments || !Array.isArray(mapData.segments)) {
        throw new Error('Map must have "segments" array');
    }

    console.log(`✅ Map validation passed: ${mapData.name}`);

    // CRITICAL: Save to localStorage and reload page
    // This is necessary because script.js caches stadium structure on init
    // Simply changing the global variable doesn't update the cached data
    console.log('💾 Saving stadium to localStorage...');
    localStorage.setItem('customStadium', JSON.stringify(mapData));

    // Apply custom physics to localStorage as well
    if (mapData.ballPhysics) {
        localStorage.setItem('customBallPhysics', JSON.stringify(mapData.ballPhysics));
    }
    if (mapData.playerPhysics) {
        localStorage.setItem('customPlayerPhysics', JSON.stringify(mapData.playerPhysics));
    }

    console.log('🔄 Reloading page to apply new stadium...');

    // Reload page to re-initialize with new stadium
    location.reload();
}

/**
 * Process stadium structure (already exists in script.js, but ensure it's called)
 */
function processStadiumStructure(std) {
    // Apply traits to discs/segments
    if (std.traits) {
        // Process discs with traits
        if (std.discs) {
            std.discs.forEach(disc => {
                if (disc.trait && std.traits[disc.trait]) {
                    Object.assign(disc, std.traits[disc.trait]);
                }
            });
        }

        // Process segments with traits
        if (std.segments) {
            std.segments.forEach(seg => {
                if (seg.trait && std.traits[seg.trait]) {
                    Object.assign(seg, std.traits[seg.trait]);
                }
            });
        }

        // Process vertexes with traits
        if (std.vertexes) {
            std.vertexes.forEach(vtx => {
                if (vtx.trait && std.traits[vtx.trait]) {
                    Object.assign(vtx, std.traits[vtx.trait]);
                }
            });
        }
    }

    // Ensure all vertexes have x, y
    std.vertexes.forEach((v, i) => {
        if (v.x === undefined || v.y === undefined) {
            console.warn(`Vertex ${i} missing coordinates, setting to (0, 0)`);
            v.x = v.x || 0;
            v.y = v.y || 0;
        }
    });

    // Process segments (collision transformation, curves)
    std.segments.forEach(seg => {
        collisionTransformation(seg, std.vertexes);
        if (seg.curve !== undefined) {
            getCurveFSegment(seg);
        }
        getStuffSegment(seg);
    });

    // Process stadium discs (goals, etc.)
    if (std.discs) {
        std.discs.forEach(disc => {
            collisionTransformation(disc);
        });
    }

    // Process goals
    if (std.goals) {
        std.goals.forEach(goal => {
            if (typeof goal.team === 'string') {
                goal.team = getTeamByID(
                    goal.team === 'red' ? 1 : goal.team === 'blue' ? 2 : 0
                );
            }
        });
    }

    console.log('✅ Stadium structure processed');
}

/**
 * Reset game state with new map
 */
function resetGameWithNewMap() {
    console.log('🔄 Resetting game with new map...');

    // IMPORTANT: Store current players before clearing
    const currentPlayers = playersArray.slice();
    console.log(`👥 Preserving ${currentPlayers.length} players`);

    // Clear old discs
    discs = [];

    // Recreate ball at center with FULL initialization
    var ballDisc = new Disc();
    ballDisc.ballPhysics(); // Sets default ball physics
    ballDisc.x = 0;
    ballDisc.y = 0;
    ballDisc.xspeed = 0;
    ballDisc.yspeed = 0;

    // CRITICAL: Set collision flags so ball is visible and interacts
    ballDisc.cMask = haxball.collisionFlags.all;
    ballDisc.cGroup = haxball.collisionFlags.ball;

    // Apply custom ball physics from stadium if specified
    if (stadium.ballPhysics) {
        if (stadium.ballPhysics.radius) ballDisc.radius = stadium.ballPhysics.radius;
        if (stadium.ballPhysics.bCoef !== undefined) ballDisc.bCoef = stadium.ballPhysics.bCoef;
        if (stadium.ballPhysics.invMass !== undefined) ballDisc.invMass = stadium.ballPhysics.invMass;
        if (stadium.ballPhysics.damping !== undefined) ballDisc.damping = stadium.ballPhysics.damping;
    }

    discs.push(ballDisc);
    console.log(`⚽ Ball created:`, {
        position: `(${ballDisc.x}, ${ballDisc.y})`,
        radius: ballDisc.radius,
        cMask: ballDisc.cMask,
        cGroup: ballDisc.cGroup,
        invMass: ballDisc.invMass
    });

    // Add stadium discs (goal posts, etc.)
    if (stadium.discs) {
        stadium.discs.forEach((discData, i) => {
            if (i > 0) { // Skip ball (already added)
                var disc = new Disc();

                // Handle both pos array [x, y] and x/y object format
                if (discData.pos) {
                    disc.x = discData.pos[0];
                    disc.y = discData.pos[1];
                } else {
                    disc.x = discData.x || 0;
                    disc.y = discData.y || 0;
                }

                disc.radius = discData.radius || 8;
                disc.invMass = discData.invMass !== undefined ? discData.invMass : 0;
                disc.bCoef = discData.bCoef !== undefined ? discData.bCoef : 0.5;
                disc.cMask = discData.cMask || 0;
                disc.cGroup = discData.cGroup || 0;
                disc.xspeed = 0;
                disc.yspeed = 0;
                disc.damping = discData.damping || 0.99;
                discs.push(disc);
            }
        });
    }

    // Reset players with new stadium properties
    console.log(`🔄 Resetting ${currentPlayers.length} players with new map`);
    currentPlayers.forEach(player => {
        // CRITICAL: Clear disc reference so setPlayerDefaultProperties recreates it
        player.disc = null;
        setPlayerDefaultProperties(player);
    });

    // Reset game state
    game.red = 0;
    game.blue = 0;
    game.time = 0;
    game.state = 0;
    resetPositionDiscs();

    // Update score UI
    var scores = document.getElementsByClassName('score');
    if (scores.length >= 2) {
        scores[0].textContent = '0';
        scores[1].textContent = '0';
    }

    // Load background if specified (this will trigger render via callback)
    if (stadium.bg && stadium.bg.type) {
        console.log(`🖼️ Loading background: ${stadium.bg.type}`);
        load_tile(stadium.bg.type);
    } else {
        // No custom background - render immediately
        render(stadium);
    }

    // Force immediate render to ensure new stadium displays
    console.log('🎨 Force rendering new stadium...');
    render(stadium);

    console.log('✅ Game reset complete');
    console.log(`   Stadium: ${stadium.width}x${stadium.height}`);
    console.log(`   Players active: ${playersArray.length}`);
    console.log(`   Discs total: ${discs.length}`);
}

/**
 * Initialize map loader event listeners
 */
/**
 * Initialize map loader event listeners and UI
 */
function initMapLoader() {
    const mapUpload = document.getElementById('map-upload');
    const resetMap = document.getElementById('reset-map');
    const mapNameEl = document.getElementById('current-map-name');

    // Update map name display on load
    var customStadiumData = localStorage.getItem('customStadium');
    if (customStadiumData && mapNameEl) {
        try {
            var customStadium = JSON.parse(customStadiumData);
            mapNameEl.textContent = customStadium.name || 'Custom';
        } catch (e) {
            console.error('Failed to parse custom stadium name:', e);
        }
    }

    if (mapUpload) {
        mapUpload.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                console.log(`📂 Loading file: ${e.target.files[0].name}`);
                loadCustomMap(e.target.files[0]);
                // Clear input so same file can be loaded again
                e.target.value = '';
            }
        });
        console.log('✅ Map upload listener attached');
    } else {
        console.warn('⚠️ Map upload element not found');
    }

    if (resetMap) {
        resetMap.addEventListener('click', function () {
            console.log('🔄 Resetting to Classic map');
            // Clear custom stadium from localStorage
            localStorage.removeItem('customStadium');
            localStorage.removeItem('customBallPhysics');
            localStorage.removeItem('customPlayerPhysics');
            // Reload page
            location.reload();
        });
        console.log('✅ Reset map listener attached');
    } else {
        console.warn('⚠️ Reset map element not found');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMapLoader);
} else {
    initMapLoader();
}

console.log('✅ Map Loader module loaded!');

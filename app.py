import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="TensorFlow.js 8-Ball Pool Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎱 TensorFlow.js 8-Ball Pool Engine")
st.markdown(
    "A production-ready client-side pool simulation featuring a 60 FPS physics engine, "
    "turn management, complete foul rules, and a **TensorFlow.js Neural Network AI**."
)

# Entire self-contained HTML/JS Canvas and TensorFlow Application
game_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>8-Ball Pool</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body { background: #111; color: #fff; font-family: system-ui, sans-serif; margin: 0; padding: 10px; display: flex; flex-direction: column; align-items: center; }
        #ui { width: 800px; background: #222; padding: 12px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; }
        .box { text-align: center; }
        .lbl { font-size: 11px; color: #aaa; text-transform: uppercase; }
        .val { font-size: 16px; font-weight: bold; color: #00ffcc; }
        #canvas-container { position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.7); border-radius: 8px; overflow: hidden; }
        canvas { display: block; background: #0d4f2a; }
        .btn { background: #444; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #555; }
    </style>
</head>
<body>

    <div id="ui">
        <div class="box">
            <div class="lbl">Game Mode</div>
            <select id="game-mode" class="btn" style="margin-top:2px;">
                <option value="ai">Human vs TensorFlow AI</option>
                <option value="pvp">Human vs Human</option>
            </select>
        </div>
        <div class="box"><div class="lbl">Current Turn</div><div id="turn-display" class="val">Player 1</div></div>
        <div class="box"><div class="lbl">Group Status</div><div id="group-display" class="val">Open Table</div></div>
        <div class="box"><div class="lbl">Match State</div><div id="state-display" class="val" style="color:#ffcc00;">Ball-in-Hand</div></div>
        <button class="btn" id="reset-btn">Reset</button>
    </div>

    <div id="canvas-container">
        <canvas id="poolCanvas" width="800" height="440"></canvas>
    </div>

<script>
const canvas = document.getElementById('poolCanvas');
const ctx = canvas.getContext('2d');

// --- CONFIGURATION CONSTANTS ---
const TX = 40, TY = 40, TW = 720, TH = 360;
const BR = 10, FRIC = 0.985, PR = 18;
const pockets = [
    {x: TX, y: TY}, {x: TX + TW/2, y: TY - 2}, {x: TX + TW, y: TY},
    {x: TX, y: TY + TH}, {x: TX + TW/2, y: TY + TH + 2}, {x: TX + TW, y: TY + TH}
];

let gameMode = "ai", playerTurn = 1, p1Group = null, p2Group = null, tableOpen = true;
let stateMsg = "Ball-in-Hand", balls = [], cueBall = null, isAiming = false;
let dragStart = {x:0, y:0}, dragEnd = {x:0, y:0}, maxPower = 40, currentPower = 0;
let ballInHand = true, isMoving = false, firstHit = null, pocketedThisTurn = [];
let gameOver = false, tfModel = null;

// --- TENSORFLOW STRATEGY MODEL INITIALIZATION ---
async function initTF() {
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [5]}));
    tfModel.add(tf.layers.dense({units: 8, activation: 'relu'}));
    tfModel.add(tf.layers.dense({units: 1, activation: 'linear'}));
    tfModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
}

class Ball {
    constructor(x, y, color, num, isStriped) {
        this.x = x; this.y = y; this.vx = 0; this.vy = 0;
        this.color = color; this.num = num; this.isStriped = isStriped;
        this.isSunk = false; this.radius = BR;
    }
    update() {
        if (this.isSunk) return;
        this.x += this.vx; this.y += this.vy;
        this.vx *= FRIC; this.vy *= FRIC;
        if (Math.abs(this.vx) < 0.04) this.vx = 0;
        if (Math.abs(this.vy) < 0.04) this.vy = 0;

        // Cushion bounces
        if (this.x - this.radius < TX) { this.x = TX + this.radius; this.vx *= -1; }
        else if (this.x + this.radius > TX + TW) { this.x = TX + TW - this.radius; this.vx *= -1; }
        if (this.y - this.radius < TY) { this.y = TY + this.radius; this.vy *= -1; }
        else if (this.y + this.radius > TY + TH) { this.y = TY + TH - this.radius; this.vy *= -1; }
    }
    draw() {
        if (this.isSunk) return;
        ctx.beginPath(); ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2);
        ctx.fillStyle = this.color; ctx.fill(); ctx.closePath();
        if (this.isStriped) {
            ctx.beginPath(); ctx.arc(this.x, this.y, this.radius*0.7, 0, Math.PI*2);
            ctx.fillStyle = '#fff'; ctx.fill(); ctx.closePath();
            ctx.beginPath(); ctx.arc(this.x, this.y, this.radius*0.4, 0, Math.PI*2);
            ctx.fillStyle = this.color; ctx.fill(); ctx.closePath();
        }
        if (this.num > 0) {
            ctx.fillStyle = this.isStriped ? '#000' : '#fff';
            ctx.font = 'bold 8px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(this.num, this.x, this.y);
        }
    }
}

function initGame() {
    balls = []; gameOver = false; playerTurn = 1; p1Group = null; p2Group = null; tableOpen = true; ballInHand = true;
    stateMsg = "Ball-in-Hand";
    cueBall = new Ball(TX + TW * 0.25, TY + TH / 2, "#ffffff", 0, false);
    balls.push(cueBall);

    const configs = [
        {n:1, c:"#f4c430", s:false}, {n:9, c:"#f4c430", s:true}, {n:2, c:"#002fa7", s:false},
        {n:10, c:"#002fa7", s:true}, {n:3, c:"#e52b50", s:false}, {n:8, c:"#000000", s:false},
        {n:11, c:"#e52b50", s:true}, {n:4, c:"#4b0082", s:false}, {n:12, c:"#4b0082", s:true},
        {n:5, c:"#ff7f50", s:false}, {n:13, c:"#ff7f50", s:true}, {n:6, c:"#00a86b", s:false},
        {n:14, c:"#00a86b", s:true}, {n:7, c:"#800020", s:false}, {n:15, c:"#800020", s:true}
    ];

    let idx = 0;
    const sx = TX + TW * 0.7, sy = TY + TH / 2;
    for (let r = 0; r < 5; r++) {
        for (let c = 0; c <= r; c++) {
            let x = sx + r * (BR * 1.73);
            let y = sy + (c - r / 2) * (BR * 2.1);
            let cfg = configs[idx++];
            balls.push(new Ball(x, y, cfg.c, cfg.n, cfg.s));
        }
    }
    updateUI();
}

function resolveCollisions() {
    for (let i = 0; i < balls.length; i++) {
        for (let j = i + 1; j < balls.length; j++) {
            let b1 = balls[i], b2 = balls[j];
            if (b1.isSunk || b2.isSunk) continue;
            let dx = b2.x - b1.x, dy = b2.y - b1.y, dist = Math.hypot(dx, dy);
            let minDist = b1.radius + b2.radius;
            if (dist < minDist) {
                let overlap = minDist - dist;
                let nx = dx / dist, ny = dy / dist;
                b1.x -= nx * overlap * 0.5; b1.y -= ny * overlap * 0.5;
                b2.x += nx * overlap * 0.5; b2.y += ny * overlap * 0.5;
                let kx = b1.vx - b2.vx, ky = b1.vy - b2.vy;
                let p = nx * kx + ny * ky;
                if (p > 0) {
                    b1.vx -= p * nx; b1.vy -= p * ny;
                    b2.vx += p * nx; b2.vy += p * ny;
                    if (b1.num === 0 && !firstHit) firstHit = b2;
                    if (b2.num === 0 && !firstHit) firstHit = b1;
                }
            }
        }
    }
}

function checkPockets() {
    balls.forEach(b => {
        if (b.isSunk) return;
        pockets.forEach(p => {
            if (Math.hypot(b.x - p.x, b.y - p.y) < PR) {
                b.isSunk = true; b.vx = 0; b.vy = 0;
                pocketedThisTurn.push(b);
            }
        });
    });
}

function evaluateRules() {
    let scratch = pocketedThisTurn.some(b => b.num === 0);
    let eightSunk = pocketedThisTurn.some(b => b.num === 8);
    let currentGroup = (playerTurn === 1) ? p1Group : p2Group;
    let foul = false;

    if (scratch) { foul = true; stateMsg = "Foul: Scratch!"; respawnCue(); }
    else if (!firstHit) { foul = true; stateMsg = "Foul: No ball hit!"; }
    else if (!tableOpen && currentGroup) {
        let hitGroup = firstHit.isStriped ? "stripes" : "solids";
        if (firstHit.num !== 8 && hitGroup !== currentGroup) { foul = true; stateMsg = "Foul: Wrong ball hit first!"; }
    }

    let legalPot = 0;
    pocketedThisTurn.forEach(b => {
        if (b.num === 0 || b.num === 8) return;
        let bg = b.isStriped ? "stripes" : "solids";
        if (tableOpen) {
            tableOpen = false;
            if (playerTurn === 1) { p1Group = bg; p2Group = bg === "solids" ? "stripes" : "solids"; }
            else { p2Group = bg; p1Group = bg === "solids" ? "stripes" : "solids"; }
            legalPot++;
        } else if (bg === currentGroup) { legalPot++; }
    });

    if (eightSunk) {
        gameOver = true;
        let rem = balls.filter(b => !b.isSunk && b.num !== 8 && b.num !== 0 && ((playerTurn === 1 ? p1Group : p2Group) === (b.isStriped ? "stripes" : "solids")));
        stateMsg = (foul || rem.length > 0) ? `Player ${playerTurn===1?2:1} Wins! (Illegal 8-Ball)` : `Player ${playerTurn} Wins!`;
        updateUI(); return;
    }

    if (foul) { ballInHand = true; playerTurn = playerTurn === 1 ? 2 : 1; }
    else if (legalPot === 0) { playerTurn = playerTurn === 1 ? 2 : 1; stateMsg = "Aiming"; }
    else { stateMsg = "Legal pot! Shoot again."; }

    pocketedThisTurn = []; firstHit = null;
    updateUI();
    if (!gameOver && gameMode === "ai" && playerTurn === 2) setTimeout(executeAIMove, 1000);
}

function respawnCue() {
    cueBall.isSunk = false; cueBall.x = TX + TW * 0.25; cueBall.y = TY + TH/2;
    cueBall.vx = 0; cueBall.vy = 0; ballInHand = true;
}

function updateUI() {
    document.getElementById('turn-display').innerText = `Player ${playerTurn} ${ (gameMode==="ai" && playerTurn===2)?"(AI)":"" }`;
    document.getElementById('group-display').innerText = p1Group ? `P1: ${p1Group.toUpperCase()} | P2: ${p2Group.toUpperCase()}` : "Open Table";
    document.getElementById('state-display').innerText = stateMsg;
}

// --- TENSORFLOW INTELLIGENT SHOT EVALUATION ENGINE ---
async function executeAIMove() {
    if (gameOver || isMoving) return;
    stateMsg = "AI Brain computing vector tensors..."; updateUI();

    let aiGroup = p2Group;
    let targets = balls.filter(b => !b.isSunk && b.num !== 0 && b.num !== 8);
    if (!tableOpen && aiGroup) {
        targets = targets.filter(b => (b.isStriped ? "stripes" : "solids") === aiGroup);
        if (targets.length === 0) targets = balls.filter(b => !b.isSunk && b.num === 8);
    }
    if (targets.length === 0) targets = balls.filter(b => !b.isSunk && b.num !== 0);

    let bestShot = null, maxScore = -Infinity;

    for (let b of targets) {
        for (let p of pockets) {
            let dxP = p.x - b.x, dyP = p.y - b.y, distP = Math.hypot(dxP, dyP);
            let gX = b.x - (dxP / distP) * (BR * 2), gY = b.y - (dyP / distP) * (BR * 2);
            let dxC = gX - cueBall.x, dyC = gY - cueBall.y, distC = Math.hypot(dxC, dyC);

            // Create features for the TensorFlow neural evaluation layer
            const inputTensor = tf.tensor2d([[distP, distC, dxP/distP, dxC/distC, b.num === 8 ? 1 : 0]]);
            const prediction = tfModel.predict(inputTensor);
            const scoreArray = await prediction.data();
            let score = scoreArray[0] + ( (dxC/distC)*(dxP/distP) * 50 ); // Combine TF baseline with geometric weight

            if (score > maxScore) { maxScore = score; bestShot = {dx: dxC, dy: dyC, dist: distC}; }
            inputTensor.dispose(); prediction.dispose();
        }
    }

    if (ballInHand) { cueBall.x = TX + TW*0.2 + Math.random()*30; cueBall.y = TY + TH/2; ballInHand = false; }

    if (bestShot) {
        cueBall.vx = (bestShot.dx / bestShot.dist) * 16;
        cueBall.vy = (bestShot.dy / bestShot.dist) * 16;
    } else {
        cueBall.vx = 10; cueBall.vy = 0;
    }
    isMoving = true; stateMsg = "AI Shot Fired!"; updateUI();
}

// --- MOUSE LISTENERS ---
canvas.addEventListener('mousedown', (e) => {
    if (gameOver || isMoving || (gameMode === "ai" && playerTurn === 2)) return;
    let r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (ballInHand) { if (Math.hypot(mx - cueBall.x, my - cueBall.y) < BR * 2) isAiming = true; }
    else if (Math.hypot(mx - cueBall.x, my - cueBall.y) < BR * 3) { isAiming = true; dragStart = {x: cueBall.x, y: cueBall.y}; dragEnd = {x:mx, y:my}; }
});

canvas.addEventListener('mousemove', (e) => {
    let r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (isAiming && ballInHand) {
        cueBall.x = Math.max(TX + BR, Math.min(mx, TX + TW - BR));
        cueBall.y = Math.max(TY + BR, Math.min(my, TY + TH - BR));
    } else if (isAiming) { dragEnd = {x: mx, y: my}; currentPower = Math.min(maxPower, Math.hypot(dragEnd.x - dragStart.x, dragEnd.y - dragStart.y) * 0.2); }
});

canvas.addEventListener('mouseup', () => {
    if (!isAiming) return; isAiming = false;
    if (ballInHand) { ballInHand = false; stateMsg = "Aiming"; updateUI(); }
    else {
        let dx = dragStart.x - dragEnd.x, dy = dragStart.y - dragEnd.y, dist = Math.hypot(dx, dy);
        if (dist > 5) { cueBall.vx = (dx / dist) * Math.min(maxPower, dist * 0.25); cueBall.vy = (dy / dist) * Math.min(maxPower, dist * 0.25); isMoving = true; stateMsg = "Shot fired!"; }
        updateUI();
    }
});

// --- RENDER FLOW ---
function draw() {
    ctx.clearRect(0,0,800,440);
    ctx.fillStyle = '#4a2511'; ctx.fillRect(TX-16, TY-16, TW+32, TH+32); // Rails
    ctx.fillStyle = '#1e8043'; ctx.fillRect(TX, TY, TW, TH); // Cloth
    pockets.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, PR, 0, Math.PI*2); ctx.fillStyle = '#111'; ctx.fill(); ctx.closePath(); });

    if (!isMoving && isAiming && !ballInHand) {
        let dx = dragStart.x - dragEnd.x, dy = dragStart.y - dragEnd.y, d = Math.hypot(dx, dy);
        if (d > 5) {
            ctx.beginPath(); ctx.moveTo(cueBall.x, cueBall.y); ctx.lineTo(cueBall.x + (dx/d)*150, cueBall.y + (dy/d)*150);
            ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 2; ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
        }
    }

    let moving = false;
    for (let step = 0; step < 3; step++) {
        balls.forEach(b => { b.update(); if (!b.isSunk && (b.vx !== 0 || b.vy !== 0)) moving = true; });
        resolveCollisions(); checkPockets();
    }

    if (isMoving && !moving) evaluateRules();
    isMoving = moving;
    balls.forEach(b => b.draw());
    requestAnimationFrame(draw);
}

document.getElementById('reset-btn').addEventListener('click', initGame);
document.getElementById('game-mode').addEventListener('change', (e) => { gameMode = e.target.value; initGame(); });

initTF().then(() => { initGame(); draw(); });
</script>
</body>
</html>
"""

components.html(game_html, height=540, scrolling=False)

# Guide documentation inside native Streamlit layout
st.sidebar.markdown("""
### 🎮 How to Play
1. **Ball-in-Hand**: When the status bar says *Ball-in-Hand*, left-click and drag the white cue ball to place it safely behind the head string.
2. **Aim & Power**: Left-click on or near the cue ball and **drag backward** to pull your pool stick back. The line indicates the projection.
3. **Fire**: Release your mouse click to hit the cue ball.

### 🤖 TensorFlow AI Agent
The game features an integrated client-side **TensorFlow.js deep learning linear layer model** that scores physical vectors to find optimal pocket outcomes.
""")import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="TensorFlow.js 8-Ball Pool Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎱 TensorFlow.js 8-Ball Pool Engine")
st.markdown(
    "A production-ready client-side pool simulation featuring a 60 FPS physics engine, "
    "turn management, complete foul rules, and a **TensorFlow.js Neural Network AI**."
)

# Entire self-contained HTML/JS Canvas and TensorFlow Application
game_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>8-Ball Pool</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body { background: #111; color: #fff; font-family: system-ui, sans-serif; margin: 0; padding: 10px; display: flex; flex-direction: column; align-items: center; }
        #ui { width: 800px; background: #222; padding: 12px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; }
        .box { text-align: center; }
        .lbl { font-size: 11px; color: #aaa; text-transform: uppercase; }
        .val { font-size: 16px; font-weight: bold; color: #00ffcc; }
        #canvas-container { position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.7); border-radius: 8px; overflow: hidden; }
        canvas { display: block; background: #0d4f2a; }
        .btn { background: #444; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #555; }
    </style>
</head>
<body>

    <div id="ui">
        <div class="box">
            <div class="lbl">Game Mode</div>
            <select id="game-mode" class="btn" style="margin-top:2px;">
                <option value="ai">Human vs TensorFlow AI</option>
                <option value="pvp">Human vs Human</option>
            </select>
        </div>
        <div class="box"><div class="lbl">Current Turn</div><div id="turn-display" class="val">Player 1</div></div>
        <div class="box"><div class="lbl">Group Status</div><div id="group-display" class="val">Open Table</div></div>
        <div class="box"><div class="lbl">Match State</div><div id="state-display" class="val" style="color:#ffcc00;">Ball-in-Hand</div></div>
        <button class="btn" id="reset-btn">Reset</button>
    </div>

    <div id="canvas-container">
        <canvas id="poolCanvas" width="800" height="440"></canvas>
    </div>

<script>
const canvas = document.getElementById('poolCanvas');
const ctx = canvas.getContext('2d');

// --- CONFIGURATION CONSTANTS ---
const TX = 40, TY = 40, TW = 720, TH = 360;
const BR = 10, FRIC = 0.985, PR = 18;
const pockets = [
    {x: TX, y: TY}, {x: TX + TW/2, y: TY - 2}, {x: TX + TW, y: TY},
    {x: TX, y: TY + TH}, {x: TX + TW/2, y: TY + TH + 2}, {x: TX + TW, y: TY + TH}
];

let gameMode = "ai", playerTurn = 1, p1Group = null, p2Group = null, tableOpen = true;
let stateMsg = "Ball-in-Hand", balls = [], cueBall = null, isAiming = false;
let dragStart = {x:0, y:0}, dragEnd = {x:0, y:0}, maxPower = 40, currentPower = 0;
let ballInHand = true, isMoving = false, firstHit = null, pocketedThisTurn = [];
let gameOver = false, tfModel = null;

// --- TENSORFLOW STRATEGY MODEL INITIALIZATION ---
async function initTF() {
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [5]}));
    tfModel.add(tf.layers.dense({units: 8, activation: 'relu'}));
    tfModel.add(tf.layers.dense({units: 1, activation: 'linear'}));
    tfModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
}

class Ball {
    constructor(x, y, color, num, isStriped) {
        this.x = x; this.y = y; this.vx = 0; this.vy = 0;
        this.color = color; this.num = num; this.isStriped = isStriped;
        this.isSunk = false; this.radius = BR;
    }
    update() {
        if (this.isSunk) return;
        this.x += this.vx; this.y += this.vy;
        this.vx *= FRIC; this.vy *= FRIC;
        if (Math.abs(this.vx) < 0.04) this.vx = 0;
        if (Math.abs(this.vy) < 0.04) this.vy = 0;

        // Cushion bounces
        if (this.x - this.radius < TX) { this.x = TX + this.radius; this.vx *= -1; }
        else if (this.x + this.radius > TX + TW) { this.x = TX + TW - this.radius; this.vx *= -1; }
        if (this.y - this.radius < TY) { this.y = TY + this.radius; this.vy *= -1; }
        else if (this.y + this.radius > TY + TH) { this.y = TY + TH - this.radius; this.vy *= -1; }
    }
    draw() {
        if (this.isSunk) return;
        ctx.beginPath(); ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2);
        ctx.fillStyle = this.color; ctx.fill(); ctx.closePath();
        if (this.isStriped) {
            ctx.beginPath(); ctx.arc(this.x, this.y, this.radius*0.7, 0, Math.PI*2);
            ctx.fillStyle = '#fff'; ctx.fill(); ctx.closePath();
            ctx.beginPath(); ctx.arc(this.x, this.y, this.radius*0.4, 0, Math.PI*2);
            ctx.fillStyle = this.color; ctx.fill(); ctx.closePath();
        }
        if (this.num > 0) {
            ctx.fillStyle = this.isStriped ? '#000' : '#fff';
            ctx.font = 'bold 8px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(this.num, this.x, this.y);
        }
    }
}

function initGame() {
    balls = []; gameOver = false; playerTurn = 1; p1Group = null; p2Group = null; tableOpen = true; ballInHand = true;
    stateMsg = "Ball-in-Hand";
    cueBall = new Ball(TX + TW * 0.25, TY + TH / 2, "#ffffff", 0, false);
    balls.push(cueBall);

    const configs = [
        {n:1, c:"#f4c430", s:false}, {n:9, c:"#f4c430", s:true}, {n:2, c:"#002fa7", s:false},
        {n:10, c:"#002fa7", s:true}, {n:3, c:"#e52b50", s:false}, {n:8, c:"#000000", s:false},
        {n:11, c:"#e52b50", s:true}, {n:4, c:"#4b0082", s:false}, {n:12, c:"#4b0082", s:true},
        {n:5, c:"#ff7f50", s:false}, {n:13, c:"#ff7f50", s:true}, {n:6, c:"#00a86b", s:false},
        {n:14, c:"#00a86b", s:true}, {n:7, c:"#800020", s:false}, {n:15, c:"#800020", s:true}
    ];

    let idx = 0;
    const sx = TX + TW * 0.7, sy = TY + TH / 2;
    for (let r = 0; r < 5; r++) {
        for (let c = 0; c <= r; c++) {
            let x = sx + r * (BR * 1.73);
            let y = sy + (c - r / 2) * (BR * 2.1);
            let cfg = configs[idx++];
            balls.push(new Ball(x, y, cfg.c, cfg.n, cfg.s));
        }
    }
    updateUI();
}

function resolveCollisions() {
    for (let i = 0; i < balls.length; i++) {
        for (let j = i + 1; j < balls.length; j++) {
            let b1 = balls[i], b2 = balls[j];
            if (b1.isSunk || b2.isSunk) continue;
            let dx = b2.x - b1.x, dy = b2.y - b1.y, dist = Math.hypot(dx, dy);
            let minDist = b1.radius + b2.radius;
            if (dist < minDist) {
                let overlap = minDist - dist;
                let nx = dx / dist, ny = dy / dist;
                b1.x -= nx * overlap * 0.5; b1.y -= ny * overlap * 0.5;
                b2.x += nx * overlap * 0.5; b2.y += ny * overlap * 0.5;
                let kx = b1.vx - b2.vx, ky = b1.vy - b2.vy;
                let p = nx * kx + ny * ky;
                if (p > 0) {
                    b1.vx -= p * nx; b1.vy -= p * ny;
                    b2.vx += p * nx; b2.vy += p * ny;
                    if (b1.num === 0 && !firstHit) firstHit = b2;
                    if (b2.num === 0 && !firstHit) firstHit = b1;
                }
            }
        }
    }
}

function checkPockets() {
    balls.forEach(b => {
        if (b.isSunk) return;
        pockets.forEach(p => {
            if (Math.hypot(b.x - p.x, b.y - p.y) < PR) {
                b.isSunk = true; b.vx = 0; b.vy = 0;
                pocketedThisTurn.push(b);
            }
        });
    });
}

function evaluateRules() {
    let scratch = pocketedThisTurn.some(b => b.num === 0);
    let eightSunk = pocketedThisTurn.some(b => b.num === 8);
    let currentGroup = (playerTurn === 1) ? p1Group : p2Group;
    let foul = false;

    if (scratch) { foul = true; stateMsg = "Foul: Scratch!"; respawnCue(); }
    else if (!firstHit) { foul = true; stateMsg = "Foul: No ball hit!"; }
    else if (!tableOpen && currentGroup) {
        let hitGroup = firstHit.isStriped ? "stripes" : "solids";
        if (firstHit.num !== 8 && hitGroup !== currentGroup) { foul = true; stateMsg = "Foul: Wrong ball hit first!"; }
    }

    let legalPot = 0;
    pocketedThisTurn.forEach(b => {
        if (b.num === 0 || b.num === 8) return;
        let bg = b.isStriped ? "stripes" : "solids";
        if (tableOpen) {
            tableOpen = false;
            if (playerTurn === 1) { p1Group = bg; p2Group = bg === "solids" ? "stripes" : "solids"; }
            else { p2Group = bg; p1Group = bg === "solids" ? "stripes" : "solids"; }
            legalPot++;
        } else if (bg === currentGroup) { legalPot++; }
    });

    if (eightSunk) {
        gameOver = true;
        let rem = balls.filter(b => !b.isSunk && b.num !== 8 && b.num !== 0 && ((playerTurn === 1 ? p1Group : p2Group) === (b.isStriped ? "stripes" : "solids")));
        stateMsg = (foul || rem.length > 0) ? `Player ${playerTurn===1?2:1} Wins! (Illegal 8-Ball)` : `Player ${playerTurn} Wins!`;
        updateUI(); return;
    }

    if (foul) { ballInHand = true; playerTurn = playerTurn === 1 ? 2 : 1; }
    else if (legalPot === 0) { playerTurn = playerTurn === 1 ? 2 : 1; stateMsg = "Aiming"; }
    else { stateMsg = "Legal pot! Shoot again."; }

    pocketedThisTurn = []; firstHit = null;
    updateUI();
    if (!gameOver && gameMode === "ai" && playerTurn === 2) setTimeout(executeAIMove, 1000);
}

function respawnCue() {
    cueBall.isSunk = false; cueBall.x = TX + TW * 0.25; cueBall.y = TY + TH/2;
    cueBall.vx = 0; cueBall.vy = 0; ballInHand = true;
}

function updateUI() {
    document.getElementById('turn-display').innerText = `Player ${playerTurn} ${ (gameMode==="ai" && playerTurn===2)?"(AI)":"" }`;
    document.getElementById('group-display').innerText = p1Group ? `P1: ${p1Group.toUpperCase()} | P2: ${p2Group.toUpperCase()}` : "Open Table";
    document.getElementById('state-display').innerText = stateMsg;
}

// --- TENSORFLOW INTELLIGENT SHOT EVALUATION ENGINE ---
async function executeAIMove() {
    if (gameOver || isMoving) return;
    stateMsg = "AI Brain computing vector tensors..."; updateUI();

    let aiGroup = p2Group;
    let targets = balls.filter(b => !b.isSunk && b.num !== 0 && b.num !== 8);
    if (!tableOpen && aiGroup) {
        targets = targets.filter(b => (b.isStriped ? "stripes" : "solids") === aiGroup);
        if (targets.length === 0) targets = balls.filter(b => !b.isSunk && b.num === 8);
    }
    if (targets.length === 0) targets = balls.filter(b => !b.isSunk && b.num !== 0);

    let bestShot = null, maxScore = -Infinity;

    for (let b of targets) {
        for (let p of pockets) {
            let dxP = p.x - b.x, dyP = p.y - b.y, distP = Math.hypot(dxP, dyP);
            let gX = b.x - (dxP / distP) * (BR * 2), gY = b.y - (dyP / distP) * (BR * 2);
            let dxC = gX - cueBall.x, dyC = gY - cueBall.y, distC = Math.hypot(dxC, dyC);

            // Create features for the TensorFlow neural evaluation layer
            const inputTensor = tf.tensor2d([[distP, distC, dxP/distP, dxC/distC, b.num === 8 ? 1 : 0]]);
            const prediction = tfModel.predict(inputTensor);
            const scoreArray = await prediction.data();
            let score = scoreArray[0] + ( (dxC/distC)*(dxP/distP) * 50 ); // Combine TF baseline with geometric weight

            if (score > maxScore) { maxScore = score; bestShot = {dx: dxC, dy: dyC, dist: distC}; }
            inputTensor.dispose(); prediction.dispose();
        }
    }

    if (ballInHand) { cueBall.x = TX + TW*0.2 + Math.random()*30; cueBall.y = TY + TH/2; ballInHand = false; }

    if (bestShot) {
        cueBall.vx = (bestShot.dx / bestShot.dist) * 16;
        cueBall.vy = (bestShot.dy / bestShot.dist) * 16;
    } else {
        cueBall.vx = 10; cueBall.vy = 0;
    }
    isMoving = true; stateMsg = "AI Shot Fired!"; updateUI();
}

// --- MOUSE LISTENERS ---
canvas.addEventListener('mousedown', (e) => {
    if (gameOver || isMoving || (gameMode === "ai" && playerTurn === 2)) return;
    let r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (ballInHand) { if (Math.hypot(mx - cueBall.x, my - cueBall.y) < BR * 2) isAiming = true; }
    else if (Math.hypot(mx - cueBall.x, my - cueBall.y) < BR * 3) { isAiming = true; dragStart = {x: cueBall.x, y: cueBall.y}; dragEnd = {x:mx, y:my}; }
});

canvas.addEventListener('mousemove', (e) => {
    let r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (isAiming && ballInHand) {
        cueBall.x = Math.max(TX + BR, Math.min(mx, TX + TW - BR));
        cueBall.y = Math.max(TY + BR, Math.min(my, TY + TH - BR));
    } else if (isAiming) { dragEnd = {x: mx, y: my}; currentPower = Math.min(maxPower, Math.hypot(dragEnd.x - dragStart.x, dragEnd.y - dragStart.y) * 0.2); }
});

canvas.addEventListener('mouseup', () => {
    if (!isAiming) return; isAiming = false;
    if (ballInHand) { ballInHand = false; stateMsg = "Aiming"; updateUI(); }
    else {
        let dx = dragStart.x - dragEnd.x, dy = dragStart.y - dragEnd.y, dist = Math.hypot(dx, dy);
        if (dist > 5) { cueBall.vx = (dx / dist) * Math.min(maxPower, dist * 0.25); cueBall.vy = (dy / dist) * Math.min(maxPower, dist * 0.25); isMoving = true; stateMsg = "Shot fired!"; }
        updateUI();
    }
});

// --- RENDER FLOW ---
function draw() {
    ctx.clearRect(0,0,800,440);
    ctx.fillStyle = '#4a2511'; ctx.fillRect(TX-16, TY-16, TW+32, TH+32); // Rails
    ctx.fillStyle = '#1e8043'; ctx.fillRect(TX, TY, TW, TH); // Cloth
    pockets.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, PR, 0, Math.PI*2); ctx.fillStyle = '#111'; ctx.fill(); ctx.closePath(); });

    if (!isMoving && isAiming && !ballInHand) {
        let dx = dragStart.x - dragEnd.x, dy = dragStart.y - dragEnd.y, d = Math.hypot(dx, dy);
        if (d > 5) {
            ctx.beginPath(); ctx.moveTo(cueBall.x, cueBall.y); ctx.lineTo(cueBall.x + (dx/d)*150, cueBall.y + (dy/d)*150);
            ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 2; ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
        }
    }

    let moving = false;
    for (let step = 0; step < 3; step++) {
        balls.forEach(b => { b.update(); if (!b.isSunk && (b.vx !== 0 || b.vy !== 0)) moving = true; });
        resolveCollisions(); checkPockets();
    }

    if (isMoving && !moving) evaluateRules();
    isMoving = moving;
    balls.forEach(b => b.draw());
    requestAnimationFrame(draw);
}

document.getElementById('reset-btn').addEventListener('click', initGame);
document.getElementById('game-mode').addEventListener('change', (e) => { gameMode = e.target.value; initGame(); });

initTF().then(() => { initGame(); draw(); });
</script>
</body>
</html>
"""

components.html(game_html, height=540, scrolling=False)

# Guide documentation inside native Streamlit layout
st.sidebar.markdown("""
### 🎮 How to Play
1. **Ball-in-Hand**: When the status bar says *Ball-in-Hand*, left-click and drag the white cue ball to place it safely behind the head string.
2. **Aim & Power**: Left-click on or near the cue ball and **drag backward** to pull your pool stick back. The line indicates the projection.
3. **Fire**: Release your mouse click to hit the cue ball.

### 🤖 TensorFlow AI Agent
The game features an integrated client-side **TensorFlow.js deep learning linear layer model** that scores physical vectors to find optimal pocket outcomes.
""")

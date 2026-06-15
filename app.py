import streamlit as st
import streamlit.components.v1 as components

# Set up page configuration
st.set_page_config(
    page_title="TensorFlow AI 8-Ball Pool Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎱 TensorFlow AI 8-Ball Pool Engine")
st.markdown(
    "A client-side 2D physics pool simulation featuring seamless animations, "
    "foul checking, and a predictive strategy engine optimized via TensorFlow.js."
)

# Complete Game Engine Code (HTML5 Canvas + TensorFlow.js Agent)
game_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>8-Ball Pool TensorFlow AI</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body {
            background-color: #111;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #ui-panel {
            width: 800px;
            background: #222;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-box {
            text-align: center;
        }
        .status-title {
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #00ffcc;
        }
        #game-container {
            position: relative;
            box-shadow: 0 10px 30px rgba(0,0,0,0.7);
            border-radius: 12px;
            overflow: hidden;
        }
        canvas {
            display: block;
            background: #0d4f2a;
        }
        .btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #666;
        }
        #power-bar-container {
            width: 800px;
            height: 15px;
            background: #333;
            margin-top: 10px;
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }
        #power-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(to right, #00ffcc, #ffcc00, #ff3333);
        }
    </style>
</head>
<body>

    <div id="ui-panel">
        <div class="status-box">
            <div class="status-title">Mode</div>
            <select id="game-mode" class="btn" style="margin-top:4px;">
                <option value="ai">Player vs TensorFlow AI</option>
                <option value="pvp">Player vs Player</option>
            </select>
        </div>
        <div class="status-box">
            <div class="status-title">Current Turn</div>
            <div id="current-turn" class="status-value">Player 1</div>
        </div>
        <div class="status-box">
            <div class="status-title">P1 Group / P2 Group</div>
            <div id="ball-assignments" class="status-value">Open Table</div>
        </div>
        <div class="status-box">
            <div class="status-title">Match State / Fouls</div>
            <div id="game-state" class="status-value" style="color:#ffcc00;">Aiming</div>
        </div>
        <div>
            <button class="btn" id="reset-btn">Reset Match</button>
        </div>
    </div>

    <div id="game-container">
        <canvas id="poolCanvas" width="800" height="440"></canvas>
    </div>
    
    <div id="power-bar-container">
        <div id="power-bar"></div>
    </div>

<script>
// --- PHYSICS AND GAME CONFIGURATION ---
const Canvas = document.getElementById('poolCanvas');
const ctx = Canvas.getContext('2d');

const TABLE_X = 40;
const TABLE_Y = 40;
const TABLE_WIDTH = 720;
const TABLE_HEIGHT = 360;
const BALL_RADIUS = 10;
const FRICTION = 0.988;
const POCKET_RADIUS = 18;

const pockets = [
    {x: TABLE_X, y: TABLE_Y}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y - 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y}, 
    {x: TABLE_X, y: TABLE_Y + TABLE_HEIGHT}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y + TABLE_HEIGHT + 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y + TABLE_HEIGHT}
];

let gameMode = "ai"; 
let playerTurn = 1; 
let p1Group = null; 
let p2Group = null;
let tableOpen = true;
let stateMessage = "Ball-in-Hand (Drag White Ball)";
let balls = [];
let cueBall = null;
let isAiming = false;
let dragStart = {x: 0, y: 0};
let dragEnd = {x: 0, y: 0};
let maxPower = 45;
let currentPower = 0;
let ballInHand = true;
let isMoving = false;
let turnFoulOccurred = false;
let firstBallHitThisTurn = null;
let pocketedThisTurn = [];
let gameOver = false;

// TensorFlow Model Placeholder Variable
let tfModel = null;

// Initialize a simple TensorFlow Policy Agent Model
async function initTensorFlowModel() {
    // A sequential model to approximate the best target scoring based on game state inputs
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [6]}));
    tfModel.add(tf.layers.dense({units: 16, activation: 'relu'}));
    tfModel.add(tf.layers.dense({units: 1, activation: 'linear'})); // Outputs a rating score for the shot
    tfModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
}

class Ball {
    constructor(x, y, color, number, isStriped = false) {
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.color = color;
        this.number = number;
        this.isStriped = isStriped;
        this.isSunk = false;
        this.radius = BALL_RADIUS;
    }

    update() {
        if (this.isSunk) return;
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= FRICTION;
        this.vy *= FRICTION;

        if (Math.abs(this.vx) < 0.04) this.vx = 0;
        if (Math.abs(this.vy) < 0.04) this.vy = 0;

        this.handleCushionCollisions();
    }

    handleCushionCollisions() {
        if (this.x - this.radius < TABLE_X) {
            this.x = TABLE_X + this.radius;
            this.vx *= -1;
        } else if (this.x + this.radius > TABLE_X + TABLE_WIDTH) {
            this.x = TABLE_X + TABLE_WIDTH - this.radius;
            this.vx *= -1;
        }
        if (this.y - this.radius < TABLE_Y) {
            this.y = TABLE_Y + this.radius;
            this.vy *= -1;
        } else if (this.y + this.radius > TABLE_Y + TABLE_HEIGHT) {
            this.y = TABLE_Y + TABLE_HEIGHT - this.radius;
            this.vy *= -1;
        }
    }

    draw() {
        if (this.isSunk) return;
        
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.closePath();

        if (this.isStriped) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.75, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
            ctx.closePath();
        }

        if (this.number > 0) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.fillStyle = "#000000";
            ctx.font = "bold 7px sans-serif";
            ctx.textAlign = "center";
            ctx.textimport streamlit as st
import streamlit.components.v1 as components

# Set up page configuration
st.set_page_config(
    page_title="TensorFlow AI 8-Ball Pool Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎱 TensorFlow AI 8-Ball Pool Engine")
st.markdown(
    "A client-side 2D physics pool simulation featuring seamless animations, "
    "foul checking, and a predictive strategy engine optimized via TensorFlow.js."
)

# Complete Game Engine Code (HTML5 Canvas + TensorFlow.js Agent)
game_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>8-Ball Pool TensorFlow AI</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body {
            background-color: #111;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #ui-panel {
            width: 800px;
            background: #222;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-box {
            text-align: center;
        }
        .status-title {
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #00ffcc;
        }
        #game-container {
            position: relative;
            box-shadow: 0 10px 30px rgba(0,0,0,0.7);
            border-radius: 12px;
            overflow: hidden;
        }
        canvas {
            display: block;
            background: #0d4f2a;
        }
        .btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #666;
        }
        #power-bar-container {
            width: 800px;
            height: 15px;
            background: #333;
            margin-top: 10px;
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }
        #power-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(to right, #00ffcc, #ffcc00, #ff3333);
        }
    </style>
</head>
<body>

    <div id="ui-panel">
        <div class="status-box">
            <div class="status-title">Mode</div>
            <select id="game-mode" class="btn" style="margin-top:4px;">
                <option value="ai">Player vs TensorFlow AI</option>
                <option value="pvp">Player vs Player</option>
            </select>
        </div>
        <div class="status-box">
            <div class="status-title">Current Turn</div>
            <div id="current-turn" class="status-value">Player 1</div>
        </div>
        <div class="status-box">
            <div class="status-title">P1 Group / P2 Group</div>
            <div id="ball-assignments" class="status-value">Open Table</div>
        </div>
        <div class="status-box">
            <div class="status-title">Match State / Fouls</div>
            <div id="game-state" class="status-value" style="color:#ffcc00;">Aiming</div>
        </div>
        <div>
            <button class="btn" id="reset-btn">Reset Match</button>
        </div>
    </div>

    <div id="game-container">
        <canvas id="poolCanvas" width="800" height="440"></canvas>
    </div>
    
    <div id="power-bar-container">
        <div id="power-bar"></div>
    </div>

<script>
// --- PHYSICS AND GAME CONFIGURATION ---
const Canvas = document.getElementById('poolCanvas');
const ctx = Canvas.getContext('2d');

const TABLE_X = 40;
const TABLE_Y = 40;
const TABLE_WIDTH = 720;
const TABLE_HEIGHT = 360;
const BALL_RADIUS = 10;
const FRICTION = 0.988;
const POCKET_RADIUS = 18;

const pockets = [
    {x: TABLE_X, y: TABLE_Y}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y - 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y}, 
    {x: TABLE_X, y: TABLE_Y + TABLE_HEIGHT}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y + TABLE_HEIGHT + 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y + TABLE_HEIGHT}
];

let gameMode = "ai"; 
let playerTurn = 1; 
let p1Group = null; 
let p2Group = null;
let tableOpen = true;
let stateMessage = "Ball-in-Hand (Drag White Ball)";
let balls = [];
let cueBall = null;
let isAiming = false;
let dragStart = {x: 0, y: 0};
let dragEnd = {x: 0, y: 0};
let maxPower = 45;
let currentPower = 0;
let ballInHand = true;
let isMoving = false;
let turnFoulOccurred = false;
let firstBallHitThisTurn = null;
let pocketedThisTurn = [];
let gameOver = false;

// TensorFlow Model Placeholder Variable
let tfModel = null;

// Initialize a simple TensorFlow Policy Agent Model
async function initTensorFlowModel() {
    // A sequential model to approximate the best target scoring based on game state inputs
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [6]}));
    tfModel.add(tf.layers.dense({units: 16, activation: 'relu'}));
    tfModel.add(tf.layers.dense({units: 1, activation: 'linear'})); // Outputs a rating score for the shot
    tfModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
}

class Ball {
    constructor(x, y, color, number, isStriped = false) {
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.color = color;
        this.number = number;
        this.isStriped = isStriped;
        this.isSunk = false;
        this.radius = BALL_RADIUS;
    }

    update() {
        if (this.isSunk) return;
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= FRICTION;
        this.vy *= FRICTION;

        if (Math.abs(this.vx) < 0.04) this.vx = 0;
        if (Math.abs(this.vy) < 0.04) this.vy = 0;

        this.handleCushionCollisions();
    }

    handleCushionCollisions() {
        if (this.x - this.radius < TABLE_X) {
            this.x = TABLE_X + this.radius;
            this.vx *= -1;
        } else if (this.x + this.radius > TABLE_X + TABLE_WIDTH) {
            this.x = TABLE_X + TABLE_WIDTH - this.radius;
            this.vx *= -1;
        }
        if (this.y - this.radius < TABLE_Y) {
            this.y = TABLE_Y + this.radius;
            this.vy *= -1;
        } else if (this.y + this.radius > TABLE_Y + TABLE_HEIGHT) {
            this.y = TABLE_Y + TABLE_HEIGHT - this.radius;
            this.vy *= -1;
        }
    }

    draw() {
        if (this.isSunk) return;
        
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.closePath();

        if (this.isStriped) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.75, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
            ctx.closePath();
        }

        if (this.number > 0) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.fillStyle = "#000000";
            ctx.font = "bold 7px sans-serif";
            ctx.textAlign = "center";
            ctx.textimport streamlit as st
import streamlit.components.v1 as components

# Set up page configuration
st.set_page_config(
    page_title="TensorFlow AI 8-Ball Pool Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎱 TensorFlow AI 8-Ball Pool Engine")
st.markdown(
    "A client-side 2D physics pool simulation featuring seamless animations, "
    "foul checking, and a predictive strategy engine optimized via TensorFlow.js."
)

# Complete Game Engine Code (HTML5 Canvas + TensorFlow.js Agent)
game_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>8-Ball Pool TensorFlow AI</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <style>
        body {
            background-color: #111;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #ui-panel {
            width: 800px;
            background: #222;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-box {
            text-align: center;
        }
        .status-title {
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #00ffcc;
        }
        #game-container {
            position: relative;
            box-shadow: 0 10px 30px rgba(0,0,0,0.7);
            border-radius: 12px;
            overflow: hidden;
        }
        canvas {
            display: block;
            background: #0d4f2a;
        }
        .btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #666;
        }
        #power-bar-container {
            width: 800px;
            height: 15px;
            background: #333;
            margin-top: 10px;
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }
        #power-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(to right, #00ffcc, #ffcc00, #ff3333);
        }
    </style>
</head>
<body>

    <div id="ui-panel">
        <div class="status-box">
            <div class="status-title">Mode</div>
            <select id="game-mode" class="btn" style="margin-top:4px;">
                <option value="ai">Player vs TensorFlow AI</option>
                <option value="pvp">Player vs Player</option>
            </select>
        </div>
        <div class="status-box">
            <div class="status-title">Current Turn</div>
            <div id="current-turn" class="status-value">Player 1</div>
        </div>
        <div class="status-box">
            <div class="status-title">P1 Group / P2 Group</div>
            <div id="ball-assignments" class="status-value">Open Table</div>
        </div>
        <div class="status-box">
            <div class="status-title">Match State / Fouls</div>
            <div id="game-state" class="status-value" style="color:#ffcc00;">Aiming</div>
        </div>
        <div>
            <button class="btn" id="reset-btn">Reset Match</button>
        </div>
    </div>

    <div id="game-container">
        <canvas id="poolCanvas" width="800" height="440"></canvas>
    </div>
    
    <div id="power-bar-container">
        <div id="power-bar"></div>
    </div>

<script>
// --- PHYSICS AND GAME CONFIGURATION ---
const Canvas = document.getElementById('poolCanvas');
const ctx = Canvas.getContext('2d');

const TABLE_X = 40;
const TABLE_Y = 40;
const TABLE_WIDTH = 720;
const TABLE_HEIGHT = 360;
const BALL_RADIUS = 10;
const FRICTION = 0.988;
const POCKET_RADIUS = 18;

const pockets = [
    {x: TABLE_X, y: TABLE_Y}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y - 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y}, 
    {x: TABLE_X, y: TABLE_Y + TABLE_HEIGHT}, 
    {x: TABLE_X + TABLE_WIDTH/2, y: TABLE_Y + TABLE_HEIGHT + 2}, 
    {x: TABLE_X + TABLE_WIDTH, y: TABLE_Y + TABLE_HEIGHT}
];

let gameMode = "ai"; 
let playerTurn = 1; 
let p1Group = null; 
let p2Group = null;
let tableOpen = true;
let stateMessage = "Ball-in-Hand (Drag White Ball)";
let balls = [];
let cueBall = null;
let isAiming = false;
let dragStart = {x: 0, y: 0};
let dragEnd = {x: 0, y: 0};
let maxPower = 45;
let currentPower = 0;
let ballInHand = true;
let isMoving = false;
let turnFoulOccurred = false;
let firstBallHitThisTurn = null;
let pocketedThisTurn = [];
let gameOver = false;

// TensorFlow Model Placeholder Variable
let tfModel = null;

// Initialize a simple TensorFlow Policy Agent Model
async function initTensorFlowModel() {
    // A sequential model to approximate the best target scoring based on game state inputs
    tfModel = tf.sequential();
    tfModel.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [6]}));
    tfModel.add(tf.layers.dense({units: 16, activation: 'relu'}));
    tfModel.add(tf.layers.dense({units: 1, activation: 'linear'})); // Outputs a rating score for the shot
    tfModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
}

class Ball {
    constructor(x, y, color, number, isStriped = false) {
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.color = color;
        this.number = number;
        this.isStriped = isStriped;
        this.isSunk = false;
        this.radius = BALL_RADIUS;
    }

    update() {
        if (this.isSunk) return;
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= FRICTION;
        this.vy *= FRICTION;

        if (Math.abs(this.vx) < 0.04) this.vx = 0;
        if (Math.abs(this.vy) < 0.04) this.vy = 0;

        this.handleCushionCollisions();
    }

    handleCushionCollisions() {
        if (this.x - this.radius < TABLE_X) {
            this.x = TABLE_X + this.radius;
            this.vx *= -1;
        } else if (this.x + this.radius > TABLE_X + TABLE_WIDTH) {
            this.x = TABLE_X + TABLE_WIDTH - this.radius;
            this.vx *= -1;
        }
        if (this.y - this.radius < TABLE_Y) {
            this.y = TABLE_Y + this.radius;
            this.vy *= -1;
        } else if (this.y + this.radius > TABLE_Y + TABLE_HEIGHT) {
            this.y = TABLE_Y + TABLE_HEIGHT - this.radius;
            this.vy *= -1;
        }
    }

    draw() {
        if (this.isSunk) return;
        
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.closePath();

        if (this.isStriped) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.75, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
            ctx.closePath();
        }

        if (this.number > 0) {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.closePath();

            ctx.fillStyle = "#000000";
            ctx.font = "bold 7px sans-serif";
            ctx.textAlign = "center";
            ctx.text

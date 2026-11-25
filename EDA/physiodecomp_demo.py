"""Run with: python -m streamlit run physiodecomp_demo.py."""

import streamlit as st
from streamlit.components.v1 import html


WALKTHROUGH = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How do blood tests relate to mood?</title>
<style>
    * { box-sizing: border-box; }
    :root {
        color-scheme: dark;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        color: #eff9fa;
    }
    body { margin: 0; }
    main {
        max-width: 1120px;
        margin: 0 auto;
        padding: 22px 24px;
        border-radius: 22px;
        background: #0a1720;
    }
    header, nav, footer, .key { display: flex; align-items: center; gap: .6rem; }
    header { justify-content: space-between; align-items: flex-start; margin-bottom: 1.1rem; }
    h1 { font-size: clamp(1.6rem, 3.8vw, 2.65rem); line-height: 1.08; margin: .2rem 0 0; letter-spacing: -.035em; }
    p { margin: 0; }
    .context, .hint { color: #afc9cf; font-size: .84rem; }
    button {
        border: 0;
        border-radius: .65rem;
        background: #172b35;
        color: #dfedef;
        font: inherit;
        font-size: .87rem;
        font-weight: 600;
        padding: .62rem .8rem;
        cursor: pointer;
    }
    button:hover { background: #23424d; }
    button:focus-visible, canvas:focus-visible { outline: 2px solid #69e5dc; outline-offset: 3px; }
    button[aria-current="step"] { background: #75e5db; color: #0b262a; }
    #play { white-space: nowrap; }
    nav { flex-wrap: wrap; }
    .scene {
        position: relative;
        margin-top: .85rem;
        border-radius: 1rem;
        background:
            radial-gradient(ellipse at 46% 48%, #163a46 0, #102934 34%, #0b1923 72%);
    }
    canvas {
        display: block;
        width: 100%;
        height: clamp(285px, 47vh, 390px);
        cursor: grab;
        touch-action: pan-y;
    }
    canvas:active { cursor: grabbing; }
    footer { justify-content: space-between; flex-wrap: wrap; padding: .72rem 1rem; }
    .key { flex-wrap: wrap; font-size: .82rem; gap: .4rem 1rem; color: #e0edef; }
    .key span { display: inline-flex; align-items: center; gap: .4rem; white-space: nowrap; }
    .key i { width: .62rem; height: .62rem; border-radius: 50%; display: inline-block; }
    .story { margin: .95rem 0 .1rem; }
    h2 { font-size: clamp(1.1rem, 2.4vw, 1.45rem); margin: 0 0 .28rem; }
    #cue, #detail { line-height: 1.42; max-width: 68ch; }
    #cue { color: #edf8fa; font-weight: 600; margin-bottom: .25rem; }
    #detail { color: #c5d9dd; }
    @media (max-width: 580px) {
        main { padding: 15px 13px; border-radius: 15px; }
        header { flex-wrap: wrap; margin-bottom: .8rem; }
        nav { gap: .35rem; }
        button { padding: .5rem .6rem; }
        canvas { height: 265px; }
        footer { padding: .6rem .7rem; }
        #cue, #detail { font-size: .9rem; }
    }
</style>
</head>
<body>
<main>
    <header>
        <div>
            <p class="context">Simulated research walkthrough</p>
            <h1>How do blood tests relate to mood?</h1>
        </div>
        <button type="button" id="play">Play walkthrough</button>
    </header>
    <nav aria-label="Walkthrough stages">
        <button type="button" data-stage="0" aria-current="step">Blood tests</button>
        <button type="button" data-stage="1">Combine</button>
        <button type="button" data-stage="2">Mood survey</button>
        <button type="button" data-stage="3">Split patterns</button>
        <button type="button" data-stage="4">Check again</button>
    </nav>
    <div class="scene">
        <canvas id="cloud" tabindex="0" role="img"
                aria-label="Two floating clouds represent the blood count and blood-sugar panel."></canvas>
        <footer>
            <div class="key" id="key" aria-label="Cloud colours"></div>
            <p class="hint">Drag to rotate</p>
        </footer>
    </div>
    <div class="story" role="status" aria-live="polite" aria-atomic="true">
        <h2 id="headline"></h2>
        <p id="cue"></p>
        <p id="detail"></p>
    </div>
</main>
<script>
"use strict";

const PALETTE = {
    cyan: [97, 236, 221],
    violet: [173, 152, 255],
    ice: [175, 230, 252],
    amber: [255, 195, 130],
};
const STAGES = [
    {
        title: "Start with two sets of blood tests",
        cue: "Teal is the blood count. Purple is the blood-sugar panel.",
        detail: "We use red-cell measures such as haemoglobin from the complete blood count (CBC), alongside glucose and insulin.",
        alt: "Two floating clouds represent the blood count and the blood-sugar panel.",
        key: [["cyan", "Blood count"], ["violet", "Blood-sugar panel"]],
    },
    {
        title: "Look at the whole profile",
        cue: "The two clouds come together.",
        detail: "We combine each person's blood-test results into one profile, then look for patterns across people.",
        alt: "The two clouds come together into one layered cloud representing combined blood-test patterns.",
        key: [["cyan", "Blood count"], ["violet", "Blood-sugar panel"]],
    },
    {
        title: "Compare with what people report",
        cue: "The gold cloud represents answers to a separate mood questionnaire.",
        detail: "We ask whether people with similar blood-test patterns also report similar symptoms.",
        alt: "The combined blood-test cloud sits beside a gold cloud of separately measured symptom responses.",
        key: [["ice", "Blood patterns"], ["amber", "Symptom responses"]],
    },
    {
        title: "See which changes go together",
        cue: "Four clouds show what the panels share, where they differ, and what varies within each panel.",
        detail: "For example, blood-sugar readings can vary while red-cell measurements stay steady.",
        alt: "The blood-test cloud opens into four lobes representing shared, diverging, blood-count and blood-sugar patterns.",
        key: [["ice", "Shared"], ["amber", "Mismatch"],
              ["cyan", "Blood count only"], ["violet", "Blood sugar only"]],
    },
    {
        title: "See what holds up in another group",
        cue: "The two clouds represent groups studied at different times.",
        detail: "We check whether the same blood patterns show up again and whether the links to symptoms hold.",
        alt: "Two clouds with similar silhouettes represent patterns assessed in earlier and later groups.",
        key: [["cyan", "Earlier group"], ["violet", "Later group"]],
    },
];

// Seeded display geometry. The shapes illustrate the workflow with simulated
// points and do not encode a fitted transform, measurements, or outcomes.
let seed = 1271;
function random() {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
}

const bloodCount = 840;
const particles = Array.from({length: bloodCount + 220}, (_, index) => {
    const azimuth = 2 * Math.PI * random();
    const polar = 2 * random() - 1;
    const radius = Math.pow(random(), 0.43);
    const flat = radius * Math.sqrt(1 - polar * polar);
    return {
        x: flat * Math.cos(azimuth),
        y: radius * polar,
        z: flat * Math.sin(azimuth),
        size: 0.55 + 0.9 * random(),
        opacity: 0.32 + 0.55 * random(),
        group: index % 2,
        lobe: index % 4,
        symptom: index >= bloodCount,
    };
});

const LOBES = [
    [-0.83, 0.51, 0.05, "ice"],
    [0.83, 0.51, 0.08, "amber"],
    [-0.83, -0.51, -0.08, "cyan"],
    [0.83, -0.51, -0.05, "violet"],
];

function shape(p, stage) {
    const {x, y, z, group, lobe, symptom} = p;
    const side = group === 0 ? -1 : 1;
    const tone = group === 0 ? "cyan" : "violet";
    if (symptom) {
        const visibility = stage === 2 ? 1 : 0;
        return {x: 1.05 + 0.69 * x, y: 0.52 * y + 0.12 * z,
                z: 0.53 * z, tone: "amber", visibility};
    }
    if (stage === 0) {
        return {x: side * 1.03 + 0.77 * x, y: 0.76 * y + side * 0.09,
                z: 0.62 * z + 0.12 * x, tone, visibility: 1};
    }
    if (stage === 1) {
        const shift = 1.35 * x + 0.26 * side;
        return {x: shift, y: 0.60 * y + 0.22 * shift + 0.18 * side,
                z: 0.58 * z + 0.20 * Math.sin(shift * 2.3), tone, visibility: 1};
    }
    if (stage === 2) {
        return {x: (1.12 * x + 0.13 * side) - 0.67, y: 0.57 * y + 0.20 * x,
                z: 0.59 * z + 0.12 * Math.sin(x * 3), tone: "ice", visibility: 1};
    }
    if (stage === 3) {
        const [cx, cy, cz, ink] = LOBES[lobe];
        return {x: cx + 0.52 * x, y: cy + 0.42 * y + 0.10 * x,
                z: cz + 0.40 * z, tone: ink, visibility: 1};
    }
    return {x: side * 0.98 + 0.73 * x,
            y: 0.43 * y + 0.50 * Math.sin(2 * x) + (side === 1 ? 0.14 * z : 0),
            z: 0.58 * z + 0.16 * x, tone, visibility: 1};
}

const spaces = STAGES.map((_, stage) => particles.map(p => shape(p, stage)));
const canvas = document.getElementById("cloud");
const context = canvas.getContext("2d");
const key = document.getElementById("key");
const play = document.getElementById("play");
const steps = [...document.querySelectorAll("[data-stage]")];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const duration = reducedMotion ? 0 : 960;
const waitBetweenStages = 6500;

function makeSprite(rgb) {
    const tile = document.createElement("canvas");
    tile.width = tile.height = 32;
    const ctx = tile.getContext("2d");
    const color = rgb.join(",");
    const glow = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    glow.addColorStop(0, "rgba(255,255,255,.93)");
    glow.addColorStop(.18, "rgba(" + color + ",.91)");
    glow.addColorStop(.41, "rgba(" + color + ",.44)");
    glow.addColorStop(.72, "rgba(" + color + ",.11)");
    glow.addColorStop(1, "rgba(" + color + ",0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, 32, 32);
    return tile;
}

const sprites = Object.fromEntries(
    Object.entries(PALETTE).map(([tone, rgb]) => [tone, makeSprite(rgb)])
);
let current = 0;
let from = spaces[0];
let to = spaces[0];
let started = -Infinity;
let frame = 0;
let timer = 0;
let playing = false;
let width = 0;
let height = 0;
let yaw = 0.14;
let pitch = -0.13;
let orbitPhase = 0;
let pointer = null;
let previousFrame = 0;
let openingUntil = performance.now() + 1900;

function tween(now) {
    if (duration === 0) return 1;
    const t = Math.max(0, Math.min(1, (now - started) / duration));
    return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function interpolate(a, b, t) {
    return {
        x: lerp(a.x, b.x, t),
        y: lerp(a.y, b.y, t),
        z: lerp(a.z, b.z, t),
        visibility: lerp(a.visibility, b.visibility, t),
    };
}

function scheduleDraw() {
    if (!frame) frame = requestAnimationFrame(draw);
}

function draw(now) {
    frame = 0;
    const advance = previousFrame ? Math.min(now - previousFrame, 34) : 0;
    previousFrame = now;
    const floating = !reducedMotion && (playing || now < openingUntil);
    if (floating && !pointer) orbitPhase += advance * 0.00065;

    const t = tween(now);
    const viewYaw = yaw + Math.sin(orbitPhase) * 0.23;
    const cosYaw = Math.cos(viewYaw);
    const sinYaw = Math.sin(viewYaw);
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const scale = Math.min(width * 0.258, height * 0.43);
    const points = [];
    for (let i = 0; i < particles.length; i++) {
        const a = from[i];
        const b = to[i];
        const p = t === 1 ? b : interpolate(a, b, t);
        if (p.visibility < 0.005) continue;
        const rotatedX = p.x * cosYaw + p.z * sinYaw;
        const rotatedZ = -p.x * sinYaw + p.z * cosYaw;
        const rotatedY = p.y * cosPitch - rotatedZ * sinPitch;
        const depth = p.y * sinPitch + rotatedZ * cosPitch;
        const perspective = 4.4 / (4.4 - depth);
        const radius = (2.15 + 1.85 * particles[i].size) * perspective;
        points.push({
            x: width / 2 + rotatedX * scale * perspective,
            y: height / 2 - rotatedY * scale * perspective,
            radius, depth, a: a.tone, b: b.tone,
            alpha: Math.min(1, particles[i].opacity * (0.88 + .19 * depth)) * p.visibility,
        });
    }
    context.clearRect(0, 0, width, height);
    points.sort((a, b) => a.depth - b.depth);
    for (const point of points) {
        const {x, y, radius, alpha, a, b} = point;
        if (t < 1 && a !== b) {
            context.globalAlpha = alpha * (1 - t);
            context.drawImage(sprites[a], x - radius, y - radius, 2 * radius, 2 * radius);
        }
        context.globalAlpha = a === b ? alpha : alpha * t;
        context.drawImage(sprites[b], x - radius, y - radius, 2 * radius, 2 * radius);
    }
    context.globalAlpha = 1;
    if (t < 1 || floating) scheduleDraw();
}

function resize() {
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    scheduleDraw();
}

function showStage(index) {
    if (index !== current) {
        const now = performance.now();
        const t = tween(now);
        from = t === 1 ? to : from.map((point, i) => ({
            ...interpolate(point, to[i], t),
            tone: t < 0.5 ? point.tone : to[i].tone,
        }));
        to = spaces[index];
        started = now;
        current = index;
    }
    const stage = STAGES[index];
    document.getElementById("headline").textContent = stage.title;
    document.getElementById("cue").textContent = stage.cue;
    document.getElementById("detail").textContent = stage.detail;
    canvas.setAttribute("aria-label", stage.alt);
    key.replaceChildren(...stage.key.map(([tone, label]) => {
        const item = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.style.background = "rgb(" + PALETTE[tone].join(",") + ")";
        swatch.setAttribute("aria-hidden", "true");
        item.append(swatch, document.createTextNode(label));
        return item;
    }));
    steps.forEach((button, i) => {
        if (i === index) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
    });
    scheduleDraw();
}

function updatePlay() {
    play.textContent = playing ? "Pause walkthrough"
        : current === STAGES.length - 1 ? "Replay walkthrough" : "Play walkthrough";
}

function stop() {
    playing = false;
    openingUntil = -Infinity;
    clearTimeout(timer);
    updatePlay();
}

function advance() {
    if (!playing) return;
    if (current === STAGES.length - 1) {
        stop();
        return;
    }
    showStage(current + 1);
    timer = setTimeout(advance, waitBetweenStages);
}

play.addEventListener("click", () => {
    if (playing) {
        stop();
        return;
    }
    if (current === STAGES.length - 1) showStage(0);
    playing = true;
    updatePlay();
    scheduleDraw();
    timer = setTimeout(advance, waitBetweenStages);
});

steps.forEach((button, index) => button.addEventListener("click", () => {
    stop();
    showStage(index);
    updatePlay();
}));

canvas.addEventListener("pointerdown", event => {
    openingUntil = -Infinity;
    pointer = {id: event.pointerId, x: event.clientX, y: event.clientY};
    canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
    if (!pointer || pointer.id !== event.pointerId) return;
    yaw += (event.clientX - pointer.x) * 0.009;
    pitch = Math.max(-1.2, Math.min(1.2, pitch + (event.clientY - pointer.y) * 0.009));
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    scheduleDraw();
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
    canvas.addEventListener(name, event => {
        if (pointer && pointer.id === event.pointerId) pointer = null;
    });
}
canvas.addEventListener("keydown", event => {
    if (event.key === "ArrowLeft") yaw -= 0.13;
    else if (event.key === "ArrowRight") yaw += 0.13;
    else if (event.key === "ArrowUp") pitch = Math.max(-1.2, pitch - 0.13);
    else if (event.key === "ArrowDown") pitch = Math.min(1.2, pitch + 0.13);
    else return;
    event.preventDefault();
    scheduleDraw();
});

new ResizeObserver(resize).observe(canvas);
showStage(0);
resize();
</script>
</body>
</html>
"""


def main():
    st.set_page_config(page_title="How do blood tests relate to mood?", layout="wide")
    st.markdown(
        "<style>.block-container{max-width:1180px;padding-top:1.65rem;"
        "padding-left:.7rem;padding-right:.7rem;padding-bottom:.7rem}</style>",
        unsafe_allow_html=True,
    )
    html(WALKTHROUGH, height=695, scrolling=False)


if __name__ == "__main__":
    main()


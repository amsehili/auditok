"""Interactive audio visualization widget for Jupyter notebooks.

Renders an HTML5 Canvas waveform with clickable detection regions and
Web Audio API playback.  The widget is entirely self-contained (no
external JS libraries) and is injected into the notebook output cell
via ``IPython.display.display()``.
"""

import base64
import io
import json
import uuid
import wave

import numpy as np

_RULER_H = 25

_WIDGET_HTML = """\
<style>
  #{wid} .atk-btn {{
    border:none; border-radius:4px; padding:4px 12px;
    cursor:pointer; font-weight:bold; font-size:12px;
    display:inline-flex; align-items:center; gap:5px;
    transition: filter 0.2s ease;
  }}
  #{wid} .atk-btn:hover {{ filter:brightness(1.15); }}
  #{wid} .atk-btn:active {{ filter:brightness(0.85); }}
  #{wid} .atk-btn svg {{ vertical-align:middle; }}
</style>
<div id="{wid}" style="
    background:{bg}; border-radius:6px; padding:10px; margin:6px 0;
    font-family:monospace; font-size:12px; color:#ccc;
    box-sizing:border-box; width:100%;
">
  <div style="margin-bottom:6px">
    <b style="color:#40d970">auditok</b>
    <span style="margin-left:8px">{title}</span>
  </div>
  <canvas id="{wid}_cv"
    style="cursor:pointer; display:block; border-radius:4px; width:100%; height:{height}px">
  </canvas>
  <div style="margin-top:6px; display:flex; align-items:center; gap:10px">
    <button id="{wid}_play" class="atk-btn" style="
        background:#40d970; color:#282a36; min-width:90px; justify-content:center;
    "><svg width="12" height="12" viewBox="0 0 12 12"><polygon points="2,0 2,12 11,6" fill="currentColor"/></svg> Play all</button>
    <button id="{wid}_stop" class="atk-btn" style="
        background:#e31f8f; color:#fff;
    "><svg width="12" height="12" viewBox="0 0 12 12"><rect x="1" y="1" width="10" height="10" rx="1" fill="currentColor"/></svg> Stop</button>
    <span id="{wid}_time" style="
        color:#40d970; font-size:12px; font-weight:bold;
        user-select:all; -webkit-user-select:all; cursor:text;
    "></span>
    <span id="{wid}_info" style="color:#999; font-size:11px"></span>
  </div>
</div>
<script>
(function() {{
  var W = document.getElementById("{wid}");
  var cv = document.getElementById("{wid}_cv");
  var ctx = cv.getContext("2d");
  var info = document.getElementById("{wid}_info");
  var btnPlay = document.getElementById("{wid}_play");
  var btnStop = document.getElementById("{wid}_stop");
  var timeSpan = document.getElementById("{wid}_time");

  var sr = {sr};
  var nch = {nch};
  var duration = {duration};
  var peaks = {peaks_json};
  var detections = {detections_json};
  var regionB64 = {regions_b64_json};
  var ethNorm = {eth_json};  // normalised energy threshold per channel (or null)

  var RULER_H = {ruler_h};
  var audioCtx = null;
  var currentSource = null;
  var fullAudioB64 = "{full_audio_b64}";
  var fullAudioBuf = null;  // cached decoded AudioBuffer for seeking
  var hoveredRegion = -1;
  var selectedRegion = -1;
  var rulerHoverX = -1;     // x pixel when hovering ruler, -1 if not
  var animId = null;
  var playStartTime = 0;
  var playOffset = 0;
  var playDuration = 0;
  var playGen = 0;          // generation counter to cancel stale async decodes
  var playing = false;      // true while audio is actively playing
  var pausedAt = -1;        // signal time where playback was paused (-1 = not paused)
  var pausedLabel = "";     // info label to restore on resume
  var pxPerSec = 1;

  var dpr = window.devicePixelRatio || 1;
  var cssW = 0;   // logical width in CSS pixels (all drawing uses this)
  var cssH = {height};

  function sizeCanvas() {{
    var w = cv.clientWidth || cv.parentElement.clientWidth;
    if (w <= 0) return;
    cssW = w;
    // Buffer at device resolution, CSS pins the display size
    cv.width = cssW * dpr;
    cv.height = cssH * dpr;
    cv.style.height = cssH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // scale once, draw in CSS px
    pxPerSec = cssW / duration;
  }}
  sizeCanvas();

  function b64ToArrayBuffer(b64) {{
    var bin = atob(b64);
    var buf = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return buf.buffer;
  }}

  // Choose a "nice" tick interval so ticks are readable but not crowded.
  function tickInterval() {{
    var nice = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    var target = duration / (cssW / 80);  // ~80px between ticks
    for (var i = 0; i < nice.length; i++) {{
      if (nice[i] >= target) return nice[i];
    }}
    return nice[nice.length - 1];
  }}

  function formatTime(t) {{
    if (duration < 60) return t.toFixed(2) + "s";
    if (duration < 3600) {{
      var m = Math.floor(t / 60);
      var s = (t % 60).toFixed(1);
      return m + ":" + (s < 10 ? "0" : "") + s;
    }}
    var h = Math.floor(t / 3600);
    var m = Math.floor((t % 3600) / 60);
    var s = Math.floor(t % 60);
    return h + ":" + (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }}

  function drawRuler() {{
    var w = cssW;
    // Ruler background
    ctx.fillStyle = "#1e1a2e";
    ctx.fillRect(0, 0, w, RULER_H);
    // Bottom border
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, snap(RULER_H - 1));
    ctx.lineTo(w, snap(RULER_H - 1));
    ctx.stroke();

    var interval = tickInterval();
    ctx.fillStyle = "#999";
    ctx.strokeStyle = "#666";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.lineWidth = 1;

    var t = 0;
    while (t <= duration) {{
      var x = snap(t * pxPerSec);
      // Tick mark
      ctx.beginPath();
      ctx.moveTo(x, RULER_H - 1);
      ctx.lineTo(x, RULER_H - 7);
      ctx.stroke();
      // Label (skip if too close to the edge)
      if (x > 20 && x < w - 20) {{
        ctx.fillText(formatTime(t), x, RULER_H - 9);
      }}
      t += interval;
    }}
  }}

  function drawNeedle(x) {{
    // Downward triangle at the ruler bottom
    var sz = 5;
    ctx.fillStyle = "#e31f8f";
    ctx.beginPath();
    ctx.moveTo(x - sz, RULER_H - 1);
    ctx.lineTo(x + sz, RULER_H - 1);
    ctx.lineTo(x, RULER_H + sz);
    ctx.closePath();
    ctx.fill();
    // Vertical dashed line through the waveform area
    ctx.strokeStyle = "rgba(227,31,143,0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, RULER_H);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Snap to nearest half-pixel for crisp 1px lines
  function snap(v) {{ return Math.round(v) + 0.5; }}

  function drawWaveform() {{
    var w = cssW, h = cssH;
    var waveH = h - RULER_H;
    var chH = waveH / nch;

    ctx.fillStyle = "{plot_bg}";
    ctx.fillRect(0, 0, w, h);

    // Draw detections background (waveform area only)
    for (var d = 0; d < detections.length; d++) {{
      var det = detections[d];
      var x0 = det[0] * pxPerSec;
      var x1 = det[1] * pxPerSec;
      var isHovered = (d === hoveredRegion);
      var isSelected = (d === selectedRegion);
      ctx.fillStyle = isSelected ? "rgba(255,140,26,0.45)"
                     : isHovered ? "rgba(255,140,26,0.30)"
                     : "rgba(119,119,119,0.5)";
      ctx.fillRect(x0, RULER_H, x1 - x0, waveH);

      if (isSelected || isHovered) {{
        ctx.strokeStyle = "#ff8c1a";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.strokeRect(Math.round(x0), RULER_H, Math.round(x1 - x0), waveH);
      }}
    }}

    // Draw waveform per channel
    for (var ch = 0; ch < nch; ch++) {{
      var chPeaks = peaks[ch];
      var mid = RULER_H + chH * ch + chH / 2;
      ctx.strokeStyle = "{signal_color}";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (var i = 0; i < chPeaks.length; i++) {{
        var x = snap((i / chPeaks.length) * w);
        var minV = chPeaks[i][0];
        var maxV = chPeaks[i][1];
        ctx.moveTo(x, snap(mid - maxV * chH * 0.45));
        ctx.lineTo(x, snap(mid - minV * chH * 0.45));
      }}
      ctx.stroke();

      // Energy threshold line
      if (ethNorm[ch] != null) {{
        var thY = snap(mid - ethNorm[ch] * chH * 0.45);
        ctx.strokeStyle = "#e31f8f";
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(0, thY);
        ctx.lineTo(w, thY);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // Channel separator
      if (ch > 0) {{
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, snap(RULER_H + chH * ch));
        ctx.lineTo(w, snap(RULER_H + chH * ch));
        ctx.stroke();
      }}
    }}

    // Detection labels
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    for (var d = 0; d < detections.length; d++) {{
      var det = detections[d];
      var x0 = det[0] * pxPerSec;
      var x1 = det[1] * pxPerSec;
      if (x1 - x0 > 18) {{
        ctx.fillStyle = (d === hoveredRegion || d === selectedRegion)
            ? "#ff8c1a" : "#ccc";
        ctx.fillText(d + 1, (x0 + x1) / 2, RULER_H + 12);
      }}
    }}

    // Ruler on top (drawn last so it's above detection highlights)
    drawRuler();

    // Needle cursor when hovering ruler
    if (rulerHoverX >= 0) drawNeedle(rulerHoverX);
  }}

  function drawPlayhead(t) {{
    var x = snap(t * pxPerSec);
    ctx.strokeStyle = "#e31f8f";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, RULER_H);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    // Small triangle at ruler bottom
    var sz = 4;
    ctx.fillStyle = "#e31f8f";
    ctx.beginPath();
    ctx.moveTo(x - sz, RULER_H);
    ctx.lineTo(x + sz, RULER_H);
    ctx.lineTo(x, RULER_H + sz);
    ctx.closePath();
    ctx.fill();
  }}

  function formatTimePrecise(t) {{
    var m = Math.floor(t / 60);
    var s = t % 60;
    return (m > 0 ? m + ":" : "") + (m > 0 && s < 10 ? "0" : "") + s.toFixed(3) + "s";
  }}

  var ICO_PLAY = '<svg width="12" height="12" viewBox="0 0 12 12"><polygon points="2,0 2,12 11,6" fill="currentColor"/></svg>';
  var ICO_PAUSE = '<svg width="12" height="12" viewBox="0 0 12 12"><rect x="1" y="0" width="3.5" height="12" rx="0.5" fill="currentColor"/><rect x="7.5" y="0" width="3.5" height="12" rx="0.5" fill="currentColor"/></svg>';

  function setPlaying(on) {{
    playing = on;
    btnPlay.innerHTML = on ? ICO_PAUSE + " Pause" : ICO_PLAY + " Play all";
  }}

  function updateTimestamp(t) {{
    timeSpan.textContent = formatTimePrecise(t);
  }}

  function animatePlayhead() {{
    if (!audioCtx || !currentSource) return;
    var elapsed = audioCtx.currentTime - playStartTime;
    if (elapsed > playDuration) {{
      stopPlayback();
      return;
    }}
    var curTime = playOffset + elapsed;
    drawWaveform();
    drawPlayhead(curTime);
    updateTimestamp(curTime);
    animId = requestAnimationFrame(animatePlayhead);
  }}

  function getAudioCtx() {{
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }}

  function stopPlayback() {{
    playGen++;  // invalidate any pending async decode callbacks
    if (currentSource) {{
      currentSource.onended = null;  // prevent stale callback from clobbering next source
      try {{ currentSource.stop(); }} catch(e) {{}}
      currentSource = null;
    }}
    if (animId) {{
      cancelAnimationFrame(animId);
      animId = null;
    }}
    setPlaying(false);
    pausedAt = -1;
    pausedLabel = "";
    info.textContent = "";
    timeSpan.textContent = "";
    drawWaveform();
  }}

  function pausePlayback() {{
    if (!playing || !currentSource) return;
    var elapsed = audioCtx.currentTime - playStartTime;
    pausedAt = playOffset + elapsed;
    pausedLabel = info.textContent;
    playGen++;
    currentSource.onended = null;
    try {{ currentSource.stop(); }} catch(e) {{}}
    currentSource = null;
    if (animId) {{
      cancelAnimationFrame(animId);
      animId = null;
    }}
    setPlaying(false);
    // Keep timestamp and playhead visible at paused position
    drawWaveform();
    drawPlayhead(pausedAt);
    updateTimestamp(pausedAt);
  }}

  function destroy() {{
    stopPlayback();
    if (audioCtx) {{
      try {{ audioCtx.close(); }} catch(e) {{}}
      audioCtx = null;
    }}
    fullAudioBuf = null;
  }}

  // Play a region from its base64 WAV (used for detection click-to-play)
  function playB64(b64, offset, dur, label) {{
    stopPlayback();
    var gen = playGen;
    var actx = getAudioCtx();
    var buf = b64ToArrayBuffer(b64);
    actx.decodeAudioData(buf, function(audioBuf) {{
      if (gen !== playGen) return;  // superseded by another play/stop
      _startSource(audioBuf, 0, audioBuf.duration, offset, label);
    }});
  }}

  // Play the full audio from a given time offset (used for ruler seek)
  function playFullFrom(seekTime) {{
    stopPlayback();
    var gen = playGen;
    var actx = getAudioCtx();

    function _play(audioBuf) {{
      if (gen !== playGen) return;  // superseded by another play/stop
      var off = Math.max(0, Math.min(seekTime, audioBuf.duration));
      var remaining = audioBuf.duration - off;
      _startSource(audioBuf, off, remaining, off,
        "Playing from " + formatTime(off));
    }}

    if (fullAudioBuf) {{
      _play(fullAudioBuf);
    }} else {{
      var buf = b64ToArrayBuffer(fullAudioB64);
      actx.decodeAudioData(buf, function(audioBuf) {{
        fullAudioBuf = audioBuf;
        _play(audioBuf);
      }});
    }}
  }}

  function _startSource(audioBuf, bufOffset, dur, displayOffset, label) {{
    var actx = getAudioCtx();
    var src = actx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(actx.destination);
    currentSource = src;
    playOffset = displayOffset;
    playDuration = dur;
    playStartTime = actx.currentTime;
    pausedAt = -1;
    info.textContent = label;
    setPlaying(true);
    src.onended = function() {{
      currentSource = null;
      if (animId) {{ cancelAnimationFrame(animId); animId = null; }}
      setPlaying(false);
      pausedAt = -1;
      pausedLabel = "";
      info.textContent = "";
      timeSpan.textContent = "";
      drawWaveform();
    }};
    src.start(0, bufOffset, dur);
    animatePlayhead();
  }}

  // --- Hit testing ---

  function getMousePos(e) {{
    var rect = cv.getBoundingClientRect();
    return {{
      x: (e.clientX - rect.left) * (cssW / rect.width),
      y: (e.clientY - rect.top) * (cssH / rect.height)
    }};
  }}

  function hitTestRegion(pos) {{
    if (pos.y < RULER_H) return -1;  // in ruler area
    var t = pos.x / pxPerSec;
    for (var d = 0; d < detections.length; d++) {{
      if (t >= detections[d][0] && t <= detections[d][1]) return d;
    }}
    return -1;
  }}

  function isInRuler(pos) {{
    return pos.y < RULER_H;
  }}

  cv.addEventListener("mousemove", function(e) {{
    var pos = getMousePos(e);
    if (isInRuler(pos)) {{
      // In ruler: show needle, clear region hover
      if (hoveredRegion !== -1) hoveredRegion = -1;
      rulerHoverX = pos.x;
      cv.style.cursor = "pointer";
      drawWaveform();
      return;
    }}
    // In waveform area
    if (rulerHoverX >= 0) {{
      rulerHoverX = -1;
    }}
    var idx = hitTestRegion(pos);
    if (idx !== hoveredRegion) {{
      hoveredRegion = idx;
      cv.style.cursor = idx >= 0 ? "pointer" : "default";
      drawWaveform();
    }}
  }});

  cv.addEventListener("mouseleave", function() {{
    var changed = (hoveredRegion !== -1 || rulerHoverX >= 0);
    hoveredRegion = -1;
    rulerHoverX = -1;
    if (changed) drawWaveform();
  }});

  cv.addEventListener("click", function(e) {{
    var pos = getMousePos(e);
    if (isInRuler(pos)) {{
      // Seek to the clicked time
      var t = Math.max(0, Math.min(pos.x / pxPerSec, duration));
      selectedRegion = -1;
      playFullFrom(t);
      return;
    }}
    var idx = hitTestRegion(pos);
    if (idx >= 0 && regionB64[idx]) {{
      selectedRegion = idx;
      var det = detections[idx];
      var dur = det[1] - det[0];
      playB64(regionB64[idx], det[0], dur,
        "Event " + (idx+1) + ": " + det[0].toFixed(3) + "s \u2013 " + det[1].toFixed(3) + "s");
    }}
  }});

  btnPlay.addEventListener("click", function() {{
    if (playing) {{
      // Currently playing -> pause
      pausePlayback();
    }} else if (pausedAt >= 0) {{
      // Paused -> resume from paused position
      var resumeFrom = pausedAt;
      selectedRegion = -1;
      playFullFrom(resumeFrom);
    }} else {{
      // Stopped -> play from beginning
      selectedRegion = -1;
      playFullFrom(0);
    }}
  }});

  btnStop.addEventListener("click", function() {{
    selectedRegion = -1;
    stopPlayback();
    info.textContent = "";
  }});

  // Stop playback when the widget is removed from the DOM (e.g. cell re-run)
  var observer = new MutationObserver(function(mutations) {{
    if (!document.contains(W)) {{
      destroy();
      observer.disconnect();
      window.removeEventListener("resize", onResize);
    }}
  }});
  observer.observe(document.body, {{childList: true, subtree: true}});

  // Redraw on window resize to fill the new width
  function onResize() {{
    sizeCanvas();
    drawWaveform();
  }}
  window.addEventListener("resize", onResize);

  drawWaveform();
}})();
</script>
"""


def _in_notebook():
    """Return True if running inside a Jupyter notebook kernel."""
    try:
        shell = get_ipython().__class__.__name__  # noqa: F821
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def _audio_to_wav_b64(data, sr, sw, ch):
    """Encode raw PCM bytes as a base64 WAV string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setframerate(sr)
        wf.setsampwidth(sw)
        wf.setnchannels(ch)
        wf.writeframes(data)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _downsample_peaks(samples, n_bins):
    """Compute (min, max) peaks per bin for waveform rendering.

    Parameters
    ----------
    samples : 1-D numpy array (one channel, normalised to [-1, 1])
    n_bins : int
        Number of horizontal pixels / bins.

    Returns
    -------
    list of [min, max] pairs, length *n_bins*.
    """
    n = len(samples)
    if n <= n_bins:
        return [[float(s), float(s)] for s in samples]
    bin_size = n / n_bins
    peaks = []
    for i in range(n_bins):
        start = int(i * bin_size)
        end = int((i + 1) * bin_size)
        chunk = samples[start:end]
        peaks.append([float(chunk.min()), float(chunk.max())])
    return peaks


def display_interactive(
    audio_region, regions, energy_threshold=None, height=None
):
    """Build and display an interactive waveform widget in a Jupyter notebook.

    The widget stretches to fill the full cell output width and resizes
    automatically when the browser window is resized.

    Parameters
    ----------
    audio_region : AudioRegion
        The full audio signal.
    regions : list of AudioRegion
        Detected events (each must have ``start`` and ``end`` set).
    energy_threshold : float, optional
        Currently unused (reserved for future threshold line overlay).
    height : int, optional
        Canvas height in pixels.  Defaults to 150 per channel, capped at 400.
    """
    from IPython.display import HTML, display

    sr = audio_region.sr
    sw = audio_region.sw
    ch = audio_region.ch
    duration = audio_region.duration

    if height is None:
        height = min(150 * ch, 400)
    height += _RULER_H  # add space for the time ruler

    # Compute waveform peaks per channel — use 2000 bins (redrawn to actual
    # canvas width by JS, so this just controls the data resolution)
    y = np.asarray(audio_region)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n_bins = min(2000, y.shape[1])
    peaks = []
    # Compute normalised threshold per channel (same space as peaks)
    if energy_threshold is not None:
        eth_log10 = energy_threshold * np.log(10) / 10
        amplitude_threshold = np.sqrt(np.exp(eth_log10))
    else:
        amplitude_threshold = None
    eth_normalized = []
    for c in range(y.shape[0]):
        channel = y[c].astype(float)
        mx = np.abs(channel).max()
        if mx > 0:
            channel = channel / mx
            if amplitude_threshold is not None:
                eth_normalized.append(min(float(amplitude_threshold / mx), 1.0))
            else:
                eth_normalized.append(None)
        else:
            eth_normalized.append(None)
        peaks.append(_downsample_peaks(channel, n_bins))

    # Detection time ranges
    detections = [[r.start, r.end] for r in regions]

    # Encode each region as a WAV for click-to-play
    regions_b64 = [_audio_to_wav_b64(r.data, sr, sw, ch) for r in regions]

    # Encode the full audio
    full_audio_b64 = _audio_to_wav_b64(audio_region.data, sr, sw, ch)

    wid = "auditok_" + uuid.uuid4().hex[:10]
    title = "{:.2f}s &middot; {} Hz &middot; {}&#x2011;bit &middot; {} ch &middot; {} events".format(
        duration, sr, sw * 8, ch, len(regions)
    )

    html = _WIDGET_HTML.format(
        wid=wid,
        height=height,
        ruler_h=_RULER_H,
        sr=sr,
        nch=ch,
        duration=duration,
        peaks_json=json.dumps(peaks),
        detections_json=json.dumps(detections),
        regions_b64_json=json.dumps(regions_b64),
        eth_json=json.dumps(eth_normalized),
        full_audio_b64=full_audio_b64,
        title=title,
        bg="#482a36",
        plot_bg="#282a36",
        signal_color="#40d970",
    )
    display(HTML(html))

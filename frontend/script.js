class BackgroundRemover {
  constructor() {
    this.currentImage = null;
    this.imageData = null;
    this.mask = null;
    this.originalMask = null;
    this.currentTool = "red"; // Start with red brush after API processing
    this.brushSize = 20;
    this.tolerance = 30;
    this.isDrawing = false;
    this.history = [];
    this.historyStep = -1;
    this.lastX = 0;
    this.lastY = 0;
    this.apiBaseUrl = "https://phraai.com/bgremover";
    this.processingSteps = ["step1", "step2", "step3", "step4", "step5"];
    this.currentStep = 0;
    this.zoom = 1;
    this.minZoom = 0.25;
    this.maxZoom = 8;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;
    this.scrollStartLeft = 0;
    this.scrollStartTop = 0;

    this.initializeElements();

    // Restore last chosen preset/format (if saved)
    const savedFmt = localStorage.getItem("fmt");
    if (savedFmt && this.formatSelect) this.formatSelect.value = savedFmt;
    const savedPreset = localStorage.getItem("preset");
    if (savedPreset && this.presetSelect) this.presetSelect.value = savedPreset;

    const savedBg = localStorage.getItem("bgColor");
    const savedAlpha = localStorage.getItem("bgAlpha");

    if (this.bgColorInput && savedBg) this.bgColorInput.value = savedBg;
    if (this.bgAlpha && savedAlpha) {
      const aPct = Math.round(parseFloat(savedAlpha) * 100);
      this.bgAlpha.value = Number.isFinite(aPct) ? aPct : 100;
    }

    // Apply if either was saved
    if (savedBg || savedAlpha) this.updateBackgroundColor();

    this.setupEventListeners();

    // Presets for /remove-pro parameters
    this.presets = {
      general: {},
      portraits: { feather_sigma: "0.9", shrink_px: "0", guided: "true" },
      product: {
        shrink_px: "1",
        rim_px: "5",
        weight_power: "2.2",
        t_power: "0.7",
      },
      logo: { guided: "false", feather_sigma: "0.0", shrink_px: "1" },
      speed: { max_dim: "1600" },
    };

    // UI selects
    this.presetSelect = document.getElementById("presetSelect");
    this.formatSelect = document.getElementById("formatSelect");
  }

  initializeElements() {
    this.uploadArea = document.getElementById("uploadArea");
    this.fileInput = document.getElementById("fileInput");
    this.loadingSpinner = document.getElementById("loadingSpinner");
    this.editorSection = document.getElementById("editorSection");
    this.mainCanvas = document.getElementById("mainCanvas");
    this.overlayCanvas = document.getElementById("overlayCanvas");
    this.canvasWrapper = document.getElementById("canvasWrapper");
    this.brushCursor = document.getElementById("brushCursor");
    this.canvasContainer = document.querySelector(".canvas-container");
    this.handTool = document.getElementById("handTool");
    this.zoomInBtn = document.getElementById("zoomInBtn");
    this.zoomOutBtn = document.getElementById("zoomOutBtn");
    this.zoomLabel = document.getElementById("zoomLabel");
    this.presetSelect = document.getElementById("presetSelect");
    this.formatSelect = document.getElementById("formatSelect");

    // Tools
    this.greenBrush = document.getElementById("greenBrush");
    this.redBrush = document.getElementById("redBrush");
    this.magicWand = document.getElementById("magicWand");
    this.brushSizeSlider = document.getElementById("brushSize");
    this.brushSizeValue = document.getElementById("brushSizeValue");
    this.toleranceSlider = document.getElementById("tolerance");
    this.toleranceValue = document.getElementById("toleranceValue");
    this.brushControls = document.getElementById("brushControls");
    this.magicControls = document.getElementById("magicControls");
    this.undoBtn = document.getElementById("undoBtn");
    this.redoBtn = document.getElementById("redoBtn");
    this.resetBtn = document.getElementById("resetBtn");
    this.downloadBtn = document.getElementById("downloadBtn");
    this.newImageBtn = document.getElementById("newImageBtn");
    this.bgColorInput = document.getElementById("bgColorInput");
    this.bgAlpha = document.getElementById("bgAlpha");
    this.bgClearBtn = document.getElementById("bgClearBtn");

    // Get contexts
    this.ctx = this.mainCanvas.getContext("2d", { willReadFrequently: true });
    this.overlayCtx = this.overlayCanvas.getContext("2d", {
      willReadFrequently: true,
    });
  }

  setupEventListeners() {
    // Upload functionality
    this.uploadArea.addEventListener("click", () => this.fileInput.click());
    this.uploadArea.addEventListener(
      "dragenter",
      this.handleDragEnter.bind(this)
    );
    this.uploadArea.addEventListener(
      "dragover",
      this.handleDragOver.bind(this)
    );
    this.uploadArea.addEventListener(
      "dragleave",
      this.handleDragLeave.bind(this)
    );
    this.uploadArea.addEventListener("drop", this.handleDrop.bind(this));
    this.fileInput.addEventListener("change", this.handleFileSelect.bind(this));

    // Tool selection
    this.greenBrush.addEventListener("click", () => this.setTool("green"));
    this.redBrush.addEventListener("click", () => this.setTool("red"));
    this.magicWand.addEventListener("click", () => this.setTool("magic"));
    this.handTool.addEventListener("click", () => this.setTool("hand"));

    // BG color UI
    if (this.bgColorInput) {
      this.bgColorInput.addEventListener("input", () => {
        this.updateBackgroundColor();
      });
    }
    if (this.bgAlpha)
      this.bgAlpha.addEventListener("input", () =>
        this.updateBackgroundColor()
      );
    if (this.bgClearBtn)
      this.bgClearBtn.addEventListener("click", () =>
        this.clearBackgroundColor()
      );

    // Brush size and tolerance
    this.brushSizeSlider.addEventListener(
      "input",
      this.updateBrushSize.bind(this)
    );
    this.toleranceSlider.addEventListener(
      "input",
      this.updateTolerance.bind(this)
    );

    // Zoom buttons
    this.zoomInBtn.addEventListener("click", () => this.zoomBy(1.25));
    this.zoomOutBtn.addEventListener("click", () => this.zoomBy(1 / 1.25));

    // History controls
    this.undoBtn.addEventListener("click", this.undo.bind(this));
    this.redoBtn.addEventListener("click", this.redo.bind(this));
    this.resetBtn.addEventListener("click", this.reset.bind(this));

    // Canvas drawing
    this.mainCanvas.addEventListener(
      "mousedown",
      this.handleMouseDown.bind(this)
    );
    this.mainCanvas.addEventListener(
      "mousemove",
      this.handleMouseMove.bind(this)
    );
    this.mainCanvas.addEventListener("mouseup", this.handleMouseUp.bind(this));
    this.mainCanvas.addEventListener(
      "mouseout",
      this.handleMouseOut.bind(this)
    );

    // Touch events for mobile
    this.mainCanvas.addEventListener("touchstart", this.handleTouch.bind(this));
    this.mainCanvas.addEventListener("touchmove", this.handleTouch.bind(this));
    this.mainCanvas.addEventListener("touchend", this.stopDrawing.bind(this));

    // Download and reset
    this.downloadBtn.addEventListener("click", this.downloadResult.bind(this));
    this.newImageBtn.addEventListener("click", this.resetApp.bind(this));

    // Custom cursor
    this.canvasWrapper.addEventListener(
      "mousemove",
      this.updateCursor.bind(this)
    );
    this.canvasWrapper.addEventListener(
      "mouseenter",
      this.showCursor.bind(this)
    );
    this.canvasWrapper.addEventListener(
      "mouseleave",
      this.hideCursor.bind(this)
    );

    if (this.formatSelect) {
      this.formatSelect.addEventListener("change", () => {
        localStorage.setItem("fmt", this.formatSelect.value);
      });
    }
    if (this.presetSelect) {
      this.presetSelect.addEventListener("change", () => {
        localStorage.setItem("preset", this.presetSelect.value);
      });
    }
  }

  handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    this.uploadArea.classList.add("dragover");
  }

  handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    this.uploadArea.classList.add("dragover");
  }

  handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();

    // Only remove dragover class if we're leaving the upload area entirely
    // Check if the related target is not a child of uploadArea
    if (!this.uploadArea.contains(e.relatedTarget)) {
      this.uploadArea.classList.remove("dragover");
    }
  }

  handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    this.uploadArea.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith("image/")) {
      this.processFile(files[0]);
    } else if (files.length > 0) {
      alert("Please drop a valid image file.");
    }
  }

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.processFile(file);
    }
  }

  animateProcessingSteps() {
    this.currentStep = 0;

    // Reset all steps first
    this.processingSteps.forEach((stepId) => {
      const el = document.getElementById(stepId);
      if (el) el.classList.remove("active");
    });

    const stepDelay = 500; // start delay (ms)  — increased
    const stepDuration = 1800; // gap between steps — increased

    const animateStep = () => {
      if (this.currentStep < this.processingSteps.length) {
        const el = document.getElementById(
          this.processingSteps[this.currentStep]
        );
        if (el) el.classList.add("active");
        this.currentStep++;
        setTimeout(animateStep, stepDuration);
      }
    };

    setTimeout(animateStep, stepDelay);
  }

  async processFile(file) {
    // Validate file type
    if (!file || !file.type || !file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    // Check file size (10MB limit to match API)
    if (file.size > 10 * 1024 * 1024) {
      alert("File too large. Maximum size is 10MB.");
      return;
    }

    // Show spinner but keep upload section container visible
    this.showLoading();
    await new Promise(requestAnimationFrame); // let the UI paint the spinner
    this.animateProcessingSteps();

    try {
      // One server call: RMBG -> (server chains to main.py /compose) -> refined image
      const form = new FormData();
      form.append("file", file);

      const resp = await fetch("https://phraai.com/rmbg/remove_bg_final", {
        method: "POST",
        body: form,
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(`Remove BG (final) failed: ${resp.status} - ${t}`);
      }
      const refinedImageBlob = await resp.blob();

      // (optional but nice) keep the full-res server result for downloads
      this.fullResBlob = refinedImageBlob;

      // Load both original and final refined images
      const [originalImg, refinedImg] = await Promise.all([
        this.loadImageFromFile(file),
        this.loadImageFromBlob(refinedImageBlob),
      ]);

      this.currentImage = originalImg;

      // Initialize editor with results
      await this.setupCanvases(originalImg, refinedImg);
      this.generateMaskFromProcessed(refinedImg);
      this.applyMask();
      this.saveState();

      this.showEditor();
      this.setTool("red");
    } catch (error) {
      console.error("Error processing image:", error);
      alert(`Error processing image: ${error.message || error}`);
      this.showUpload();
    } finally {
      this.hideLoading();
    }
  }

  async removeBackgroundAPI(file) {
    const formData = new FormData();
    formData.append("file", file);

    const fmt =
      this.formatSelect && this.formatSelect.value
        ? this.formatSelect.value
        : "png";
    formData.append("fmt", fmt);

    // Apply selected preset params
    const presetKey =
      this.presetSelect && this.presetSelect.value
        ? this.presetSelect.value
        : "general";
    const cfg = this.presets[presetKey] || {};
    for (const [k, v] of Object.entries(cfg)) {
      formData.append(k, String(v));
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/remove-pro`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `API request failed: ${response.status} - ${errorText}`
        );
      }

      return await response.blob();
    } catch (error) {
      if (error.name === "TypeError" && error.message.includes("fetch")) {
        throw new Error(
          "Network error. Please check your internet connection."
        );
      }
      throw error;
    }
  }

  async refineWithRemoveBg(imageBlob) {
    const formData = new FormData();

    // Convert blob to File object for the second API
    const file = new File([imageBlob], "temp.png", { type: imageBlob.type });
    formData.append("file", file);
    formData.append("mode", "cutout");

    try {
      const response = await fetch(`https://phraai.com/rmbg/remove_bg`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Remove BG API failed: ${response.status} - ${errorText}`
        );
      }

      return await response.blob();
    } catch (error) {
      if (error.name === "TypeError" && error.message.includes("fetch")) {
        throw new Error(
          "Network error during refinement. Please check your internet connection."
        );
      }
      throw error;
    }
  }

  loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        reject(new Error("Failed to load image"));
      };
      img.src = URL.createObjectURL(file);
    });
  }

  loadImageFromBlob(blob) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        reject(new Error("Failed to load processed image"));
      };
      img.src = URL.createObjectURL(blob);
    });
  }

  async setupCanvases(originalImg, processedImg) {
    const maxWidth = 800;
    const maxHeight = 600;
    let { width, height } = originalImg;

    // Scale down large images
    if (width > maxWidth || height > maxHeight) {
      const scale = Math.min(maxWidth / width, maxHeight / height);
      width = Math.floor(width * scale);
      height = Math.floor(height * scale);
    }

    // Set canvas dimensions
    this.mainCanvas.width = width;
    this.mainCanvas.height = height;
    this.overlayCanvas.width = width;
    this.overlayCanvas.height = height;

    // Draw original image and store its data
    this.ctx.drawImage(originalImg, 0, 0, width, height);
    this.imageData = this.ctx.getImageData(0, 0, width, height);

    // We'll display the processed image initially, but keep original data for mask operations
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.drawImage(processedImg, 0, 0, width, height);
    this.setZoom(1);
    this.centerCanvasInView();
  }

  centerCanvasInView() {
    requestAnimationFrame(() => {
      const c = this.canvasContainer;
      if (!c) return;
      c.scrollLeft = Math.max(0, (c.scrollWidth - c.clientWidth) / 2);
      c.scrollTop = Math.max(0, (c.scrollHeight - c.clientHeight) / 2);
    });
  }

  generateMaskFromProcessed(processedImg) {
    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;

    // Create a temporary canvas to extract alpha channel
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = width;
    tempCanvas.height = height;

    // Draw processed image to temp canvas
    tempCtx.drawImage(processedImg, 0, 0, width, height);
    const processedData = tempCtx.getImageData(0, 0, width, height);

    // Extract alpha channel as mask
    this.mask = new Uint8Array(width * height);
    const data = processedData.data;

    for (let i = 0; i < width * height; i++) {
      // Alpha channel is every 4th value (RGBA)
      this.mask[i] = data[i * 4 + 3];
    }

    // Store original mask for reset functionality
    this.originalMask = new Uint8Array(this.mask);
  }

  handleMouseDown(e) {
    if (this.currentTool === "hand") {
      this.startPan(e);
      return;
    }
    if (this.currentTool === "magic") {
      this.magicSelect(e);
    } else {
      this.startDrawing(e);
    }
  }

  handleMouseMove(e) {
    if (this.currentTool === "hand") {
      this.panMove(e);
      return;
    }
    if (this.currentTool !== "magic") {
      this.draw(e);
    }
  }

  handleMouseUp(e) {
    if (this.currentTool === "hand") {
      this.endPan();
      return;
    }
    if (this.currentTool !== "magic") {
      this.stopDrawing();
    }
  }

  updateBackgroundColor() {
    if (!this.canvasWrapper || !this.bgColorInput) return;
    const hex = this.bgColorInput.value; // "#rrggbb"
    const a = this.bgAlpha
      ? Math.min(1, Math.max(0, parseInt(this.bgAlpha.value, 10) / 100))
      : 1;
    const [r, g, b] = hex.match(/[0-9a-f]{2}/gi).map((h) => parseInt(h, 16));
    const rgba = `rgba(${r}, ${g}, ${b}, ${a})`;
    this.canvasWrapper.classList.add("bg-solid");
    this.canvasWrapper.style.setProperty("--bg-solid", rgba);
    localStorage.setItem("bgColor", hex);
    localStorage.setItem("bgAlpha", String(a));
  }

  clearBackgroundColor() {
    if (!this.canvasWrapper) return;
    this.canvasWrapper.classList.remove("bg-solid");
    this.canvasWrapper.style.removeProperty("--bg-solid");

    if (this.bgColorInput) this.bgColorInput.value = "#ffffff";
    if (this.bgAlpha) this.bgAlpha.value = 100;

    localStorage.removeItem("bgColor");
    localStorage.removeItem("bgAlpha"); // <-- add this
  }

  magicSelect(e) {
    const rect = this.mainCanvas.getBoundingClientRect();
    const scaleX = this.mainCanvas.width / rect.width;
    const scaleY = this.mainCanvas.height / rect.height;

    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    if (
      x < 0 ||
      x >= this.mainCanvas.width ||
      y < 0 ||
      y >= this.mainCanvas.height
    ) {
      return;
    }

    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;
    const data = this.imageData.data;

    // Get clicked pixel color from original image
    const clickIdx = (y * width + x) * 4;
    const clickColor = {
      r: data[clickIdx],
      g: data[clickIdx + 1],
      b: data[clickIdx + 2],
    };

    // Flood fill based on color similarity
    const visited = new Uint8Array(width * height);
    const selected = new Uint8Array(width * height);
    const queue = [{ x, y }];
    let selectedCount = 0;

    while (queue.length > 0 && selectedCount < width * height) {
      const pos = queue.shift();
      const px = pos.x;
      const py = pos.y;

      if (px < 0 || px >= width || py < 0 || py >= height) continue;

      const idx = py * width + px;
      if (visited[idx]) continue;

      visited[idx] = 1;

      const pixelIdx = idx * 4;
      const pixelColor = {
        r: data[pixelIdx],
        g: data[pixelIdx + 1],
        b: data[pixelIdx + 2],
      };

      // Calculate color difference
      const colorDiff =
        Math.abs(pixelColor.r - clickColor.r) +
        Math.abs(pixelColor.g - clickColor.g) +
        Math.abs(pixelColor.b - clickColor.b);

      if (colorDiff < this.tolerance * 3) {
        selected[idx] = 1;
        selectedCount++;

        // Add neighbors
        queue.push({ x: px + 1, y: py });
        queue.push({ x: px - 1, y: py });
        queue.push({ x: px, y: py + 1 });
        queue.push({ x: px, y: py - 1 });
      }
    }

    // Apply selection to mask
    const isAdditive = e.shiftKey;
    const isSubtractive = e.ctrlKey || e.metaKey;

    for (let i = 0; i < width * height; i++) {
      if (selected[i]) {
        if (isSubtractive || (!isAdditive && !isSubtractive)) {
          // Default behavior or explicit subtract: remove from foreground
          this.mask[i] = 0;
        } else if (isAdditive) {
          // Add to foreground
          this.mask[i] = 255;
        }
      }
    }

    this.applyMask();
    this.saveState();
  }

  startPan(e) {
    this.isPanning = true;
    this.canvasContainer.classList.add("dragging");
    this.mainCanvas.style.cursor = "grabbing";
    this.panStartX = e.clientX;
    this.panStartY = e.clientY;
    this.scrollStartLeft = this.canvasContainer.scrollLeft;
    this.scrollStartTop = this.canvasContainer.scrollTop;
  }

  panMove(e) {
    if (!this.isPanning) return;
    const dx = e.clientX - this.panStartX;
    const dy = e.clientY - this.panStartY;
    this.canvasContainer.scrollLeft = this.scrollStartLeft - dx;
    this.canvasContainer.scrollTop = this.scrollStartTop - dy;
  }

  endPan() {
    this.isPanning = false;
    this.canvasContainer.classList.remove("dragging");
    this.mainCanvas.style.cursor = "grab";
  }

  zoomBy(factor) {
    this.setZoom(this.zoom * factor);
  }

  setZoom(nextZoom) {
    const prev = this.zoom || 1;
    this.zoom = Math.min(
      this.maxZoom || 8,
      Math.max(this.minZoom || 0.25, nextZoom)
    );

    // Resize BOTH canvases via inline CSS (not attributes)
    const displayW = Math.round(this.mainCanvas.width * this.zoom);
    const displayH = Math.round(this.mainCanvas.height * this.zoom);
    this.mainCanvas.style.width = displayW + "px";
    this.mainCanvas.style.height = displayH + "px";
    this.overlayCanvas.style.width = displayW + "px";
    this.overlayCanvas.style.height = displayH + "px";

    // Keep viewport centered on the same spot
    const c = this.canvasContainer; // document.querySelector('.canvas-container')
    const cx = c.scrollLeft + c.clientWidth / 2;
    const cy = c.scrollTop + c.clientHeight / 2;
    const scale = this.zoom / prev;
    c.scrollLeft = Math.max(0, cx * scale - c.clientWidth / 2);
    c.scrollTop = Math.max(0, cy * scale - c.clientHeight / 2);

    // Keep the visible brush ring in sync with zoom
    if (this.brushCursor) {
      this.brushCursor.style.width = this.brushSize * this.zoom + "px";
      this.brushCursor.style.height = this.brushSize * this.zoom + "px";
    }

    // Update label if you have one
    if (this.zoomLabel)
      this.zoomLabel.textContent = Math.round(this.zoom * 100) + "%";
  }

  startDrawing(e) {
    this.isDrawing = true;
    const rect = this.mainCanvas.getBoundingClientRect();
    this.lastX = e.clientX - rect.left;
    this.lastY = e.clientY - rect.top;
    this.draw(e);
  }

  draw(e) {
    const rect = this.mainCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (this.isDrawing) {
      this.drawLine(this.lastX, this.lastY, x, y);
      this.lastX = x;
      this.lastY = y;
      this.applyMask();
    }
  }

  drawLine(x1, y1, x2, y2) {
    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;
    const rect = this.mainCanvas.getBoundingClientRect();

    // Scale coordinates
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    x1 = Math.round(x1 * scaleX);
    y1 = Math.round(y1 * scaleY);
    x2 = Math.round(x2 * scaleX);
    y2 = Math.round(y2 * scaleY);

    // Draw line using Bresenham's algorithm
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const sx = x1 < x2 ? 1 : -1;
    const sy = y1 < y2 ? 1 : -1;
    let err = dx - dy;

    while (true) {
      this.drawBrush(x1, y1);

      if (x1 === x2 && y1 === y2) break;

      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x1 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y1 += sy;
      }
    }
  }

  drawBrush(centerX, centerY) {
    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;

    // Make the radius integral so our loops use integer indices
    const radius = Math.floor(this.brushSize / 2);
    const value = this.currentTool === "green" ? 255 : 0;

    const x0 = Math.max(0, Math.floor(centerX - radius));
    const x1 = Math.min(width - 1, Math.floor(centerX + radius));
    const y0 = Math.max(0, Math.floor(centerY - radius));
    const y1 = Math.min(height - 1, Math.floor(centerY + radius));

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= radius) {
          const idx = y * width + x; // always an integer now
          if (distance > radius - 2) {
            const blend = (radius - distance) / 2;
            this.mask[idx] = Math.round(
              this.mask[idx] * (1 - blend) + value * blend
            );
          } else {
            this.mask[idx] = value;
          }
        }
      }
    }
  }

  stopDrawing() {
    if (this.isDrawing) {
      this.isDrawing = false;
      this.saveState();
    }
  }

  handleMouseOut(e) {
    this.stopDrawing();
  }

  handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
      e.type === "touchstart" ? "mousedown" : "mousemove",
      {
        clientX: touch.clientX,
        clientY: touch.clientY,
      }
    );
    this.mainCanvas.dispatchEvent(mouseEvent);
  }

  applyMask() {
    const width = this.mainCanvas.width;
    const height = this.mainCanvas.height;

    // Create new image data with transparency
    const outputData = this.ctx.createImageData(width, height);
    const data = this.imageData.data;
    const output = outputData.data;

    for (let i = 0; i < width * height; i++) {
      const pixelIdx = i * 4;
      const maskValue = this.mask[i];

      // Copy RGB values from original image
      output[pixelIdx] = data[pixelIdx];
      output[pixelIdx + 1] = data[pixelIdx + 1];
      output[pixelIdx + 2] = data[pixelIdx + 2];

      // Set alpha based on mask
      output[pixelIdx + 3] = maskValue;
    }

    // Clear and redraw
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.putImageData(outputData, 0, 0);

    // Update overlay
    this.updateOverlay();
  }

  updateOverlay() {
    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;

    this.overlayCtx.clearRect(0, 0, width, height);

    // Create overlay showing mask areas
    const overlayData = this.overlayCtx.createImageData(width, height);
    const overlay = overlayData.data;

    for (let i = 0; i < width * height; i++) {
      const pixelIdx = i * 4;
      const maskValue = this.mask[i];

      if (maskValue < 128) {
        // Background - show red tint
        overlay[pixelIdx] = 255;
        overlay[pixelIdx + 1] = 0;
        overlay[pixelIdx + 2] = 0;
        overlay[pixelIdx + 3] = 100;
      } else {
        // Foreground - show green tint
        overlay[pixelIdx] = 0;
        overlay[pixelIdx + 1] = 255;
        overlay[pixelIdx + 2] = 0;
        overlay[pixelIdx + 3] = 30;
      }
    }

    this.overlayCtx.putImageData(overlayData, 0, 0);
  }

  setTool(tool) {
    this.currentTool = tool;

    // Update UI
    document
      .querySelectorAll(".tool-btn")
      .forEach((btn) => btn.classList.remove("active"));

    if (tool === "green" || tool === "red") {
      (tool === "green" ? this.greenBrush : this.redBrush).classList.add(
        "active"
      );

      // Show brush cursor and color it
      this.brushCursor.style.display = ""; // clear any inline 'none'
      this.brushCursor.className = `brush-cursor ${tool}`;

      // Controls
      this.brushControls.style.display = "flex";
      this.magicControls.style.display = "none";

      // Cursors
      this.canvasContainer.classList.remove("hand", "dragging");
      this.mainCanvas.style.cursor = "crosshair";
      this.showCursor();
    } else if (tool === "magic") {
      this.magicWand.classList.add("active");

      // Hide brush cursor, show magic controls
      this.brushCursor.classList.remove("active");
      this.brushCursor.style.display = "none";
      this.brushControls.style.display = "none";
      this.magicControls.style.display = "flex";

      this.canvasContainer.classList.remove("hand", "dragging");
      this.mainCanvas.style.cursor = "crosshair";
    } else if (tool === "hand") {
      this.handTool.classList.add("active");

      // Hide brush UI/cursor
      this.brushCursor.classList.remove("active");
      this.brushCursor.style.display = "none";
      this.brushControls.style.display = "none";
      this.magicControls.style.display = "none";

      // Pan cursor via container + canvas
      this.canvasContainer.classList.add("hand");
      this.mainCanvas.style.cursor = "grab";
    }
  }

  updateBrushSize() {
    this.brushSize = parseInt(this.brushSizeSlider.value);
    this.brushSizeValue.textContent = this.brushSize;
    this.brushCursor.style.width = this.brushSize * this.zoom + "px";
    this.brushCursor.style.height = this.brushSize * this.zoom + "px";
  }

  updateTolerance() {
    this.tolerance = parseInt(this.toleranceSlider.value);
    this.toleranceValue.textContent = this.tolerance;
  }

  updateCursor(e) {
    if (this.currentTool === "magic" || this.currentTool === "hand") return;
    const rect = this.mainCanvas.getBoundingClientRect();
    this.brushCursor.style.left = e.clientX - rect.left + "px";
    this.brushCursor.style.top = e.clientY - rect.top + "px";
  }

  showCursor() {
    if (this.currentTool !== "magic") {
      this.brushCursor.classList.add("active");
      this.brushCursor.style.width = this.brushSize * this.zoom + "px";
      this.brushCursor.style.height = this.brushSize * this.zoom + "px";
    }
  }

  hideCursor() {
    this.brushCursor.classList.remove("active");
  }

  saveState() {
    this.historyStep++;
    this.history = this.history.slice(0, this.historyStep);
    this.history.push(new Uint8Array(this.mask));

    // Limit history size
    if (this.history.length > 50) {
      this.history.shift();
      this.historyStep--;
    }
  }

  undo() {
    if (this.historyStep > 0) {
      this.historyStep--;
      this.mask = new Uint8Array(this.history[this.historyStep]);
      this.applyMask();
    }
  }

  redo() {
    if (this.historyStep < this.history.length - 1) {
      this.historyStep++;
      this.mask = new Uint8Array(this.history[this.historyStep]);
      this.applyMask();
    }
  }

  reset() {
    this.mask = new Uint8Array(this.originalMask);
    this.applyMask();
    this.saveState();
  }

  downloadResult() {
    const fmt =
      this.formatSelect && this.formatSelect.value
        ? this.formatSelect.value
        : "png";
    const filename = `background-removed.${fmt}`;

    // If we have a full-res server blob, download that directly
    if (this.fullResBlob instanceof Blob) {
      const link = document.createElement("a");
      link.download = filename;
      link.href = URL.createObjectURL(this.fullResBlob);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 0);
      return;
    }

    // Fallback: export the current canvas
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = this.mainCanvas.width;
    tempCanvas.height = this.mainCanvas.height;
    tempCtx.drawImage(this.mainCanvas, 0, 0);

    const mime = fmt === "webp" ? "image/webp" : "image/png";
    const link = document.createElement("a");
    link.download = filename;
    link.href = tempCanvas.toDataURL(mime);
    link.click();
  }

  resetApp() {
    this.editorSection.style.display = "none";
    document.querySelector(".upload-section").style.display = "block";
    this.fileInput.value = "";
    this.history = [];
    this.historyStep = -1;
    this.mask = null;
    this.originalMask = null;
    this.imageData = null;
    this.currentImage = null;
    this.showUpload();
  }

  showLoading() {
    // Keep the section visible so the spinner can render
    document.querySelector(".upload-section").style.display = "block";
    this.uploadArea.style.display = "none";
    if (this.presetSelect) this.presetSelect.disabled = true;
    if (this.formatSelect) this.formatSelect.disabled = true;
    this.loadingSpinner.style.display = "block";
    this.editorSection.style.display = "none";
  }

  hideLoading() {
    // Just stop the spinner; do NOT change which view is shown
    this.loadingSpinner.style.display = "none";
    if (this.presetSelect) this.presetSelect.disabled = false;
    if (this.formatSelect) this.formatSelect.disabled = false;
  }

  showUpload() {
    document.querySelector(".upload-section").style.display = "block";
    this.uploadArea.style.display = "block";
    if (this.presetSelect) this.presetSelect.disabled = false;
    if (this.formatSelect) this.formatSelect.disabled = false;
    this.loadingSpinner.style.display = "none";
    this.editorSection.style.display = "none";
  }

  showEditor() {
    // When result is ready, hide the entire upload section (incl. spinner)
    document.querySelector(".upload-section").style.display = "none";
    this.loadingSpinner.style.display = "none";
    this.uploadArea.style.display = "none";
    this.editorSection.style.display = "block";
    this.centerCanvasInView();
  }
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new BackgroundRemover();
});

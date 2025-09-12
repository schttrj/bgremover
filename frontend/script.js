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

    this.initializeElements();
    this.setupEventListeners();
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

    // Brush size and tolerance
    this.brushSizeSlider.addEventListener(
      "input",
      this.updateBrushSize.bind(this)
    );
    this.toleranceSlider.addEventListener(
      "input",
      this.updateTolerance.bind(this)
    );

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
      const element = document.getElementById(stepId);
      if (element) {
        element.classList.remove("active");
      }
    });

    const animateStep = () => {
      // Activate current step
      if (this.currentStep < this.processingSteps.length) {
        const currentStepElement = document.getElementById(
          this.processingSteps[this.currentStep]
        );
        if (currentStepElement) {
          currentStepElement.classList.add("active");
          console.log(
            `Activating step: ${this.processingSteps[this.currentStep]}`
          ); // Debug log
        }
        this.currentStep++;
        setTimeout(animateStep, 1200); // Increased timing to 1.2 seconds
      }
    };

    // Start the animation after a short delay
    setTimeout(animateStep, 300);
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
      // Send image to API for background removal
      const processedImageBlob = await this.removeBackgroundAPI(file);

      // Load both original and processed images in parallel
      const [originalImg, processedImg] = await Promise.all([
        this.loadImageFromFile(file),
        this.loadImageFromBlob(processedImageBlob),
      ]);

      this.currentImage = originalImg;

      // Initialize editor with results
      await this.setupCanvases(originalImg, processedImg);
      this.generateMaskFromProcessed(processedImg);
      this.applyMask();
      this.saveState();

      // Switch to the editor view (do not toggle spinner here)
      this.showEditor();

      // Set default tool to red brush for refinement
      this.setTool("red");
    } catch (error) {
      console.error("Error processing image:", error);
      alert(`Error processing image: ${error.message || error}`);
      // On failure, return to upload view
      this.showUpload();
    } finally {
      // Stop the spinner without changing which view is visible
      this.hideLoading();
    }
  }

  async removeBackgroundAPI(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", "u2net");
    formData.append("post_process_mask", "true");
    formData.append("fmt", "png");

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
    if (this.currentTool === "magic") {
      this.magicSelect(e);
    } else {
      this.startDrawing(e);
    }
  }

  handleMouseMove(e) {
    if (this.currentTool !== "magic") {
      this.draw(e);
    }
  }

  handleMouseUp(e) {
    if (this.currentTool !== "magic") {
      this.stopDrawing();
    }
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
    const radius = this.brushSize / 2;
    const value = this.currentTool === "green" ? 255 : 0;

    // Draw circular brush
    for (
      let y = Math.max(0, centerY - radius);
      y < Math.min(height, centerY + radius);
      y++
    ) {
      for (
        let x = Math.max(0, centerX - radius);
        x < Math.min(width, centerX + radius);
        x++
      ) {
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        if (distance <= radius) {
          const idx = y * width + x;
          // Smooth edges
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

    // Update UI button state
    document
      .querySelectorAll(".tool-btn")
      .forEach((btn) => btn.classList.remove("active"));

    if (tool === "green" || tool === "red") {
      // Activate the right brush button
      (tool === "green" ? this.greenBrush : this.redBrush).classList.add(
        "active"
      );

      // Clear the inline 'display:none' set by Magic Select and color the cursor
      this.brushCursor.style.display = "";
      this.brushCursor.className = `brush-cursor ${tool}`;

      // Show brush controls, hide magic controls
      this.brushControls.style.display = "flex";
      this.magicControls.style.display = "none";

      // Make the cursor appear immediately
      this.showCursor();
    } else if (tool === "magic") {
      this.magicWand.classList.add("active");

      // Hide brush cursor for Magic Select
      this.brushCursor.classList.remove("active");
      this.brushCursor.style.display = "none";

      // Swap controls
      this.brushControls.style.display = "none";
      this.magicControls.style.display = "flex";
    }
  }

  updateBrushSize() {
    this.brushSize = parseInt(this.brushSizeSlider.value);
    this.brushSizeValue.textContent = this.brushSize;
    this.brushCursor.style.width = this.brushSize + "px";
    this.brushCursor.style.height = this.brushSize + "px";
  }

  updateTolerance() {
    this.tolerance = parseInt(this.toleranceSlider.value);
    this.toleranceValue.textContent = this.tolerance;
  }

  updateCursor(e) {
    if (this.currentTool === "magic") return;

    const rect = this.canvasWrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.brushCursor.style.left = x + "px";
    this.brushCursor.style.top = y + "px";
  }

  showCursor() {
    if (this.currentTool !== "magic") {
      this.brushCursor.classList.add("active");
      this.brushCursor.style.width = this.brushSize + "px";
      this.brushCursor.style.height = this.brushSize + "px";
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
    // Create a temporary canvas for the download
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = this.mainCanvas.width;
    tempCanvas.height = this.mainCanvas.height;

    // Copy the result with transparency
    tempCtx.drawImage(this.mainCanvas, 0, 0);

    // Download
    const link = document.createElement("a");
    link.download = "background-removed.png";
    link.href = tempCanvas.toDataURL("image/png");
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
    this.loadingSpinner.style.display = "block";
    this.editorSection.style.display = "none";
  }

  hideLoading() {
    // Just stop the spinner; do NOT change which view is shown
    this.loadingSpinner.style.display = "none";
  }

  showUpload() {
    document.querySelector(".upload-section").style.display = "block";
    this.uploadArea.style.display = "block";
    this.loadingSpinner.style.display = "none";
    this.editorSection.style.display = "none";
  }

  showEditor() {
    // When result is ready, hide the entire upload section (incl. spinner)
    document.querySelector(".upload-section").style.display = "none";
    this.loadingSpinner.style.display = "none";
    this.uploadArea.style.display = "none";
    this.editorSection.style.display = "block";
  }
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new BackgroundRemover();
});

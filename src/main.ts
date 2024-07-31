// Ensure these elements exist in your HTML
const demosSection = document.getElementById("demos") as HTMLElement;
let gestureRecognizer: any;
let runningMode = "IMAGE";
let enableWebcamButton: HTMLButtonElement;
let webcamRunning: boolean = false;
const videoHeight = "360px";
const videoWidth = "480px";

// Check if Mediapipe classes are defined
console.log("FilesetResolver:", typeof FilesetResolver);
console.log("GestureRecognizer:", typeof GestureRecognizer);
console.log("DrawingUtils:", typeof DrawingUtils);

// Before we can use HandLandmarker class we must wait for it to finish loading.
const createGestureRecognizer = async () => {
  try {
    console.log("Initializing gesture recognizer...");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        delegate: "GPU"
      },
      runningMode: runningMode
    });
    demosSection.classList.remove("invisible");
    console.log("Gesture recognizer initialized.");
  } catch (error) {
    console.error("Failed to initialize gesture recognizer:", error);
  }
};
createGestureRecognizer().then(() => {
  console.log("Gesture recognizer loaded.");
}).catch((error) => {
  console.error("Failed to initialize gesture recognizer:", error);
});

const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
  (imageContainers[i].children[0] as HTMLElement).addEventListener("click", handleClick);
}

async function handleClick(event: Event) {
  if (!gestureRecognizer) {
    alert("Please wait for gestureRecognizer to load");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await gestureRecognizer.setOptions({ runningMode: "IMAGE" });
  }

  const target = event.target as HTMLImageElement;

  // Remove all previous landmarks
  const allCanvas = (target.parentNode as HTMLElement).getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode!.removeChild(n);
  }

  const results = gestureRecognizer.recognize(target);

  console.log(results);
  if (results.gestures.length > 0) {
    const p = target.parentNode!.childNodes[3] as HTMLElement;
    p.setAttribute("class", "info");

    const categoryName = results.gestures[0][0].categoryName;
    const categoryScore = parseFloat(
      (results.gestures[0][0].score * 100).toFixed(2)
    );
    const handedness = results.handednesses[0][0].displayName;

    p.innerText = `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore}%\n Handedness: ${handedness}`;
    p.style.cssText = `left: 0px; top: ${target.height}px; width: ${target.width - 10}px;`;

    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", target.naturalWidth + "px");
    canvas.setAttribute("height", target.naturalHeight + "px");
    canvas.style.cssText = `left: 0px; top: 0px; width: ${target.width}px; height: ${target.height}px;`;

    target.parentNode!.appendChild(canvas);
    const canvasCtx = canvas.getContext("2d")!;
    const drawingUtils = new DrawingUtils(canvasCtx);
    for (const landmarks of results.landmarks) {
      drawingUtils.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: "#FF0000",
        lineWidth: 1
      });
    }
  }
}

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById("output_canvas") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d")!;
const gestureOutput = document.getElementById("gesture_output") as HTMLElement;

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton") as HTMLButtonElement;
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam(event: Event) {
  if (!gestureRecognizer) {
    alert("Please wait for gestureRecognizer to load");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results: any;
async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await gestureRecognizer.setOptions({ runningMode: "VIDEO" });
  }
  let nowInMs = Date.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    results = gestureRecognizer.recognizeForVideo(video, nowInMs);
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  const drawingUtils = new DrawingUtils(canvasCtx);

  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawingUtils.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: "#FF0000",
        lineWidth: 2
      });
    }
  }
  canvasCtx.restore();
  if (results.gestures.length > 0) {
    gestureOutput.style.display = "block";
    gestureOutput.style.width = videoWidth;
    const categoryName = results.gestures[0][0].categoryName;
    const categoryScore = parseFloat(
      (results.gestures[0][0].score * 100).toFixed(2)
    );
    const handedness = results.handednesses[0][0].displayName;
    gestureOutput.innerText = `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore} %\n Handedness: ${handedness}`;
  } else {
    gestureOutput.style.display = "none";
  }
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
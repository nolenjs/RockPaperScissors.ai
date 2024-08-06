document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('video-feed');
    const gestureOutputElement = document.getElementById('gesture_output');
    const resultElement = document.getElementById('result');

    const gestures = ['Rock', 'Paper', 'Scissors'];
    const countdownMessages = ['Rock', 'Paper', 'Scissors', 'Shoot'];
    let lastGestureTime = Date.now();
    const gestureInterval = 2000;  // Time interval in milliseconds between gesture recognitions
    let countdownIndex = 0;
    let consistentGesture = null;

    // Function to get a random gesture for the AI
    function getRandomGesture() {
        return gestures[Math.floor(Math.random() * gestures.length)];
    }

    // Function to determine the winner
    function determineWinner(userGesture, aiGesture) {
        if (userGesture === aiGesture) {
            return 'It\'s a tie!';
        }
        if ((userGesture === 'Rock' && aiGesture === 'Scissors') ||
            (userGesture === 'Paper' && aiGesture === 'Rock') ||
            (userGesture === 'Scissors' && aiGesture === 'Paper')) {
            return 'You win!';
        }
        return 'AI wins!';
    }

    // Function to handle the game logic
    function handleGame(userGesture) {
        const aiGesture = getRandomGesture();
        const result = determineWinner(userGesture, aiGesture);
        resultElement.innerText = `You: ${userGesture} - AI: ${aiGesture}\n${result}`;
        console.log(`You: ${userGesture} - AI: ${aiGesture}\n${result}`);
    }

    // Function to process the gesture data
    function processGestureData(data) {
        const currentTime = Date.now();
        if (currentTime - lastGestureTime >= gestureInterval) {
            if (countdownIndex < countdownMessages.length) {
                gestureOutputElement.innerText = `Countdown: ${countdownMessages[countdownIndex]}`;
                countdownIndex++;
                lastGestureTime = currentTime;  // Update the last gesture time
            } else if (data.gesture) {
                gestureOutputElement.innerText = `Gesture: ${data.gesture}`;
                consistentGesture = data.gesture;
                handleGame(consistentGesture);
                countdownIndex = 0;  // Reset countdown
                consistentGesture = null;  // Reset consistent gesture
            }
        }
    }

    // Function to start the video feed and listen for gestures
    async function startVideoFeed() {
        console.log("Starting video feed");
        const response = await fetch('/video_feed');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        let jsonBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            console.log("Received text:", text);

            const parts = text.split('--frame');

            for (let part of parts) {
                if (part.includes('Content-Type: image/jpeg')) {
                    const imgPart = part.split('Content-Type: image/jpeg')[1].trim();
                    const imgSrc = `data:image/jpeg;base64,${imgPart}`;
                    videoElement.src = imgSrc;
                } else if (part.includes('Content-Type: application/json')) {
                    const jsonPart = part.split('Content-Type: application/json')[1].trim();
                    try {
                        const gestureData = JSON.parse(jsonPart);
                        processGestureData(gestureData);
                    } catch (e) {
                        console.error("Error parsing gesture data", e);
                    }
                }
            }
        }
    }

    startVideoFeed();
});

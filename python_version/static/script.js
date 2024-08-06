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
    function startVideoFeed() {
        console.log("Starting video feed");

        const socket = io();

        socket.on('connect', () => {
            console.log('Connected to the server');
            socket.emit('request_video_feed');
        });

        socket.on('video_feed', (data) => {
            console.log("Received data:", data);

            const frame = `data:image/jpeg;base64,${data.frame}`;
            videoElement.src = frame;

            if (data.gesture) {
                processGestureData(data);
            }
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from the server');
        });
    }

    startVideoFeed();
});

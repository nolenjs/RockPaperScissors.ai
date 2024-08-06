document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('webcam');
    const gestureOutputElement = document.getElementById('gesture_output');
    const resultElement = document.getElementById('result');
    
    const gestures = ['Rock', 'Paper', 'Scissors'];
    
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
    }

    // Function to process the gesture data
    function processGestureData(data) {
        if (data.gesture) {
            gestureOutputElement.innerText = `Gesture: ${data.gesture}`;
            handleGame(data.gesture);
        }
    }

    // Function to start the video feed and listen for gestures
    async function startVideoFeed() {
        const response = await fetch('/video_feed');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        let jsonBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            const parts = text.split('--gesture');

            if (parts.length > 1) {
                jsonBuffer += parts[1];
                try {
                    const gestureData = JSON.parse(jsonBuffer.trim());
                    processGestureData(gestureData);
                    jsonBuffer = ''; // Reset the buffer after successful parsing
                } catch (e) {
                    // Wait for more data
                }
            }
        }
    }

    startVideoFeed();
});

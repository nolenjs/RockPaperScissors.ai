document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('video-feed');
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
        console.log(`You: ${userGesture} - AI: ${aiGesture}\n${result}`);
    }

    // WebSocket connection
    const socket = io();

    socket.on('connect', () => {
        console.log('Connected to the server');
    });

    socket.on('frame_data', (data) => {
        const parsedData = JSON.parse(data);
        if (parsedData.frame) {
            const imgSrc = `data:image/jpeg;base64,${parsedData.frame}`;
            videoElement.src = imgSrc;
        }
        if (parsedData.gesture) {
            gestureOutputElement.innerText = `Gesture: ${parsedData.gesture}`;
            handleGame(parsedData.gesture);
        }
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from the server');
    });
});

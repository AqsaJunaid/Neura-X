// static/js/voice-chat.js

document.addEventListener('DOMContentLoaded', () => {
    const micButton = document.getElementById('voice-mic-btn');
    const statusDiv = document.getElementById('voice-status');
    const input = document.getElementById('chatbotInput');
    const sendBtn = document.getElementById('send-message-btn');
    const form = document.getElementById('chatbotForm');
    const stopTtsBtn = document.getElementById('stop-tts-btn');
    let recognition = null;
    let isListening = false;

    // ───────────────────────────────
    // Initialize Speech Recognition
    // ───────────────────────────────
    function initRecognition() {
        if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
            statusDiv.textContent = "Voice input not supported in this browser";
            micButton.disabled = true;
            return false;
        }

        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';           // ← change to 'hi-IN', 'ta-IN', etc if needed

        recognition.onstart = () => {
            isListening = true;
            micButton.classList.add('listening');
            statusDiv.textContent = "Listening... speak now";
            statusDiv.classList.remove('hidden');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript.trim();
            if (transcript) {
                input.value = transcript;
                statusDiv.textContent = "Understood: " + transcript.substring(0, 40) + "...";

                // Auto-submit after short delay (feels natural)
                setTimeout(() => {
                    if (form.checkValidity()) {
                        sendBtn.click();
                    }
                }, 600);
            }
        };

        recognition.onerror = (event) => {
            console.error(event.error);
            statusDiv.textContent = "Voice error: " + event.error;
            stopListening();
        };

        recognition.onend = stopListening;

        return true;
    }

    function startListening() {
        if (!recognition) {
            if (!initRecognition()) return;
        }
        recognition.start();
    }

    function stopListening() {
        if (recognition && isListening) {
            recognition.stop();
        }
        isListening = false;
        micButton.classList.remove('listening');
        statusDiv.textContent = "Click 🎤 to speak";
    }

    // ───────────────────────────────
    // Text-to-Speech: Read AI answer
    // ───────────────────────────────
    function speak(text) {
        if (!('speechSynthesis' in window)) return;

        // Cancel any ongoing speech first (this is already good)
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.08;
        utterance.pitch = 1.0;
        utterance.volume = 0.92;

        const voices = window.speechSynthesis.getVoices();
        const natural = voices.find(v =>
            v.name.includes("Google") ||
            v.name.includes("Microsoft") ||
            v.name.includes("Natural") ||
            v.name.includes("Samantha") ||
            v.name.includes("Karen")
        );
        if (natural) utterance.voice = natural;

        // Show mute button **immediately** when we start speaking
        if (stopTtsBtn) {
            stopTtsBtn.classList.add('enabled');
        }

        // Only hide mute button when speech queue is completely empty
        utterance.onend = () => {
            // Check if there's still anything speaking/queued
            if (!window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
                if (stopTtsBtn) {
                    stopTtsBtn.classList.remove('enabled');
                }
            }
        };

        utterance.onerror = () => {
            // Same check on error
            if (!window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
                if (stopTtsBtn) {
                    stopTtsBtn.classList.remove('enabled');
                }
            }
        };

        window.speechSynthesis.speak(utterance);
    }
    // ───────────────────────────────
    // Hook into AI response
    // ───────────────────────────────
    const originalAddChatbotMessage = addChatbotMessage;

    window.addChatbotMessage = function (message, isUser = false) {
        originalAddChatbotMessage(message, isUser);

        // NEW: speak only AI messages (not user messages)
        if (!isUser && message) {
            speak(message);
        }
    };

    // Click handler
    micButton.addEventListener('click', () => {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    });

    // Optional: stop listening when form is submitted manually
    form.addEventListener('submit', () => {
        stopListening();
    });
    if (stopTtsBtn) {
        stopTtsBtn.addEventListener('click', () => {
            window.speechSynthesis.cancel();
            stopTtsBtn.classList.remove('enabled');
            // No "Voice stopped" message — keeps UI clean
        });
    }
});
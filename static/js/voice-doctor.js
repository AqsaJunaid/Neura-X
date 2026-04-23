// static/js/voice-doctor.js

document.addEventListener('DOMContentLoaded', () => {
    // ──── IDs from specialist.html ────
    const micBtn = document.getElementById('voice-mic-btn-specialist');
    const statusDiv = document.getElementById('voice-status-specialist');
    const input = document.getElementById('chatInput');           // ← this is correct in your HTML
    const sendBtn = document.getElementById('sendChatBtn');         // ← your send button ID
    const form = document.querySelector('#chatContainer .border-t-2'); // parent container (no form id)

    const stopTtsBtn = document.getElementById('stop-tts-btn');
    let recognition = null;
    let isListening = false;
    let shouldAutoSendOnStop = true;   // ← new line — we want auto-send when stopping



    // Initialize browser speech recognition
    function initRecognition() {
        if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
            statusDiv.textContent = "Voice input not supported in this browser";
            micBtn.disabled = true;
            return false;
        }

        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';           // ← change to 'hi-IN', etc. if needed

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add('listening');
            statusDiv.textContent = "Listening... speak now";
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript.trim();
            if (transcript) {
                input.value = transcript;
                statusDiv.textContent = "→ " + transcript.substring(0, 50) + "...";

                // Auto-send after short delay (feels natural for dictation)
                setTimeout(() => {
                    if (form.checkValidity() && input.value.trim()) {
                        sendBtn.click();
                    }
                }, 800);
            }
        };

        recognition.onerror = (event) => {
            console.error('Voice error:', event.error);
            statusDiv.textContent = "Error: " + event.error;
            stopListening();
        };

        recognition.onend = stopListening;

        return true;
    }

    function startListening() {
        if (!recognition && !initRecognition()) return;
        recognition.start();
    }

    function stopListening() {
        if (recognition && isListening) {
            recognition.stop();
        }
        isListening = false;
        micBtn.classList.remove('listening');
        statusDiv.textContent = "Click 🎤";

        // NEW: Auto-send when user stops recording (second click)
        if (shouldAutoSendOnStop && input.value.trim()) {
            // Give a tiny delay so user sees the text before send
            setTimeout(() => {
                if (input.value.trim()) {
                    // Trigger send — use your actual send button
                    sendBtn.click();           // this simulates clicking "Send"
                    // or if you prefer: form.dispatchEvent(new Event('submit'));
                }
            }, 400);  // 0.4 seconds — feels instant but user can see text
        }
    }

    // ───────────────────────────────
    // Text-to-Speech: Read incoming messages aloud
    // ───────────────────────────────
    function speak(text) {
        if (!('speechSynthesis' in window)) return;

        window.speechSynthesis.cancel(); // clear queue

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        utterance.volume = 0.9;

        const voices = window.speechSynthesis.getVoices();
        const naturalVoice = voices.find(v =>
            v.name.includes("Google") ||
            v.name.includes("Microsoft") ||
            v.name.includes("Natural") ||
            v.name.includes("Samantha")
        );
        if (naturalVoice) utterance.voice = naturalVoice;

        // Show stop button when speaking starts
        stopTtsBtn.classList.add('enabled');

        utterance.onend = () => {
            // Hide stop button when speech finishes naturally
            stopTtsBtn.classList.remove('enabled');
        };

        utterance.onerror = () => {
            stopTtsBtn.classList.remove('enabled');
        };

        window.speechSynthesis.speak(utterance);
    }

    // Hook into message rendering – speak only received messages (not sent ones)
    const originalChatMessageRender = window.addMessage || function () { }; // if you have custom function

    // Override or extend message display logic
    // This assumes messages are added via DOM manipulation in loadChatMessages()
    // We'll observe the chat container for new messages
    const chatContainer = document.getElementById('chatMessages');

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) { // element node
                        // Look for new received message bubbles
                        const receivedBubble = node.querySelector('.chat-bubble-received');
                        if (receivedBubble) {
                            const text = receivedBubble.querySelector('p.text-sm.leading-relaxed')?.textContent;
                            if (text && text.trim()) {
                                speak(text.trim());
                            }
                        }
                    }
                });
            }
        });
    });

    observer.observe(chatContainer, { childList: true, subtree: true });

    // Button click handler
    if (micBtn) {
        micBtn.addEventListener('click', () => {
            if (isListening) {
                stopListening();
            } else {
                startListening();
            }
        });
    }
    form.addEventListener('submit', () => {
        stopListening();
    });
    if (stopTtsBtn) {
        stopTtsBtn.addEventListener('click', () => {
            window.speechSynthesis.cancel();
            stopTtsBtn.classList.remove('enabled');

            // Visual feedback (optional but nice)
            statusDiv.textContent = "Voice stopped";
            setTimeout(() => {
                statusDiv.textContent = "Click 🎤";
            }, 1500);
        });
    }
    else {
        console.warn("Microphone button not found - check ID: voice-mic-btn-specialist");
    }

    // Optional: stop on send
    if (sendBtn) {
        sendBtn.addEventListener('click', stopListening);
    }
});
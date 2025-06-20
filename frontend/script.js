document.addEventListener('DOMContentLoaded', () => {
    const passwordInput = document.getElementById('password-input');
    const passwordDisplay = document.getElementById('password-display');
    const instructionMessage = document.getElementById('instruction-message');
    const errorMessage = document.getElementById('error-message');
    const typingFeedback = document.getElementById('typing-feedback');
    const subjectIdInput = document.getElementById('subject-id');
    const resetButton = document.getElementById('reset-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsDisplay = document.getElementById('results-display');

    const resClaimantId = document.getElementById('res-claimant-id');
    const resAuthStatus = document.getElementById('res-auth-status');
    const resJustification = document.getElementById('res-justification');
    const resLstm = document.getElementById('res-lstm');
    const resSvm = document.getElementById('res-svm');
    const resRf = document.getElementById('res-rf');
    const resFinalPred = document.getElementById('res-final-pred');

    const TARGET_PASSWORD = ".tie5Roanl";
    const EXPECTED_FEATURES_COUNT = 28;

    let sequenceEvents = [];
    let passwordIndex = 0;

    // --- Normalize key names ---
    function normalizeKeyName(char) {
         if (char === '.') return 'DOT';
         return char;
    }
   


    function setStatusColor(element, status) {
        element.classList.remove('status-na', 'status-success', 'status-fail', 'status-info');
        if (status === "Authenticated") {
            element.classList.add('status-success');
        } else if (status === "Not Authenticated" || status.includes("Error") || status.includes("Mismatch")) {
            element.classList.add('status-fail');
        } else if (status.includes("Processing")) {
            element.classList.add('status-info');
        } else {
            element.classList.add('status-na');
        }
    }

    function clearResults() {
        resClaimantId.textContent = 'N/A';
        resAuthStatus.textContent = 'N/A';
        resJustification.textContent = 'N/A';
        resLstm.textContent = 'N/A';
        resSvm.textContent = 'N/A';
        resRf.textContent = 'N/A';
        resFinalPred.textContent = 'N/A';

        setStatusColor(resAuthStatus, 'N/A');
        setStatusColor(resLstm, 'N/A');
        setStatusColor(resSvm, 'N/A');
        setStatusColor(resRf, 'N/A');
        setStatusColor(resFinalPred, 'N/A');

        errorMessage.textContent = '';
        errorMessage.classList.add('hidden');
        instructionMessage.textContent = '';
        instructionMessage.classList.add('hidden');

        loadingIndicator.classList.add('hidden');
        resultsDisplay.classList.add('hidden');
    }

    function resetTyping() {
        sequenceEvents = [];
        passwordIndex = 0;
        passwordInput.value = "";
        passwordInput.disabled = false;

        typingFeedback.textContent = "Ready";
        typingFeedback.classList.remove('typing-active');
        typingFeedback.classList.add('typing-inactive');

        instructionMessage.textContent = `Type the password: "${TARGET_PASSWORD}" and press Enter.`;
        instructionMessage.classList.remove('hidden');
        instructionMessage.classList.add('info-message');

        clearResults();
        passwordInput.focus();
    }

    resetButton.addEventListener('click', resetTyping);
    resetTyping();

    passwordInput.onkeydown = (e) => {
        const now = performance.now();
        const expectedChar = TARGET_PASSWORD[passwordIndex];
        const key = e.key;

        errorMessage.classList.add('hidden');
        instructionMessage.classList.add('hidden');

        if (typingFeedback.textContent === "Ready") {
            typingFeedback.textContent = "Typing...";
            typingFeedback.classList.remove('typing-inactive');
            typingFeedback.classList.add('typing-active');
            clearResults();
        }

        if (key === 'Enter') {
            e.preventDefault();
            if (passwordIndex === TARGET_PASSWORD.length) {
                processKeystrokeSequenceAndSend();
            } else {
                errorMessage.textContent = "Please type the full password before pressing Enter.";
                errorMessage.classList.remove('hidden');
            }
            return;
        }

        if (key === 'Backspace') {
            if (passwordIndex > 0) {
                errorMessage.textContent = "Backspace detected. Please reset and re-type for accurate capture.";
                errorMessage.classList.remove('hidden');
                setTimeout(resetTyping, 2000);
                return;
            }
        }

        const pressedChar = (key === 'Period') ? '.' : key;

        if (pressedChar === expectedChar) {
            sequenceEvents.push({ type: 'keydown', char: pressedChar, time: now });
        } else if (pressedChar !== expectedChar && pressedChar.length === 1 && TARGET_PASSWORD.includes(pressedChar)) {
            errorMessage.textContent = `Wrong key! Expected '${expectedChar}', got '${pressedChar}'. Please reset and try again.`;
            errorMessage.classList.remove('hidden');
            setTimeout(resetTyping, 2000);
            return;
        }
    };

    passwordInput.onkeyup = (e) => {
        const now = performance.now();
        const key = e.key;

        if (key === 'Enter' || key === 'Backspace') return;

        const releasedChar = (key === 'Period') ? '.' : key;

        if (releasedChar === TARGET_PASSWORD[passwordIndex]) {
            sequenceEvents.push({ type: 'keyup', char: releasedChar, time: now });
            passwordIndex++;
        } else if (releasedChar !== TARGET_PASSWORD[passwordIndex] && releasedChar.length === 1 && TARGET_PASSWORD.includes(releasedChar)) {
            errorMessage.textContent = `Keyup mismatch! Expected '${TARGET_PASSWORD[passwordIndex]}', got '${releasedChar}'. Please reset and try again.`;
            errorMessage.classList.remove('hidden');
            setTimeout(resetTyping, 2000);
        }
    };

    function processKeystrokeSequenceAndSend() {
        if (passwordIndex !== TARGET_PASSWORD.length || sequenceEvents.length < TARGET_PASSWORD.length * 2) {
            errorMessage.textContent = "Invalid password sequence captured. Please reset and try again.";
            errorMessage.classList.remove('hidden');
            return;
        }

        loadingIndicator.classList.remove('hidden');
        resultsDisplay.classList.add('hidden');
        passwordInput.disabled = true;

        typingFeedback.textContent = "Processing...";
        typingFeedback.classList.remove('typing-active');
        typingFeedback.classList.add('typing-inactive');

        const featuresToSend = [];
        const keydownTimes = {};
        const keyupTimes = {};

        const uniqueSequenceEvents = [];
        const seenEvents = new Set();

        sequenceEvents.forEach(event => {
            const eventId = `${event.type}-${event.char}-${event.time}`;
            if (!seenEvents.has(eventId)) {
                uniqueSequenceEvents.push(event);
                seenEvents.add(eventId);
            }
        });

        uniqueSequenceEvents.sort((a, b) => a.time - b.time);

        uniqueSequenceEvents.forEach(event => {
            if (event.type === 'keydown') keydownTimes[event.char] = event.time;
            else if (event.type === 'keyup') keyupTimes[event.char] = event.time;
        });

        TARGET_PASSWORD.split('').forEach(char => {
            const H_time = keyupTimes[char] - keydownTimes[char];
            featuresToSend.push({ key: `H.${normalizeKeyName(char)}`, interval: Math.round(H_time) });
        });

        for (let i = 0; i < TARGET_PASSWORD.length - 1; i++) {
    const char1 = TARGET_PASSWORD[i];
    const char2 = TARGET_PASSWORD[i + 1];

    const DD_time = keydownTimes[char2] - keydownTimes[char1];
    const UD_time = keydownTimes[char2] - keyupTimes[char1];

    featuresToSend.push({
        key: `DD.${normalizeKeyName(char1)}.${normalizeKeyName(char2)}`,
        interval: Math.round(DD_time)
    });
    featuresToSend.push({
        key: `UD.${normalizeKeyName(char1)}.H${normalizeKeyName(char2)}`,
        interval: Math.round(UD_time)
    });
}


        if (featuresToSend.length !== EXPECTED_FEATURES_COUNT) {
            errorMessage.textContent = `Feature extraction error: Expected ${EXPECTED_FEATURES_COUNT} features, but got ${featuresToSend.length}.`;
            errorMessage.classList.remove('hidden');
            loadingIndicator.classList.add('hidden');
            return;
        }

        const subjectId = subjectIdInput.value || 'unknown_subject';
        const payload = {
            subject_id: subjectId,
            keystrokes: featuresToSend
        };
        console.log("Features:", featuresToSend.map(f => f.key));


        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
        .then(response => {
            loadingIndicator.classList.add('hidden');
            resultsDisplay.classList.remove('hidden');
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Network error');
                });
            }
            return response.json();
        })
        .then(data => {
            resClaimantId.textContent = data.claimant_subject_id;
            resAuthStatus.textContent = data.system_authentication_status_for_claimant;
            setStatusColor(resAuthStatus, data.system_authentication_status_for_claimant);

            resJustification.textContent = data.justification;
            resLstm.textContent = data.predictions.LSTM;
            resSvm.textContent = data.predictions.SVM;
            resRf.textContent = data.predictions.RandomForest;
            resFinalPred.textContent = data.final_prediction_for_target_subject;

            setStatusColor(resLstm, data.predictions.LSTM);
            setStatusColor(resSvm, data.predictions.SVM);
            setStatusColor(resRf, data.predictions.RandomForest);
            setStatusColor(resFinalPred, data.final_prediction_for_target_subject);

            instructionMessage.textContent = "Results displayed. Click 'Reset Demo' to try again.";
            instructionMessage.classList.remove('hidden');
            typingFeedback.textContent = "Finished";
        })
        .catch((error) => {
            loadingIndicator.classList.add('hidden');
            resultsDisplay.classList.add('hidden');
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
            typingFeedback.textContent = "Error";
        });
    }
});

const BACKEND_URL = "http://127.0.0.1:8000/chat";

// Agent configuration with colors
const AGENT_CONFIG = [
    { 
        name: "🧠 Agent Alpha", 
        subtitle: "OpenAI",
        key: "alpha", 
        color: "#38bdf8" 
    },
    { 
        name: "💬 Agent Beta", 
        subtitle: "Gemini Flash",
        key: "beta", 
        color: "#22c55e" 
    },
    { 
        name: "⚖️ Agent Gamma", 
        subtitle: "Gemini Pro",
        key: "gamma", 
        color: "#facc15" 
    }
];

// Main think function
async function think() {
    const topic = document.getElementById("topic").value.trim();
    const style = document.getElementById("style").value;
    const agentsDiv = document.getElementById("agents");
    const finalDiv = document.getElementById("final");
    const finalSection = document.getElementById("finalSection");
    const agentsSection = document.getElementById("agentsSection");
    const loading = document.getElementById("loading");
    const thinkBtn = document.getElementById("thinkBtn");

    // Validation
    if (!topic) {
        showNotification("Please enter a question first.", "error");
        return;
    }

    // Reset UI
    agentsDiv.innerHTML = "";
    finalDiv.innerHTML = "";
    finalSection.style.display = "none";
    agentsSection.style.display = "none";
    loading.style.display = "block";
    thinkBtn.disabled = true;

    try {
        const res = await axios.post(BACKEND_URL, {
            topic: topic,
            preferred_style: style || null
        });

        const data = res.data;

        // Show agents section
        agentsSection.style.display = "block";

        // Render agent responses with staggered animation
        AGENT_CONFIG.forEach((agent, index) => {
            setTimeout(() => {
                renderAgentCard(agent, data.agents[agent.key]);
            }, index * 150);
        });

        // Show final verdict with delay
        setTimeout(() => {
            finalSection.style.display = "block";
            renderFinalVerdict(data.final);
            loading.style.display = "none";
            thinkBtn.disabled = false;
        }, AGENT_CONFIG.length * 150 + 300);

    } catch (err) {
        console.error("Error:", err);
        loading.style.display = "none";
        thinkBtn.disabled = false;
        
        const errorMsg = err.response?.data?.detail 
            || "Error connecting to backend. Make sure it's running on 127.0.0.1:8000 with CORS enabled.";
        
        showNotification(errorMsg, "error");
    }
}

// Render individual agent card
function renderAgentCard(agent, response) {
    const agentsDiv = document.getElementById("agents");
    const div = document.createElement("div");
    div.className = "agent";
    
    const responseText = response || "(No response received)";
    
    div.innerHTML = `
        <h2 style="color:${agent.color}">
            ${agent.name}
            <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">${agent.subtitle}</span>
        </h2>
        <div class="response" style="border-left-color: ${agent.color};">${escapeHtml(responseText)}</div>
        <div class="vote">
            <button onclick="vote('${agent.name}', '${agent.key}')">
                👍 Vote Best Response
            </button>
        </div>
    `;
    
    agentsDiv.appendChild(div);
}

// Render final verdict
function renderFinalVerdict(finalText) {
    const finalDiv = document.getElementById("final");
    finalDiv.innerHTML = `
        <strong>🏆 Final Verdict:</strong>
        
${escapeHtml(finalText)}
    `;
}

// Vote handler
function vote(agentName, agentKey) {
    showNotification(`You voted for ${agentName} as the best response!`, "success");
    
    // Optional: Send vote to backend
    // axios.post(`${BACKEND_URL}/vote`, { agent: agentKey })
    //     .catch(err => console.error("Vote error:", err));
}

// Notification system
function showNotification(message, type = "info") {
    // Remove existing notification
    const existing = document.querySelector(".notification");
    if (existing) {
        existing.remove();
    }

    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 10px;
        font-weight: 600;
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;

    if (type === "error") {
        notification.style.background = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)";
        notification.style.color = "#fff";
    } else if (type === "success") {
        notification.style.background = "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)";
        notification.style.color = "#fff";
    } else {
        notification.style.background = "linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%)";
        notification.style.color = "#0f172a";
    }

    document.body.appendChild(notification);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = "slideOutRight 0.3s ease-out";
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add notification animations to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Allow Enter key to submit (Shift+Enter for new line)
document.getElementById("topic").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        think();
    }
});
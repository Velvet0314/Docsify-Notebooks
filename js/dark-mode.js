document.addEventListener('DOMContentLoaded', function () {
    applyInitialMode();
    addToggleButton();
    injectStyles();
});

function applyInitialMode() {
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'dark' || (!currentTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.body.classList.add('dark-mode');
    }
}

function addToggleButton() {
    const button = document.createElement('button');
    button.id = 'theme-toggle';
    button.setAttribute('aria-label', 'Toggle dark/light theme');
    button.className = 'theme-toggle';
    button.innerHTML = document.body.classList.contains('dark-mode') ? getSunIcon() : getMoonIcon();

    button.onclick = function () {
        document.body.classList.toggle('dark-mode');
        const isDarkMode = document.body.classList.contains('dark-mode');
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        button.innerHTML = isDarkMode ? getSunIcon() : getMoonIcon();
        updateStyles(isDarkMode);
    };

    // 查找封面元素，然后在其后添加按钮
    const coverElement = document.querySelector('.cover');
    if (coverElement) {
        coverElement.parentNode.insertBefore(button, coverElement.nextSibling);
    } else {
        // 如果没有找到封面元素，回退到在body的第一个子元素前添加
        document.body.insertBefore(button, document.body.firstChild);
    }
}

function getMoonIcon() {
    return `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 16 16'>
        <path fill='currentColor' d='M9.598 1.591a.749.749 0 0 1 .785-.175 7.001 7.001 0 1 1-8.967 8.967.75.75 0 0 1 .961-.96 5.5 5.5 0 0 0 7.046-7.046.75.75 0 0 1 .175-.786m1.616 1.945a7 7 0 0 1-7.678 7.678 5.499 5.499 0 1 0 7.678-7.678' />
    </svg>`;
}

function getSunIcon() {
    return `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 16 16'>
        <path fill='currentColor' d='M8 12a4 4 0 1 1 0-8a4 4 0 0 1 0 8m0-1.5a2.5 2.5 0 1 0 0-5a2.5 2.5 0 0 0 0 5m5.657-8.157a.75.75 0 0 1 0 1.061l-1.061 1.06a.749.749 0 0 1-1.275-.326a.749.749 0 0 1 .215-.734l1.06-1.06a.75.75 0 0 1 1.06 0Zm-9.193 9.193a.75.75 0 0 1 0 1.06l-1.06 1.061a.75.75 0 1 1-1.061-1.06l1.06-1.061a.75.75 0 0 1 1.061 0M8 0a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0V.75A.75.75 0 0 1 8 0M3 8a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1 0-1.5h1.5A.75.75 0 0 1 3 8m13 0a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1 0-1.5h1.5A.75.75 0 0 1 16 8m-8 5a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 13m3.536-1.464a.75.75 0 0 1 1.06 0l1.061 1.06a.75.75 0 0 1-1.06 1.061l-1.061-1.06a.75.75 0 0 1 0-1.061M2.343 2.343a.75.75 0 0 1 1.061 0l1.06 1.061a.751.75 0 0 1-.018 1.042a.751.75 0 0 1-1.042.018l-1.06-1.06a.75.75 0 0 1 0-1.06Z' />
    </svg>`;
}

function injectStyles() {
    const styles = `
        .theme-toggle {
            position: fixed;
            top: 91.5%;
            right: 97.25%;
            background-color: transparent;
            border: none;
            cursor: pointer;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 30;
            transition: color 0.35s ease; /* Transition for icon color */
        }
        .theme-toggle svg {
            fill: currentColor;
            color: #333;  /* Light mode icon color */
        }
        .dark-mode .theme-toggle svg {
            transform: translateX(0.5px) translateY(1px);
            color: #fff;  /* Dark mode icon color */
        }
        /* 全局过渡效果定义 */
        body, a, .sidebar, .sidebar-toggle, b, strong {
            transition: color 0.35s ease; /* 为颜色变化添加平滑过渡效果 */
        }
        /* 深色模式的样式 */
        .dark-mode {
            background-color: #212121; /* 深色背景 */
            color: #fff; /* 白色文本 */
        }
        .dark-mode a, .dark-mode b, .dark-mode strong {
            color: #ddd; /* 淡灰色链接 */
        }
        .dark-mode .sidebar {
            background-color: #171717; /* 深色侧边栏背景 */
            color: #ccc; /* 浅灰色文本 */
        }
        .dark-mode .sidebar-toggle {
            background-color: #171717; /* 深色切换按钮背景 */
        }
        body.dark-mode.close .sidebar-toggle {
            background-color: #212121;  /* 暗模式下关闭状态的深色背景 */
        }
        /* 浅色模式样式，确保切换回浅色模式时也有过渡效果 */
        .light-mode {
            background-color: #fff; /* 浅色背景 */
            color: #000; /* 黑色文本 */
        }
        .light-mode a, .light-mode b, .light-mode strong {
            color: #000; /* 黑色链接 */
        } 
        .light-mode .sidebar {
            background-color: #fff; /* 浅色侧边栏背景 */
            color: #000; /* 黑色文本 */
        }
        .light-mode .sidebar-toggle {
            background-color: #fff; /* 浅色切换按钮背景 */
        }ground-color: #171717; /* Darker background for the toggle button */
    }
    `;

    const styleSheet = document.createElement('style');
    styleSheet.innerText = styles;
    document.head.appendChild(styleSheet);
}



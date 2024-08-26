(function() {
    // Docsify 插件
    function addCopyButton(hook, vm) {
        hook.doneEach(function() {
            // 找到所有的 code 块
            const blocks = document.querySelectorAll('pre[data-lang]');

            // 为每个 code 块添加复制按钮
            blocks.forEach(function(block) {
                // 创建一个按钮元素
                const button = document.createElement('button');
                button.innerText = 'Copy';
                button.className = 'copy-btn';
                block.appendChild(button);

                // 鼠标进入按钮时显示按钮
                button.addEventListener('mouseenter', function() {
                    button.style.opacity = '1'; // 设置按钮为可见
                });

                // 鼠标离开按钮时隐藏按钮
                button.addEventListener('mouseleave', function() {
                    if (button.innerText !== 'Copied!') {
                        button.style.opacity = '0'; // 鼠标离开时按钮再次隐藏
                    }
                });

                // 点击按钮时复制代码
                button.addEventListener('click', function() {
                    const code = block.querySelector('code').innerText;
                    copyToClipboard(code);

                    // 显示 "Copied!" 提示
                    button.innerText = 'Copied!';

                    // 等待1秒后将按钮隐藏并恢复为初始状态
                    setTimeout(() => {
                        button.style.opacity = '0'; // 隐藏按钮
                    }, 1000); // 1秒后恢复

                    setTimeout(() => {
                        button.innerText = 'Copy';
                    }, 1100);
                    
                });
            });
        });
    }

    // 复制到剪贴板功能
    function copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }

    // 注册插件
    window.$docsify.plugins = [].concat(addCopyButton, window.$docsify.plugins || []);
})();

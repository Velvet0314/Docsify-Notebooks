(function() {
  function lazyLoadMathPlugin(hook, vm) {
    function renderMathJax(element) {
      return new Promise((resolve, reject) => {
        if (window.MathJax) {
          MathJax.typesetPromise([element]).then(() => {
            resolve();
          }).catch((err) => {
            reject(err);
          });
        } else {
          reject('MathJax is not loaded');
        }
      });
    }

    function initializeLazyLoading() {
      const inlineMaths = document.querySelectorAll('span.math');
      const blockMaths = document.querySelectorAll('div.math');

      // 使用一个变量来存储当前正在渲染的元素队列，避免多次渲染
      let renderingQueue = [];

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          const element = entry.target;

          if (entry.isIntersecting) {
            // 当元素进入视口时，批量渲染
            if (!element.dataset.rendered) {
              renderingQueue.push(() => {
                element.innerHTML = element.dataset.originalContent; // 恢复原始内容
                renderMathJax(element).then(() => {
                  element.dataset.rendered = "true"; // 标记为已渲染
                }).catch((err) => {
                  console.error('MathJax rendering failed: ', err);
                });
              });
            }
          } else {
            // 当元素离开视口时，取消渲染
            if (element.dataset.rendered) {
              element.innerHTML = ''; // 清空内容取消渲染
              delete element.dataset.rendered; // 移除渲染标记
            }
          }
        });

        // 使用 requestAnimationFrame 批量处理渲染队列
        if (renderingQueue.length > 0) {
          requestAnimationFrame(() => {
            while (renderingQueue.length > 0) {
              const render = renderingQueue.shift();
              render();
            }
          });
        }
      }, { rootMargin: '100px 0px', threshold: 0.2 });

      inlineMaths.forEach(element => {
        element.dataset.originalContent = element.innerHTML;
        element.innerHTML = ''; // 清空内容防止自动渲染
        observer.observe(element);
      });

      blockMaths.forEach(element => {
        element.dataset.originalContent = element.innerHTML;
        element.innerHTML = ''; // 清空内容防止自动渲染
        observer.observe(element);
      });
    }

    // 在每次路由切换后调用
    hook.doneEach(function() {
      initializeLazyLoading();
    });

    // 在初始化完成后调用一次
    hook.mounted(function() {
      initializeLazyLoading();
    });
  }

  // 将插件添加到Docsify
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(lazyLoadMathPlugin);
})();

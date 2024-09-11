(function () {
  function lazyLoadMathPlugin(hook, vm) {
    function renderMathJax(element) {
      return new Promise((resolve, reject) => {
        if (window.MathJax) {
          MathJax.typesetPromise([element])
            .then(() => {
              resolve();
            })
            .catch((err) => {
              reject(err);
            });
        } else {
          reject("MathJax is not loaded");
        }
      });
    }

    function initializeLazyLoading() {
      const inlineMaths = document.querySelectorAll("span.math");
      const blockMaths = document.querySelectorAll("div.math");

      // 使用一个变量来存储当前正在渲染的元素队列，避免多次渲染
      let renderingQueue = [];

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const element = entry.target;

            if (entry.isIntersecting) {
              // 当元素进入视口时，批量渲染
              if (!element.dataset.rendered) {
                renderingQueue.push(() => {
                  element.innerHTML = element.dataset.originalContent; // 恢复原始内容
                  renderMathJax(element)
                    .then(() => {
                      element.dataset.rendered = "true"; // 标记为已渲染
                    })
                    .catch((err) => {
                      console.error("MathJax rendering failed: ", err);
                    });
                });
              }
            } else {
              // 当元素离开视口时，保留占位符并清空内容
              if (element.dataset.rendered) {
                element.innerHTML = "<span style='display:inline-block; width:100%; height:1em;'></span>"; // 保留占位符，避免高度突然变化
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
        },
        { rootMargin: "150px 0px", threshold: 0.15 }
      );

      inlineMaths.forEach((element) => {
        element.dataset.originalContent = element.innerHTML;
        element.innerHTML = "<span style='display:inline-block; width:100%; height:1em;'></span>"; // 初始设置占位符
        observer.observe(element);
      });

      blockMaths.forEach((element) => {
        element.dataset.originalContent = element.innerHTML;
        element.innerHTML = "<div style='width:100%; height:2em;'></div>"; // 初始设置占位符
        observer.observe(element);
      });
    }

    // 调用初始化 MathJax 的函数
    function initializeMathJax() {
      const hash = window.location.hash;

      // 检查是否是首页路由
      const isHomePage = (hash === '' || hash === '#/');

      if (!isHomePage) {
        MathJax.startup.defaultReady();
        console.log("MathJax initialized.");
      }
    }

    // 在每次路由切换后调用
    hook.doneEach(function () {
      initializeMathJax();
      initializeLazyLoading();
    });

    // 在初始化完成后调用一次
    hook.mounted(function () {
      initializeMathJax();
      initializeLazyLoading();
    });
  }

  // 将插件添加到Docsify
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(
    lazyLoadMathPlugin
  );
})();

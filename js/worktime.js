(function () {
  // 默认配置选项
  let defaultCommitStatsOptions = {
    text: "该仓库的提交天数为：{commit-days} 天",
    repo: "", // GitHub 仓库的名称，格式为 'username/repo'
    branch: "main", // 分支名称，默认为 main
    commitsPath: "commits.json", // commits.json 文件在仓库中的路径
    whereToPlace: "bottom", // 插入位置，"top" 或 "bottom"
  };

  function calculateCommitDays(commits) {
    const commitDates = new Set();

    commits.forEach((commit) => {
      const date = new Date(commit.commit.author.date).toDateString();
      commitDates.add(date);
    });

    return commitDates.size;
  }

  // Docsify 插件主函数
  function commitStatsPlugin(hook, vm) {
    let options = Object.assign({}, defaultCommitStatsOptions, vm.config.commitStats);

    let commitDaysPromise = new Promise(async (resolve, reject) => {
      const { repo, branch, commitsPath } = options;

      if (!repo) {
        console.error("GitHub repository is not specified.");
        return resolve(0);
      }

      const fileUrl = `https://raw.githubusercontent.com/${repo}/${branch}/${commitsPath}`;

      try {
        const response = await fetch(fileUrl);
        const commits = await response.json();

        // 统计提交天数
        const commitDays = calculateCommitDays(commits);
        resolve(commitDays);
      } catch (error) {
        console.error("Error fetching commits file from GitHub:", error);
        resolve(0); // 如果有错误，返回 0 天
      }
    });

    hook.beforeEach(async function (content, next) {
      const { text, whereToPlace } = options;

      // 等待异步操作完成并获取 commitDays
      const commitDays = await commitDaysPromise;

      const updatedText = text.replace("{commit-days}", commitDays);

      // 插入到文档内容中
      if (whereToPlace === "top") {
        content = updatedText + "\n\n" + content;
      } else {
        content = content + "\n\n" + updatedText;
      }

      next(content);
    });
  }

  // 注册插件
  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(commitStatsPlugin);
})();

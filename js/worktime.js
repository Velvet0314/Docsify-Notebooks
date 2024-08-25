(function () {
    // 默认配置选项
    let defaultCommitStatsOptions = {
      text: "该仓库的提交天数为：{commit-days} 天",
      repo: "",  // GitHub 仓库的名称，格式为 'username/repo'
      branch: "main",  // 分支名称，默认为 main
      token: "",  // 可选的 GitHub Personal Access Token，用于提高速率限制
      whereToPlace: "bottom"  // 插入位置，"top" 或 "bottom"
    };
  
    // 计算提交天数
    function calculateCommitDays(commits) {
      const commitDates = new Set();
  
      commits.forEach(commit => {
        const date = new Date(commit.commit.author.date).toDateString();
        commitDates.add(date);
      });
  
      return commitDates.size;
    }
  
    // Docsify 插件主函数
    function commitStatsPlugin(hook, vm) {
      let options = Object.assign({}, defaultCommitStatsOptions, vm.config.commitStats);
  
      hook.init(async function () {
        const { repo, branch, token } = options;
  
        if (!repo) {
          console.error("GitHub repository is not specified.");
          return;
        }
  
        const apiUrl = `https://api.github.com/repos/${repo}/commits?sha=${branch}&per_page=100`;
        const headers = token ? { Authorization: `token ${token}` } : {};
        let allCommits = [];
        let page = 1;
        let moreCommits = true;
  
        // Fetch all pages of commits to cover the entire repository
        while (moreCommits) {
          const response = await fetch(`${apiUrl}&page=${page}`, { headers });
          const data = await response.json();
  
          if (data && data.length > 0) {
            allCommits = allCommits.concat(data);
            page++;
          } else {
            moreCommits = false;
          }
        }
  
        // 统计整个仓库的提交天数
        const commitDays = calculateCommitDays(allCommits);
  
        // 保存提交天数以便后续使用
        vm.config.commitStats.commitDays = commitDays;
      });
  
      hook.beforeEach(function (content, next) {
        const { text, whereToPlace } = options;
        const commitDays = vm.config.commitStats.commitDays || 0;
        const updatedText = text.replace('{commit-days}', commitDays);
  
        // 插入到文档内容中
        if (whereToPlace === 'top') {
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
  
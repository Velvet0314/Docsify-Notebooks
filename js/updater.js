let defaultDocsifyUpdatedOptions = {
    text: "> Last Modify: {docsify-updated}",
    formatUpdated: "{YYYY}-{MM}-{DD} {HH}:{mm}:{ss}",
    whereToPlace: "bottom",
    repo: "",  // GitHub 仓库的名称，格式为 'username/repo'
    branch: "main",  // 分支名称，默认为 main
    token: ""  // 可选的 GitHub Personal Access Token，用于提高速率限制
  };
  
  function formatDate(date, format) {
    const map = {
      YYYY: date.getFullYear(),
      MM: ('0' + (date.getMonth() + 1)).slice(-2),
      DD: ('0' + date.getDate()).slice(-2),
      HH: ('0' + date.getHours()).slice(-2),
      mm: ('0' + date.getMinutes()).slice(-2),
      ss: ('0' + date.getSeconds()).slice(-2)
    };
    return format.replace(/YYYY|MM|DD|HH|mm|ss/gi, matched => map[matched]);
  }
  
  function plugin(hook, vm) {
    let options = vm.config.timeUpdater;
    let text = options.text;
    let formatUpdated = options.formatUpdated;
    let whereToPlace = String(options.whereToPlace).toLowerCase();
    let repo = options.repo;
    let branch = options.branch || "main";
    let token = options.token;
  
    hook.beforeEach(function(content, next) {
      const path = vm.route.file;
      
      if (!repo) {
        console.error("GitHub repository is not specified.");
        next(content);
        return;
      }
  
      // 构造 GitHub API 请求 URL
      const apiUrl = `https://api.github.com/repos/${repo}/commits?path=${path}&sha=${branch}&per_page=1`;
  
      // 设置请求头，如果提供了 GitHub Token
      const headers = token ? { Authorization: `token ${token}` } : {};
  
      fetch(apiUrl, { headers })
        .then(response => response.json())
        .then(data => {
          if (data && data.length > 0) {
            const lastCommitDate = new Date(data[0].commit.author.date);
            const formattedDate = formatDate(lastCommitDate, formatUpdated);
            const updatedText = text.replace('{docsify-updated}', formattedDate);
  
            if (whereToPlace === 'top') {
              content = updatedText + "\n\n" + content;
            } else {
              content = content + "\n\n" + updatedText;
            }
          } else {
            console.warn("No commit history found for this file.");
          }
          next(content);
        })
        .catch(err => {
          console.error("Failed to fetch last modified date from GitHub:", err);
          next(content);
        });
    });
  }
  
  window.$docsify = window.$docsify || {};
  window.$docsify.timeUpdater = Object.assign(defaultDocsifyUpdatedOptions, window.$docsify.timeUpdater);
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(plugin);
  
# 🔧 GitHub Actions 设置指南

## 权限问题解决方案

如果您遇到 `403 Forbidden` 错误，请按照以下步骤解决：

### 方法1：设置仓库权限（推荐）

1. **进入仓库设置**：
   - 在您的 GitHub 仓库页面
   - 点击 "Settings" 标签
   - 在左侧菜单中找到 "Actions" → "General"

2. **配置工作流权限**：
   - 找到 "Workflow permissions" 部分
   - 选择 "Read and write permissions"
   - 勾选 "Allow GitHub Actions to create and approve pull requests"
   - 点击 "Save"

3. **配置分支保护**（如果有）：
   - 在 "Settings" → "Branches" 中
   - 找到您的分支保护规则
   - 确保 "Allow GitHub Actions to create and approve pull requests" 已启用

### 方法2：使用 Personal Access Token

如果方法1不起作用，可以创建 Personal Access Token：

1. **创建 Token**：
   - 进入 GitHub Settings → Developer settings → Personal access tokens
   - 点击 "Generate new token (classic)"
   - 选择 "repo" 权限
   - 复制生成的 token

2. **添加 Token 到仓库**：
   - 在仓库 Settings → Secrets and variables → Actions
   - 点击 "New repository secret"
   - Name: `PAT_TOKEN`
   - Value: 粘贴您的 token

3. **更新工作流**：
   将工作流中的 `GITHUB_TOKEN` 替换为 `PAT_TOKEN`：

```yaml
env:
  GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
```

### 方法3：简化工作流

使用我们提供的简化工作流 `simple-update.yml`：

```bash
# 删除复杂的工作流文件
rm .github/workflows/arxiv_bot.yml
rm .github/workflows/rss-service.yml
rm .github/workflows/config-update.yml

# 只保留简化版本
# .github/workflows/simple-update.yml
```

## 工作流触发条件

当前配置的工作流会在以下情况触发：

### 主工作流 (arxiv_bot.yml)
- **Push 触发**：当以下文件被修改时
  - `config.json` - 配置更改
  - `arxiv_bot.py` - 机器人代码更改
  - `CUSTOM_RULES.md` - 自定义规则文档
  - `example_custom_rules.py` - 自定义规则代码
  - `rss_service.py` - RSS 服务代码

- **定时触发**：每天 12:00 UTC
- **手动触发**：在 Actions 页面手动运行

### 简化工作流 (simple-update.yml)
- **Push 触发**：当以下文件被修改时
  - `config.json` - 配置更改
  - `arxiv_bot.py` - 机器人代码更改

## 测试工作流

1. **本地测试**：
```bash
python test_workflow.py
```

2. **触发测试**：
```bash
# 修改配置文件
echo '{"test": "update"}' >> config.json
git add config.json
git commit -m "test: trigger workflow"
git push origin master
```

3. **检查 Actions**：
- 进入仓库的 "Actions" 标签
- 查看工作流是否成功运行

## 常见问题

### Q: 工作流没有触发？
A: 检查：
- 文件路径是否正确
- 分支名称是否正确 (main/master)
- 文件是否真的被修改

### Q: 权限被拒绝？
A: 按照上面的方法1或方法2设置权限

### Q: 推送失败？
A: 确保：
- 使用了正确的 git 配置
- 有足够的权限
- 网络连接正常

## 推荐设置

对于大多数用户，推荐使用：

1. **简化工作流** (`simple-update.yml`)
2. **设置仓库权限** (方法1)
3. **定期检查 Actions 日志**

这样可以避免复杂的权限问题，同时保持功能完整。 
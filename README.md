<p align="center">
  <img src="pages/settings/logo.svg" width="100" alt="映境 Visora" />
</p>

# 映境 Visora

把想象，映成图景。

映境是由 [TenmaGabriel0721](https://github.com/TenmaGabriel0721) 独立维护的 AstrBot 图像与视频创作插件。连接已有模型供应商，在对话中生成图片、改图、创作短视频，并为多个 Bot 保留各自的人物身份和自拍参考图。

需要 **AstrBot ≥ 4.29.4、Python ≥ 3.12**，以及支持相应图像或视频协议的模型供应商。

## 安装与开始使用

1. 在 AstrBot 管理面板的插件管理中，通过仓库地址安装：

   ```text
   https://github.com/TenmaGabriel0721/astrbot_plugin_visora
   ```

2. 在 AstrBot 的模型供应商设置中配置模型、API 地址和密钥。
3. 打开左侧拓展页中的 **映境 Visora 管理 → 模型顺序**，选择任务类型，添加供应商和请求协议，保存设置。
4. 在对话中发送 `~生图 一只在窗边看雨的猫`，或启用 LLM 工具后直接提出创作要求。

管理页中的供应商来自 AstrBot 已有配置，不需要重复输入 API 地址或密钥。

## 模型顺序

文生图、图生图、人格自拍和管理员生图分别拥有自己的模型列表。启用的模型按从上到下的顺序尝试；使用 **上移 / 下移** 交换顺序，保存时会自动同步所有优先级。

每项只使用选定的请求协议。失败后切换到同一任务类型的下一个模型，不会向当前供应商轮询其他协议。视频生成的供应商也可以在此页选择。

| 协议 | 请求路径 | 适用任务 |
| --- | --- | --- |
| Gemini 原生（`gemini`） | `v1beta/models/{model}:generateContent` | 文生图、图生图、人格自拍 |
| OpenAI Chat（`openai_chat`） | `v1/chat/completions` | 文生图、图生图、人格自拍 |
| OpenAI Images（`openai_images`） | `v1/images/generations` | 文生图 |
| OpenAI Edits（`openai_edits`） | `v1/images/edits` | 图生图、人格自拍 |

特殊接口可选择“自定义路径”。路径仍需使用上述协议之一的请求体和响应结构；全新的协议需要新增适配代码。

原 `image_model_routes` 表单作为高级入口保留。任务类型为 `txt2img`、`img2img`、`persona`、`admin`；`priority` 越小越优先，`enabled` 控制是否参与生成。

## 各 Bot 独立自拍

打开 **映境 Visora 管理 → 各 Bot 自拍**：

1. 新建人物资料，填写名称并从下拉列表选择要绑定的 Bot。
2. 填写人物身份补充，选择真人摄影、动漫插画或跟随本次要求，以及默认分辨率。
3. 保存资料，再上传这个人物的参考图。

每套资料最多 4 张 PNG、JPEG 或 WebP 图片；管理页每张最多 10 MB，同时受插件图片大小限制。不同 Bot 的资料与图片独立保存。绑定使用 AstrBot 消息平台的 **实例 ID**，工具自动读取收到消息的 Bot 所绑定的资料，用户和 LLM 不能通过参数切换到另一个人物。

人物参考图锁定身份、发色、瞳色与体型。本次消息或引用中的额外图片只作为最后一张穿搭、姿势或构图参考，不采用 @ 用户头像。输入数量受 `max_reference_images_per_request` 限制，同时使用人物和穿搭参考时至少需要 2 张额度。

未绑定资料或没有参考图时，工具会提示配置，不提交任务、不扣积分。用户原消息明确指定的分辨率优先，其次是 Bot 的默认值，再其次是插件的自拍默认值。成片使用第三人称人像构图。

参考图命令也会自动选择当前 Bot：

```text
~自拍参考图
~自拍参考图 上传
~自拍参考图 删除 <序号或文件名>
```

查看向所有用户开放；上传和删除需要管理员权限。

## 对话命令

以下示例使用 `~` 前缀，实际以前缀配置为准。

| 命令 | 用途 |
| --- | --- |
| `~生图 <提示词或预设名称> [补充要求]` | 自动根据参考图选择文生图或图生图 |
| `~文生图 <提示词>` | 纯文字生成图片 |
| `~管理员生图 <提示词>` | 使用独立管理员模型列表 |
| `~生视频 <提示词>` | 根据文字或参考图生成视频 |
| `~文生视频 <提示词>` | 纯文字生成视频 |
| `~生图帮助` / `~视频帮助` | 查看用法 |
| `~预设` | 查看可用预设 |
| `~预设 添加 <名称>:<提示词>` | 添加预设，需要管理员权限 |
| `~预设 删除 <名称>` | 删除预设，需要管理员权限 |

普通图生图和图生视频的参考图依次取自：当前消息图片、引用消息图片、@ 用户头像。

预设支持普通提示词，也可以指定比例和分辨率：

```text
~预设 添加 像素风:pixel art, warm colors, clean composition
~生图 像素风 一只猫
~像素风 一只猫
```

```text
海报:{"prompt":"minimalist travel poster","aspect_ratio":"3:4","resolution":"2K"}
```

比例支持 `1:1`、`2:3`、`3:2`、`3:4`、`4:3`、`4:5`、`5:4`、`9:16`、`16:9`、`21:9`、`auto`。分辨率支持 `1K`、`2K`、`4K`，实际输出取决于模型。

## LLM 工具与后台任务

启用 `enable_llm_tool` 后提供以下工具：

| 工具名 | 功能 |
| --- | --- |
| `visora_generate_image` | 文生图、图生图与头像参考 |
| `visora_persona_photo` | 根据当前 Bot 的独立资料生成生活人像或动漫插画 |
| `visora_image_task_status` | 查询当前用户在当前会话中的图片任务 |
| `visora_generate_video` | 文生视频与图生视频 |

图片任务在后台执行。LLM 先告知已受理，完成后插件发送图片；确认发送成功后，再进入原会话让 LLM 按当前人格自然收尾。会话切换、重置或插件重载后，无法恢复的旧上下文会使用固定完成提示。

后台队列和任务记录持久化保存。任务总时限 `task_timeout` 包含各模型的重试、回退与结果下载，默认 600 秒。发生 4K 降级或输出尺寸不足时，结果会说明实际分辨率。

所有生成入口沿用频率、并发、参考图大小和可选积分限制；管理员生图需要 AstrBot 管理员权限。

## 视频配置

视频能力面向 NewAPI / OpenAI 兼容服务。选择支持视频生成的供应商，再根据服务配置：

| 配置 | 用途 |
| --- | --- |
| `video_provider_id` | 视频模型供应商 |
| `video_api_request_formats` | 提交路径，默认 `v1/videos/generations` |
| `video_task_query_format` | 异步任务查询路径模板；响应提供 `status_url` 时可留空 |
| `video_timeout` / `video_poll_interval` | 等待时限与轮询间隔 |
| `default_video_duration` / `default_video_ratio` | 默认时长与比例 |
| `video_public_base_url` | 需要公网参考图时，填写外部可访问的 AstrBot 地址 |

支持直接视频 URL、Base64 结果以及任务 ID 加轮询的常见响应结构。具体时长、分辨率和图片输入能力以所选服务为准。

## 积分系统

默认关闭。启用后，每个用户维护自己的积分余额；生成失败或明确取消的任务按任务状态退款。

```text
~祈福
~查询
~货币 <新名称>
~充值 <用户ID> <数量>
~充值 <数量> @用户
```

修改名称和充值需要管理员权限。消耗、签到奖励和初始积分可在插件配置中调整。

## 常用配置

| 配置 | 用途 |
| --- | --- |
| `persona_profiles` | Bot 绑定、人物身份、画风和分辨率 |
| `default_aspect_ratio` / `default_resolution` | 普通生图的默认比例与分辨率 |
| `persona_photo_default_resolution` | 人格自拍的默认分辨率 |
| `timeout` / `task_timeout` | 单次请求与完整图片任务时限 |
| `max_concurrent_generations` / `max_queued_generations` | 同时执行数量与队列容量 |
| `max_requests_per_minute` | 每个用户的请求频率上限 |
| `max_image_size_mb` / `max_reference_images_per_request` | 参考图大小与数量上限 |
| `max_avatar_refs_per_request` | 单次头像参考数量上限 |
| `max_retry_attempts` | 请求失败时的重试次数 |
| `enable_economy` / `cost_per_image` / `cost_per_video` | 积分开关与消耗 |
| `proxy` | 上游请求代理 |

## 数据与旧版迁移

运行数据位于 AstrBot 的 `data/plugin_data/astrbot_plugin_visora/`：

```text
economy.json                           # 积分账本
persona_profiles/<资料 ID>/            # 独立人物参考图
background_tasks/                      # 后台任务数据库与任务文件
```

配置保存在 `data/config/astrbot_plugin_visora_config.json`；多配置实例带相应后缀。源码仓库不包含 API 密钥、实际 Bot 配置、账本或人物照片。

从旧的 `astrbot_plugin_gemini_image` 迁移时，先停用旧插件，备份并迁移配置文件、完整插件数据目录，再启用映境。模型顺序和人物资料无需重建。旧源码目录中的 `data.json` 和 `persona_refs/` 如尚未迁入持久化目录，需一并保留；插件会在持久化目标不存在时进行一次性导入，不覆盖已有数据。

如其他插件或人格提示词写死了旧工具名，按以下映射调整：

| 旧名称 | 映境名称 |
| --- | --- |
| `gemini_generate_image` | `visora_generate_image` |
| `gemini_image_task_status` | `visora_image_task_status` |
| `gemini_generate_video` | `visora_generate_video` |
| `shoubanhua_persona_photo` | `visora_persona_photo` |

旧管理页“Gemini 生图管理”对应新入口“映境 Visora 管理”。请通过本仓库更新映境，避免使用旧插件的更新源。

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

测试覆盖模型路由、请求协议、分辨率降级、后台任务与退款、Bot 身份隔离、参考图管理及配置保存。GitHub Actions 在提交和拉取请求时执行检查。

## 许可证与来源

映境使用 [GNU AGPL v3](LICENSE)，由 TenmaGabriel0721 独立维护。部分早期图像生成代码源自 [Railgun19457/astrbot_plugin_gemini_image](https://github.com/Railgun19457/astrbot_plugin_gemini_image)，保留其许可证；独立品牌和后续开发信息见 [NOTICE](NOTICE)。

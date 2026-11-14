"use strict";

(async function () {
  const bridge = window.AstrBotPluginView;
  const get = id => document.getElementById(id);
  const modes = ["txt2img", "img2img", "persona", "admin"];
  const state = { routes: {}, profiles: [], providers: [], bots: [], formats: [], mode: "txt2img", profileId: "", revision: "", dirty: false, busy: false, savedProfileIds: new Set() };
  let referenceRequest = 0;

  function status(message, error = false) {
    get("status").textContent = message;
    get("status").classList.toggle("error", error);
  }

  function changed() {
    state.dirty = true;
    status("更改尚未保存");
  }

  function busy(value) {
    state.busy = value;
    get("editor").disabled = value || !state.revision;
    get("save").disabled = value || !state.revision;
    for (const id of ["refresh", "reload", "models-tab", "personas-tab"]) get(id).disabled = value;
    const profile = state.profiles.find(item => item.id === state.profileId);
    get("reference-upload").disabled = value || !profile || !state.savedProfileIds.has(profile.id) || profile.reference_count >= state.max_reference_count;
  }

  function option(select, value, label, disabled = false) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    node.disabled = disabled;
    select.append(node);
  }

  function populateProviders(select, value, optional = false) {
    select.replaceChildren();
    option(select, "", optional ? "暂不配置" : "请选择模型供应商");
    const preferred = document.createElement("optgroup");
    preferred.label = state.mode === "video" ? "视频模型" : "图片模型";
    const others = document.createElement("optgroup");
    others.label = "其他模型";
    const matcher = state.mode === "video" ? /video|veo|sora|kling|wan/i : /image|imagen|dall.?e|flux|z-image|stable.?diffusion/i;
    for (const provider of state.providers) option(matcher.test(provider.model) ? preferred : others, provider.id, provider.id);
    if (preferred.children.length) select.append(preferred);
    if (others.children.length) select.append(others);
    if (value && !state.providers.some(item => item.id === value)) option(select, value, value + "（当前不可用）");
    select.value = value;
  }

  function suggestedFormat(providerId, mode) {
    const provider = state.providers.find(item => item.id === providerId);
    const model = provider?.model || providerId;
    const alias = /gemini|imagen/i.test(model) ? "gemini" : (mode === "img2img" || mode === "persona" ? "openai_edits" : "openai_images");
    return state.formats.find(item => item.value === alias)?.endpoint || "v1/images/generations";
  }

  function renderRoutes() {
    const isVideo = state.mode === "video";
    get("video-settings").hidden = !isVideo;
    get("deadline-settings").hidden = isVideo;
    get("add-route").hidden = isVideo;
    get("route-help").hidden = isVideo;
    get("routes").replaceChildren();
    if (isVideo) {
      populateProviders(get("video-provider"), state.video_provider_id, true);
      return;
    }
    const items = state.routes[state.mode] || [];
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "此任务类型尚未配置模型。添加模型后即可使用。";
      get("routes").append(empty);
    }
    let rank = 0;
    items.forEach((route, index) => {
      const card = document.createElement("article");
      card.className = "card route-card" + (route.enabled ? "" : " off");
      card.dataset.provider = route.provider_id;
      const head = document.createElement("div");
      head.className = "route-head";
      const title = document.createElement("div");
      title.className = "route-title";
      const order = document.createElement("span");
      order.className = "route-rank";
      order.textContent = route.enabled ? (rank++ === 0 ? "首选模型" : "回退 " + (rank - 1)) : "已停用";
      const toggle = document.createElement("label");
      toggle.className = "toggle";
      const enabled = document.createElement("input");
      enabled.type = "checkbox";
      enabled.checked = route.enabled;
      enabled.addEventListener("change", () => { route.enabled = enabled.checked; renderRoutes(); changed(); });
      toggle.append(enabled, document.createTextNode("启用"));
      title.append(order, toggle);
      const actions = document.createElement("div");
      actions.className = "actions";
      for (const [label, step] of [["上移", -1], ["下移", 1], ["移除", 0]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.setAttribute("aria-label", label + " " + route.provider_id);
        button.disabled = step !== 0 && (index + step < 0 || index + step >= items.length);
        button.addEventListener("click", () => {
          if (step === 0) items.splice(index, 1);
          else [items[index], items[index + step]] = [items[index + step], items[index]];
          renderRoutes();
          changed();
        });
        actions.append(button);
      }
      head.append(title, actions);
      const fields = document.createElement("div");
      fields.className = "fields";
      const providerField = document.createElement("label");
      providerField.className = "field";
      providerField.textContent = "模型供应商";
      const provider = document.createElement("select");
      populateProviders(provider, route.provider_id);
      provider.addEventListener("change", () => {
        if (!provider.value || items.some(item => item !== route && item.provider_id === provider.value)) {
          provider.value = route.provider_id;
          status("该模型已在当前列表中，请选择其他供应商", true);
          return;
        }
        if (route.request_format === suggestedFormat(route.provider_id, state.mode)) route.request_format = suggestedFormat(provider.value, state.mode);
        route.provider_id = provider.value;
        renderRoutes();
        changed();
      });
      providerField.append(provider);
      const protocolField = document.createElement("label");
      protocolField.className = "field";
      protocolField.textContent = "请求协议";
      const protocol = document.createElement("select");
      const formats = state.formats.filter(item => {
        if ((state.mode === "txt2img" || state.mode === "admin") && item.value === "openai_edits") return false;
        if ((state.mode === "img2img" || state.mode === "persona") && item.value === "openai_images") return false;
        return true;
      });
      for (const format of formats) option(protocol, format.endpoint, format.label);
      option(protocol, "custom", "自定义路径");
      const isCustom = !formats.some(item => item.endpoint === route.request_format);
      protocol.value = isCustom ? "custom" : route.request_format;
      protocol.addEventListener("change", () => {
        route.request_format = protocol.value === "custom" ? "" : protocol.value;
        renderRoutes();
        changed();
      });
      protocolField.append(protocol);
      fields.append(providerField, protocolField);
      card.append(head, fields);
      if (isCustom) {
        const label = document.createElement("label");
        label.className = "field custom-path";
        label.textContent = "自定义请求路径";
        const path = document.createElement("input");
        path.type = "text";
        path.maxLength = 512;
        path.value = route.request_format;
        path.placeholder = "例如 v2/images/generations";
        path.addEventListener("input", () => { route.request_format = path.value; changed(); });
        label.append(path);
        card.append(label);
      }
      get("routes").append(card);
    });
  }

  async function loadReferences(profileId) {
    const requestId = ++referenceRequest;
    get("references").replaceChildren();
    if (!state.savedProfileIds.has(profileId)) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "先保存新资料，再上传这个 Bot 的参考图。";
      get("references").append(empty);
      return;
    }
    try {
      const data = await bridge.apiGet("page/references/" + encodeURIComponent(profileId));
      if (requestId !== referenceRequest || state.profileId !== profileId) return;
      const profile = state.profiles.find(item => item.id === profileId);
      if (profile) profile.reference_count = data.references.length;
      get("references").replaceChildren();
      if (!data.references.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "还没有参考图。上传后，这个 Bot 的自拍会使用自己的形象。";
        get("references").append(empty);
      }
      for (const reference of data.references) {
        const card = document.createElement("article");
        card.className = "reference";
        if (reference.data_url) {
          const preview = document.createElement("img");
          preview.src = reference.data_url;
          preview.alt = profile.name + "的身份参考图";
          card.append(preview);
        }
        const name = document.createElement("p");
        name.className = "filename";
        name.textContent = reference.name + (reference.data_url ? "" : "（无法预览，请检查图片）");
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "danger-text";
        remove.textContent = "删除参考图";
        remove.addEventListener("click", async () => {
          busy(true);
          try {
            await bridge.apiPost("page/references/" + encodeURIComponent(profileId) + "/delete", { names: [reference.name] });
            await loadReferences(profileId);
            status("已删除所选参考图" + (state.dirty ? "；资料更改尚未保存" : ""));
          } catch (error) { status(error.message || String(error), true); }
          finally { busy(false); }
        });
        card.append(name, remove);
        get("references").append(card);
      }
      get("reference-help").textContent = `${data.references.length} / ${state.max_reference_count} 张 · PNG、JPEG 或 WebP，每张最多 ${Math.floor(state.max_reference_bytes / 1024 / 1024)} MB`;
      busy(state.busy);
    } catch (error) {
      if (requestId === referenceRequest && state.profileId === profileId) status(error.message || String(error), true);
    }
  }

  function renderProfile() {
    get("profile").replaceChildren();
    for (const profile of state.profiles) option(get("profile"), profile.id, profile.name);
    if (!state.profiles.some(item => item.id === state.profileId)) state.profileId = state.profiles[0]?.id || "";
    get("profile").value = state.profileId;
    const profile = state.profiles.find(item => item.id === state.profileId);
    get("profile-empty").hidden = Boolean(profile);
    get("profile-editor").hidden = !profile;
    get("remove-profile").disabled = !profile;
    if (!profile) { referenceRequest++; return; }
    get("profile-name").value = profile.name;
    get("profile-identity").value = profile.identity_prompt;
    get("profile-style").value = profile.style;
    get("profile-resolution").value = profile.resolution;
    const botSelect = get("profile-bot");
    botSelect.replaceChildren();
    option(botSelect, "", "暂不绑定");
    for (const bot of state.bots) {
      const occupied = state.profiles.some(item => item.id !== profile.id && item.bot_id === bot.id);
      option(botSelect, bot.id, `${bot.name} · ${bot.type}（${bot.id}）` + (occupied ? " · 已绑定" : ""), occupied);
    }
    if (profile.bot_id && !state.bots.some(item => item.id === profile.bot_id)) option(botSelect, profile.bot_id, profile.bot_id + "（当前未配置）");
    botSelect.value = profile.bot_id;
    get("reference-help").textContent = "每套资料单独保存参考图。";
    get("reference-upload").value = "";
    busy(state.busy);
    loadReferences(profile.id);
  }

  function applySettings(data) {
    for (const mode of modes) state.routes[mode] = data.routes.filter(item => item.__template_key === mode);
    for (const key of ["profiles", "providers", "bots", "formats", "revision", "task_timeout", "video_provider_id", "max_reference_count", "max_reference_bytes"]) state[key] = data[key];
    state.savedProfileIds = new Set(state.profiles.map(item => item.id));
    state.dirty = false;
    get("task-timeout").value = state.task_timeout;
    renderRoutes();
    renderProfile();
  }

  async function saveSettings() {
    busy(true);
    status("正在保存…");
    try {
      const routes = modes.flatMap(mode => state.routes[mode].map((item, priority) => ({ ...item, __template_key: mode, priority })));
      const data = await bridge.apiPost("page/settings", { routes, profiles: state.profiles, revision: state.revision, task_timeout: Number(get("task-timeout").value), video_provider_id: state.video_provider_id });
      applySettings(data);
      status("已保存并应用");
    } catch (error) { status(error.message || String(error), true); }
    finally { busy(false); }
  }

  for (const [tab, panel] of [["models-tab", "models"], ["personas-tab", "personas"]]) {
    get(tab).addEventListener("click", () => {
      for (const id of ["models", "personas"]) {
        const active = id === panel;
        get(id).hidden = !active;
        const button = get(id + "-tab");
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      }
    });
  }
  get("mode").addEventListener("change", () => { state.mode = get("mode").value; renderRoutes(); });
  get("add-route").addEventListener("click", () => {
    const items = state.routes[state.mode];
    const available = state.providers.filter(provider => !items.some(item => item.provider_id === provider.id));
    const provider = available.find(item => /image|imagen|dall.?e|flux|z-image/i.test(item.model)) || available[0];
    if (!provider) { status("没有可添加的供应商，请先在 AstrBot 模型配置中添加模型", true); return; }
    items.push({ provider_id: provider.id, request_format: suggestedFormat(provider.id, state.mode), enabled: true });
    renderRoutes();
    changed();
  });
  get("video-provider").addEventListener("change", () => { state.video_provider_id = get("video-provider").value; changed(); });
  get("task-timeout").addEventListener("input", changed);
  get("profile").addEventListener("change", () => { state.profileId = get("profile").value; renderProfile(); });
  for (const [id, key] of [["profile-name", "name"], ["profile-bot", "bot_id"], ["profile-style", "style"], ["profile-resolution", "resolution"], ["profile-identity", "identity_prompt"]]) {
    get(id).addEventListener(id === "profile-name" || id === "profile-identity" ? "input" : "change", () => {
      const profile = state.profiles.find(item => item.id === state.profileId);
      if (!profile) return;
      profile[key] = get(id).value;
      if (key === "name") {
        const selected = Array.from(get("profile").options).find(item => item.value === profile.id);
        if (selected) selected.textContent = profile.name || "未命名资料";
      }
      changed();
    });
  }
  get("add-profile").addEventListener("click", () => {
    if (state.profiles.length >= 32) { status("最多保留 32 套自拍资料", true); return; }
    const bot = state.bots.find(item => !state.profiles.some(profile => profile.bot_id === item.id));
    const id = (globalThis.crypto?.randomUUID?.() || `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`).replaceAll("-", "");
    state.profiles.push({ __template_key: "profile", id, name: bot?.name || "新自拍资料", bot_id: bot?.id || "", identity_prompt: "", style: "photo", resolution: "inherit", reference_count: 0 });
    state.profileId = id;
    renderProfile();
    changed();
  });
  get("remove-profile").addEventListener("click", () => {
    state.profiles = state.profiles.filter(item => item.id !== state.profileId);
    state.profileId = "";
    renderProfile();
    changed();
  });
  get("reference-upload").addEventListener("change", async event => {
    const profile = state.profiles.find(item => item.id === state.profileId);
    const files = Array.from(event.target.files || []);
    if (!profile || !files.length) return;
    if (profile.reference_count + files.length > state.max_reference_count || files.some(file => file.size > state.max_reference_bytes || !["image/png", "image/jpeg", "image/webp"].includes(file.type))) {
      event.target.value = "";
      status(`每套最多 ${state.max_reference_count} 张 PNG、JPEG 或 WebP，请检查图片数量与大小`, true);
      return;
    }
    busy(true);
    try {
      for (const file of files) await bridge.upload("page/references/" + encodeURIComponent(profile.id), file);
      status(`已为${profile.name}上传 ${files.length} 张参考图` + (state.dirty ? "；资料更改尚未保存" : ""));
    } catch (error) { status(error.message || String(error), true); }
    finally { event.target.value = ""; await loadReferences(profile.id); busy(false); }
  });
  get("save").addEventListener("click", saveSettings);
  get("reload").addEventListener("click", async () => {
    busy(true);
    try { applySettings(await bridge.apiGet("page/settings")); status("已重新加载当前设置"); }
    catch (error) { status(error.message || String(error), true); }
    finally { busy(false); }
  });
  get("refresh").addEventListener("click", async () => {
    busy(true);
    try {
      const data = await bridge.apiGet("page/settings");
      state.providers = data.providers;
      state.bots = data.bots;
      renderRoutes();
      renderProfile();
      status("已刷新供应商和 Bot 列表" + (state.dirty ? "；更改尚未保存" : ""));
    } catch (error) { status(error.message || String(error), true); }
    finally { busy(false); }
  });

  if (!bridge) { status("请从 AstrBot 管理面板打开此页面", true); return; }
  try {
    await bridge.ready();
    bridge.onContext(context => { document.documentElement.dataset.theme = context?.isDark ? "dark" : "light"; });
    applySettings(await bridge.apiGet("page/settings"));
    status("已加载当前设置");
    busy(false);
  } catch (error) { status(error.message || String(error), true); }
})();

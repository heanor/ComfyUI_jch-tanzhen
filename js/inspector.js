import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "JCH.TensorInspector.UI",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "JCH_TensorInspector") {
			
			// --- 1. 找回并保留原本的节点创建逻辑 ---
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function() {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				
				// 【关键修改】在节点创建时就初始化组件，确保它被注册到工作流 JSON 中实现持久化
				if (!this.widgets || !this.widgets.find(w => w.name === "display_text")) {
					const widget = ComfyWidgets["STRING"](this, "display_text", ["STRING", { multiline: true }], app).widget;
					widget.inputEl.readOnly = true; 
					widget.inputEl.style.opacity = 0.9;
					widget.inputEl.style.backgroundColor = "#222"; 
					widget.inputEl.style.fontFamily = "monospace";
					widget.inputEl.style.fontSize = "12px";
				}
				
				// 设置一个初始的体面尺寸
				this.setSize([350, 200]);
				return r;
			};

			// --- 2. 劫持执行逻辑，仅负责更新数据内容 ---
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function(message) {
				onExecuted?.apply(this, arguments);

				if (message && message.text) {
					// 查找已经存在的组件（它现在一定存在，因为出生时就创建了）
					let widget = this.widgets && this.widgets.find(w => w.name === "display_text");
					
					if (widget) {
						// 将 Python 传来的文本数组合并并赋值
						widget.value = message.text.join("");
						
						// 动态调整节点高度以适应内容
						this.onResize?.(this.computeSize());
					}
				}
			};
		}
	}
});
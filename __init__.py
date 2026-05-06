import os
import torch

# 保持前端脚本握手
WEB_DIRECTORY = "./js"

class JCH_TensorInspector:
	@classmethod
	def INPUT_TYPES(cls):
		return {"required": {"anything": ("*", {}), "label": ("STRING", {"default": "Debug_Node"})}}

	RETURN_TYPES = ("*",)
	RETURN_NAMES = ("anything",)
	FUNCTION = "inspect_data"
	CATEGORY = "JCH_Tools"
	OUTPUT_NODE = True 

	def format_shape(self, shape, is_weight=False):
		"""
		精准维度映射标注：引入“高度(Output) x 宽度(Input)”建模逻辑
		"""
		s = list(shape)
		if len(s) == 5:
			return f"{s} | [批次:{s[0]}, 通道:{s[1]}, 帧:{s[2]}, 高:{s[3]}, 宽:{s[4]}]"
		
		if len(s) == 4:
			if is_weight:
				return f"{s} | [输出高度(特征):{s[0]}, 输入宽度(信号):{s[1]}, 卷积核:{s[2]}x{s[3]}]"
			return f"{s} | [批次:{s[0]}, 通道:{s[1]}, 纵向高:{s[2]}, 横向宽:{s[3]}]"
		
		if len(s) == 3:
			if is_weight:
				return f"{s} | [输出高度:{s[0]}, 输入宽度:{s[1]}, 映射深度:{s[2]}]"
			if abs(s[1] - s[2]) < max(s[1], s[2]) * 0.4:
				return f"{s} | [批次:{s[0]}, 纵向高:{s[1]}, 横向宽:{s[2]}] (空间型)"
			return f"{s} | [批次:{s[0]}, 序列长度(词数):{s[1]}, 特征维度:{s[2]}] (特征型)"
		
		if len(s) == 2:
			if is_weight:
				return f"{s} | [输出高度:{s[0]}, 输入宽度:{s[1]}]"
			return f"{s} | [纵向高:{s[0]}, 横向宽:{s[1]}]"
		return f"{s}"

	def get_vram_usage(self, tensor):
		"""显存占用物理计算 (MB)"""
		try:
			return f"{(tensor.numel() * tensor.element_size()) / (1024 * 1024):.2f} MB"
		except Exception: return "未知"

	def get_shape_interpretation(self, shape, is_weight=False):
		"""全量常数推导：锁定核心带宽识别"""
		s = list(shape)
		standard_h = {
			768: "CLIP-L", 1024: "CLIP-G", 1280: "SDXL-Core", 2048: "Gemma",
			2432: "SD3-8B", 2560: "Qwen-4B", 3072: "FLUX.1", 3584: "Qwen-7B", 4096: "T5/Llama3"
		}
		if len(s) >= 2:
			width = s[-1]
			if width == 12288: return f"🧬 融合 QKV 空间 (3.0x) | 源自 4096 核心 (Qwen3) | {self.format_shape(s, is_weight=is_weight)}"
			for h, name in standard_h.items():
				if width == h * 3: return f"🧬 融合 QKV 投影 (3.0x) | 源自 {name} | {self.format_shape(s, is_weight=is_weight)}"
				if width == h * 4: return f"🚀 MLP 膨胀空间 (4.0x) | 源自 {name} | {self.format_shape(s, is_weight=is_weight)}"
				if width == h * 3.5: return f"🚀 MLP 膨胀空间 (3.5x) | 源自 {name} | {self.format_shape(s, is_weight=is_weight)}"
			if width in standard_h: return f"标准语义特征层 ({standard_h[width]}) | {self.format_shape(s, is_weight=is_weight)}"
		return f"💡 结构: {self.format_shape(s, is_weight=is_weight)}"

	def inspect_tensor_stats(self, tensor, log, prefix="  "):
		"""深度生化指标统计"""
		vram = self.get_vram_usage(tensor)
		log.append(f"{prefix}├─ 运行环境: [设备:{tensor.device}] | [精度:{tensor.dtype}] | [物理体积: {vram}]")
		if tensor.is_floating_point():
			try:
				with torch.no_grad():
					t = tensor.detach()
					v_min, v_max = t.min().item(), t.max().item()
					# 计算均值时指定 dtype 为 float32，避免创建巨大的浮点副本导致显存溢出
					v_mean = t.mean(dtype=torch.float32).item()
					v_nan = torch.isnan(t).any().item()
					log.append(f"{prefix}├─ 数值区间: [{v_min:.3f} ~ {v_max:.3f}] (均值:{v_mean:.3f})")
					if v_nan: log.append(f"{prefix}⚠️ 警告: 探测到 NaN")
			except Exception: pass

	def find_block_width_recursive(self, block):
		"""递归寻找 Block 内部隐藏的特征维度"""
		for module in block.modules():
			if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
				return module.weight.shape[0]
			if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
				if len(module.weight.shape) > 0:
					return module.weight.shape[0]
		return "N/A"

	def inspect_data(self, anything, label):
		log = []
		pipe_type = "未知 (Unknown)"
		temp_log = []

		# --- A: 模型/VAE/CLIP 扫描 ---
		if hasattr(anything, 'first_stage_model') or 'VAE' in str(type(anything)):
			pipe_type = "VAE"
			temp_log.append("💎 [VAE 编解码器扫描]")
			vae = getattr(anything, 'first_stage_model', anything)
			try:
				temp_log.append(f"  ├─ VAE 架构: {type(vae).__name__}")
				self.inspect_tensor_stats(next(vae.parameters()), temp_log)
			except Exception: pass

		elif hasattr(anything, 'cond_stage_model') or 'CLIP' in str(type(anything)):
			pipe_type = "CLIP"
			temp_log.append("🎙️ [CLIP/文本编码器扫描]")
			clip = getattr(anything, 'cond_stage_model', anything)
			try: self.inspect_tensor_stats(next(clip.parameters()), temp_log)
			except Exception: pass

		elif hasattr(anything, 'model') and not isinstance(anything, torch.Tensor):
			pipe_type = "Model"
			temp_log.append("🧱 [模型计算体架构扫描]")
			patcher = anything
			diff = patcher.model.diffusion_model
			
			# 1. 架构身份探测
			if hasattr(diff, "input_blocks"):
				pipe_type = "SD1.5/UNet"
				try:
					first_conv = diff.input_blocks[0][0]
					if hasattr(first_conv, "weight"):
						temp_log.append(f"  ├─ 卷积入口门(conv_in): {self.get_shape_interpretation(first_conv.weight.shape, is_weight=True)}")
				except Exception: pass
				temp_log.append(f"  ├─ 📥 入场序列 (Input Stack): {len(diff.input_blocks)} Blocks")
				for i, block in enumerate(diff.input_blocks):
					channels = self.find_block_width_recursive(block)
					temp_log.append(f"  │  └─ Block_{i:02}: 宽度 {channels}")
				if hasattr(diff, "middle_block"): temp_log.append(f"  ├─ ⚙️ 中枢核 (Mid Block): 1 Block")
				if hasattr(diff, "output_blocks"): temp_log.append(f"  ├─ 📤 出场序列 (Output Stack): {len(diff.output_blocks)} Blocks")
			
			elif hasattr(diff, "joint_blocks"):
				pipe_type = "SD3/MM-DiT"
				temp_log.append(f"  ├─ 内核架构: {type(diff).__name__} ({len(diff.joint_blocks)} Joint Blocks)")
				# --- V23 增强: SD3 核心专家阵列解剖 ---
				try:
					b0 = diff.joint_blocks[0]
					if hasattr(b0, "x_block") and hasattr(b0.x_block, "attn"):
						qkv = b0.x_block.attn.qkv
						temp_log.append(f"  ├─ 🔬 [核心车间 0号抽检]: 融合 QKV 墙: {self.get_shape_interpretation(qkv.weight.shape, is_weight=True)}")
						# 新增：探测 38 个专家属性
						if hasattr(b0.x_block.attn, "num_heads"):
							nh = b0.x_block.attn.num_heads
							hd = qkv.weight.shape[1] // nh # 输入宽度除以专家数
							temp_log.append(f"  │  └─ 专家阵列: {nh} 位专家 (每位分管 {hd} 维带宽)")
				except Exception: pass
			
			elif hasattr(diff, "double_blocks"):
				pipe_type = "FLUX/DiT"
				d_blocks, s_blocks = len(diff.double_blocks), len(diff.single_blocks)
				temp_log.append(f"  ├─ 内核架构: {type(diff).__name__} ({d_blocks}D + {s_blocks}S Blocks)")
				# --- V23 增强: FLUX 核心专家阵列解剖 ---
				try:
					b0 = diff.double_blocks[0]
					if hasattr(b0, "img_attn"):
						qkv = b0.img_attn.qkv
						temp_log.append(f"  ├─ 🔬 [核心车间 0号抽检]: 图像 QKV 墙: {self.get_shape_interpretation(qkv.weight.shape, is_weight=True)}")
						# 新增：探测 FLUX 专家属性 (通常为 24)
						if hasattr(b0.img_attn, "num_heads"):
							nh = b0.img_attn.num_heads
							hd = qkv.weight.shape[1] // nh
							temp_log.append(f"  │  └─ 专家阵列: {nh} 位专家 (每位分管 {hd} 维带宽)")
				except Exception: pass

			# 2. 增强型物理门禁探测
			# --- 图像入口 ---
			for attr in ["img_in", "x_embedder"]:
				if hasattr(diff, attr):
					gate = getattr(diff, attr)
					target = gate.proj if hasattr(gate, 'proj') else (gate.linear if hasattr(gate, 'linear') else gate)
					if hasattr(target, 'weight'):
						temp_log.append(f"  ├─ 图像入口门({attr}): {self.get_shape_interpretation(target.weight.shape, is_weight=True)}")
					else:
						width = self.find_block_width_recursive(gate)
						temp_log.append(f"  ├─ 图像入口门({attr}): [复合模块, 高度:{width}]")
					break
			
			# --- 文本入口 ---
			for attr in ["txt_in", "context_embedder"]:
				if hasattr(diff, attr):
					gate = getattr(diff, attr)
					target = gate.linear if hasattr(gate, 'linear') else gate
					if hasattr(target, 'weight'):
						temp_log.append(f"  ├─ 文本入口门({attr}): {self.get_shape_interpretation(target.weight.shape, is_weight=True)}")
					break
			
			# --- 图像出口 ---
			for attr in ["final_layer", "output_layer"]:
				if hasattr(diff, attr):
					gate = getattr(diff, attr)
					target = gate.linear if hasattr(gate, 'linear') else gate
					if hasattr(target, 'weight'):
						temp_log.append(f"  ├─ 图像出口门({attr}): {self.get_shape_interpretation(target.weight.shape, is_weight=True)}")
					break

			# 3. 潜空间物理与统计指标
			if hasattr(patcher.model, "latent_format"):
				temp_log.append(f"  ├─ 潜空间物理: {type(patcher.model.latent_format).__name__} (压缩率:{getattr(patcher.model.latent_format, 'scale_factor', 1.0):.4f}x)")
			self.inspect_tensor_stats(next(diff.parameters()), temp_log)
			
			# 4. 补丁/Hook 探测
			p_count = len(patcher.patches) if hasattr(patcher, 'patches') else 0
			if p_count > 0:
				temp_log.append(f"  💉 挂载补丁: {p_count} 个外部 Hook")
				try:
					p_val = patcher.patches[next(iter(patcher.patches))]
					temp_log.append(f"  └─ 补丁分析: 探测到活跃强度约为 {p_val[0] if isinstance(p_val, (list, tuple)) else '未知'}")
				except Exception: pass
			else: temp_log.append("  └─ 挂载补丁: 0 (纯净状态)")

		# --- B: 潜空间字典探测 (保留 V21 所有细节) ---
		elif isinstance(anything, dict) and 'samples' in anything:
			pipe_type = "Latent"
			temp_log.append("📦 [潜空间字典探测]")
			v_samples = anything['samples']
			if hasattr(v_samples, 'shape'):
				temp_log.append(f"  ├─ 核心 Latent: {self.get_shape_interpretation(v_samples.shape, is_weight=False)}")
				self.inspect_tensor_stats(v_samples, temp_log)
			for k, v in anything.items():
				if k != 'samples' and hasattr(v, 'shape'):
					icon = "🎭" if "mask" in k.lower() else "附件"
					temp_log.append(f"  ├─ {icon} '{k}': {self.get_shape_interpretation(v.shape, is_weight=False)}")
					self.inspect_tensor_stats(v, temp_log)

		# --- C: 条件管道探测 (保留 V21 所有图标与隐秘隔间分析) ---
		elif isinstance(anything, (list, tuple)):
			pipe_type = "Conditioning"
			temp_log.append(f"🔗 [条件管道探测] 长度: {len(anything)}")
			try:
				if len(anything) > 0 and isinstance(anything[0], (list, tuple)):
					main_t = anything[0][0]
					if hasattr(main_t, 'shape'):
						temp_log.append(f"  ├─ 核心语义矩阵: {self.get_shape_interpretation(main_t.shape, is_weight=False)}")
						self.inspect_tensor_stats(main_t, temp_log)
					meta = anything[0][1]
					if isinstance(meta, dict):
						temp_log.append(f"  💊 [隐秘隔间分析]:")
						for k,v in meta.items():
							icon = "🖼️" if "reference" in k.lower() else "🎯" if "guidance" in k.lower() else "🎭" if "mask" in k.lower() else "⚓"
							if hasattr(v, 'shape'):
								temp_log.append(f"    ├─ {icon} '{k}': {self.get_shape_interpretation(v.shape, is_weight=False)}")
								if "guidance" in k.lower(): temp_log.append(f"    │  └─ 引导强度: {v.item():.2f}")
							elif isinstance(v, (list, tuple)):
								temp_log.append(f"    ├─ {icon} '{k}': [包含 {len(v)} 组参考序列]")
								for idx, item in enumerate(v):
									if hasattr(item, 'shape'): temp_log.append(f"    │  └─ 样板[{idx}]: {self.get_shape_interpretation(item.shape, is_weight=False)}")
			except Exception: pass

		# --- D: 通用张量审计 ---
		elif hasattr(anything, 'shape') and isinstance(anything, torch.Tensor):
			s = list(anything.shape)
			pipe_type = "Image" if len(s) == 4 and (s[1] == 3 or s[3] == 3) else "Tensor"
			temp_log.append(f"📐 尺寸结构: {self.get_shape_interpretation(anything.shape, is_weight=False)}")
			self.inspect_tensor_stats(anything, temp_log)

		# --- 最终输出 ---
		log.append(f"📍 管道类型: {pipe_type}")
		log.append(f"🔍 [JCH 深度探针: {label}]\n")
		log.extend(temp_log)
		if not temp_log: log.append(f"🧱 [未知实体] 类名: {type(anything).__name__}")
		final_output = "\n".join(log)
		return {"ui": {"text": [final_output]}, "result": (anything,)}

NODE_CLASS_MAPPINGS = {"JCH_TensorInspector": JCH_TensorInspector}
NODE_DISPLAY_NAME_MAPPINGS = {"JCH_TensorInspector": "🔍 JCH 深度探针 (全功能 V23)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
# AviSynth — Composite

[English](README.md) | **简体中文** | [日本語](README.ja.md)

AviSynth — Composite 是 AviSynthMinus 的独立图像混合与合成模块。它可以脱离 AviSynth 构建，为 Merge、Overlay、Layer 及相关通道和遮罩操作提供计算内核，同时保留普通 C 实现并使用 Google Highway 提供跨平台 SIMD 加速。

公开接口使用 C 类型和函数，实现使用 C++17，不依赖 AviSynth SDK、AvsCore 或 AvsSimd。

## 为什么拆分图像合成？

将计算内核与帧服务器分离，可以独立维护接口、数值行为、测试和性能。本模块与 AviSynthMinus 一同演进，可以作为固定版本的 Git 子模块加入宿主并静态链接。

宿主负责剪辑、脚本注册、帧分配、属性、色彩空间解释和调度。本库处理显式通道视图、遮罩、引导数据和行范围，为宿主滤镜提供构建组件；宿主集成另行维护。

## 支持的操作

| 类别 | 能力 |
|---|---|
| 平面混合 | 混合、加减、乘积、反相目标混合、引导乘法、明暗选择和带偏置差值。 |
| YUV 联合运算 | 全分辨率 Add、Subtract、Soft Light、Hard Light、Difference、Exclusion、Multiply，以及整数越界去饱和处理。F32 支持 Add、Subtract、Multiply。 |
| 遮罩与引导采样 | 444、422、420、411，居中、MPEG2、左上定位及有符号采样相位。 |
| 通道工具 | 复制、填充、仿射变换、限幅、RGB 亮度、色键和矩形求交。 |
| 兼容运算 | 独立的历史 Minus 整数混合操作。 |

存储类型为 U8/8 位、U16/9–16 位、F32/32 位。显式通道视图支持平面及带步长的打包通道。Alpha 作为普通通道处理，权重遮罩与混合目标分离，不隐式套用 Porter–Duff 公式。整数样本、遮罩和引导数据必须处于声明位深的范围内。F32 颜色支持负值和 HDR，F32 遮罩必须为 [0,1] 内的有限值。各操作的限制见[公开头文件](include/composite)。

数值行为以经过审查的 C 实现和回归测试为准。符合条件的整数 SIMD 路径允许**每次调用最多 1 LSB** 的差异；部分 F32 路径允许随数值幅度变化的舍入误差。连续操作可能累积误差。需要参考运算时可选择普通 C，整数运算也可先提高工作位深以减小舍入误差的归一化幅度。精确容差、端点行为和回退条件见[数值行为说明](NUMERICS.md)。滤镜名称相同不代表与所有历史实现的输出完全一致。

## SIMD 与 CPU 限制

`CP_TARGET_C` 选择普通 C，`CP_TARGET_NATIVE` 选择可用的本机实现，必要时回退到 C。`cp_choose_target(allowed_bits)` 从已编译、CPU 支持且调用方允许的目标中选择。`cp_get_kernels` 返回不可变函数表，显式请求不可用或非法目标时返回空指针。目标使用 Highway 位值。直接调用操作函数使用普通 C。

选择按使用者实例生效，不修改 Highway 的进程级目标限制。集成到 AviSynth 时，AvsSimd 留在宿主内解释 `SetMaxCPU` 并提供允许的目标。宿主应将 `SetMaxCPU("none")` 映射到普通 C，并缓存选定的函数表。

SIMD 覆盖像素操作，包括带步长通道和有边界保护的尾部处理，无需额外填充。不支持 FP64 的目标对需要双精度的运算保留普通 C 回退。矩形求交使用标量几何运算。更宽的 SIMD 不保证更快。

## 构建与集成

需要 CMake 3.24 或更新版本以及支持 C++17 的编译器。接口兼容 C，不暴露 STL 容器或 Highway 向量类型。接口可能演进，不承诺不同版本的二进制可互换；请配套使用头文件与库。

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCP_BUILD_TESTS=ON
cmake --build build/release --config Release --parallel
ctest --test-dir build/release -C Release --output-on-failure
```

测试无需下载依赖。仅构建库时设置 `-DCP_BUILD_TESTS=OFF`，嵌入构建默认关闭测试。`-DCP_SCALAR_ONLY=ON` 移除 SIMD 和 Highway 依赖。可选基准测试使用 `-DCP_BUILD_BENCHMARKS=ON`。受支持的非 MSVC Clang/GCC 工具链可用 `-DCP_SANITIZERS=ON` 启用 ASan/UBSan。

静态库名为 `Composite`，CMake 别名为 `AviSynth::Composite`。将仓库作为子模块加入后，直接链接目标：

```cmake
add_subdirectory(third_party/composite)
target_link_libraries(MyHost PRIVATE AviSynth::Composite)
```

独立构建使用仓库内的 Highway 1.4.0；嵌入构建复用已有的兼容 `hwy` 目标，让 Audio、Video 与宿主共享同一运行库。CMake 传递静态链接依赖，使用方无需逐项列出内核源文件。支持的集成方式是联合 CMake 构建，不是安装式二进制 SDK。

公开头文件位于 [include/composite](include/composite)。行跨度为有符号字节数，样本步长为正字节数。请遵守各接口的行起点、重叠和生命周期约定。像素操作不分配内存；依赖原始引导数据与遮罩的通道操作完成前，应保留这些数据。写入不相交输出区域的独立调用可以并发执行。

## 测试与性能

独立测试覆盖算术、兼容行为、采样、通道工具、C/SIMD 对照、数值边界、不规则尺寸、有符号行跨度、分行处理、并发和内存边界。C 使用方测试检查公开接口。测试套件包含 15 个 CTest 条目，条目内部覆盖多个用例和目标。

CI 覆盖 Windows、Linux、macOS 的 SIMD 与普通 C 配置，另有 Windows Win32 构建和 Linux ASan/UBSan。还在 Linux、FreeBSD 的 x86-64 与 ARM64 平台做过正确性和性能测试。内核验证不能代替宿主滤镜集成测试。

开启测试时，构建关闭隐式浮点收缩以便复现参考比较，但仍可使用显式 SIMD 融合运算。仅构建库时允许收缩，比较结果时应记录这一选项。

可选基准测试在计时前对照 C 验证输出，报告最小、中位和最大毫秒耗时，覆盖 8、10、16、32 位输入、连续及带步长通道和打包布局。部分上游内核可从本地仓库提取。命令、计时范围和参考实现限制见[内核基准测试说明](benchmarks/README.md)。

比较时应保持输入、布局、构建选项、CPU 限制和计时范围一致。内核测速与完整滤镜测速不同。性能报告应同时提供输出比较和耗时；对细小差异应进行适度复核。

## 开发与贡献

维护者负责技术方向、变更审核和发布。欢迎问题报告、建议与贡献；修改数值语义、公开接口或重要架构前，建议先讨论目标和方案。

本项目使用 AI 辅助实现、测试和审查。贡献应说明问题、方案、验证方法和 AI 参与方式。报告问题请提供提交版本、系统、CPU、编译器、构建选项、输入输出格式及最小复现；性能报告还应包含尺寸、CPU 目标和测量方法。

## 致谢与许可证

本模块建立在 AviSynth、AviSynth+、AviSynthMinus 及其贡献者的工作上，使用 Google Highway 提供 SIMD 能力。感谢原作者和参与测试、报告问题及改进的开发者与用户。

感谢 [烧饼论坛](https://sb.sb) 赞助本项目开发使用的 LLM 订阅。

项目采用 GPL 第 2 版或更新版本，保留继承自 AviSynth 的链接例外原文和适用范围，完整条款见 [LICENSE](LICENSE)。源文件保留版权声明，第三方组件遵循各自许可证。提供新的 C 接口不会扩大原有例外的适用范围。

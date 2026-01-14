import ROOT
import sys

# 抑制 ROOT 命令行/图形警告
sys.argv = []
ROOT.PyConfig.IgnoreCommandLineOptions = True

# 1. 加载文件并检查
file_path = "./Am_600s.root"
f = ROOT.TFile.Open(file_path)
if not f or f.IsZombie():
    print(f"错误：无法打开文件 {file_path}")
    sys.exit(1)

tree = f.Get("Tree")
if not tree:
    print("错误：树 'Tree' 不存在。")
    f.ls()
    sys.exit(1)

print("文件和树加载成功。")
# tree.Print()  # 已知结构，注释以加速

# 2. 定义 RooFit 变量
energy_var = ROOT.RooRealVar("cluster_energy", "Energy (keV)", 40, 70)

# 3. 读取数据（使用 TTreeReaderArray，无需 ncluster 分支）
data = ROOT.RooDataSet("data", "dataset", ROOT.RooArgSet(energy_var))
reader = ROOT.TTreeReader("Tree", f)

# TTreeReaderArray for variable-length array
rv_energy = ROOT.TTreeReaderArray(ROOT.double_t)(reader, "cluster_energy")

print("正在读取并展平数据...")
count_total_events = 0
count_total_clusters = 0
count_filtered = 0
sample_vals = []

while reader.Next():
    count_total_events += 1
    n_clust = rv_energy.GetSize()  # 使用 GetSize() 获取当前事件簇数
    count_total_clusters += n_clust
    
    for i in range(n_clust):
        val = rv_energy[i]
        sample_vals.append(val)
        if 40 <= val <= 70:
            energy_var.setVal(val)
            data.add(ROOT.RooArgSet(energy_var))
            count_filtered += 1

# 限事件数防慢（可选）
if count_total_events > 10000:
    print("警告：事件数多，已全读。")

print(f"总事件数: {count_total_events}")
if count_total_events > 0:
    print(f"总簇数: {count_total_clusters} (平均/事件: {count_total_clusters / count_total_events :.1f})")
else:
    print("错误：无事件读取。检查 reader.Next()。")
    sys.exit(1)

print(f"过滤后数据点数 (40-70 keV): {count_filtered}")
if sample_vals:
    print(f"样本值 (前10): {sample_vals[:10]}")
    print(f"所有样本范围: {min(sample_vals):.1f} - {max(sample_vals):.1f} keV")

if count_filtered == 0:
    print("错误:无数据在范围内。调整过滤或检查单位(keV?）。")
    # 打印更多样本以诊断
    print("扩展样本 (前20):", sample_vals[:20])
    sys.exit(1)

# 4. 构建模型 (高斯信号 + 线性本底)
mean = ROOT.RooRealVar("mean", "mean", 59.5, 55, 65)  # 紧缩 Am-241 峰
sigma = ROOT.RooRealVar("sigma", "sigma", 1.5, 0.5, 5.0)  # 典型分辨率
gauss = ROOT.RooGaussian("gauss", "signal", energy_var, mean, sigma)

# 线性本底: a0 + a1*x
a0 = ROOT.RooRealVar("a0", "a0", 0.0, -5, 5)
a1 = ROOT.RooRealVar("a1", "a1", 0.0, -0.5, 0.5)
bkg_coeffs = ROOT.RooArgList(a0, a1)
bkg = ROOT.RooPolynomial("bkg", "background", energy_var, bkg_coeffs)

# 产量（基于过滤数据）
n_sig = ROOT.RooRealVar("n_sig", "yield_signal", count_filtered * 0.4, 0, count_filtered)
n_bkg = ROOT.RooRealVar("n_bkg", "yield_background", count_filtered * 0.6, 0, count_filtered)
model = ROOT.RooAddPdf("model", "sig+bkg", 
                       ROOT.RooArgList(gauss, bkg), 
                       ROOT.RooArgList(n_sig, n_bkg))

# 5. 拟合
fit_result = model.fitTo(data, 
                         ROOT.RooFit.Extended(ROOT.kTRUE), 
                         ROOT.RooFit.Range(40, 70), 
                         ROOT.RooFit.Save(ROOT.kTRUE),
                         ROOT.RooFit.PrintLevel(1))  # 打印拟合进度
print("\n拟合结果：")
fit_result.Print("v")  # 详细打印参数/误差

# 6. 绘图
canvas = ROOT.TCanvas("canvas", "Am-241 Fit (40-70 keV)", 800, 600)
xframe = energy_var.frame(ROOT.RooFit.Title("Energy Spectrum Fit"))

# 修正 binning：使用 RooUniformBinning(低, 高, nbins)
binning = ROOT.RooUniformBinning(40, 70, 30)
data.plotOn(xframe, ROOT.RooFit.Binning(binning))
model.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kBlue))
model.plotOn(xframe, ROOT.RooFit.Components("bkg"), 
             ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

model.paramOn(xframe, ROOT.RooFit.Layout(0.6, 0.9, 0.8))
xframe.Draw()
canvas.SaveAs("fit_result.png")
print("保存图像: fit_result.png")

# 额外：信号统计
print(f"\n信号产量: {n_sig.getVal():.0f} ± {n_sig.getError():.0f}")
print(f"峰位置: {mean.getVal():.2f} ± {mean.getError():.2f} keV")
print(f"分辨率 (sigma): {sigma.getVal():.2f} ± {sigma.getError():.2f} keV")

# 保持 canvas 打开（可选，在交互 ROOT 中）
# input("按 Enter 关闭...")
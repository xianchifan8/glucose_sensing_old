from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape

OUT = Path("visualization/gated_residual_late_fusion_architecture.pptx")

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

EMU = 914400
SLIDE_W = int(13.333333 * EMU)
SLIDE_H = int(7.5 * EMU)


def emu(v):
    return int(v * EMU)


def color(hex_color):
    return hex_color.replace("#", "").upper()


def text_runs(text, size=16, bold=False, fill="#1f2933"):
    lines = text.split("\n")
    paras = []
    for line in lines:
        paras.append(
            f'<a:p><a:r><a:rPr lang="zh-CN" sz="{size * 100}" b="{1 if bold else 0}">'
            f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
            f'<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:rPr>'
            f'<a:t>{escape(line)}</a:t></a:r></a:p>'
        )
    return "".join(paras)


def shape(shape_id, name, x, y, w, h, fill, line="#D0D7E2", radius=True, text="", size=16, bold=False):
    geom = "roundRect" if radius else "rect"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="152400" tIns="91440" rIns="152400" bIns="91440"/>
        <a:lstStyle/>
        {text_runs(text, size=size, bold=bold)}
      </p:txBody>
    </p:sp>"""


def line(shape_id, x1, y1, x2, y2, fill="#3F4A56", width=2.0):
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="Arrow {shape_id}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/><a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="{int(width * 12700)}">
          <a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>
          <a:tailEnd type="triangle"/>
        </a:ln>
      </p:spPr>
    </p:cxnSp>"""


def slide_xml():
    items = []
    sid = 2
    items.append(shape(sid, "Background", 0, 0, 13.333, 7.5, "#F7FAFC", "#F7FAFC", False)); sid += 1
    items.append(shape(sid, "Title", 0.45, 0.25, 8.6, 0.55, "#F7FAFC", "#F7FAFC", False,
                       "Gated Residual 多模态血糖预测结构", 24, True)); sid += 1
    items.append(shape(sid, "Subtitle", 0.48, 0.82, 11.9, 0.34, "#F7FAFC", "#F7FAFC", False,
                       "频谱作为主模态；PPG/ICM 辅助模态只学习有界门控残差，用于小幅修正频谱特征。", 12, False)); sid += 1

    items.append(shape(sid, "Spectrum Input", 0.65, 2.0, 2.05, 0.85, "#D9ECFF", text="Spectrum 输入\n频谱窗口 / 单点频谱", size=13, bold=True)); sid += 1
    items.append(shape(sid, "Spectrum Encoder", 3.25, 2.0, 2.35, 0.85, "#EAF4FF", text="频谱特征编码器\nbase_model.extract_features\nz_s ∈ R^d", size=12, bold=True)); sid += 1
    items.append(shape(sid, "Aux Input", 0.65, 4.35, 2.05, 0.85, "#DFF6EC", text="PPG / ICM 输入\n动态状态 / 运动质量特征", size=13, bold=True)); sid += 1
    items.append(shape(sid, "Aux Encoder", 3.25, 4.35, 2.35, 0.85, "#EDF9F3", text="AuxFeatureEncoder\nLinear → LN → ReLU\n a ∈ R^k", size=12, bold=True)); sid += 1

    items.append(shape(sid, "Concat", 6.25, 2.8, 1.8, 0.7, "#FFFFFF", text="条件拼接\nu = [z_s ; a]", size=12, bold=True)); sid += 1
    items.append(shape(sid, "Residual", 6.15, 4.25, 1.8, 0.9, "#E7F7EF", text="Residual 分支\nr = tanh(f_r(u))", size=12, bold=True)); sid += 1
    items.append(shape(sid, "Gate", 8.45, 4.25, 1.8, 0.9, "#E7F7EF", text="Gate 分支\ng = sigmoid(f_g(u))", size=12, bold=True)); sid += 1
    items.append(shape(sid, "Scale", 8.5, 2.45, 1.7, 0.75, "#F3EDFF", text="有界 Scale\nα ∈ [min, 1]", size=12, bold=True)); sid += 1
    items.append(shape(sid, "Correction", 6.85, 5.85, 2.75, 0.75, "#F0FBF6", text="z' = z_s + α · g ⊙ r", size=14, bold=True)); sid += 1
    items.append(shape(sid, "Head", 10.65, 3.55, 1.45, 1.05, "#EFE5FF", text="预测头\nMLP Head\nŷ", size=13, bold=True)); sid += 1
    items.append(shape(sid, "LR Note", 0.9, 6.85, 11.3, 0.35, "#FFFFFF", "#D8DEE7", True,
                       "可选差异学习率：base_model → lr；AuxEncoder / Residual / Gate / Head → fusion_lr；residual_scale → scale_lr。", 10, False)); sid += 1

    for coords in [
        (2.7, 2.43, 3.25, 2.43), (2.7, 4.78, 3.25, 4.78),
        (5.6, 2.43, 6.25, 3.05), (5.6, 4.78, 6.25, 3.28),
        (7.15, 3.5, 7.05, 4.25), (7.15, 3.5, 9.35, 4.25),
        (9.35, 3.2, 9.35, 4.25), (7.05, 5.15, 7.75, 5.85),
        (9.35, 5.15, 8.75, 5.85), (9.35, 3.2, 8.95, 5.85),
        (5.6, 2.43, 6.85, 6.2), (9.6, 6.2, 10.65, 4.1),
    ]:
        items.append(line(sid, *coords)); sid += 1

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(items)}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


files = {
    "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
</Types>""",
    "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""",
    "ppt/presentation.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""",
    "ppt/_rels/presentation.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>""",
    "ppt/slides/slide1.xml": slide_xml(),
    "ppt/slides/_rels/slide1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""",
    "ppt/slideLayouts/slideLayout1.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}" type="blank">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""",
    "ppt/slideLayouts/_rels/slideLayout1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""",
    "ppt/slideMasters/slideMaster1.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>""",
    "ppt/slideMasters/_rels/slideMaster1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""",
    "ppt/theme/theme1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Simple">
  <a:themeElements><a:clrScheme name="Simple"><a:dk1><a:srgbClr val="17202A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2933"/></a:dk2><a:lt2><a:srgbClr val="F7FAFC"/></a:lt2><a:accent1><a:srgbClr val="7EADE2"/></a:accent1><a:accent2><a:srgbClr val="79C7A5"/></a:accent2><a:accent3><a:srgbClr val="D0B4F2"/></a:accent3><a:accent4><a:srgbClr val="F1D590"/></a:accent4><a:accent5><a:srgbClr val="98A2B3"/></a:accent5><a:accent6><a:srgbClr val="667085"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="Simple"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme><a:fmtScheme name="Simple"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements>
</a:theme>""",
}

with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
    for path, content in files.items():
        zf.writestr(path, content)

print(OUT)

"""word_exporter.py — 用 python-docx 将总结与对比结果排版导出为 Word。

对外接口：
    export_report(summary, comparison, meta, output_path) -> str
"""

from __future__ import annotations

import os
from datetime import datetime

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

HEADING_FONT = "微软雅黑"
BODY_FONT = "宋体"

_HEADING_SIZE = {0: 18, 1: 15, 2: 13, 3: 12}
_HEADING_COLOR = (0x1F, 0x3B, 0x66)


def _set_run_font(run, name: str = BODY_FONT, size: float = None, bold: bool = None,
                  color: tuple = None) -> None:
    """同时设置西文与东亚字体，保证中文不乱码。"""
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def _add_heading(doc, text: str, level: int):
    p = doc.add_paragraph(style=f"Heading {level}")
    run = p.add_run(text)
    _set_run_font(run, HEADING_FONT, _HEADING_SIZE.get(level, 12), bold=True,
                  color=_HEADING_COLOR if level <= 1 else (0x33, 0x33, 0x33))
    return p


def _add_body(doc, text: str, size: float = 11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, BODY_FONT, size)
    return p


def _add_meta_line(doc, label: str, value: str):
    p = doc.add_paragraph()
    r1 = p.add_run(f"{label}：")
    _set_run_font(r1, BODY_FONT, 10.5, bold=True)
    r2 = p.add_run(value or "（无）")
    _set_run_font(r2, BODY_FONT, 10.5)
    return p


def _add_bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(str(it))
        _set_run_font(run, BODY_FONT, 11)
    return p


def _add_table(doc, headers: list[str], rows: list[list]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        run = cell.paragraphs[0].add_run(str(h))
        _set_run_font(run, HEADING_FONT, 10.5, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            run = cells[i].paragraphs[0].add_run(str(val))
            _set_run_font(run, BODY_FONT, 10)
    return table


def export_report(summary: dict, comparison: dict, meta: dict, output_path: str) -> str:
    """把总结与对比结果排版导出为 Word，返回保存路径。"""
    summary = summary or {}
    comparison = comparison or {}
    meta = meta or {}

    doc = Document()

    # 标题
    title = summary.get("title") or meta.get("title") or "未命名视频"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    _set_run_font(run, HEADING_FONT, 18, bold=True)

    _add_meta_line(doc, "视频链接", meta.get("url", ""))
    _add_meta_line(doc, "BVID", meta.get("bvid", ""))
    _add_meta_line(doc, "生成时间", meta.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # 一、视频总结
    _add_heading(doc, "一、视频总结", 1)
    _add_body(doc, summary.get("summary") or "（无）")

    # 二、知识点分类
    _add_heading(doc, "二、知识点分类", 1)
    kps = summary.get("knowledge_points") or []
    if not kps:
        _add_body(doc, "（未提炼出知识点）")
    for i, kp in enumerate(kps, 1):
        _add_heading(doc, f"{i}. {kp.get('topic') or '未命名知识点'}", 2)
        cat = kp.get("category") or ""
        tags = "、".join(kp.get("tags") or [])
        _add_meta_line(doc, "分类", cat or "未分类")
        if tags:
            _add_meta_line(doc, "标签", tags)
        _add_body(doc, kp.get("content") or "")
        if kp.get("key_points"):
            _add_bullets(doc, kp["key_points"])

    # 三、与历史知识库对比
    _add_heading(doc, "三、与历史知识库对比", 1)

    _add_heading(doc, "3.1 重叠知识点", 2)
    overlap = comparison.get("overlap") or []
    if overlap:
        _add_table(
            doc,
            ["新视频知识点", "历史知识点", "关系", "说明"],
            [
                [
                    o.get("new_topic", ""),
                    o.get("old_topic", ""),
                    o.get("relation", ""),
                    o.get("description", ""),
                ]
                for o in overlap
            ],
        )
    else:
        _add_body(doc, "（无重叠知识点）")

    _add_heading(doc, "3.2 新增与扩充", 2)
    expansion = comparison.get("expansion") or []
    if expansion:
        for j, e in enumerate(expansion, 1):
            _add_heading(doc, f"新增 {j}. {e.get('topic') or '未命名'}", 3)
            _add_body(doc, e.get("description") or "")
    else:
        _add_body(doc, "（无新增知识点）")

    _add_heading(doc, "3.3 整体分析", 2)
    _add_body(doc, comparison.get("analysis") or "（无）")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    return output_path

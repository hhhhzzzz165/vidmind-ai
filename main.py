"""main.py — 测试入口。

串联：解析视频 -> AI总结 -> 读历史JSON -> AI对比 -> 导出Word -> 保存JSON。
用法：
    python main.py <B站视频URL>
    python main.py <占位> --mock          # 离线测试，不联网不耗 API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

from dotenv import load_dotenv

from summarizer import DeepSeekClient, SummarizeError
from video_parser import VideoParseError, parse_video
from word_exporter import export_report

DEFAULT_KB = "knowledge_base.json"
DEFAULT_OUT_DIR = "output"
SAMPLE_TRANSCRIPT = os.path.join("sample", "sample_transcript.txt")


def load_knowledge_base(path: str = DEFAULT_KB) -> dict:
    if not os.path.exists(path):
        return {"videos": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "videos" not in data:
        raise ValueError(f"知识库文件 {path} 结构不合法（应含 videos 列表），请检查或备份后处理")
    return data


def save_knowledge_base(path: str, kb: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _upsert_video(kb: dict, entry: dict) -> None:
    bvid = entry.get("bvid")
    videos = kb.setdefault("videos", [])
    for i, v in enumerate(videos):
        if bvid and v.get("bvid") == bvid:
            videos[i] = entry
            return
    videos.append(entry)


def _safe_filename(title: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n]+', "_", title or "video").strip()
    s = re.sub(r"\s+", "_", s)
    return s[:80] or "video"


MOCK_TITLE = "密室犯罪就非得是密室吗？【点到为止54】"
MOCK_URL = "https://www.bilibili.com/video/BV1RNH56QEtB"
MOCK_BVID = "BV1RNH56QEtB"


def _mock_summary(title: str = MOCK_TITLE) -> dict:
    """离线演示数据：取自一次真实运行的《密室犯罪就非得是密室吗？》总结。"""
    return {
        "title": MOCK_TITLE,
        "summary": (
            "本期视频通过一起密室杀人案，探讨了密室犯罪中凶手脱困的多种可能性，"
            "并介绍了双侦探组合如何通过细节推理破解案件。案件表面上是凶手用电锯在墙上开洞逃脱，"
            "但侦探发现油漆罐没有摔痕、桌子腿被锯短等疑点，最终揭示真正的第一现场在厨房，"
            "凶手为了转移尸体和桌子而伪造密室。视频还介绍了侦探事务所的设定和推理小说《敲响密室之门》的特点。"
        ),
        "knowledge_points": [
            {
                "topic": "密室脱困的常见方法",
                "category": "基础概念",
                "content": "在推理小说中，凶手被困在犯罪现场时，有多种方法可以脱困并制造密室假象，例如从窗户离开、利用工具破坏墙壁等。但无论哪种方法，最终结果都是房间里只有死者，凶手消失。",
                "key_points": ["脱困方法多样，但都会导向凶手消失的结果", "制造密室不一定需要复杂手法", "简单粗暴的方法（如砸墙）也可能有效"],
                "tags": ["密室", "脱困", "推理小说"],
            },
            {
                "topic": "案件现场分析：油漆罐的疑点",
                "category": "方法",
                "content": "侦探通过观察发现，装满油漆的罐子从高处摔落却没有任何痕迹，说明油漆是被人特意取下后泼洒的，而非打斗中意外掉落。这推翻了凶手因门口油漆而无法从正门离开的假设。",
                "key_points": ["油漆罐无摔痕，表明是人为泼洒", "凶手故意用油漆堵住门口，制造开洞的借口", "油漆罐无标签，凶手需熟悉木屋才能知道内容物"],
                "tags": ["现场勘查", "细节推理", "油漆罐"],
            },
            {
                "topic": "桌子腿被锯短的真相",
                "category": "案例",
                "content": "侦探发现未完工的桌子腿高低不平，但血迹扩散形态表明案发时桌子是平的。凶手锯短桌腿是为了让桌子看起来未完工，从而掩盖桌子曾被搬动的事实。",
                "key_points": ["血迹形态显示桌子曾水平放置", "桌腿被故意锯短以伪装未完工状态", "凶手忽略血迹凝固和重心偏移的细节"],
                "tags": ["血迹分析", "桌子", "伪装"],
            },
            {
                "topic": "真正的第一现场与开洞动机",
                "category": "原理",
                "content": "凶手在墙上开洞并非为了自己逃脱，而是为了将桌子和尸体一起搬回木屋，以掩盖真正的第一现场在厨房附近。开洞是为了让桌子能通过墙壁进入木屋。",
                "key_points": ["第一现场在厨房附近，而非木屋", "开洞是为了搬运桌子，而非逃生", "油漆和开洞都是伪造密室的伪装"],
                "tags": ["第一现场", "开洞动机", "伪装"],
            },
            {
                "topic": "双侦探组合的分工与推理方式",
                "category": "方法",
                "content": "侦探事务所的两位侦探分工明确：一位负责破解犯罪手法（物理矛盾），另一位侧重犯罪动机和心理。两人通过敲门声等细节进行推理，相辅相成。",
                "key_points": ["分工：手法破解与动机分析", "通过敲门声推理访客性格和目的", "合作避免吵架，提高破案效率"],
                "tags": ["侦探", "分工", "推理方法"],
            },
            {
                "topic": "嫌疑人不在场证明分析",
                "category": "案例",
                "content": "案件中的嫌疑人包括死者妻子、弟弟、女儿和侄子。妻子生病睡觉无证明，弟弟散步但手机没电无证明，女儿和侄子互相作证但在厨房看电视理由牵强。最终通过桌子搬运需要两人，锁定女儿和侄子。",
                "key_points": ["弟弟手机没电，不在场证明薄弱", "女儿和侄子互相作证但理由可疑", "搬运桌子需两人，排除单独作案"],
                "tags": ["不在场证明", "嫌疑人", "推理"],
            },
            {
                "topic": "推理小说《敲响密室之门》介绍",
                "category": "工具",
                "content": "视频提及日本推理作家青崎有吾的短篇小说系列《敲响密室之门》，以双主角侦探展开，保持逻辑流推理水准，故事小巧精致，但犯罪动机有时较草率。",
                "key_points": ["作者青崎有吾有“平成奎因”美誉", "系列以双侦探为主角，逻辑推理强", "动机设计可能较弱，但推理精彩"],
                "tags": ["推理小说", "青崎有吾", "敲响密室之门"],
            },
            {
                "topic": "视频收尾与互动引导",
                "category": "案例",
                "content": "主讲人阿怪在片段结尾向观众道别，鼓励观众推荐给朋友，并预告后续还有新的考验。",
                "key_points": ["主讲人自称阿怪", "鼓励观众分享推荐给朋友", "预告后续存在新的考验", "以“点到为止”作为栏目收尾语"],
                "tags": ["视频结尾", "互动", "预告"],
            },
        ],
    }


_MOCK_EXPANSION = [
    {"topic": "密室脱困的常见方法", "description": "推理小说中凶手被困犯罪现场时的多种脱困方式及制造密室假象的基本概念，属于全新领域。", "is_new": True},
    {"topic": "案件现场分析：油漆罐的疑点", "description": "通过油漆罐无摔痕推断油漆系人为泼洒，推翻凶手无法从正门离开的假设，属于全新推理方法。", "is_new": True},
    {"topic": "桌子腿被锯短的真相", "description": "利用血迹扩散形态与桌腿高低不平的矛盾，揭示锯短桌腿是为掩盖桌子被搬动，属于全新案例推理。", "is_new": True},
    {"topic": "真正的第一现场与开洞动机", "description": "墙上开洞并非为凶手逃脱，而是为搬运桌子和尸体以掩盖厨房附近的真实第一现场，属于全新原理。", "is_new": True},
    {"topic": "双侦探组合的分工与推理方式", "description": "两位侦探分别负责物理矛盾破解与犯罪动机心理分析，通过敲门声等细节相辅相成，属于全新方法。", "is_new": True},
    {"topic": "嫌疑人不在场证明分析", "description": "对妻子、弟弟、女儿、侄子四名嫌疑人的不在场证明逐一分析，并通过搬运桌子需两人锁定真凶，属于全新案例。", "is_new": True},
    {"topic": "推理小说《敲响密室之门》介绍", "description": "介绍青崎有吾短篇系列的双主角设定、逻辑流推理水准及动机草率等特点，属于全新工具类知识。", "is_new": True},
    {"topic": "视频收尾与互动引导", "description": "主讲人阿怪道别、鼓励分享并预告后续考验，属于全新内容。", "is_new": True},
]


def _mock_compare(old_kb: dict = None, new_summary: dict = None) -> dict:
    """离线演示对比数据（与真实运行一致：旧库为 DSH 相关内容，全部新增）。"""
    return {
        "overlap": [],
        "expansion": list(_MOCK_EXPANSION),
        "analysis": (
            "旧知识库全部围绕 DeepSeek Harness 桌面端及其插件生态，涵盖产品定位、核心技术、插件安装与功能"
            "（上下文压缩、插件市场、费用监控、边栏增强、图像桥接、远程控制、安全拦截、搜索、手机同步、TUI、"
            "浏览器观察窗、主动聊天、记忆备份等），属于 AI 工具与自动化工作流领域。新视频总结则是一期推理类内容，"
            "围绕密室杀人案的脱困方法、现场疑点分析、双侦探分工、嫌疑人不在场证明及推理小说介绍展开，"
            "属于推理文化与逻辑分析领域。两者在主题、概念和知识体系上完全无交集，不存在相同、相似、包含或部分重叠的知识点，"
            "因此 overlap 为空，新视频的全部知识点均作为新增内容列入 expansion。"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VidMind AI：总结 B站视频并对比历史知识库，导出 Word")
    p.add_argument("url", help="B站视频链接（--mock 时可传任意占位串）")
    p.add_argument("--kb", default=DEFAULT_KB, help="历史知识库 JSON 路径（默认 knowledge_base.json）")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录（默认 output）")
    p.add_argument("--cookies", default=None, help="可选：Netscape 格式 cookies.txt，用于需登录的字幕")
    p.add_argument("--mock", action="store_true", help="离线 dry-run：使用内置示例字幕与假总结，不联网不耗 API")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    load_dotenv()

    out_dir = os.path.abspath(args.out_dir)
    kb_path = os.path.abspath(args.kb)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1) 解析视频
        print("[1/6] 解析视频与字幕...")
        if args.mock:
            with open(SAMPLE_TRANSCRIPT, "r", encoding="utf-8") as f:
                transcript = f.read()
            title, bvid, url = "示例视频（mock）", "BV_mock", args.url
        else:
            info = parse_video(args.url, cookies_file=args.cookies)
            title, bvid, url, transcript = info.title, info.bvid, info.url, info.transcript
        print(f"      标题：{title}")
        print(f"      字幕长度：{len(transcript)} 字符")

        # 2) AI 总结
        print("[2/6] AI 总结并分类...")
        if args.mock:
            summary = _mock_summary(title)
        else:
            summary = DeepSeekClient().summarize(title, transcript)
        print(f"      知识点数量：{len(summary.get('knowledge_points') or [])}")

        # 3) 读历史知识库
        print("[3/6] 读取历史知识库...")
        old_kb = load_knowledge_base(kb_path)
        print(f"      历史视频数：{len(old_kb.get('videos') or [])}")

        # 4) AI 对比
        print("[4/6] AI 对比知识点...")
        if args.mock:
            comparison = _mock_compare(old_kb, summary)
        else:
            comparison = DeepSeekClient().compare(old_kb, summary)
        print(
            f"      重叠：{len(comparison.get('overlap') or [])} 条；"
            f"新增：{len(comparison.get('expansion') or [])} 条"
        )

        # 5) 导出 Word
        print("[5/6] 导出 Word...")
        meta = {
            "url": url,
            "bvid": bvid,
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        base = _safe_filename(title)
        docx_path = os.path.join(out_dir, base + ".docx")
        export_report(summary, comparison, meta, docx_path)
        print(f"      Word：{docx_path}")

        summary_json_path = os.path.join(out_dir, base + ".summary.json")
        comparison_json_path = os.path.join(out_dir, base + ".comparison.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(comparison_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        # 6) 保存知识库
        print("[6/6] 保存知识库...")
        entry = {
            "bvid": bvid,
            "title": title,
            "url": url,
            "summarized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": summary.get("summary") or "",
            "knowledge_points": summary.get("knowledge_points") or [],
        }
        _upsert_video(old_kb, entry)
        save_knowledge_base(kb_path, old_kb)
        print(f"      知识库：{kb_path}")

        print("\n[OK] 全部完成。")
        print(f"   Word      ：{docx_path}")
        print(f"   总结 JSON ：{summary_json_path}")
        print(f"   对比 JSON ：{comparison_json_path}")
        print(f"   知识库     ：{kb_path}")
        return 0
    except (VideoParseError, SummarizeError, ValueError) as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"\n[文件缺失] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

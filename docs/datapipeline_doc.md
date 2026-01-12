# Pipeline

1. 使用关键词搜索（如 "US stock analysis"）通过 YouTube API 获取候选频道列表，
   构建初步的 KOL_list。

2. 对 KOL_list 中的每个 KOL 调用 video_count.py，
   统计其在指定三年（如 2022–2024）内的视频发布频率，
   过滤出发布量较高、活跃度较强的 KOL，形成 refined_KOL_list。

3. 对 refined_KOL_list 中的每位 KOL 调用 summary_pipeline.fetch_video_ids 与
   summary_pipeline.fetch_video_details，抓取指定时间区间内的所有视频 ID、
   标题与基础元信息，并保存为 CSV 文件。

4. 使用 summary_pipeline.download_transcript，
   基于 yt-dlp 工具批量下载所有目标视频的 transcript（字幕文本），
   作为后续大模型处理的输入。

5. 调用 Gemini 2.5 Flash API（summary_pipeline.find_company_llm），
   对每段 transcript 执行金融实体抽取与情绪总结，
   输出字段包括：

      - company: 公司名称
      - excerpt: 视频中与该公司相关的关键内容摘要
      - confidence: 模型识别置信度
      - sentiment: 情绪倾向（如正向/负向/中性）

   然后将结果写入对应 JSON 文件，并与最初的视频元数据（title、videoId、
   publish_date 等）合并，最终生成模型训练/推理所需的原始 CSV 数据。

flowchart TD

```mermaid
A[YouTube 关键词搜索<br/>US stock analysis] --> B[初步 KOL_list]

B --> C[video_count.py<br/>统计 3 年视频发布频率]
C --> D[筛选活跃 KOL<br/>refined_KOL_list]

D --> E[summary_pipeline.fetch_video_ids<br/>获取视频ID]
E --> F[summary_pipeline.fetch_video_details<br/>获取视频标题/元信息]
F --> G[保存 video_details.csv]

G --> H[summary_pipeline.download_transcript<br/>使用 yt-dlp 下载字幕]

H --> I[summary_pipeline.find_company_llm<br/>Gemini 2.5 Flash 总结]
I --> J[company / excerpt / sentiment / confidence]

J --> K[保存 JSON 与 video_details 合并]
K --> L[输出最终模型输入 CSV]
```



# 相关工具



## video_count.py

使用 **YouTube Data API v3**，通过频道名称查询频道 ID，并统计该频道在指定年份区间内发布的视频数量。

```py
start_dt = _dt.datetime(start_year, 1, 1, 0, 0, 0)
end_dt = _dt.datetime(end_year, 12, 31, 23, 59, 59)
published_after = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
published_before = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

count = 0
page_token = None
while True:
    resp = (
        youtube.search()
        .list(
            part="id",
            channelId=channel_id,
            publishedAfter=published_after,
            publishedBefore=published_before,
            maxResults=50,
            order="date",
            type="video",
            pageToken=page_token,
        )
        .execute()
    )
```



## summary_pipeline.py

包括获取id，detail，以及summary之后的find_company_llm	

```py
def fetch_video_ids(channel_id: str):
    video_ids, token = [], None
    while True:
        resp = youtube.search().list(
            part="id",
            channelId=channel_id,
            publishedAfter=PUBLISHED_AFTER,
            publishedBefore=PUBLISHED_BEFORE,
            maxResults=50,
            order="date",
            type="video",
            pageToken=token,
        ).execute()
        video_ids += [
            item["id"].get("videoId", "")
            for item in resp["items"]
            if "videoId" in item["id"]
        ]
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.4)  # keep within rate limits
    return video_ids


def fetch_video_details(video_ids, channel_name):
    details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet", id=",".join(batch)
        ).execute()
        for item in resp["items"]:
            snip = item["snippet"]
            details.append(
                {
                    "channel_name": channel_name,
                    "video_id": item.get("id", ""),
                    "publishedAt": snip.get("publishedAt", ""),
                    "title": snip.get("title", ""),
                    "description": snip.get("description", ""),
                }
            )
        time.sleep(0.3)
    return details


def download_transcript(video_id):
    vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")
    try:
        cmd = [
            "yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--skip-download",
            "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)

        if not os.path.exists(vtt_path):
            vtts = [f for f in os.listdir(output_dir) if f.startswith(video_id) and f.endswith(".vtt")]
            if vtts:
                vtt_path = os.path.join(output_dir, vtts[0])
            else:
                return ""

        text_lines = []
        with open(vtt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("WEBVTT") or re.match(r"^\d\d:\d\d:\d\d\.\d\d\d -->", line):
                    continue
                cleaned_line = re.sub(r"<[^>]+>", "", line)
                cleaned_line = re.sub(r"http\S+", "", cleaned_line)
                text_lines.append(cleaned_line)

        deduped = []
        prev = ""
        for line in text_lines:
            if line != prev:
                deduped.append(line)
            prev = line

        return " ".join(deduped)
    finally:
        for f in os.listdir(output_dir):
            if f.startswith(video_id) and f.endswith(".vtt"):
                try:
                    os.remove(os.path.join(output_dir, f))
                except:
                    pass
                
def find_company_llm(content, title: str = "", description: str = ""):
    """
    Use Gemini to extract distinct company names with a short supporting excerpt.
    Considers title/transcript (ignore ads/promos). Returns [{"company": str, "excerpt": str}].
    """
    global _gemini_model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if _gemini_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are extracting company mentions.
- Use the title and transcript to find distinct company names.
- Ignore ads/promo/CTA content when picking excerpts.
- For each company, return all the related sentences combined together from the transcript that references it.
- After extracting excerpt, score:
  * confidence: 0-1 (certainty the excerpt refers to that company)
  * sentiment: -1 to 1 (negative to positive tone toward the company)
- Output JSON array only, schema:
[{{"company": "<name>", "excerpt": "<text from transcript>", "confidence": <float>, "sentiment": <float>}}]
- No markdown, no prose outside JSON.

Title:
{title}

Transcript:
{content}
"""
    resp = _gemini_model.generate_content(prompt)
    text = (resp.text or "").strip()
    try:
        items = json.loads(text)
        if isinstance(items, list):
            cleaned = []
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("company", "")).strip()
                excerpt = str(obj.get("excerpt", "")).strip()
                conf = obj.get("confidence", None)
                sent = obj.get("sentiment", None)
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = None
                try:
                    sent_f = float(sent)
                except Exception:
                    sent_f = None
                if name and excerpt:
                    cleaned.append(
                        {
                            "company": name,
                            "excerpt": excerpt,
                            "confidence": conf_f,
                            "sentiment": sent_f,
                        }
                    )
            if cleaned:
                return cleaned
    except Exception:
        pass

    # Fallback: basic split heuristic if JSON parsing fails
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return [{"company": p, "excerpt": "", "confidence": None, "sentiment": None} for p in parts]
```
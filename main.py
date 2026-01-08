import csv
import json
from datetime import datetime
from collections import Counter, defaultdict
import statistics
import matplotlib.pyplot as plt


def parse_trending_date(date_str: str) -> datetime:
    #The dataset sometimes uses different formats for trending date
    for fmt in ("%Y-%m-%d", "%y.%d.%m"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unknown trending_date format: {date_str}")


def load_videos(filename: str) -> list[dict]:
    videos = []

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            publish_time = datetime.fromisoformat(row["publish_time"].replace("Z", ""))

            video = {
                "video_id": row["video_id"],
                "trending_date": parse_trending_date(row["trending_date"]),
                "title": row["title"],
                "channel_title": row["channel_title"],
                "category_id": int(row["category_id"]),
                "publish_time": publish_time,
                "tags": row["tags"],
                "views": int(row["views"]),
                "likes": int(row["likes"]),
                "dislikes": int(row["dislikes"]),
                "comment_count": int(row["comment_count"]),
                "thumbnail_link": row["thumbnail_link"],
            }
            videos.append(video)

    return videos


def total_unique_videos_and_channels(videos: list[dict]) -> tuple[int, int]:
    video_ids = set()
    channel_titles = set()

    for v in videos:
        video_ids.add(v["video_id"])
        channel_titles.add(v["channel_title"])

    return len(video_ids), len(channel_titles)


def videos_per_category(videos: list[dict]) -> Counter:
    counts = Counter()
    for v in videos:
        counts[v["category_id"]] += 1
    return counts


def find_video_records_by_id(videos: list[dict], video_id: str) -> list[dict]:
    matches = []
    for v in videos:
        if v["video_id"] == video_id:
            matches.append(v)
    return matches


def print_video_record(video: dict) -> None:
    print("\n--- Video Record ---")
    print("Video ID:", video["video_id"])
    print("Title:", video["title"])
    print("Channel:", video["channel_title"])
    print("Category ID:", video["category_id"])
    print("Trending Date:", video["trending_date"])
    print("Publish Time:", video["publish_time"])
    print("Views:", video["views"])
    print("Likes:", video["likes"])
    print("Dislikes:", video["dislikes"])
    print("Comments:", video["comment_count"])
    print("Tags:", video["tags"])
    print("Thumbnail Link:", video["thumbnail_link"])


def best_record_per_video(videos: list[dict]) -> list[dict]:
    #I keep one record per video id (the one with the highest views).To avoid same video showing multiple times.
    best = {}

    for v in videos:
        vid = v["video_id"]
        if vid not in best:
            best[vid] = v
        else:
            if v["views"] > best[vid]["views"]:
                best[vid] = v

    return list(best.values())


def top_10_by_metric(videos: list[dict], metric: str) -> list[dict]:
    return sorted(videos, key=lambda x: x[metric], reverse=True)[:10]


def print_top_10(top10: list[dict], metric: str) -> None:
    print(f"\n--- Top 10 videos by {metric.upper()} ---")
    for i, v in enumerate(top10, start=1):
        print(f"{i}. {v['title']} | {metric}: {v[metric]} | Channel: {v['channel_title']}")


def plot_category_pie_chart(category_counts: Counter, min_percent: float = 3.0) -> None:
    #smaller groups are put as 'others' to make the pie chart easier to read
    total = sum(category_counts.values())
    labels = []
    sizes = []
    other_total = 0

    for category, count in category_counts.items():
        percent = (count / total) * 100
        if percent < min_percent:
            other_total += count
        else:
            labels.append(f"Category {category}")
            sizes.append(count)

    if other_total > 0:
        labels.append("Other")
        sizes.append(other_total)

    plt.figure(figsize=(9, 7))
    plt.pie(sizes, startangle=90, autopct="%1.1f%%")
    plt.title("Distribution of videos across categories")
    plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_histogram(videos: list[dict], metric: str, bins: int = 30) -> None:
    values = [v[metric] for v in videos if v[metric] > 0]

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.xscale("log")
    plt.title(f"Histogram of {metric} (log scale)")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def average_engagement_by_category(videos: list[dict]) -> dict:
    grouped = defaultdict(lambda: {"likes": [], "dislikes": [], "comments": []})

    for v in videos:
        cat = v["category_id"]
        grouped[cat]["likes"].append(v["likes"])
        grouped[cat]["dislikes"].append(v["dislikes"])
        grouped[cat]["comments"].append(v["comment_count"])

    averages = {}
    for cat, data in grouped.items():
        averages[cat] = {
            "likes": statistics.mean(data["likes"]) if data["likes"] else 0,
            "dislikes": statistics.mean(data["dislikes"]) if data["dislikes"] else 0,
            "comments": statistics.mean(data["comments"]) if data["comments"] else 0,
        }
    return averages


def print_average_engagement(averages: dict) -> None:
    print("\n--- Average Engagement Per Category ---")
    for cat in sorted(averages.keys()):
        a = averages[cat]
        print(
            f"Category {cat}: "
            f"avg likes={a['likes']:.2f}, "
            f"avg dislikes={a['dislikes']:.2f}, "
            f"avg comments={a['comments']:.2f}"
        )


def trending_duration_per_video(videos: list[dict]) -> dict:
    #using a set so same dates are not counted twice
    dates_by_video = defaultdict(set)
    for v in videos:
        dates_by_video[v["video_id"]].add(v["trending_date"])
    return {vid: len(dates) for vid, dates in dates_by_video.items()}


def print_top_trending_durations(durations: dict, top_n: int = 10) -> None:
    print(f"\n--- Top {top_n} Videos by Trending Duration ---")
    top = sorted(durations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (vid, days) in enumerate(top, start=1):
        print(f"{i}. video_id={vid} | days trending={days}")


def like_dislike_ratio_outliers(videos: list[dict], threshold: float = 50) -> list:
    outliers = []
    for v in videos:
        #+1 to avoid division by zero when dislikes are 0
        ratio = v["likes"] / (v["dislikes"] + 1)
        if ratio >= threshold:
            outliers.append((v, ratio))

    outliers.sort(key=lambda x: x[1], reverse=True)
    return outliers


def print_outliers(outliers: list, top_n: int = 10) -> None:
    print(f"\n--- Top {top_n} Like/Dislike Ratio Outliers ---")
    for i, (v, ratio) in enumerate(outliers[:top_n], start=1):
        print(
            f"{i}. {v['title']} | ratio={ratio:.2f} | "
            f"likes={v['likes']} dislikes={v['dislikes']} | "
            f"channel={v['channel_title']}"
        )

def month_key(dt: datetime) -> str:
    return f"{dt.year}-{dt.month:02d}"


def average_trending_duration_by_category_over_time(videos: list[dict]) -> dict:
    """
    Returns:
    { category_id: { "YYYY-MM": avg_duration_days, ...}, ...}
    """
    dates_by_video = defaultdict(set)
    video_to_cat = {}
    video_to_month = {}

    for v in videos:
        vid = v["video_id"]
        dates_by_video[vid].add(v["trending_date"])
        video_to_cat[vid] = v["category_id"]

        #assign month based on first time we see the video
        if vid not in video_to_month:
            video_to_month[vid] = month_key(v["trending_date"])

    durations_by_cat_month = defaultdict(list)

    for vid, dates in dates_by_video.items():
        duration_days = len(dates)
        cat=video_to_cat.get(vid)
        m = video_to_month.get(vid)
        if cat is not None and m is not None:
            durations_by_cat_month[(cat, m)].append(duration_days)

    result = defaultdict(dict)
    for (cat,  m),durations in durations_by_cat_month.items():
        result[cat][m] = statistics.mean(durations)

    return result

def plot_avg_trending_duration_linechart(avg_by_cat_month: dict, top_categories: int = 5) -> None:
    cats_sorted = sorted(
        avg_by_cat_month.keys(),
        key=lambda c: len(avg_by_cat_month[c]),
        reverse=True
    )[:top_categories]

    plt.figure(figsize=(10, 5))

    for cat in cats_sorted:
        month_to_avg = avg_by_cat_month[cat]
        months = sorted(month_to_avg.keys())
        values = [month_to_avg[m] for m in months]
        plt.plot(months, values, marker="o", label=f"Category {cat}")

    plt.title("Average Trending Duration per Category Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average days on trending")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_engagement_bar_chart(top_videos: list[dict]) -> None:
    titles = [
        v["title"][:20] + "..." if len(v["title"]) > 20 else v["title"]
        for v in top_videos
    ]
    likes = [v["likes"] for v in top_videos]
    dislikes = [v["dislikes"] for v in top_videos]
    comments = [v["comment_count"] for v in top_videos]

    x = list(range(len(top_videos)))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar([i - width for i in x], likes, width=width, label="Likes")
    plt.bar(x, dislikes, width=width, label="Dislikes")
    plt.bar([i + width for i in x], comments, width=width, label="Comments")

    plt.title("Engagement Metrics for Top Videos (by views)")
    plt.xlabel("Top Videos")
    plt.ylabel("Count")
    plt.xticks(x, titles, rotation=45, ha="right")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def export_video_to_json(video: dict, filename: str) -> None:
    export_data = video.copy()
    export_data["trending_date"] = export_data["trending_date"].isoformat()
    export_data["publish_time"] = export_data["publish_time"].isoformat()

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4)

    print(f"Exported selected video to {filename}")

def export_top10_to_csv(top_videos: list[dict], filename: str) -> None:
    fieldnames = [
        "video_id", "title", "channel_title", "category_id",
        "views", "likes", "dislikes", "comment_count"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in top_videos:
            writer.writerow({
                "video_id": v["video_id"],
                "title": v["title"],
                "channel_title": v["channel_title"],
                "category_id": v["category_id"],
                "views": v["views"],
                "likes": v["likes"],
                "dislikes": v["dislikes"],
                "comment_count": v["comment_count"]
            })

    print(f"Exported top 10 videos to {filename}")


def main():
    filename = "youtube_trending_videos.csv"
    videos = load_videos(filename)

    if not videos:
        print("No data loaded. Check the CSV file.")
        return

    print("Loaded records:", len(videos))
    print("First trending_date:", videos[0]["trending_date"])

    #basic processing
    unique_videos, unique_channels = total_unique_videos_and_channels(videos)
    print("Unique videos:", unique_videos)
    print("Unique channels:", unique_channels)

    category_counts = videos_per_category(videos)
    print("\nVideos per category:")
    for category, count in category_counts.items():
        print(f"Category {category}: {count} videos")
    print("Total category count:", sum(category_counts.values()))

    #basic visualisation
    plot_category_pie_chart(category_counts)
    plot_histogram(videos, "views")
    plot_histogram(videos, "likes")
    plot_histogram(videos, "comment_count")

    #top 10 lists
    unique_video_records = best_record_per_video(videos)
    top10_by_views = top_10_by_metric(unique_video_records, "views")
    print_top_10(top10_by_views, "views")
    export_top10_to_csv(top10_by_views, "top10_videos.csv")

    print_top_10(top_10_by_metric(unique_video_records, "likes"), "likes")
    print_top_10(top_10_by_metric(unique_video_records, "comment_count"), "comment_count")

    #intermediate processing
    averages = average_engagement_by_category(videos)
    print_average_engagement(averages)

    durations = trending_duration_per_video(videos)
    print_top_trending_durations(durations, top_n=10)

    outliers = like_dislike_ratio_outliers(videos, threshold=50)
    print_outliers(outliers, top_n=10)

    #intermediate visualisations
    avg_by_cat_month = average_trending_duration_by_category_over_time(videos)
    plot_avg_trending_duration_linechart(avg_by_cat_month, top_categories=5)

    plot_engagement_bar_chart(top10_by_views)

    #search by id
    print("\n--- Search video by ID ---")
    search_id = input("Enter a video_id to search: ").strip()
    results = find_video_records_by_id(videos, search_id)

    if not results:
        print("No video found with that video_id.")
    else:
        print(f"Found {len(results)} record(s) for video_id {search_id}")
        for record in results:
            print_video_record(record)

        export_video_to_json(results[0], "selected_video.json")

if __name__ == "__main__":
    main()

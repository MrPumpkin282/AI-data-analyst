import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_chart(df: pd.DataFrame, plan: dict):
    if not plan.get("need_chart") or df.empty:
        return None

    chart_type = plan.get("chart_type", "bar")
    group_by = plan.get("group_by") or []
    target_col = plan.get("target_column")

    if not target_col or target_col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and len(group_by) > 0:
        x_col = group_by[0]
        if x_col not in df.columns:
            return None
        plot_df = df.head(15)

        if chart_type == "bar":
            bars = ax.bar(range(len(plot_df)), plot_df[target_col], color=sns.color_palette("husl", len(plot_df)))
            ax.set_xticks(range(len(plot_df)))
            ax.set_xticklabels(plot_df[x_col], rotation=45, ha='right')
            ax.set_ylabel(target_col, fontsize=12)
            ax.set_xlabel(x_col, fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, plot_df[target_col])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:,.0f}' if abs(val) > 100 else f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9)

        elif chart_type == "line":
            ax.plot(range(len(plot_df)), plot_df[target_col], marker='o', linewidth=2, markersize=8)
            ax.set_xticks(range(len(plot_df)))
            ax.set_xticklabels(plot_df[x_col], rotation=45, ha='right')
            ax.set_ylabel(target_col, fontsize=12)
            ax.set_xlabel(x_col, fontsize=12)
            ax.grid(True, alpha=0.3)

        elif chart_type == "pie" and len(plot_df) <= 10:
            ax.pie(plot_df[target_col], labels=plot_df[x_col], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

        else:
            bars = ax.bar(range(len(plot_df)), plot_df[target_col])
            ax.set_xticks(range(len(plot_df)))
            ax.set_xticklabels(plot_df[x_col], rotation=45, ha='right')

        title = f"{target_col} by {x_col}"
    else:
        ax.bar([target_col], [df[target_col].iloc[0]])
        title = f"{target_col}"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf

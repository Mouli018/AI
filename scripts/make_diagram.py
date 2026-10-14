"""Draw the architecture figure -> paper/figs/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, text, fc="#E8F0FE", ec="#1A56B0", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, text="", style="-|>", color="#333", rad=0.0, dy=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", fontsize=7, color=color)


box(1, 31, 12, 9, "User\nquestion")
box(18, 31, 16, 9, "Query Analyzer\n(LLM, contract:\nQueryPlan)", fc="#FFF4E5", ec="#C77700")
box(39, 31, 16, 9, "Strategy Planner\n(policy: type \u2192\nbudget B, top-k)", fc="#FFF4E5", ec="#C77700")
box(60, 31, 16, 9, "Retriever Agent\n(TF-IDF/dense;\nmulti-query pool)")
box(81, 31, 18, 9, "Context Engineer\n(dedupe \u00b7 sentence\nscoring \u00b7 pack to B)", fc="#E6F4EA", ec="#1E8E3E", fs=8)
box(81, 12, 18, 9, "Answerer\n(LLM, contract:\nAnswerDraft)")
box(60, 12, 16, 9, "Critic\n(LLM, contract:\nCritique)", fc="#FCE8E6", ec="#C5221F")
box(39, 12, 16, 9, "Final answer\n+ trace")
box(1, 1.5, 98, 6.5, "Shared typed state (LangGraph):  plan \u00b7 decision \u00b7 candidate pool \u00b7 engineered context \u00b7 draft \u00b7 critiques \u00b7 iteration \u00b7 usage meter", fc="#F3E8FD", ec="#7B1FA2", fs=8)
arrow(13, 35.5, 18, 35.5); arrow(34, 35.5, 39, 35.5); arrow(55, 35.5, 60, 35.5); arrow(76, 35.5, 81, 35.5)
arrow(90, 31, 90, 21, "context\n\u2264 B tokens", dy=-0.5)
arrow(81, 16.5, 76, 16.5, "draft")
arrow(60, 16.5, 55, 16.5, "grounded &\nsufficient", dy=1.0)
arrow(68, 21, 68, 31, "", color="#C5221F")
ax.text(70.5, 25.6, "follow-up query\n(iterate \u2264 3)", fontsize=7, color="#C5221F")
ax.text(50, 10.2, "queries typed \"direct\" skip retrieval and go Planner \u2192 Answerer", fontsize=7.5, ha="center", color="#555")
ax.set_title("ContextRouter: agents, typed contracts and control flow", fontsize=10)
plt.savefig("paper/figs/architecture.png", dpi=220, bbox_inches="tight")
print("saved paper/figs/architecture.png")

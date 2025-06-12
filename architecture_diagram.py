import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Set background color
ax.set_facecolor('#f5f5f5')

# Define component colors
colors = {
    'frontend': '#4CAF50',
    'backend': '#2196F3',
    'tools': '#FF9800',
    'dataset': '#9C27B0',
    'memory': '#E91E63'  # Added memory color
}

# Draw components
def draw_box(x, y, width, height, color, label, fontsize=12):
    box = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

# Frontend
draw_box(3, 7, 4, 2, colors['frontend'], 'Streamlit\nChat Interface')

# Backend
draw_box(3, 4, 4, 2, colors['backend'], 'ReAct Agent')

# Tools
tools_x = 8
tools_y = 2
tools_width = 4
tools_height = 6
draw_box(tools_x, tools_y, tools_width, tools_height, colors['tools'], 'Tools', fontsize=14)

# Add tool names
tool_names = [
    'select_semantic_intent()',
    'select_semantic_category()',
    'count_category()',
    'count_intent()',
    'show_examples()',
    'summarize()',
    'get_intent_distribution()',
    'get_category_distribution()',
    'finish()'
]

for i, tool in enumerate(tool_names):
    y_pos = tools_y + tools_height - 0.7 - i * 0.65
    ax.text(tools_x + 0.3, y_pos, tool, fontsize=9, ha='left', va='center')

# Dataset
draw_box(3, 1, 4, 2, colors['dataset'], 'Customer Service\nDataset')

# Memory (new component)
memory_x = 0
memory_y = 3
memory_width = 2.5
memory_height = 4
draw_box(memory_x, memory_y, memory_width, memory_height, colors['memory'], 'Memory\nSystem', fontsize=12)

# Add memory components
memory_components = [
    'Interactions',
    'Summaries',
    'Insights',
    'Retrieval'
]

for i, component in enumerate(memory_components):
    y_pos = memory_y + memory_height - 0.7 - i * 0.8
    ax.text(memory_x + 0.3, y_pos, component, fontsize=9, ha='left', va='center')

# Draw arrows
def draw_arrow(x1, y1, x2, y2, color='black'):
    ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.3, fc=color, ec=color, length_includes_head=True)

# User to Frontend
draw_arrow(1, 8, 3, 8)
ax.text(2, 8.2, 'Query', ha='center', va='bottom')

# Frontend to Backend
draw_arrow(5, 7, 5, 6)
ax.text(5.2, 6.5, 'Question', ha='left', va='center')

# Backend to Frontend
draw_arrow(4, 6, 4, 7)
ax.text(3.8, 6.5, 'Answer', ha='right', va='center')

# Backend to Tools
draw_arrow(7, 5, 8, 5)
ax.text(7.5, 5.2, 'Function Calls', ha='center', va='bottom')

# Tools to Backend
draw_arrow(8, 4, 7, 4)
ax.text(7.5, 3.8, 'Results', ha='center', va='top')

# Tools to Dataset
draw_arrow(8, 3, 7, 2)
ax.text(7.5, 2.5, 'Query', ha='center', va='center')

# Dataset to Tools
draw_arrow(7, 1.5, 8, 2.5)
ax.text(7.5, 1.8, 'Data', ha='center', va='center')

# Memory to Backend
draw_arrow(2.5, 5, 3, 5)
ax.text(2.75, 5.2, 'Context', ha='center', va='bottom')

# Backend to Memory
draw_arrow(3, 4.5, 2.5, 4.5)
ax.text(2.75, 4.3, 'Store', ha='center', va='top')

# Set limits and remove axes
ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis('off')

# Add title
ax.set_title('Customer Service Dataset Q&A Agent Architecture', fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    patches.Patch(facecolor=colors['frontend'], edgecolor='black', alpha=0.7, label='Frontend'),
    patches.Patch(facecolor=colors['backend'], edgecolor='black', alpha=0.7, label='Backend'),
    patches.Patch(facecolor=colors['tools'], edgecolor='black', alpha=0.7, label='Tools'),
    patches.Patch(facecolor=colors['dataset'], edgecolor='black', alpha=0.7, label='Dataset'),
    patches.Patch(facecolor=colors['memory'], edgecolor='black', alpha=0.7, label='Memory')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Save the figure
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

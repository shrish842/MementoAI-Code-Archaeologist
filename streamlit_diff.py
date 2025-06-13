import streamlit as st
from typing import List, Optional
import html

def diff_viewer(
    old_lines: List[str],
    new_lines: List[str],
    old_title: str = "Old Version",
    new_title: str = "New Version",
    lang: Optional[str] = None,
    show_gutter: bool = True,
    show_line_numbers: bool = True,
    wrap_lines: bool = True,
    height: Optional[int] = None,
):
    """
    Display a side-by-side diff viewer in Streamlit.
    
    Args:
        old_lines: List of lines from the old version
        new_lines: List of lines from the new version
        old_title: Title for the old version panel
        new_title: Title for the new version panel
        lang: Language for syntax highlighting (None for no highlighting)
        show_gutter: Whether to show the gutter (line change markers)
        show_line_numbers: Whether to show line numbers
        wrap_lines: Whether to wrap long lines
        height: Fixed height in pixels (None for automatic)
    """
    # Generate HTML for the diff viewer
    html_content = f"""
    <div class="diff-container" style="height: {height}px" if height else ""}>
        <div class="diff-pane">
            <h4>{old_title}</h4>
            <pre><code class="language-{lang}">{generate_diff_html(old_lines, 'old', show_gutter, show_line_numbers, wrap_lines)}</code></pre>
        </div>
        <div class="diff-pane">
            <h4>{new_title}</h4>
            <pre><code class="language-{lang}">{generate_diff_html(new_lines, 'new', show_gutter, show_line_numbers, wrap_lines)}</code></pre>
        </div>
    </div>
    """
    
    # Display in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Add syntax highlighting if language is specified
    if lang:
        st.markdown(f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        """, unsafe_allow_html=True)

def generate_diff_html(
    lines: List[str],
    version: str,
    show_gutter: bool,
    show_line_numbers: bool,
    wrap_lines: bool,
) -> str:
    """
    Generate HTML for one side of the diff viewer.
    
    Args:
        lines: List of lines to display
        version: 'old' or 'new' (for styling)
        show_gutter: Whether to show the gutter
        show_line_numbers: Whether to show line numbers
        wrap_lines: Whether to wrap lines
    
    Returns:
        HTML string
    """
    html_lines = []
    
    for i, line in enumerate(lines, 1):
        line_class = ""
        gutter_class = ""
        
        # Determine line type (added, removed, context)
        if version == 'old' and i < len(lines) and lines[i].startswith('+') and not lines[i].startswith('++'):
            line_class = "diff-removed"
            gutter_class = "removed"
        elif version == 'new' and line.startswith('+') and not line.startswith('++'):
            line_class = "diff-added"
            gutter_class = "added"
        
        # Build line HTML
        line_html = []
        
        if show_gutter:
            line_html.append(f'<span class="diff-gutter {gutter_class}"></span>')
        
        if show_line_numbers:
            line_html.append(f'<span class="diff-line-number">{i}</span>')
        
        escaped_line = html.escape(line.replace('\t', '    '))
        line_html.append(f'<span class="diff-line {line_class}">{escaped_line}</span>')
        
        html_lines.append(''.join(line_html))
    
    return '\n'.join(html_lines)
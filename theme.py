"""
Theme definition for ElevenLabs Sample Splitter
This file contains color schemes and styling for the application UI.
The default theme is inspired by Discord's color palette.
"""

# Discord-inspired theme
DISCORD = {
    # Main colors
    "primary": "#5865F2",       # Discord Blurple
    "secondary": "#57F287",     # Discord Green
    "tertiary": "#EB459E",      # Discord Pink
    "accent": "#FEE75C",        # Discord Yellow
    "danger": "#ED4245",        # Discord Red
    
    # Background colors
    "bg_dark": "#202225",       # Discord Dark Background
    "bg_mid": "#2F3136",        # Discord Medium Background
    "bg_light": "#36393F",      # Discord Light Background
    "bg_ultra_light": "#40444B", # Discord Ultra Light Background
    
    # Text colors
    "text_normal": "#DCDDDE",   # Discord Normal Text
    "text_muted": "#A3A6AA",    # Discord Muted Text
    "text_heading": "#FFFFFF",  # Discord Heading Text
    "text_link": "#00B0F4",     # Discord Link Text
    
    # UI element colors
    "input_bg": "#40444B",      # Discord Input Background
    "input_border": "#202225",  # Discord Input Border
    "button_secondary": "#4F545C", # Discord Secondary Button
    "scrollbar": "#202225",     # Discord Scrollbar
    "scrollbar_thumb": "#4F545C", # Discord Scrollbar Thumb
    
    # Misc
    "divider": "#2D2F33",       # Discord Divider
    "shadow": "rgba(0, 0, 0, 0.3)", # Discord Shadow
    
    # Style sheets (using Qt style syntax)
    "window": """
        QMainWindow {
            background-color: #36393F;
            color: #DCDDDE;
        }
    """,
    
    "button": """
        QPushButton {
            background-color: #5865F2;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #4752C4;
        }
        QPushButton:pressed {
            background-color: #3C45A5;
        }
        QPushButton:disabled {
            background-color: #4F545C;
            color: #A3A6AA;
        }
    """,
    
    "secondary_button": """
        QPushButton {
            background-color: #4F545C;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #5D636B;
        }
        QPushButton:pressed {
            background-color: #6A707A;
        }
        QPushButton:disabled {
            background-color: #4F545C;
            color: #A3A6AA;
        }
    """,
    
    "danger_button": """
        QPushButton {
            background-color: #ED4245;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #C03536;
        }
        QPushButton:pressed {
            background-color: #A12D2F;
        }
    """,
    
    "input": """
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #40444B;
            color: #DCDDDE;
            border: 1px solid #202225;
            border-radius: 4px;
            padding: 5px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #5865F2;
        }
    """,
    
    "group_box": """
        QGroupBox {
            background-color: #2F3136;
            color: #FFFFFF;
            border: 1px solid #202225;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #FFFFFF;
        }
    """,
    
    "list_widget": """
        QListWidget {
            background-color: #2F3136;
            color: #DCDDDE;
            border: 1px solid #202225;
            border-radius: 4px;
            padding: 2px;
        }
        QListWidget::item {
            border-radius: 2px;
            padding: 4px;
        }
        QListWidget::item:selected {
            background-color: #5865F2;
            color: #FFFFFF;
        }
        QListWidget::item:hover {
            background-color: #4752C4;
        }
    """,
    
    "progress_bar": """
        QProgressBar {
            background-color: #2F3136;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #5865F2;
            border-radius: 4px;
        }
    """,
    
    "tab_widget": """
        QTabWidget::pane {
            border: 1px solid #202225;
            background-color: #2F3136;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #4F545C;
            color: #DCDDDE;
            padding: 8px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #5865F2;
            color: #FFFFFF;
        }
        QTabBar::tab:hover:!selected {
            background-color: #5D636B;
        }
    """,
    
    "scrollbar": """
        QScrollBar:vertical {
            background-color: #2F3136;
            width: 16px;
            margin: 16px 0 16px 0;
        }
        QScrollBar::handle:vertical {
            background-color: #4F545C;
            min-height: 25px;
            border-radius: 7px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #5D636B;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 15px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        
        QScrollBar:horizontal {
            background-color: #2F3136;
            height: 16px;
            margin: 0 16px 0 16px;
        }
        QScrollBar::handle:horizontal {
            background-color: #4F545C;
            min-width: 25px;
            border-radius: 7px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #5D636B;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
            width: 15px;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }
    """,
    
    "check_box": """
        QCheckBox {
            color: #DCDDDE;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 1px solid #4F545C;
        }
        QCheckBox::indicator:unchecked {
            background-color: #2F3136;
        }
        QCheckBox::indicator:checked {
            background-color: #5865F2;
        }
        QCheckBox::indicator:hover {
            border: 1px solid #5865F2;
        }
    """,
    
    "radio_button": """
        QRadioButton {
            color: #DCDDDE;
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 1px solid #4F545C;
        }
        QRadioButton::indicator:unchecked {
            background-color: #2F3136;
        }
        QRadioButton::indicator:checked {
            background-color: #5865F2;
        }
        QRadioButton::indicator:hover {
            border: 1px solid #5865F2;
        }
    """,
    
    "slider": """
        QSlider::groove:horizontal {
            border: none;
            height: 8px;
            background-color: #2F3136;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background-color: #5865F2;
            border: none;
            width: 16px;
            height: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background-color: #4752C4;
        }
    """,
    
    "label": """
        QLabel {
            color: #DCDDDE;
        }
    """,
    
    "heading_label": """
        QLabel {
            color: #FFFFFF;
            font-weight: bold;
            font-size: 14px;
        }
    """,
    
    "status_label": """
        QLabel {
            color: #5865F2;
            font-style: italic;
        }
    """,
    
    "warning_label": """
        QLabel {
            color: #ED4245;
            font-weight: bold;
        }
    """,
    
    "dialog": """
        QDialog {
            background-color: #36393F;
            color: #DCDDDE;
        }
    """
}

# Theme access function
def get_theme(theme_name="discord"):
    """
    Get the specified theme dictionary
    
    Args:
        theme_name (str): Name of the theme (default: "discord")
        
    Returns:
        dict: Theme color and style definitions
    """
    theme_name = theme_name.lower()
    
    if theme_name == "discord":
        return DISCORD
    
    # Default to Discord theme if the specified theme doesn't exist
    return DISCORD

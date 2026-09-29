# VHS Tape Script for wm_fuzzy prompt_infer demo
Output demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1200
Set Height 800
Set Padding 20

# Hide compilation output
Hide
Type "cargo build --quiet"
Enter
Sleep 3s
Ctrl+L
Show

# Start running the demo binary
Type "cargo run --example demo"
Sleep 500ms
Enter

# Wait until prompt_infer requests the first input
Sleep 1.5s

# Type input value for attribute 1 (e.g., temperature)
Type "22.5"
Sleep 600ms
Enter

# Type input value for attribute 2 (e.g., humidity)
Sleep 800ms
Type "45.0"
Sleep 600ms
Enter

# Pause to view batch results and output artifacts
Sleep 4s

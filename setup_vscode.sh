#!/bin/bash
# Script to setup VS Code for the interactive video processor branch

echo "=== GenTbD - VS Code Setup Script ==="
echo ""

# Check we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "❌ Error: Not in GenTbD directory"
    echo "Please run this script from the GenTbD root directory"
    exit 1
fi

echo "✅ In GenTbD directory"
echo ""

# Check git remote
echo "📡 Checking git remote..."
REMOTE=$(git remote get-url origin 2>/dev/null)
if [[ "$REMOTE" == *"MikCas/GenTbD"* ]]; then
    echo "✅ Connected to MikCas/GenTbD"
else
    echo "⚠️  Remote: $REMOTE"
fi
echo ""

# Fetch all remote branches
echo "📥 Fetching remote branches..."
git fetch origin
echo ""

# Show all branches
echo "📋 Available branches:"
git branch -a | grep -E "(base|interactive)" | head -10
echo ""

# Check if interactive branch exists
if git show-ref --verify --quiet refs/remotes/origin/claude/interactive-ui-t-0176UsizjhUcLiaGqcSwhsxp; then
    echo "✅ Interactive UI branch found on remote!"
    echo ""

    # Ask if user wants to checkout
    read -p "Do you want to checkout the interactive-ui branch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout claude/interactive-ui-t-0176UsizjhUcLiaGqcSwhsxp
        echo ""
        echo "✅ Checked out interactive-ui branch"
        echo ""
        echo "You can now run:"
        echo "  python examples/demo_interactive.py --webcam"
    fi
else
    echo "❌ Interactive UI branch not found on remote"
    echo "Current branches:"
    git branch -a
fi

echo ""
echo "=== Setup Complete ==="

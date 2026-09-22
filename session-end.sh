#!/bin/bash
# session-end.sh — run at the end of every working session

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       End of session cleanup         ║"
echo "╚══════════════════════════════════════╝"
echo ""

echo "📋 Current git status:"
git status --short
echo ""

read -p "Commit changes now? (y/n): " commit_choice
if [[ "$commit_choice" == "y" ]]; then
  git add .
  read -p "Commit message: " commit_msg
  git commit -m "$commit_msg" || echo "Nothing new to commit."
  git push
  echo "✅ Pushed to GitHub."
fi

echo ""
echo "🐳 Docker cleanup options:"
echo "  1) Stop container only (keep image — fastest restart)"
echo "  2) Stop container + remove image (saves disk)"
echo "  3) Full prune — stop, remove image, remove all unused Docker data"
echo "  4) Skip Docker cleanup"
echo ""
read -p "Choose (1/2/3/4): " docker_choice

CONTAINER_ID=$(docker ps -q --filter "label=devcontainer.local_folder=${PWD}")

case $docker_choice in
  1)
    docker stop $CONTAINER_ID 2>/dev/null || true
    echo "✅ Container stopped. Image kept."
    ;;
  2)
    IMAGE_ID=$(docker inspect --format='{{.Image}}' $CONTAINER_ID 2>/dev/null)
    docker stop $CONTAINER_ID 2>/dev/null || true
    docker rmi $IMAGE_ID 2>/dev/null || true
    echo "✅ Container stopped and image removed."
    ;;
  3)
    docker stop $CONTAINER_ID 2>/dev/null || true
    docker system prune -f
    echo "✅ Full Docker prune complete."
    ;;
  4)
    echo "⏭  Skipping Docker cleanup."
    ;;
esac

echo ""
echo "📁 Workspace folder is safe at: $(pwd)/workspace"
echo "   → notebooks/   ✅"
echo "   → outputs/     ✅"
echo "   → data/        ✅ (local only, not pushed)"
echo ""
echo "✅ Session ended. See you next time!"
echo ""

@echo off
echo ==================================================
echo Automatically Pushing Offroad Segmentation to Github
echo ==================================================
echo.

cd /d "C:\Users\divya\OneDrive\Desktop\Efinity\Offroad_Segmentation_Scripts"

echo [1/5] Initializing Git Repository...
git init

echo [2/5] Staging files (ignoring 1.1GB checkpoint history)...
git add .

echo [3/5] Committing changes...
git commit -m "Fixed Kaggle Checkpoint Extraction and implemented correct Global IoU metrics"

echo [4/5] Setting main branch and Remote URL...
git branch -M main
git remote add origin https://github.com/hariomThacker751/Enfinity.git

echo [5/5] Force Pushing to GitHub...
git push -u origin main --force

echo.
echo ==================================================
echo PUSH COMPLETE! Check your github page!
echo ==================================================
pause

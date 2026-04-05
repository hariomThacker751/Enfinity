@echo off
echo ==================================================
echo Automatically Updating Github Repository
echo ==================================================
echo.

cd /d "C:\Users\divya\OneDrive\Desktop\Efinity\Offroad_Segmentation_Scripts"

echo [1/3] Staging new and modified files...
git add .

echo [2/3] Committing changes...
git commit -m "Minor code update and training refinements"

echo [3/3] Pushing to GitHub...
git push origin main

echo.
echo ==================================================
echo UPDATE COMPLETE! Check your github page!
echo ==================================================
pause

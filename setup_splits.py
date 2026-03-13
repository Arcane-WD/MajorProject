import shutil
import os

# Copy files
shutil.copy('app.py', 'main_direct.py')
shutil.copy('pipeline.py', 'pipeline_direct.py')
shutil.copy('app.py', 'demons_main.py')
shutil.copy('pipeline.py', 'pipeline_demons.py')

# Modify imports in main_direct.py
with open('main_direct.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('import pipeline', 'import pipeline_direct as pipeline')
with open('main_direct.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Modify imports in demons_main.py
with open('demons_main.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('import pipeline', 'import pipeline_demons as pipeline')
with open('demons_main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Files copied and imports updated successfully.")

import os

directories = {
    'storage': ['eval', 'logs', 'octis', 'queries', 'tipster/corpora_unzipped']
}

for k, v in directories.items():
    for dir in v:
        try:
            os.makedirs(f'{k}/{dir}', exist_ok=True)
        except Exception as e:
            print(f'An error occurred while creating {k}/{dir}: {e}')

print('ALL DIRECTORIES CREATED.\nPlease populate them as indicated in readme.md.')

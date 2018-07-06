import subprocess, json, os

# extract lines from tar.gz file
filepath = os.path.dirname(os.path.abspath(__file__))
subprocess.call(['bash', os.path.join(filepath, 'sogou.sh'), filepath])
tmp = {
	'content.txt':[],
	'contenttitle.txt':[],
	'url.txt':[],
	'docno.txt':[]
}

# remove tag like this <content> from lines
for file in  ['content.txt', 'contenttitle.txt', 'url.txt', 'docno.txt']:
	with open(os.path.join(filepath, file), 'r', encoding='utf-8') as f:
		tag = file.replace('.txt', '')
		for line in f:
			tmp[file].append(line.replace('<{}>'.format(tag), '').replace('</{}>'.format(tag), ''))

# turn lines into json format
result = []
for content, contenttitle, url, docno in zip(tmp['content.txt'], tmp['contenttitle.txt'], tmp['url.txt'], tmp['docno.txt']):
	result.append(
		{
			'content':content,
			'contenttitle':contenttitle,
			'url':url,
			'docno':docno
		}
	)
json.dump(result, open(os.path.join(filepath, 'sogou.json'), 'w', encoding='utf-8'))
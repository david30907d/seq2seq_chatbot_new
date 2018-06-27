import subprocess, json
subprocess.call(['bash', 'sogou.sh'])
tmp = {
	'content.txt':[],
	'contenttitle.txt':[],
	'url.txt':[],
	'docno.txt':[]
}
for file in  ['content.txt', 'contenttitle.txt', 'url.txt', 'docno.txt']:
	with open(file, 'r', encoding='utf-8') as f:
		tag = file.replace('.txt', '')
		for line in f:
			tmp[file].append(line.replace('<{}>'.format(tag), '').replace('</{}>'.format(tag), ''))

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
json.dump(result, open('sogou.json', 'w', encoding='utf-8'))

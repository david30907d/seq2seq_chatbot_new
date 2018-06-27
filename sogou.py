import subprocess, json

# extract lines from tar.gz file
subprocess.call(['bash', 'sogou.sh'])
tmp = {
	'content.txt':[],
	'contenttitle.txt':[],
	'url.txt':[],
	'docno.txt':[]
}

# remove tag like this <content> from lines
for file in  ['content.txt', 'contenttitle.txt', 'url.txt', 'docno.txt']:
	with open(file, 'r', encoding='utf-8') as f:
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
json.dump(result, open('sogou.json', 'w', encoding='utf-8'))

# remove redundent file
subprocess.Popen('rm *.txt', shell=True)
subprocess.Popen('rm *.dat', shell=True)
subprocess.Popen('rm *.tar.gz', shell=True)
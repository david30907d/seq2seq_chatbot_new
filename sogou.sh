cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<content>" | opencc > content.txt
cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<contenttitle>" | opencc > contenttitle.txt
cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<url>" | opencc > url.txt
cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<docno>" | opencc > docno.txt

# small dataset
# wget http://download.labs.sogou.com/dl/sogoulabdown/SogouCA/news_tensite_xml.smarty.tar.gz
# tar zxvf news_tensite_xml.smarty.tar.gz
# cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<content>" | opencc > content.txt
# cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<contenttitle>" | opencc > contenttitle.txt
# cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<url>" | opencc > url.txt
# cat ./news_tensite_xml.smarty.dat | iconv -f gbk -t utf-8 -c | grep "<docno>" | opencc > docno.txt

# full dataset
wget http://www.sogou.com/labs/resource/ftp.php?dir=/Data/SogouCS/news_sohusite_xml.full.tar.gz
tar zxvf news_sohusite_xml.full.tar.gz
cat ./news_sohusite_xml.full.dat | iconv -f gbk -t utf-8 -c | grep "<content>" | opencc > content.txt
cat ./news_sohusite_xml.full.dat | iconv -f gbk -t utf-8 -c | grep "<contenttitle>" | opencc > contenttitle.txt
cat ./news_sohusite_xml.full.dat | iconv -f gbk -t utf-8 -c | grep "<url>" | opencc > url.txt
cat ./news_sohusite_xml.full.dat | iconv -f gbk -t utf-8 -c | grep "<docno>" | opencc > docno.txt


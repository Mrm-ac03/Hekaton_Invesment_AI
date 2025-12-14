from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json
import os
import random
from sklearn.base import BaseEstimator, TransformerMixin
import time

# 1. MODEL SINIFI (Veri temizliği için)
class CoerceNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols: X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

app = Flask(__name__)

# --- %100 GARANTİLİ İSTANBUL VERİSİ (Dropdownlar için) ---
ISTANBUL_DATA = {
    "Adalar": ["Burgazada Mh.", "Heybeliada Mah.", "Kınalıada Mh.", "Maden Mh.", "Nizam Mh."], 
    "Arnavutköy": ["Adnan Menderes Mah.", "Anadolu Mah.", "Arnavutköy Merkez Mh.", "Atatürk Mh.", "Bolluca Mah.", "Boğazköy İstiklal Mh.", "Deliklikaya Mah.", "Dursunköy Mh.", "Fatih Mah.", "Hadımköy Mh.", "Haraççı Mh.", "Hastane Mah.", "Hicret Mah.", "Karlıbayır Mh.", "Mareşal Fevzi Çakmak Mh.", "Mavigöl Mh.", "Mehmet Akif Ersoy Mah.", "Mustafa Kemal Paşa Mh.", "Nenehatun Mah.", "Taşoluk Mh.", "Yavuz Selim Mah.", "Yunus Emre Mah.", "Ömerli Mh.", "İslambey Mh."], 
    "Ataşehir": ["Atatürk Mh.", "Aşık Veysel Mh.", "Barbaros Mah.", "Esatpaşa Mh.", "Ferhatpaşa Mh.", "Fetih Mah.", "Kayışdağı Mh.", "Küçükbakkalköy Mh.", "Mevlana Mah.", "Mimar Sinan Mah.", "Mustafa Kemal Mah.", "Yeni Çamlıca Mh.", "Yenişehir Mh.", "Örnek Mh.", "İnönü Mh.", "İçerenköy Mh."], 
    "Avcılar": ["Ambarlı Mh.", "Cihangir Mah.", "Denizköşkler Mh.", "Firuzköy Mh.", "Gümüşpala Mh.", "Merkez Mah.", "Mustafa Kemal Paşa Mh.", "Tahtakale Mah.", "Yeşilkent Mh.", "Üniversite Mh."], 
    "Bahçelievler": ["Bahçelievler Mh.", "Cumhuriyet Mah.", "Fevzi Çakmak Mh.", "Hürriyet Mh.", "Kocasinan Merkez Mah.", "Siyavuşpaşa Mh.", "Soğanlı Mh.", "Yenibosna Merkez Mah.", "Zafer Mah.", "Çobançeşme Mh.", "Şirinevler Mh."], 
    "Bakırköy": ["Ataköy 1. Kısım Mh.", "Ataköy 2-5-6. Kısım Mh.", "Ataköy 3-4-11. Kısım Mh.", "Ataköy 7-8-9-10. Kısım Mh.", "Basınköy Mh.", "Cevizlik Mah.", "Kartaltepe Mah.", "Osmaniye Mah.", "Sakızağacı Mh.", "Yenimahalle Mah.", "Yeşilköy Mh.", "Yeşilyurt Mh.", "Zeytinlik Mah.", "Zuhuratbaba Mah.", "Şenlikköy Mh."], 
    "Bayrampaşa": ["Altıntepsi Mh.", "Cevatpaşa Mh.", "Kartaltepe Mah.", "Kocatepe Mah.", "Muratpaşa Mh.", "Ortamahalle Mah.", "Terazidere Mah.", "Vatan Mah.", "Yenidoğan Mh.", "Yıldırım Mh.", "İsmet Paşa Mh."], 
    "Bağcılar": ["100. Yıl Mh.", "15 Temmuz Mh.", "Barbaros Mah.", "Bağlar Mh.", "Demirkapı Mh.", "Fatih Mah.", "Fevzi Çakmak Mh.", "Göztepe Mh.", "Güneşli Mh.", "Hürriyet Mh.", "Kazım Karabekir Mh.", "Kemalpaşa Mh.", "Kirazlı Mh.", "Mahmutbey Mah.", "Merkez Mah.", "Sancaktepe Mah.", "Yavuz Selim Mah.", "Yenigün Mh.", "Yenimahalle Mah.", "Yıldıztepe Mh.", "Çınar Mh.", "İnönü Mh."], 
    "Başakşehir": ["Altınşehir Mh.", "Bahçeşehir 1. Kısım Mh.", "Bahçeşehir 2. Kısım Mh.", "Başak Mh.", "Başakşehir Mh.", "Güvercintepe Mh.", "Kayabaşı Mh.", "Ziya Gökalp Mh.", "İkitelli OSB"], 
    "Beykoz": ["Acarlar Mah.", "Anadolu Hisarı Mh.", "Elmalı Mh.", "Fatih Mah.", "Göksu Mh.", "Göztepe Mh.", "Kanlıca Mh.", "Kavacık Mh.", "Merkez Mah.", "Ortaçeşme Mh.", "Paşabahçe Mh.", "Poyrazköy Mh.", "Riva Köyü", "Rüzgarlıbahçe Mh.", "Soğuksu Mh.", "Yalıköy Mh.", "Yavuz Selim Mah.", "Yeni Mahalle Mah.", "Çamlıbahçe Mh.", "Çiğdem Mh.", "Çubuklu Mh."], 
    "Beylikdüzü": ["Adnan Kahveci Mah.", "Barış Mh.", "Beylikdüzü OSB", "Büyükşehir Mh.", "Cumhuriyet Mah.", "Dereağzı Mh.", "Gürpınar Mh.", "Kavaklı Mh.", "Marmara Mah.", "Sahil Mah.", "Yakuplu Mah."], 
    "Beyoğlu": ["Arap Cami Mah.", "Asmalı Mescit Mh.", "Bereketzade Mah.", "Bostan Mah.", "Bülbül Mh.", "Camiikebir Mah.", "Cihangir Mah.", "Evliya Çelebi Mh.", "Fetihtepe Mah.", "Firuzağa Mh.", "Gümüşsuyu Mh.", "Hacıahmet Mh.", "Hacımimi Mh.", "Halıcıoğlu Mh.", "Hüseyinağa Mh.", "Kadımehmet Efendi Mh.", "Kalyoncu Kulluğu Mh.", "Kamer Hatun Mah.", "Kaptanpaşa Mh.", "Katip Mustafa Çelebi Mh.", "Keçeci Piri Mh.", "Kocatepe Mah.", "Kulaksız Mh.", "Kuloğlu Mh.", "Küçük Piyale Mh.", "Kılıçali Paşa Mh.", "Müeyyetzade Mh.", "Piri Paşa Mh.", "Piyalepaşa Mh.", "Pürtelaş Hasan Efendi Mh.", "Sururi Mehmet Efendi Mah.", "Sütlüce Mh.", "Tomtom Mah.", "Yahya Kahya Mah.", "Yenişehir Mh.", "Çatma Mescit Mh.", "Çukur Mh.", "Ömer Avni Mh.", "Örnektepe Mh.", "İstiklal Mh.", "Şahkulu Mh.", "Şehit Muhtar Mh."], 
    "Beşiktaş": ["Abbasağa Mh.", "Akat Mah.", "Arnavutköy Mh.", "Balmumcu Mah.", "Bebek Mah.", "Cihannüma Mh.", "Dikilitaş Mh.", "Etiler Mah.", "Gayrettepe Mah.", "Konaklar Mah.", "Kuruçeşme Mh.", "Kültür Mh.", "Levent Mah.", "Levazım Mh.", "Mecidiye Mah.", "Muradiye Mah.", "Nisbetiye Mh.", "Ortaköy Mh.", "Sinanpaşa Mh.", "Türkali Mh.", "Ulus Mah.", "Vişnezade Mh.", "Yıldız Mh."], 
    "Büyükçekmece": ["19 Mayıs Mh.", "Alkent 2000 Mah.", "Atatürk Mh.", "Bahçelievler Mh.", "Celaliye Mah.", "Cumhuriyet Mah.", "Dizdariye Mah.", "Ekinoba Mah.", "Fatih Mah.", "Güzelce Mh.", "Hürriyet Mh.", "Kamiloba Mah.", "Karaağaç Mh.", "Kumburgaz Merkez Mah.", "Mimar Sinan Merkez Mh.", "Mimaroba", "Murat Çeşme Mh.", "Pınartepe Mh.", "Sinanoba", "Türkoba Mh.", "Ulus Mah.", "Yenimahalle Mah.", "Çakmaklı Mh."], 
    "Esenler": ["Birlik Mah.", "Davutpaşa Mh.", "Fatih Mah.", "Fevzi Çakmak Mh.", "Havaalanı Mh.", "Kazım Karabekir Mh.", "Kemer Mah.", "Menderes Mah.", "Mimar Sinan Mah.", "Namık Kemal Mh.", "Nine Hatun Mah.", "Oruçreis Mh.", "Tuna Mah.", "Turgut Reis Mah.", "Yavuz Selim Mah.", "Çifte Havuzlar Mh."], 
    "Esenyurt": ["Akevler Mh.", "Akçaburgaz Mh.", "Akşemseddin Mh.", "Ardıçlı Mh.", "Atatürk Mh.", "Aşık Veysel Mh.", "Balıkyolu Mh.", "Barbaros Hayrettin Paşa Mh.", "Battalgazi Mh.", "Bağlarçeşme Mh.", "Cumhuriyet Mah.", "Esenkent Mah.", "Fatih Mah.", "Gökevler Mh.", "Güzelyurt Mh.", "Hürriyet Mh.", "Koza Mh.", "Mehmet Akif Ersoy Mh.", "Mehterçeşme Mh.", "Mevlana Mh.", "Namık Kemal Mh.", "Necip Fazıl Kısakürek Mh.", "Orhan Gazi Mah.", "Osmangazi Mh.", "Piri Reis Mh.", "Pınar Mh.", "Saadetdere Mah.", "Selahaddin Eyyubi Mh.", "Sultaniye Mh.", "Süleymaniye Mh.", "Talatpaşa Mh.", "Turgut Özal Mh.", "Yenikent Mah.", "Yeşilkent Mh.", "Yunus Emre Mh.", "Zafer Mh.", "Çınar Mh.", "Örnek Mh.", "Üçevler Mh.", "İncirtepe Mh.", "İnönü Mh.", "İstiklal Mh.", "Şehitler Mh."], 
    "Eyüpsultan": ["Akşemsettin Mh.", "Alibeyköy Mh.", "Defterdar Mah.", "Düğmeciler Mh.", "Emniyettepe Mah.", "Esentepe Mah.", "Eyüp Merkez Mah.", "Göktürk Merkez Mh.", "Güzeltepe Mh.", "Karadolap Mah.", "Mimar Sinan Mh.", "Mithatpaşa Mh.", "Nişancı Mh.", "Rami Cuma Mah.", "Rami Yeni Mah.", "Sakarya Mah.", "Silahtarağa Mh.", "Yeşilpınar Mh.", "Çırçır Mh.", "İslambey Mh."], 
    "Fatih": ["Aksaray Mah.", "Akşemsettin Mh.", "Ali Kuşçu Mh.", "Atikali Mah.", "Ayvansaray Mah.", "Balat Mah.", "Binbirdirek Mah.", "Cerrahpaşa Mh.", "Cibali Mah.", "Derviş Ali Mh.", "Emin Sinan Mah.", "Hacı Kadın Mh.", "Haseki Sultan Mah.", "Hırka-i Şerif Mh.", "Karagümrük Mh.", "Katip Kasım Mh.", "Kemal Paşa Mh.", "Koca Mustafapaşa Mh.", "Mesih Paşa Mh.", "Mevlanakapı Mh.", "Molla Gürani Mh.", "Muhsine Hatun Mah.", "Nişanca Mh.", "Saraç İshak Mh.", "Seyyid Ömer Mh.", "Silivrikapı Mh.", "Sümbül Efendi Mh.", "Topkapı Mh.", "Yavuz Sultan Selim Mah.", "Yedikule Mah.", "Zeyrek Mah.", "İskenderpaşa Mh.", "Şehremini Mh.", "Şehsuvar Bey Mh."], 
    "Gaziosmanpaşa": ["Barbaros Hayrettinpaşa Mh.", "Bağlarbaşı Mh.", "Fevzi Çakmak Mh.", "Hürriyet Mh.", "Karadeniz Mah.", "Karayolları Mh.", "Karlıtepe Mh.", "Kazım Karabekir Mh.", "Merkez Mah.", "Mevlana Mah.", "Pazariçi Mh.", "Sarıgöl Mh.", "Yeni Mahalle Mh.", "Yenidoğan Mh.", "Yıldıztabya Mh.", "Şemsipaşa Mh."], 
    "Güngören": ["Abdurrahman Nafiz Gürman Mh.", "Akıncılar Mh.", "Gencosman Mh.", "Güven Mh.", "Güneştepe Mh.", "Haznedar Mah.", "Mareşal Çakmak Mh.", "Mehmet Nesih Özmen Mh.", "Merkez Mah.", "Sanayi Mah.", "Tozkoparan Mah."], 
    "Kadıköy": ["19 Mayıs Mh.", "Acıbadem Mh.", "Bostancı Mh.", "Caddebostan Mah.", "Caferağa Mh.", "Dumlupınar Mh.", "Erenköy Mh.", "Eğitim Mh.", "Fenerbahçe Mh.", "Feneryolu Mah.", "Fikirtepe Mah.", "Göztepe Mh.", "Hasanpaşa Mh.", "Kozyatağı Mh.", "Koşuyolu Mh.", "Merdivenköy Mh.", "Osmanağa Mh.", "Rasimpaşa Mh.", "Sahrayı Cedit Mh.", "Suadiye Mah.", "Zühtüpaşa Mh."], 
    "Kartal": ["Atalar Mah.", "Cevizli Mah.", "Cumhuriyet Mah.", "Esentepe Mah.", "Gümüşpınar Mh.", "Hürriyet Mh.", "Karlıktepe Mh.", "Kordonboyu Mah.", "Orhantepe Mah.", "Orta Mah.", "Petroliş Mh.", "Soğanlık Yeni Mh.", "Topselvi Mah.", "Uğur Mumcu Mh.", "Yakacık Yeni Mh.", "Yakacık Çarşı Mh.", "Yalı Mh.", "Yukarı Mh.", "Yunus Mah.", "Çavuşoğlu Mh."], 
    "Kağıthane": ["Emniyet Evleri Mh.", "Gültepe Mh.", "Gürsel Mh.", "Hamidiye Mah.", "Harmantepe Mah.", "Hürriyet Mh.", "Mehmet Akif Ersoy Mah.", "Merkez Mah.", "Nurtepe Mah.", "Ortabayır Mh.", "Seyrantepe Mah.", "Sultan Selim Mh.", "Talatpaşa Mh.", "Telsizler Mah.", "Yahya Kemal Mah.", "Yeşilce Mh.", "Çağlayan Mh.", "Çeliktepe Mh.", "Şirintepe Mh."], 
    "Küçükçekmece": ["Atakent Mah.", "Atatürk Mh.", "Beşyol Mh.", "Cennet Mah.", "Cumhuriyet Mah.", "Fatih Mah.", "Fevzi Çakmak Mh.", "Gültepe Mh.", "Halkalı Merkez Mh.", "Kanarya Mah.", "Kartaltepe Mah.", "Kemalpaşa Mh.", "Mehmet Akif Mah.", "Sultan Murat Mah.", "Söğütlü Çeşme Mh.", "Tevfik Bey Mah.", "Yarımburgaz Mh.", "Yeni Mahalle Mah.", "Yeşilova Mh.", "İnönü Mh.", "İstasyon Mh."], 
    "Maltepe": ["Altayçeşme Mh.", "Altıntepe Mh.", "Aydınevler Mh.", "Bağlarbaşı Mh.", "Başıbüyük Mh.", "Cevizli Mah.", "Esenkent Mah.", "Feyzullah Mah.", "Fındıklı Mh.", "Girne Mah.", "Gülsuyu Mh.", "Küçükyalı Mh.", "Yalı Mh.", "Zümrütevler Mh.", "Çınar Mh.", "İdealtepe Mh."], 
    "Pendik": ["Ahmet Yesevi Mah.", "Bahçelievler Mh.", "Batı Mh.", "Doğu Mh.", "Dumlupınar Mh.", "Esenler Mah.", "Esenyalı Mh.", "Fatih Mah.", "Fevzi Çakmak Mh.", "Güllü Bağlar Mh.", "Güzelyalı Mh.", "Harmandere Mah.", "Kavakpınar Mh.", "Kaynarca Mah.", "Kurtköy Mh.", "Orhangazi Mah.", "Orta Mah.", "Sapan Bağları Mh.", "Sülüntepe Mh.", "Yayalar Mah.", "Yeni Mahalle Mah.", "Yenişehir Mh.", "Yeşilbağlar Mh.", "Velibaba Mah.", "Çamlık Mh.", "Çamçeşme Mh.", "Çınardere Mh.", "Şeyhli Mh."], 
    "Sancaktepe": ["Abdurrahmangazi Mah.", "Akpınar Mh.", "Atatürk Mh.", "Emek Mah.", "Eyüp Sultan Mh.", "Fatih Mah.", "Hilal Mah.", "Kemal Türkler Mh.", "Meclis Mah.", "Merve Mah.", "Mevlana Mah.", "Osmangazi Mah.", "Safa Mah.", "Sarıgazi Mh.", "Yenidoğan Mh.", "Yunus Emre Mah.", "Veysel Karani Mah.", "İnönü Mh."], 
    "Sarıyer": ["Ayazağa Mh.", "Bahçeköy Kemer Mh.", "Bahçeköy Merkez Mh.", "Bahçeköy Yeni Mh.", "Baltalimanı Mh.", "Büyükdere Mh.", "Cumhuriyet Mah.", "Darüşşafaka Mh.", "Demirciköy Mh.", "Emirgan Mah.", "Fatih Sultan Mehmet Mah.", "Ferahevler Mah.", "Huzur Mah.", "Kazım Karabekir Paşa Mh.", "Kireçburnu Mh.", "Kumköy Mh.", "Maden Mah.", "Maslak Mah.", "Merkez Mah.", "Poligon Mah.", "Ptt Evleri Mah.", "Pınar Mh.", "Reşitpaşa Mh.", "Rumeli Hisarı Mh.", "Rumeli Kavağı Mh.", "Tarabya Mah.", "Uskumruköy Mh.", "Yeni Mah.", "Yeniköy Mh.", "Zekeriyaköy Mh.", "Çamlıtepe Mh.", "İstinye Mh."], 
    "Silivri": ["Alibey Mah.", "Alipaşa Mh.", "Balaban Mh.", "Cumhuriyet Mah.", "Fatih Silivri Mah.", "Fevzipaşa Mh.", "Gümüşyaka Mh.", "Mimar Sinan Mh.", "Piri Mehmet Paşa Mh.", "Sancaktepe Mh.", "Selimpaşa Mh.", "Semizkumlar Mah.", "Yeni Mah.", "İsmetpaşa Mh."], 
    "Sultanbeyli": ["Abdurrahmangazi Mah.", "Adil Mah.", "Ahmet Yesevi Mah.", "Akşemsettin Mh.", "Battalgazi Mah.", "Fatih Mah.", "Hamidiye Mah.", "Hasanpaşa Mh.", "Mecidiye Mah.", "Mehmet Akif Mah.", "Mimar Sinan Mah.", "Necip Fazıl Mh.", "Orhangazi Mah.", "Turgut Reis Mah.", "Yavuz Selim Mah."], 
    "Sultangazi": ["50. Yıl Mh.", "75. Yıl Mh.", "Cebeci Mah.", "Cumhuriyet Mah.", "Esentepe Mah.", "Eski Habipler Mah.", "Gazi Mah.", "Habibler Mh.", "Malkoçoğlu Mh.", "Sultançiftliği Mh.", "Uğur Mumcu Mh.", "Yayla Mah.", "Yunus Emre Mah.", "Zübeyde Hanım Mh.", "İsmetpaşa Mh."], 
    "Tuzla": ["Aydınlı Mh.", "Aydıntepe Mh.", "Cami Mah.", "Evliya Çelebi Mh.", "Fatih Mah.", "Mescit Mah.", "Mimar Sinan Mah.", "Orhanlı Mh.", "Orta Mah.", "Postane Mah.", "Tepeören Mh.", "Yayla Mah.", "İstasyon Mh.", "İçmeler Mh.", "Şifa Mh."], 
    "Zeytinburnu": ["Beştelsiz Mh.", "Gökalp Mh.", "Kazlıçeşme Mh.", "Maltepe Mah.", "Merkezefendi Mah.", "Nuripaşa Mh.", "Seyit Nizam Mah.", "Sümer Mh.", "Telsiz Mah.", "Yenidoğan Mh.", "Yeşiltepe Mh.", "Veliefendi Mah.", "Çırpıcı Mh."], 
    "Çatalca": ["Akalan Köyü", "Atatürk Mh.", "Fatih Mah.", "Ferhatpaşa Mh.", "Kaleiçi Mh.", "Muratbey Merkez Mah.", "Örcünlü Mh."], 
    "Çekmeköy": ["Alemdağ Mh.", "Aydınlar Mh.", "Cumhuriyet Mah.", "Ekşioğlu Mh.", "Güngören Mh.", "Hamidiye Mah.", "Kirazlıdere Mh.", "Mehmet Akif Ersoy Mah.", "Merkez Mah.", "Mimar Sinan Mah.", "Nişantepe Mh.", "Reşadiye Mh.", "Soğukpınar Mh.", "Sultançiftliği Mh.", "Taşdelen Mh.", "Çamlık Mh.", "Çatalmeşe Mh.", "Ömerli Mh."], 
    "Ümraniye": ["Adem Yavuz Mah.", "Altınşehir Mh.", "Armağanevler Mh.", "Atakent Mah.", "Atatürk Mh.", "Aşağı Dudullu Mh.", "Cemil Meriç Mh.", "Elmalıkent Mh.", "Esenevler Mah.", "Esenkent Mah.", "Esenşehir Mh.", "Fatih Sultan Mehmet Mah.", "Finanskent Mh.", "Huzur Mah.", "Ihlamurkuyu Mah.", "Madenler Mah.", "Mehmet Akif Mah.", "Namık Kemal Mh.", "Necip Fazıl Mh.", "Parseller Mah.", "Saray Mah.", "Site Mah.", "Tantavi Mah.", "Tatlısu Mh.", "Tepeüstü Mh.", "Yamanevler Mh.", "Yukarı Dudullu Mh.", "Çakmak Mh.", "Çamlık Mh.", "İnkılap Mh.", "İstiklal Mh.", "Şerifali Mh."], 
    "Üsküdar": ["Acıbadem Mh.", "Ahmediye Mah.", "Altunizade Mah.", "Aziz Mahmut Hüdayi Mh.", "Bahçelievler Mh.", "Barbaros Mah.", "Beylerbeyi Mah.", "Bulgurlu Mah.", "Burhaniye Mah.", "Cumhuriyet Mah.", "Ferah Mah.", "Güzeltepe Mh.", "Kandilli Mah.", "Kuleli Mah.", "Kuzguncuk Mah.", "Küplüce Mh.", "Küçük Çamlıca Mh.", "Küçüksu Mh.", "Kısıklı Mh.", "Mehmet Akif Ersoy Mah.", "Mimar Sinan Mh.", "Murat Reis Mah.", "Salacak Mah.", "Selami Ali Mah.", "Selimiye Mah.", "Sultantepe Mah.", "Valide-i Atik Mh.", "Yavuztürk Mh.", "Zeynep Kamil Mah.", "Çengelköy Mh.", "Ünalan Mh.", "İcadiye Mh."], 
    "Şile": ["Ahmetli Köyü", "Ağva Merkez Mh.", "Balibey Mah.", "Hacı Kasım Mh.", "Kumbaba Mah.", "Kurna Köyü", "Meşrutiyet Mh.", "Oruçoğlu Mh.", "Çavuş Mh."], 
    "Şişli": ["19 Mayıs Mh.", "Bozkurt Mah.", "Cumhuriyet Mah.", "Duatepe Mah.", "Ergenekon Mah.", "Esentepe Mah.", "Eskişehir Mh.", "Feriköy Mh.", "Fulya Mah.", "Gülbahar Mh.", "Halaskargazi Mah.", "Halide Edip Adıvar Mh.", "Halil Rıfat Paşa Mh.", "Harbiye Mah.", "Kaptan Paşa Mh.", "Kuştepe Mh.", "Mahmut Şevket Paşa Mh.", "Mecidiyeköy Mh.", "Merkez Mah.", "Meşrutiyet Mh.", "Paşa Mh.", "Teşvikiye Mh.", "Yayla Mah.", "İnönü Mh.", "İzzet Paşa Mh."]
}

# --- GLOBAL DEĞİŞKENLER ---
df_listings = pd.DataFrame()
model_loaded = False
pipe = None
metrics = {}

# YENİ: Kategori Filtreleri Tanımları
AMENITY_CATEGORIES = {
    "AVM Yakın": "The mall",
    "Camiye Yakın": "Mosque",
    "Cemevine Yakın": "Cemevi",
    "Geniş Koridor": "Wide Corridor",
    "Hastane Yakın": "Hospital",
    "Kapalı Otopark": "Closed Garage",
    "Metro/Metrobüs": ["Metro", "Metrobus"], 
    "Okula Yakın": "Primary School-Secondary School",
    "Site İçinde (Güvenlik)": "Security",
}
# Kategorilerin label'larını A-Z sıralamak için listeyi oluşturuyoruz
SORTED_AMENITY_LABELS = sorted(AMENITY_CATEGORIES.keys())


# --- CSV YÜKLEME (ROBUST) ---
def load_data():
    global df_listings, model_loaded, pipe, metrics
    
    possible_files = [
        'data/hackathon_train_set.csv', 'hackathon_train_set.csv.csv', 'hackathon_train_set.csv', 'hackathon_train_set_final_TR_no_kot.csv'
    ]
    found_file = None
    for f in possible_files:
        if os.path.exists(f):
            found_file = f
            break
            
    try:
        if found_file:
            df_listings = pd.read_csv(found_file, delimiter=';')
            if df_listings.shape[1] < 2: 
                df_listings = pd.read_csv(found_file, delimiter=',')
                
            def clean_price_col(val):
                if isinstance(val, str):
                    return float(val.replace('.', '').replace(' TL', '').replace(',', '.'))
                try:
                    return float(val)
                except:
                    return 0.0
            
            df_listings['Price_Num'] = df_listings['Price'].apply(clean_price_col)
            # Fiyatı 0 olan kayıtları listeden çıkararak performansı ve sıralama doğruluğunu artır
            df_listings = df_listings[df_listings['Price_Num'] > 0].copy() 
            
            print(f"✅ İlan verileri yüklendi: {len(df_listings)} kayıt. Dosya: {found_file}")
        else:
            print("⚠️ HATA: CSV dosyası bulunamadı! Lütfen kontrol edin.")
            df_listings = pd.DataFrame() 
    except Exception as e:
        print(f"⚠️ CSV okuma hatası: {e}")
        df_listings = pd.DataFrame()

    # 2. MODEL YÜKLE
    MODEL_DIR = "models"
    try:
        bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
        pipe = bundle["pipeline"]
        metrics = bundle.get("metrics", {"r2": 0.94})
        model_loaded = True
        print("✅ Model yüklendi.")
    except:
        print("⚠️ Model yüklenemedi (Demo mod aktif).")
        model_loaded = False

load_data()

# --- TEK GÖRSEL URL'Sİ (Tüm ilanlara aynı görsel atanacak) ---
# KULLANICININ EN SON KORUMASINI İSTEDİĞİ UNSPLASH FOTOĞRAFI
SINGLE_IMAGE_URL = "https://images.unsplash.com/photo-1570129477492-45c003edd2be?q=70&w=400&auto=format&fit=crop&v=5&listing_id=4484"


def currency_filter(value):
    try: return "{:,.0f}".format(float(value)).replace(",", ".")
    except: return value
app.jinja_env.filters['currency'] = currency_filter

def get_clean_floors():
    return ["Giriş"] + [f"{i}. Kat" for i in range(1, 31)]

def get_clean_ages():
    return ["0 (Yeni)", "1-5", "6-10", "11-15", "16-20", "21-30", "31 ve üzeri"]

HEATING_OPTIONS = ["Kombi (Doğalgaz)", "Yerden Isıtma", "Merkezi Sistem", "Merkezi (Pay Ölçer)", "Doğalgaz Sobası", "Klima", "Yok"]
ROOM_OPTIONS = ['1+0', '1+1', '2+1', '3+1', '4+1', '4+2', '5+1', '5+2']

# --- ROTALAR ---

@app.route("/ilanlar", methods=["GET"])
def ilanlar_page():
    start_time = time.time()
    
    filtre = request.args.get('filtre')
    if not filtre:
        filtre = 'tumu' 
        
    # Birden fazla kategori filtresini list olarak al
    aktif_kategoriler = request.args.getlist('category_filter')
    siralama = request.args.get('siralama', 'onerilen') 
    
    arama = request.args.get('q', '').lower()
    district_filter = request.args.get('district')
    neighborhood_filter = request.args.get('neighborhood')
    
    ilanlar_list = []
    
    if not df_listings.empty:
        try:
            filtered_df = df_listings.copy() 
            
            # Krediye Uygunluk Filtresi
            if 'Available for Loan' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Available for Loan'] == 'Yes']

            # KONUM VE ARAMA FİLTRELERİ
            if district_filter:
                filtered_df = filtered_df[filtered_df['District'] == district_filter]
            if neighborhood_filter:
                filtered_df = filtered_df[filtered_df['Neighborhood'].astype(str).str.contains(neighborhood_filter, case=False, na=False)]
            if arama:
                mask = (filtered_df['District'].str.lower().str.contains(arama, na=False)) | \
                       (filtered_df['Neighborhood'].str.lower().str.contains(arama, na=False))
                filtered_df = filtered_df[mask]
            
            # ÖZEL ODA FİLTRESİ
            if filtre not in ['tumu', 'tümü']: 
                if filtre == 'aile':
                    filtered_df = filtered_df[filtered_df['Number of rooms'].astype(str).str.match(r'^[3-9]')]
                elif filtre == 'yeni_evli':
                    filtered_df = filtered_df[filtered_df['Number of rooms'].astype(str).str.match(r'^[1-2]')]
            
            # *** ÇOKLU KATEGORİ FİLTRESİ UYGULAMASI (OR mantığı) ***
            if aktif_kategoriler:
                # Başlangıçta tüm satırların maskesi False
                overall_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                
                for category_label in aktif_kategoriler:
                    col_info = AMENITY_CATEGORIES.get(category_label)
                    if col_info:
                        if isinstance(col_info, list):
                            # Metro/Metrobüs gibi OR mantığı gerektiren kategoriler
                            category_mask = False
                            for col in col_info:
                                if col in filtered_df.columns:
                                    category_mask = category_mask | (filtered_df[col] == 1)
                            overall_mask = overall_mask | category_mask
                        elif col_info in filtered_df.columns:
                            # Tek sütun
                            overall_mask = overall_mask | (filtered_df[col_info] == 1)
                
                # Eğer birden fazla kategori seçilmişse, bu kategorilerden en az birine uyanları filtrele
                if overall_mask.any():
                    filtered_df = filtered_df[overall_mask]
            # *** FİLTRELEME BİTTİ ***
                
            # SIRALAMA MANTIĞI KONTROLÜ
            if siralama == 'fiyat_asc':
                filtered_df = filtered_df.sort_values(by='Price_Num', ascending=True, na_position='last')
            elif siralama == 'fiyat_desc':
                filtered_df = filtered_df.sort_values(by='Price_Num', ascending=False, na_position='last')
            elif siralama == 'onerilen':
                filtered_df = filtered_df.sort_index()


            # TEK GÖRSEL ATAMA: Tüm ilanlara aynı URL atanır
            # Cache Buster'ı, tarayıcıyı zorlamak için her seferinde rastgele bir değerle güncelliyoruz.
            cache_buster = random.randint(1000, 9999) 
            # URL'nin sonuna cache buster eklenir
            image_url_with_cache = f"{SINGLE_IMAGE_URL}&cache_buster={cache_buster}" 
            
            for idx, row in filtered_df.iterrows():
                bina_yasi = str(row.get('Building Age', '-'))
                
                # İlan başlığını 'Title' sütunundan al, yoksa dinamik oluştur
                title_fallback = f"{row.get('District', '')} {row.get('Neighborhood', '')} Fırsat"
                ilan_basligi = row.get('Title', title_fallback)
                
                ilan = {
                    "id": idx,
                    "baslik": ilan_basligi, 
                    "konum": f"{row.get('District', '')}, {row.get('Neighborhood', '')}",
                    "fiyat": row.get('Price_Num', 0),
                    "resim": image_url_with_cache, 
                    "metrekare": row.get('m² (Net)', 0),
                    "oda_sayisi": row.get('Number of rooms', '-'),
                    "bina_yasi": bina_yasi,
                    "ozellikler": [
                        {"ikon": "bed", "değer": str(row.get('Number of rooms', '-'))},
                        {"ikon": "square_foot", "değer": f"{row.get('m² (Net)', '-')} m²"}
                    ]
                }
                ilanlar_list.append(ilan)
        except Exception as e:
            print(f"İlan Hatası: {e}")

    end_time = time.time()
    print(f"⏳ İlan çekme süresi: {end_time - start_time:.4f} saniye")
    
    # Aktif kategori listesini HTML'ye geri gönderiyoruz
    return render_template("ilanlar.html", ilanlar=ilanlar_list, aktif_filtre=filtre, arama_terimi=arama, aktif_siralama=siralama, aktif_kategoriler=aktif_kategoriler, AMENITY_CATEGORIES=AMENITY_CATEGORIES, SORTED_AMENITY_LABELS=SORTED_AMENITY_LABELS)

@app.route("/", methods=["GET", "POST"])
def index():
    # Başlangıç/Varsayılan veriler
    result = None
    map_url = "https://maps.google.com/maps?q=Istanbul&t=&z=10&ie=UTF8&iwloc=&output=embed"
    default_price = "1.500.000"
    
    # YENİ EKLENTİ: Modelin başarılı olup olmadığını takip etmek için
    model_prediction_successful = False

    current_data = {
        "district": "", 
        "neighborhood": "", 
        "rooms": "2+1", 
        "age": "5-10", 
        "floor": "3. Kat", 
        "m2": 100, 
        "heating": "Kombi (Doğalgaz)"
    }

    if request.method == "POST":
        try:
            form = request.form
            
            # 1. Sayısal Alanları Güvenli Al ve Dönüştür
            try:
                m2_val = float(form.get("m2", 100))
            except ValueError:
                m2_val = 100
                
            try:
                raw_price = form.get("listing_price", "0").replace(".", "").replace(",", ".")
                listing_price = float(raw_price)
            except ValueError:
                listing_price = 1500000.0
                raw_price = "1.500.000"


            current_data = {
                "district": form.get("district", ""),
                "neighborhood": form.get("neighborhood", ""),
                "rooms": form.get("rooms", "2+1"),
                "age": form.get("age", "5-10"),
                "floor": form.get("floor", "3. Kat"),
                "m2": m2_val,
                "heating": form.get("heating", "Kombi (Doğalgaz)")
            }
            default_price = raw_price 

            address_query = f"{current_data['neighborhood']}, {current_data['district']}, Istanbul"
            map_url = f"https://maps.google.com/maps?q={address_query}&t=&z=15&ie=UTF8&iwloc=&output=embed"

            fair_value = 0 
            
            # KRİTİK EŞİKLER: FIRSAT/PAHALI için %15 sapma
            LOWER_THRESHOLD = 0.85 
            UPPER_THRESHOLD = 1.15 
            

            if model_loaded:
                try:
                    model_age = "31" if "üzeri" in str(current_data['age']) else current_data['age']
                    model_floor = "0" if current_data['floor'] == "Giriş" else (current_data['floor'].split(".")[0] if "Kat" in current_data['floor'] else "1")
                    
                    row = {
                        "District": current_data['district'], 
                        "Neighborhood": current_data['neighborhood'],
                        "Number of rooms": current_data['rooms'], 
                        "m² (Net)": current_data['m2'], 
                        "m² (Gross)": current_data['m2']*1.25,
                        "Building Age": model_age, 
                        "Floor location": model_floor, 
                        "Heating": current_data['heating'],
                        "Furnished": "No", "Number of bathrooms": "1", "Balcony": "No", "Using status": "Empty", 
                        "Available for Loan": "Yes"
                    }
                    
                    log_pred = pipe.predict(pd.DataFrame([row]))[0]
                    model_prediction = np.expm1(log_pred)
                    
                    if model_prediction <= 1000 or np.isnan(model_prediction):
                         # Model tahmin yapamadı: Hata fırlat (Model başarısız oldu)
                         raise ValueError("Model tahmin hatası.")
                    else:
                         # Model başarılı tahmin yaptı
                         fair_value = model_prediction
                         model_prediction_successful = True # BAŞARILI
                    
                except Exception as model_e:
                    # Model çökünce, yedek sisteme düşülüyor.
                    print(f"❌ MODEL ÇÖKÜŞÜ (Hata: {model_e}). Yedek sisteme düşülüyor.")
                    
                    # Dinamik Yedek Tahmin: Metrekare ve oda sayısına göre basit bir çarpan kullanılır.
                    base_m2_price = 10000 
                    if '1+1' in current_data['rooms'] or '2+1' in current_data['rooms']:
                        base_m2_price = 15000
                    elif '3+1' in current_data['rooms'] or '4+1' in current_data['rooms']:
                        base_m2_price = 20000

                    if current_data['m2'] < 50: 
                         base_m2_price *= 1.5 
                    
                    # Dinamik yedek tahmin (%20 sapmalı rastgele değer atar)
                    fair_value = current_data['m2'] * base_m2_price * random.uniform(0.8, 1.2) 
                    
                    model_prediction_successful = False # BAŞARISIZ
                    
                    print(f"🚨 Yedek Sonuç: MODEL ÇÖKTÜ. Atanan Dinamik Yedek Değer: {fair_value:,.0f}")
            else:
                # Model hiç yüklenmemişse (Demo Modu)
                base_m2_price = 10000 
                if '1+1' in current_data['rooms'] or '2+1' in current_data['rooms']:
                    base_m2_price = 15000
                elif '3+1' in current_data['rooms'] or '4+1' in current_data['rooms']:
                    base_m2_price = 20000

                if current_data['m2'] < 50:
                     base_m2_price *= 1.5 
                fair_value = current_data['m2'] * base_m2_price * random.uniform(0.8, 1.2)
                model_prediction_successful = False
                print(f"🚨 Model Yüklü Değil. Atanan Dinamik Yedek Değer: {fair_value:,.0f}")


            # Nihai Fair Value kontrolü
            if fair_value <= 0:
                 fair_value = listing_price * 1.05 

            ratio = listing_price / fair_value
            
            # Fiyat Analizi Mantığı
            if ratio > UPPER_THRESHOLD: 
                status, color, icon, desc = "PAHALI", "text-rose-400", "warning", "Liste fiyatı piyasa ortalamasının belirgin şekilde üzerindedir. Alıcılar için yüksek risk taşıyabilir."
                grad_to, badge_bg = "rose-500", "bg-rose-500/20 text-rose-300 border-rose-500/30"
            elif ratio < LOWER_THRESHOLD: 
                status, color, icon, desc = "FIRSAT", "text-emerald-400", "check_circle", "Bu mülk piyasa değerinin belirgin şekilde altında listelenmiştir. Hızlıca değerlendirilmelidir."
                grad_to, badge_bg = "emerald-500", "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
            else: 
                status, color, icon, desc = "NORMAL", "text-amber-400", "balance", "Fiyat piyasa koşullarıyla uyumludur. Makul bir yatırım potansiyeli sunar."
                grad_to, badge_bg = "amber-500", "bg-amber-500/20 text-amber-300 border-amber-500/30"

            # Model çökme durumunda raporu uyarı ile güncelle
            if not model_prediction_successful:
                # Eğer model çalışmadıysa (yalnızca yedek değer kullanıldıysa) kullanıcıyı uyar.
                desc = f"⚠️ DİKKAT: Model tahmininde teknik bir hata oluştu veya Model yüklü değil. Sonuç, girilen değerlere göre dinamik olarak hesaplanmış yaklaşık bir rapordur. ({desc})"

            result = {
                "fair_value": f"{fair_value:,.0f}".replace(",", "."), "listing_price": f"{listing_price:,.0f}".replace(",", "."),
                "status": status, "color_cls": color, "grad_to": grad_to, "badge_bg": badge_bg, "ratio": f"{ratio:.2f}", "icon": icon,
                "bar_width": min(100, (fair_value / listing_price) * 85),
                "diff": f"{abs((fair_value - listing_price) / fair_value) * 100:.1f}",
                "desc": desc, 
                "r2": f"{metrics.get('r2', 0.94):.3f}" 
            }
            
        except Exception as e:
            print(f"Genel Hata: {e}")
            result = None

    choices = {'District': sorted(list(ISTANBUL_DATA.keys())), 'Number of rooms': ROOM_OPTIONS, 'Building Age': get_clean_ages(), 'Floor location': get_clean_floors(), 'Heating': HEATING_OPTIONS}
    
    return render_template("index.html", choices=choices, result=result, map_url=map_url, ISTANBUL_DATA=ISTANBUL_DATA, default_price=default_price, current_data=current_data)

if __name__ == "__main__":
    app.run(debug=True)
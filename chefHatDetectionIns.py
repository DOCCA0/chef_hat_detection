import pymysql
import time


host='localhost'
port=3306
user='root'
passwd='2947'
db1='chefhat_detection'
charset='utf8'
db = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db1, charset=charset)
cursor = db.cursor()

#Kitchen
class Kitchen:

    def __init__(self):
        global db, cursor

    def insert(self,name,location):
        self.name = name
        self.location = location

        sql = "INSERT INTO `kitchen`(`name`, `location`) VALUES ('%s','%s')"%(self.name,self.location)
        cursor.execute(sql)
        db.commit()

    def setName(self,newname,oldname):

        sql = " UPDATE `kitchen` SET `name`='%s' WHERE name='%s'" % (newname,oldname)
        cursor.execute(sql)
        db.commit()

    def setLocation(self,newloc,oldloc):

        sql = " UPDATE `kitchen` SET `location`='%s' WHERE location='%s'" % (newloc,oldloc)
        cursor.execute(sql)
        db.commit()

#HistoryVideo
class HistoryVideo:

    def __init__(self):
        global db, cursor

    def insert(self,id,videoURL):
        self.id = id
        self.videoURL = videoURL
        sql = "INSERT INTO `historyvideo`(`id`, `videoURL`, `date`) VALUES (%d,'%s','%s')"%(self.id,self.videoURL,time.strftime("%Y-%m-%d"))
        cursor.execute(sql)
        db.commit()

#Alarm
class Alarm:
    def __init__(self):
        global db, cursor

    def insert(self,id,type,picPath):
        self.id = id
        self.picPath = picPath
        self.type= type

        sql = "INSERT INTO `alarm`(`id`, `type`, `picPath`,`datetime`) VALUES (%d,'%s','%s','%s')"%(self.id,self.type,self.picPath,time.strftime("%Y-%m-%d %H:%M:%S"))
        cursor.execute(sql)
        db.commit()





















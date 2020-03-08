# -*- coding:utf-8 -*-
import cqplus
import re
import datetime as dt
import pymysql
from urllib import request
from os import mkdir,listdir
from shutil import make_archive
import numpy as np
from PIL import Image
from io import StringIO,BytesIO
import matplotlib.pyplot as plt

CQ_dir = R'C:\Users\xiong35\Desktop\酷Q Air\data\image\ '[:-1]
save_dir = R'C:\Users\xiong35\Desktop\images\ '[:-1]
ftp_dir = R'C:\xiong35_ftp\ '[:-1]

conn = pymysql.connect('localhost', 'root', 'xyl0321', 'cq')
cursor = conn.cursor()

NOW = dt.datetime.now()
TODAY = dt.date.today()
FIVEMIN = dt.timedelta(minutes=5)
ONEWEEK = dt.timedelta(weeks=1)
ONEDAY = dt.timedelta(days=1)
ONEHOUR = dt.timedelta(hours=1)

# TODO other feature's description
README = """
##使用指南_(:з」∠)_##
[+] 指示接下来进行备忘之类的操作
[#] 指示接下来进行管理组别之类的操作
详情可输入
+REAMDME 和 #README
来查看"""

plusReadMe = """
"""


hashReanME = """
# 设置3224609972为[管理员,用户]
# 删除xxx
# 注册xxx
# xxx是什么身份
# 查看所有
"""


class ReadMe:
    def respondRM(self, request):
        if request == "README":
            return README
        if request == "+README":
            return plusReadMe
        if request == "#README":
            return hashReanME


class ReminderInfo:
    def __init__(self, from_id=None, from_type='private', ddl=None,
                 content='干活', begin_time=None, interval=None):
        self.from_id = from_id
        self.from_type = from_type
        self.ddl = ddl
        self.content = content
        self.begin_time = begin_time
        self.interval = interval


class HandleReminder:
    def __init__(self, msg=None):
        self.msg = msg
    # check whether to remind
    # return fetch tuple,'ddl'|'begin' / None,None

    def check_remind(self):
        sql_msg = """SELECT
                     `id`, `from_id`, `from_type`, `ddl`,
                     `content`, `begin_time`, `interval`
                     FROM reminder;"""
        cursor.execute(sql_msg)
        reminders = cursor.fetchall()
        now = dt.datetime.now()
        for item in reminders:
            ddl = item[3]
            if now > ddl:
                del_ddl = """DELETE FROM reminder
                             WHERE id={};""".format(item[0])
                cursor.execute(del_ddl)
                conn.commit()
                return item, True
            begin_time = item[5]
            if now > begin_time:
                delta_time = dt.timedelta(minutes=item[6])
                next_time = (
                    begin_time+delta_time).strftime('%Y-%m-%d %H:%M:%S')
                set_next_time = """UPDATE reminder
                                   SET begin_time ="{}"
                                   WHERE id = {};""".format(next_time, item[0])
                cursor.execute(set_next_time)
                conn.commit()
                return item, False
        return None, None

    def set_reminder(self,  RI):
        msg = """
        INSERT INTO `reminder`
        (`from_id`,`from_type`,`ddl`,`content`,`begin_time`,`interval`)
        VALUES
        ("{}","{}","{}","{}","{}",{});
        """.format(RI.from_id, RI.from_type,
                   RI.ddl, RI.content,
                   RI.begin_time, RI.interval)
        cursor.execute(msg)
        conn.commit()
        return '成功设置在{}提醒你{} ฅ( ̳• ◡ • ̳)ฅ'.format(RI.ddl, RI.content)
        # self.logging.debug(return_msg)

    # detect time key words in self.msg
    # return True/False

    def is_reminder_event(self):
        key_words = ['天', '星期', '周', '秒', '分钟', '时',
                     '月', '点', '早', '晚', '.', '号', ':', '：']
        for word in key_words:
            if re.search(word, self.msg):
                return True
        return False

    def parse(self, qq_id, from_type):
        msg = preprocess(self.msg)
        if msg == '+查看所有' or msg == '+查看全部':
            return self.show_reminder(qq_id)
        if '+提前' in msg:
            before = re.search(
                r"\+提前(\d+)个?小?(时|分钟?)[, ，]?每隔?(\d+)分钟?\D*1?\D*(\d*)", self.msg)
            if before:
                sql_msg = "SELECT `ddl` FROM reminder WHERE from_id='{}' AND id={};".format(
                    qq_id, before.group(4))
                cursor.execute(sql_msg)
                return_msg = cursor.fetchall()
                if not return_msg:
                    return '在你的备忘里没找到这个编号呢, 重新检查一下输入吧'
                ddl = return_msg[0][0]
                time_dic = {'时': 60, '分': 1}
                ahead = int(before.group(1))*time_dic[before.group(2)[0]]
                ahead = dt.timedelta(minutes=ahead)
                begin = (ddl - ahead).strftime('%Y-%m-%d %H:%M:%S')
                interval = before.group(3)
                return self.reset_bg_it(before.group(4), begin, interval)
            else:
                return '不太明白你的意思呢QwQ, 你可以输入"+README"查看详细信息'
        if '+删除' in msg:
            delete = re.search(r"\+删除(\d+)", msg)
            return self.del_reminder(int(delete.group(1)), qq_id)
        if not self.is_reminder_event():
            return "我看不懂你输入的指令呢, 能换个说法重新输一遍么QwQ"
        RI = ReminderInfo(qq_id, from_type)
        pattern = re.search(parsable+r'(.*)', msg)
        if pattern:
            to_parse = pattern.group(1)
            time = parse_time(to_parse)
            if time.__class__ == str:
                return time
            if pattern.group(2):
                content = pattern.group(2)
        else:
            return "我看不懂你输入的指令呢, 能换个说法重新输一遍么QwQ"
        RI.ddl = time.strftime('%Y-%m-%d %H:%M:%S')
        RI.content = content
        RI.begin_time = (time-FIVEMIN*2).strftime('%Y-%m-%d %H:%M:%S')
        RI.interval = 7
        return self.set_reminder(RI)

    def show_reminder(self, qq_id):
        sql_msg = """SELECT `id`,`ddl`,`content`
                     FROM `reminder`
                     WHERE from_id="{}";""".format(qq_id)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg
        else:
            return '你还没有添加备忘呢(°ー°〃)'

    # reset begin > interval
    def reset_bg_it(self,  idnum, begin, interval):
        sql_msg = """UPDATE reminder
                     SET `begin_time` ="{}",`interval`={}
                     WHERE `id` = {} ;""".format(begin, interval, idnum)
        cursor.execute(sql_msg)
        conn.commit()
        return '成功更改{}事件为从{}开始,每{}分钟提醒一次(´･ω･`)'.format(idnum, begin, interval)

    def del_reminder(self, idnum, qq_id):
        sql_msg = "SELECT * FROM reminder WHERE from_id='{}' AND id={}".format(
            qq_id, idnum)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if not return_msg:
            return '在你的备忘里没找到这个编号呢, 重新检查一下输入吧'
        sql_msg = "DELETE FROM reminder WHERE `id`={} AND `from_id`={};".format(
            idnum, qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功删除编号为{}的备忘_(:з」∠)_".format(idnum)


class MemberInfo:
    def __init__(self, name, qq_id, classNum,
                 group_name='user', state='not_yet', last_change=None, path=''):
        self.name = name
        self.qq_id = qq_id
        self.classNum = classNum
        self.group_name = group_name
        self.state = state
        if last_change:
            self.last_change = last_change
        else:
            now = NOW.strftime('%Y-%m-%d %H:%M:%S')
            self.last_change = now
        self.path = path


# done
#! all input qq id need to be valid
class GroupSys:

    def __init__(self, msg, operator_qq):
        self.msg = msg
        self.operator_group = self.check_groupNclass(operator_qq)[0]

    ##
    # directly call the corespond func
    # output: msg / check_all's tuple
    def parse(self):
        if self.operator_group == '数':
            return '抱歉, 你好像还没注册呢😖, 请先找管理员注册一下吧!'
        if self.operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'
        pattern = re.search(r"#设置(\d*)为(管理员|用户)", self.msg)
        if pattern:
            qq_id = pattern.group(1)
            if pattern.group(2) == '管理员':
                group = 'admin'
            else:
                group = 'user'
            return self.set_group(qq_id, group)
        pattern = re.search(r"#(删除)(\d*)", self.msg)
        if pattern:
            qq_id = pattern.group(2)
            return self.del_group(qq_id)
        pattern = re.search(r"#注册([^\d,， ]+)[,， ]*(\d+)[,， ]*(\w+)", self.msg)
        if pattern:
            MI = MemberInfo(pattern.group(
                1), pattern.group(2), pattern.group(3))
            return self.sign_group(MI)
        pattern = re.search(r"#(\d*)是什么身份", self.msg)
        if pattern:
            qq_id = pattern.group(1)
            person_msg = self.check_groupNclass(qq_id)
            if person_msg.__class__ == str:
                return person_msg
            return '{}是{}的{}ヽ(•̀ω•́ )ゝ'.format(qq_id, person_msg[1], person_msg[0])
        pattern = re.search(r"#查看([0-9a-zA-Z]+)", self.msg)
        if pattern:
            tar_class = pattern.group(1)
            return self.show_all_user(tar_class)
        return "我不太清楚你的意思呢，试试输入#README查看操作指南?"

    # check a qq id's group
    # input: qq id to be checked
    # output: group str / err msg

    def check_groupNclass(self, qq_id):
        sql_msg = """SELECT `group_name`,`class`
                     FROM `members`
                     WHERE qq_id="{}";""".format(qq_id)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg[0]
        else:
            return '数据库里没有这个qq呢'

    # set a user to a certain group
    # input:
    # qq id to be changed
    # group to be changed to
    # operator's qq
    # output:
    # error msg/succes msg
    def set_group(self, qq_id, group):
        return_group = self.check_groupNclass(qq_id)[0]
        if return_group == '数据库里没有这个qq呢':
            return return_group
        if return_group == 'root':
            return '抱歉, 您的权限不够/摊手'
        if return_group == 'admin' and self.operator_group != 'root':
            return '抱歉, 您的权限不够/摊手'

        sql_msg = """UPDATE `members`
                      SET group_name = "{}"
                      WHERE qq_id = "{}"; """.format(group, qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功更改{}的组别为{}!".format(qq_id, group)

    # sign in as user
    # input: qq id to be signed
    # out: succes/error msg
    def sign_group(self, MI):
        if self.check_groupNclass(MI.qq_id) == '数据库里没有这个qq呢':
            sql_msg = """INSERT INTO `members`
                       ( `name`, `qq_id`, `class`, `group_name`, 
                       `state`, `last_change`, `path`)
                       VALUES
                       ("{}", "{}","{}","{}","{}","{}","{}");
                       """.format(MI.name, MI.qq_id, MI.classNum, MI.group_name,
                                  MI.state, MI.last_change, MI.path)
            cursor.execute(sql_msg)
            conn.commit()
            return "成功添加{}为普通用户!".format(MI.name)
        return "数据库里已经有这个人了呢QwQ"

    # delete a qq from list
    # input:
    # qq id to be del
    # operator id
    # output:
    # error/succes msg
    def del_group(self, qq_id):
        target_group = self.check_groupNclass(qq_id)[0]
        if target_group == '数据库里没有这个qq呢':
            return target_group
        if self.operator_group == 'admin' and target_group != 'user':
            return '抱歉, 您的权限不够/摊手'
        sql_msg = """DELETE
                     FROM `members`
                     WHERE qq_id="{}";""".format(qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功删除{}! A.A".format(qq_id)

    def show_all_user(self, tar_class):
        sql_msg = """SELECT `qq_id`, `name`
                     FROM `members`
                     WHERE `group_name`="user" 
                     AND `class` = "{}";
                     """.format(tar_class)
        cursor.execute(sql_msg)
        user_list = cursor.fetchall()
        if user_list:
            return user_list
        return "数据库里没有这个班的数据呢(°ー°〃)"


class HomeworkSys:
    operator_group = 'user'

    def __init__(self, mainHandler):
        self.mainHandler = mainHandler

    def ready_for_handin(self, qq_id):
        sql_msg = """SELECT `flag` 
                     FROM homework
                     WHERE `class`=(SELECT `class`
                                    FROM `members`
                                    WHERE qq_id = "{}")
                     ORDER BY `id` DESC;""".format(qq_id)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if not return_msg:
            return "现在没有要交的作业呢💁‍♂️"
        if return_msg[0][0] == 2:
            return "现在没有要交的作业呢💁‍♂️"
        now = NOW.strftime('%Y-%m-%d %H:%M:%S')
        sql_msg = """UPDATE `members`
                     SET `state` = "time_wait",last_change = "{}"
                     WHERE qq_id="{}";""".format(now, qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "准备接受你的作业, 请发一张作业照片给我, 如果5分钟内没收到照片我就不会等了哦"

    def reset_not_yet(self):
        # TODO set teacher
        sql_msg = """SELECT `id`, `last_change`,`qq_id`
                     FROM `members` 
                     WHERE `state`="time_wait";"""
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            sql_msg = 'UPDATE `members` SET `state`="not_yet" WHERE `id`={};'
            for item in return_msg:
                if NOW - item[1] > FIVEMIN:
                    qq_id = int(item[2])
                    cursor.execute(sql_msg.format(item[0]))
                    conn.commit()
                    self.mainHandler.api.send_private_msg(qq_id, '哼, 不等你了😒')

    def parse(self, msg, operator_qq):
        groupSystem = GroupSys(None, operator_qq)

        if groupSystem.operator_group == '数':
            return '抱歉, 你好像还没注册呢😖, 请先找管理员注册一下吧!'

        self.operator_group = groupSystem.operator_group
        class_num = groupSystem.check_groupNclass(operator_qq)[1]

        if msg == '>交作业':
            return self.ready_for_handin(operator_qq)

        show_all = re.search(r">查看(所?有?)作业", msg)
        if show_all:
            flag = 1
            if show_all.group(1):
                flag = 2
            return_msg = self.get_all_homework(class_num, flag)
            # `id`,`content`, `ddl`, `flag`
            if return_msg.__class__ == str:
                return return_msg
            respons = '你的作业如下:'
            format_dic = {2: '已截止💩', 1: '马上截止😳', 0: '还不急😈'}
            for item in return_msg:
                respons += '\n-------'+'\n编号: {}'.format(item[0])
                respons += '\n内容: {}'.format(item[1])
                respons += '\nDDL: {}'.format(item[2])
                respons += '\n状态: {}'.format(format_dic[item[3]])
            return respons

        if msg[0] == '[':
            sql_msg = 'SELECT `state`,`group_name` FROM `members` WHERE `qq_id`="{}";'.format(
                operator_qq)
            cursor.execute(sql_msg)
            return_msg = cursor.fetchall()
            if return_msg[0][1] == 'user' and return_msg[0][0] == 'time_wait':
                sql_msg = 'SELECT `name` FROM `members` WHERE `qq_id`="{}";'.format(
                    operator_qq)
                cursor.execute(sql_msg)
                return_msg = cursor.fetchall()
                st_name = return_msg[0][0]
                sql_msg = """SELECT `id` FROM homework
                            WHERE `class`= "{}"
                            ORDER BY `id` DESC;""".format(class_num)
                cursor.execute(sql_msg)
                homework_id = str(cursor.fetchall()[0][0])
                homework_dir = save_dir+class_num+'_'+homework_id+R'\ '[:-1]
                CQfile = msg[15:-1]+'.cqimg'
                filename = homework_dir + st_name + '.jpg'
                return self.get_img(CQfile, filename, operator_qq)
            if return_msg[0][1] == 'admin' and return_msg[0][0] == 'time_wait':
                sql_msg = """SELECT `id` FROM homework
                            WHERE `class`= "{}"
                            ORDER BY `id` DESC;""".format(class_num)
                cursor.execute(sql_msg)
                homework_id = str(cursor.fetchall()[0][0])
                class_dir = class_num+'_'+homework_id
                homework_dir = save_dir+class_dir+R'\ '[:-1]
                CQfile = msg[15:-1]+'.cqimg'
                filename = homework_dir + 'answer' + '.jpg'
                self.get_img(CQfile, filename, operator_qq)
                fp = open(filename, 'rb')
                img = fp.read()
                sql = "INSERT INTO "+class_dir+" (`image`,`name`) VALUES  (%s"+',"admin"'+")"
                cursor.execute(sql, img)
                conn.commit()
                sql_msg = """UPDATE `members` SET `state`="test" 
                            WHERE `class`="{}"AND group_name="admin";""".format(class_num)
                cursor.execute(sql_msg)
                conn.commit()

            return None

        if self.operator_group == 'user':
            return '我不太懂你说的呢, 试试输入">README"查看操作手册吧😉'

        if msg == '>提醒同学':
            self.hurry(class_num, 1)
            return '已经提醒同学们抓紧时间写作业了√'

        handout_pat = re.search(
            r"(>布置作业|>>>布置考试)[:： \n]?\s*内容[:： \n]?(.+)\s*[Dd]{2}[lL][:： \n]?(.+)",
            msg, re.S)
        if handout_pat:
            content = handout_pat.group(2)
            ddl = handout_pat.group(3)
            ddl = preprocess(ddl)
            time = parse_time(ddl)
            if time.__class__ == str:
                return time
            time = time.strftime('%Y-%m-%d %H:%M:%S')
            return_msg = self.handout(class_num, content, time)
            #return (return_msg,homework_id)
            if handout_pat.group(1) == '>>>布置考试':
                self.start_test(class_num,return_msg[1])
            return return_msg[0]
        return '我不太懂你说的呢，试试输入">README"查看操作手册吧😉'

    def handout(self, class_num, content, ddl):
        sql_msg = """INSERT INTO `homework`
                     (`class`, `content`, `ddl`,`flag`)
                     VALUES
                     ("{}", "{}","{}",{});""".format(class_num, content, ddl, 0)
        cursor.execute(sql_msg)
        conn.commit()
        sql_msg = """SELECT `id` FROM `homework` ORDER BY `id` DESC"""
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        homework_id = return_msg[0][0]
        new_dir = save_dir+class_num+'_'+str(homework_id)+R'\ '[:-1]
        mkdir(new_dir)
        sql_msg = """UPDATE `members`
                     SET `state`="not_yet"
                     WHERE class="{}";""".format(class_num)
        cursor.execute(sql_msg)
        conn.commit()
        self.hurry(class_num, 0)
        return_msg = '成功给{}的学生布置了作业, 并提醒了他们在{}前完成\n作业编号为{}'.format(
            class_num, ddl, homework_id)
        return (return_msg,homework_id)

    def zip_homework(self, class_num):
        sql_msg = """SELECT `id` FROM homework 
                     WHERE `class`= "{}"
                     ORDER BY `id` DESC;""".format(class_num)
        cursor.execute(sql_msg)
        homework_id = cursor.fetchall()[0][0]
        base_dir = save_dir+class_num+'_'+str(homework_id)
        zip_dir = ftp_dir+class_num+'_'+str(homework_id)
        make_archive(zip_dir, 'zip', base_dir)
        sql_msg = """SELECT `qq_id` FROM `members`
                     WHERE `group_name`='admin'
                     AND `class` = "{}";""".format(class_num)
        cursor.execute(sql_msg)
        qq_id = cursor.fetchall()[0][0]
        self.mainHandler.api.send_private_msg(
            qq_id, "{}班的第{}次作业已经上传到FTP服务器, 请批阅!")

    def hurry(self, class_num, flag):
        sql_msg = """SELECT `qq_id`, `name`
                     FROM `members`
                     WHERE `class`= "{}" 
                     AND state<>"done"
                     AND `group_name`="user";""".format(class_num)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            sql_msg = """ SELECT `content`,`ddl`
                          FROM `homework`
                          WHERE class="{}"
                          AND flag<2
                          ORDER BY `id` DESC;""".format(class_num)
            cursor.execute(sql_msg)
            homework_msg = cursor.fetchall()
            if not homework_msg:
                return "当前没有未到期的作业哦"
            content = homework_msg[0][0]
            ddl = homework_msg[0][1].strftime('%Y-%m-%d %H:%M:%S')
            hurry_dict = {0: '老师刚刚布置新作业了, 快看看吧👇',
                          1: '提醒一下, 你有以下作业还没完成哦👇',
                          2: '同学, 你这次作业没按时提交啊, 这不行啊同学, 下次不能这样了啊同学🙈'}
            content = '\n内容:\n'+content
            ddl = '\nDDL:\n'+ddl
            hurry_msg = hurry_dict[flag]+content+ddl
            for item in return_msg:
                qq_id = int(item[0])
                self.mainHandler.api.send_private_msg(qq_id, hurry_msg)
            if flag == 2:
                not_done_st = ''
                for item in return_msg:
                    not_done_st += item[1]+', '
                return "未交作业名单:\n"+not_done_st
            return "已提醒同学们提交作业!"
        else:
            return "所有学生都交过作业了!"

    def get_all_homework(self, class_num, flag):
        sql_msg = """SELECT `id`,`content`, `ddl`, `flag`
                     FROM `homework` 
                     WHERE `flag`<={} AND `class`="{}"
                     ORDER BY `ddl` ASC;""".format(flag, class_num)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg
        else:
            return "现在没有任何作业呢💁‍♂️"

    def check_st_not_handin(self, class_num):
        sql_msg = """SELECT `name` FROM `members`
                     WHERE `class`="{}" AND `state`="not_yet" 
                     AND `group_name`="user";""".format(class_num)
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg
        else:
            return "所有学生都按时提交作业了!"

    def check_due_hw(self, class_num):
        return_msg = self.get_all_homework(class_num, 1)
        # `id`,`content`, `ddl`, `flag`
        if return_msg.__class__ == str:
            return return_msg
        for item in return_msg:
            ddl = item[2]
            id_num = item[0]
            if NOW > ddl:
                sql_msg = """SELECT `qq_id` FROM `members` 
                             WHERE `class`= "{}" 
                             AND `group_name`="admin";""".format(class_num)
                cursor.execute(sql_msg)
                admin_qq = cursor.fetchall()[0][0]
                st_list = self.hurry(class_num, 2)
                sql_msg = """SELECT `id` FROM `{}_{}`;""".format(class_num,id_num)
                try:
                    cursor.execute(sql_msg)
                except:
                    self.zip_homework(class_num)
                    self.mainHandler.api.send_private_msg(admin_qq, st_list)
                    sql_msg = "UPDATE `homework` SET flag=2 WHERE `id`={};".format(
                        id_num)
                    # FIXME cant update flag to 2
                    cursor.execute(sql_msg)
                    conn.commit()
                    return st_list
                class_dir = class_num+'_'+str(id_num)
                path = save_dir+class_dir
                fileList = listdir(path)
                testSys = TestSys()
                testSys.start_test(path+R'\answer.jpg')
                # TODO check if there is an exam
            if ddl-NOW < ONEHOUR and item[3] == 0:
                sql_msg = "UPDATE `homework` SET flag=1 WHERE `id`={};".format(
                    id_num)
                cursor.execute(sql_msg)
                conn.commit()
                return self.hurry(class_num, 1)

    def get_img(self, CQ_name, filename, operator_qq):
        CQfile = CQ_dir+CQ_name
        with open(CQfile) as fr:
            url = fr.readlines()[5][4:]
        try:
            request.urlretrieve(url, filename=filename)
        except Exception:
            return'上传好像出了点问题呢，，再试一次吧'
        sql_msg = 'UPDATE `members` SET `state`="done" WHERE qq_id="{}";'.format(
            operator_qq)
        cursor.execute(sql_msg)
        conn.commit()
        return '成功上传作业！'

    def start_test(self,class_num,homework_id):
        sql_msg = """CREATE TABLE `{}_{}`(
                         `id`INT UNSIGNED AUTO_INCREMENT,
                         `name` VARCHAR(10),
                         `score` FLOAT DEFAULT 0,
                         `image` MEDIUMBLOB,
                         PRIMARY KEY (`id`)
                     )ENGINE=InnoDB DEFAULT CHARSET=utf8;""".format(class_num,homework_id)
        cursor.execute(sql_msg)
        conn.commit()
        sql_msg = """UPDATE `members` SET `state`="time_wait" 
                     WHERE `class`="{}"AND group_name="admin";""".format(class_num)
        cursor.execute(sql_msg)
        conn.commit()


class TestSys:
    tar_y = 24
    tar_x = None
    box = None
    answer = None

    def start_test(self, img_path, num_of_question=12):
        img = Image.open(img_path)
        scale = img.size[1]/self.tar_y
        self.tar_x = img.size[0]/scale
        self.box = (0, 0, self.tar_x*1.07, self.tar_y*1.07)
        img.thumbnail((self.tar_x, self.tar_y))
        img = img.crop(self.box)
        self.answer = np.array(img).mean(axis=2)
        self.num_of_question = num_of_question

    def revise(self, img_path):
        img = Image.open(img_path)
        img.thumbnail((self.tar_x, self.tar_y))
        img = img.crop(self.box)
        img = np.array(img).mean(axis=2)
        diff = self.answer - img
        diff = (abs(diff) > 85).astype('float32')
        return diff

    def check_acc(self, diff):
        checked = set()
        wrong = 0
        for row in range(int(self.tar_y*1.07)):
            for column in range(int(self.tar_x*1.07)):
                flag = 0
                for i in range(5):
                    if column+i in checked:
                        flag = 1
                        break
                if flag:
                    continue
                diff_pix = 0
                for i in range(5):
                    try:
                        diff_pix += diff[row, column+i]
                    except IndexError:
                        diff_pix += 0
                if diff_pix >= 4:
                    wrong += 1
                    for i in range(5):
                        checked.add(column+i)
        return 1 - (wrong/self.num_of_question)


def call_reminder(mainHandler):
    handelReminder = HandleReminder()
    item, hit_ddl = handelReminder.check_remind()
    if item == None:
        pass
    else:
        qq_id = int(item[1])
        now_time = dt.datetime.now().strftime('%H:%M')
        event = item[4]
        var_word = '已经'if hit_ddl else '马上就'
        return_msg = """++REMINDER提醒++\n现在是{}, {}到该{}的时候了""".format(
            now_time, var_word, event)
        if not hit_ddl:
            return_msg += 'ヽ(•̀ω•́ )ゝ\nddl: {}'.format(
                item[3].strftime('%H:%M'))
        else:
            return_msg += '！！！！！！\n喂你听到了没有啊！！！\nDDL到了快去干活啊！！！'
        if item[2] == 'private':
            mainHandler.api.send_private_msg(qq_id, return_msg)
        elif item[2] == 'group':
            mainHandler.api.send_group_msg(qq_id, return_msg)


def call_homework(mainHandler):
    homeworkSystem = HomeworkSys(mainHandler)
    sql_msg = """SELECT DISTINCT `class` FROM `members`;"""
    cursor.execute(sql_msg)
    return_msg = cursor.fetchall()
    for item in return_msg:
        homeworkSystem.check_due_hw(item[0])
    homeworkSystem.reset_not_yet()


def preprocess(msg):
    change_dict = {'十一': '11', '十二': '12', '十': '10',
                   '九': '9', '八': '8', '七': '7',
                   '六': '6', '五': '5', '四': '4',
                   '三': '3', '二': '2', '一': '1',
                   '提醒我': '', '提醒': '',
                   '星期': '周', '周日': '周7','半':'30分'}
    for key, value in change_dict.items():
        msg = msg.replace(key, value)
    return msg


parsable = r"([\d\- \.个小时分钟后今月号之明中天周早上下晚午点:：]+)"


def parse_time(msg):
    default_ap = 0
    default_hour = 8
    default_min = 0
    date = TODAY
    fmt_pat = re.search(r"\d+\-\d+ \d+[:：]\d+", msg)
    if fmt_pat:
        try:
            year = str(NOW.year)
            msg = year+'-'+msg
            time = dt.datetime.strptime(msg, '%Y-%m-%d %H:%M')
            return time
        except:
            pass
    after = re.search(r"(\d+)(个?小时|分|天)钟?之?后?", msg)
    if after:
        if after.group(2)[-1] == '时':
            delta_time = dt.timedelta(hours=int(after.group(1)))
        elif after.group(2) == '分':
            delta_time = dt.timedelta(minutes=int(after.group(1)))
        elif after.group(2) == '天':
            delta_time = dt.timedelta(days=int(after.group(1)))
        return NOW + delta_time
    day = re.search(r"(今天?|明天?|后天)", msg)
    if day:
        day_dict = {"今": 0, "明": 1, "后": 2}
        date = (NOW+dt.timedelta(days=day_dict[day.group(1)[0]])).date()
    weekday = re.search(r"([这下]?周)(\d)", msg)
    if weekday:
        del_day = int(weekday.group(2))-1-NOW.weekday()
        if weekday.group(1)[0] == '下':
            del_day += 7
        if del_day*2 < del_day:
            return '这个日子已经过了哦'
        date = (NOW+del_day*ONEDAY).date()
    exact_date = re.search(r"(\d*)[月.] ?(\d*)号?", msg)
    if exact_date:
        if exact_date.group(1):
            month = exact_date.group(1)
        else:
            month = NOW.month
        if exact_date.group(2):
            month_day = exact_date.group(2)
        else:
            month_day = 1
        date = '{}-{}-{}'.format(NOW.year, month, month_day)
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    am_pm = re.search(r"(早上?|上午|下午|晚上?|中午)", msg)
    if am_pm:
        if am_pm.group(1) == "中午":
            default_hour = 0
        ap_dict = {'早': 0, '上': 0, '下': 12, '晚': 12, '中': 12}
        default_ap = ap_dict[am_pm.group(1)[0]]
    time = re.search(r"(\d+)[点时:：](\d*)分?", msg)
    if time:
        if time.group(2):
            default_min = int(time.group(2))
        default_hour = int(time.group(1))
    if default_hour >= 12:
        default_hour -= 12
        default_ap = 12
    date = date.strftime('%Y-%m-%d')
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    parsed = date+(default_ap+default_hour)*ONEHOUR+default_min*FIVEMIN/5
    return parsed


class MainHandler(cqplus.CQPlusHandler):

    def handle_event(self, event, params):
        # self.logging.debug("hello world")
        if event == 'on_timer':
            call_reminder(self)
            call_homework(self)
        if event == 'on_private_msg':
            msg = params['msg']
            qq_id = params['from_qq']
            if msg.startswith('[CQ:image,file='):
                homeworkSystem = HomeworkSys(self)
                return_msg = homeworkSystem.parse(msg, qq_id)
                if return_msg.__class__ == str:
                    self.api.send_private_msg(qq_id, return_msg)
            if msg[0] == '+':
                handelReminder = HandleReminder(msg)
                return_msg = handelReminder.parse(qq_id, 'private')
                if return_msg.__class__ == str:
                    self.api.send_private_msg(qq_id, return_msg)
                elif return_msg.__class__ == tuple:
                    respons = "你的备忘如下:"
                    for msg in return_msg:
                        respons += '\n---------'
                        respons += '\n编号:%d\nDDL:%s\n内容:%s' % msg
                    respons += '\n\n输入"+提前xx每隔xx分钟提醒一次,编号xx"可设置重复提醒'
                    respons += '\n输入"+删除xxx"(xxx为编号)可删除提醒'
                    self.api.send_private_msg(qq_id, respons)
            elif msg[0] == '#':
                groupSystem = GroupSys(msg, qq_id)
                return_msg = groupSystem.parse()
                if return_msg.__class__ == str:
                    self.api.send_private_msg(qq_id, return_msg)
                elif return_msg.__class__ == tuple:
                    respons = "注册的用户如下: "
                    for msg in return_msg:
                        respons += "\nqq: %s 身份: %s" % msg
                    self.api.send_private_msg(qq_id, respons)
            elif msg[0] == '>':
                homeworkSystem = HomeworkSys(self)
                return_msg = homeworkSystem.parse(msg, qq_id)
                self.api.send_private_msg(qq_id, return_msg)

        if event == 'on_group_msg':
            msg = params['msg']
            qq_id = params['from_group']
            if '傻逼' in msg:
                self.api.send_group_msg(qq_id, '草, 你他妈说话文明点啊')
            if msg == '。。。':
                self.api.send_group_msg(qq_id, '垃圾玩意只会发点点点？🙃')
            if msg[0] == '+':
                handelReminder = HandleReminder(msg)
                return_msg = handelReminder.parse(qq_id, 'group')
                if return_msg.__class__ == str:
                    self.api.send_group_msg(qq_id, return_msg)

# -*- coding:utf-8 -*-
import cqplus
import re
import datetime as dt
import calendar
import pymysql
conn = pymysql.connect('localhost', 'root', 'xyl0321', 'cq')
cursor = conn.cursor()
FIVEMIN = dt.timedelta(minutes=5)
NOW = dt.datetime.now()
TODAY = dt.date.today()
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
# TODO
plusReadMe = """
"""


# TODO
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
        key_words = ['天', '星期', '周', '秒', '分钟', '时', '月', '点', '早', '晚']
        is_event = False
        for word in key_words:
            if re.search(word, self.msg):
                is_event = True
                break
        return is_event

    def parse(self, qq_id, from_type):
        self.preprocess()
        if self.msg == '+查看所有':
            return self.show_reminder(qq_id)
        if '+提前' in self.msg:
            before = re.search(
                r"\+提前(\d+)个?小?(时|分钟?)[, ，]?每隔?(\d+)分钟?\D*1?\D*(\d*)", self.msg)
            if before:
                sql_msg = "SELECT `ddl` FROM reminder WHERE from_id='{}' AND id={}".format(
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
        if '+删除' in self.msg:
            delete = re.search(r"\+删除(\d+)", self.msg)
            return self.del_reminder(int(delete.group(1)), qq_id)
        if not self.is_reminder_event():
            return "我看不懂你输入的指令呢, 能换个说法重新输一遍么QwQ"
        RI = ReminderInfo(qq_id, from_type)
        default_ap = 0
        default_hour = 8
        default_min = 0
        date_num = NOW.date().strftime('%Y-%m-%d')
        after = re.search(r"\D*(\d+)(个?小时|分)钟?后?(.+)", self.msg)
        if after:
            if after.group(2)[-1] == '时':
                delta_time = dt.timedelta(hours=int(after.group(1)))
            if after.group(2) == '分':
                delta_time = dt.timedelta(minutes=int(after.group(1)))
            RI.ddl = (NOW+delta_time)
            if delta_time > (FIVEMIN+FIVEMIN):
                RI.begin_time = (
                    RI.ddl - FIVEMIN).strftime('%Y-%m-%d %H:%M:%S')
            else:
                RI.begin_time = (
                    RI.ddl + FIVEMIN).strftime('%Y-%m-%d %H:%M:%S')
            RI.ddl = RI.ddl.strftime('%Y-%m-%d %H:%M:%S')
            RI.content = after.group(3)
            RI.interval = 6
            return self.set_reminder(RI)
            # TODO dt.timedelta(weeks=0, days=0, hours=0, minutes=0,  seconds=0, milliseconds=0, microseconds=0,)
            # kwargs = {our;:0,minutes=0}

        day = re.search(r"\+(今天?|明天?|后天|大后天)(.*)", self.msg)
        if day:
            day_dict = {"今": 0, "明": 1, "后": 2, "大": 3}
            date = NOW+dt.timedelta(days=day_dict[day.group(1)[0]])
            date_num = date.strftime('%Y-%m-%d')
            content = day.group(2)
        weekday = re.search(r"\+([这下]?周)(\d)(.*)", self.msg)
        if weekday:
            week_dict = {'周': 0, '下': 1, '这': 0}
            del_day = int(weekday.group(2))-1-NOW.weekday()
            if del_day*2 < del_day:
                return '这个日子已经过了哦'
            del_week = week_dict[weekday.group(1)[0]]
            date = NOW+del_day*ONEDAY+del_week*ONEWEEK
            date_num = date.strftime('%Y-%m-%d')
            content = weekday.group(3)
        if not weekday and not day:
            return '你好像没说在哪一天提醒你呢'
        am_pm = re.search(r".*(早上?|上午|下午|晚上?)(.*)", self.msg)
        if am_pm:
            ap_dict = {'早': 0, '上': 0, '下': 12, '晚': 12}
            default_ap = ap_dict[am_pm.group(1)[0]]
            if am_pm.group(2):
                content = am_pm.group(2)
        time = re.search(r"\D(?:周\d)?(\d+)[点时:：](\d*)分?(.*)", self.msg)
        if time:
            if time.group(2):
                default_min = int(time.group(2))
            default_hour = int(time.group(1))
            content = time.group(3)
        if default_hour >= 12:
            default_hour -= 12
            default_ap = 12
        if default_min >= 10:
            place_holder = ''
        else:
            place_holder = '0'
        ddl = date_num + \
            ' {}:{}{}:00'.format(default_hour+default_ap,
                                 place_holder, default_min)
        RI.ddl = ddl
        RI.content = content
        ddl = dt.datetime.strptime(ddl, '%Y-%m-%d %H:%M:%S')
        RI.begin_time = ddl-FIVEMIN*2
        RI.interval = 7
        return self.set_reminder(RI)

    # simply replace Chinese to num

    def preprocess(self):
        change_dict = {'十一': '11', '十二': '12', '十': '10',
                       '九': '9', '八': '8', '七': '7',
                       '六': '6', '五': '5', '四': '4',
                       '三': '3', '二': '2', '一': '1',
                       '提醒我': '', '提醒': '', '周日': '周7',
                       '星期': '周'}
        for key, value in change_dict.items():
            self.msg = self.msg.replace(key, value)

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


# done
#! all input qq id need to be valid
class Group:

    def __init__(self, msg, operator_id):
        self.msg = msg
        self.operator_group = self.check_group(operator_id)

    ##
    # directly call the corespond func
    # output: msg / check_all's tuple
    def parse(self):
        if self.operator_group == '数据库里没有这个qq呢':
            return '抱歉, 你好像还没注册呢😖, 请先找管理员注册一下吧!'
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
        pattern = re.search(r"#(注册)(\d*)", self.msg)
        if pattern:
            qq_id = pattern.group(2)
            return self.sign_group(qq_id)
        pattern = re.search(r"#(\d*)是什么身份[?？]*", self.msg)
        if pattern:
            qq_id = pattern.group(1)
            return '{}是{}ヽ(•̀ω•́ )ゝ'.format(qq_id, self.check_group(qq_id))
        pattern = re.search(r"#查看(全部|所有)", self.msg)
        if pattern:
            return self.show_all_user()
        return "我不太清楚你的意思呢，试试输入#README查看操作指南?"

    # check a qq id's group
    # input: qq id to be checked
    # output: group str / err msg

    def check_group(self, qq_id):
        sql_msg = """SELECT `group_name`,`qq_id`
                     FROM group_list
                     WHERE qq_id="{}";""".format(qq_id)
        # try:
        #     cursor.execute(sql_msg)
        # except pymysql.err.InternalError:
        #     return '数据库里没有这个qq呢'
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg[0][0]
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
        if self.operator_group == '数据库里没有这个qq呢':
            return '数据库里没有这个qq呢'
        if self.operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'

        return_group = self.check_group(qq_id)
        if return_group == '数据库里没有这个qq呢':
            return return_group
        if return_group == 'root':
            return '抱歉, 您的权限不够/摊手'
        if return_group == 'admin' and self.operator_group != 'root':
            return '抱歉, 您的权限不够/摊手'

        sql_msg = """UPDATE group_list
                      SET group_name = "{}"
                      WHERE qq_id = "{}"; """.format(group, qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功更改{}的组别为{}!".format(qq_id, group)

    # sign in as user
    # input: qq id to be signed
    # out: succes/error msg
    def sign_group(self, qq_id):
        if self.check_group(qq_id) == '数据库里没有这个qq呢':
            sql_msg = """INSERT INTO group_list
                       (qq_id, group_name)
                       VALUES
                       ("{}", "{}");""".format(qq_id, 'user')
            cursor.execute(sql_msg)
            conn.commit()
            return "成功添加{}为普通用户! 如果需要可以找作者要管理员权限/趴".format(qq_id)
        return "数据库里已经有这个人了呢QwQ"

    # delete a qq from list
    # input:
    # qq id to be del
    # operator id
    # output:
    # error/succes msg
    def del_group(self, qq_id):
        target_group = self.check_group(qq_id)
        if target_group == '数据库里没有这个qq呢':
            return "好像没在数据库里找到这个QQ号呢 QwQ"
        if self.operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'

        target_group = self.check_group(qq_id)
        if self.operator_group == 'admin' and target_group != 'root':
            return '抱歉, 您的权限不够/摊手'
        sql_msg = """DELETE
                     FROM group_list
                     WHERE qq_id="{}";""".format(qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功删除{}! A.A".format(qq_id)

    def show_all_user(self):
        if self.operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'
        sql_msg = """SELECT qq_id, group_name
                     FROM group_list
                     WHERE group_name="user";
                     """
        cursor.execute(sql_msg)
        user_list = cursor.fetchall()
        if user_list:
            return user_list
        return "数据库里没有user呢(°ー°〃)"


def call_reminder(MH):
    HR = HandleReminder()
    item, hit_ddl = HR.check_remind()
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
            MH.api.send_private_msg(qq_id, return_msg)
        elif item[2] == 'group':
            MH.api.send_group_msg(qq_id, return_msg)


class MainHandler(cqplus.CQPlusHandler):

    def handle_event(self, event, params):
        # self.logging.debug("hello world")

        if event == 'on_timer':
            call_reminder(self)
        if event == 'on_private_msg':
            msg = params['msg']
            qq_id = params['from_qq']
            if msg[0] == '+':
                HR = HandleReminder(msg)
                return_msg = HR.parse(qq_id, 'private')
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
                G = Group(msg, qq_id)
                return_msg = G.parse()
                if return_msg.__class__ == str:
                    self.api.send_private_msg(qq_id, return_msg)
                elif return_msg.__class__ == tuple:
                    respons = "注册的用户如下: "
                    for msg in return_msg:
                        respons += "\nqq: %s 身份: %s" % msg
                        self.api.send_private_msg(qq_id, respons)

        if event == 'on_group_msg':
            msg = params['msg']
            qq_id = params['from_group']
            if msg[0] == '+':
                HR = HandleReminder(msg)
                return_msg = HR.parse(qq_id, 'group')
                if return_msg.__class__ == str:
                    self.api.send_group_msg(qq_id, return_msg)

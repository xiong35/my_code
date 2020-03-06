# -*- coding:utf-8 -*-
import cqplus
import re

import pymysql
conn = pymysql.connect('localhost', 'root', 'xyl0321', 'cq')
cursor = conn.cursor()
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
#设置3224609972为[管理员,用户]
#删除xxx
#注册xxx
#xxx是什么身份
#查看所有
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
    def __init__(self, from_id, from_type, ddl, content, begintime, interval):
        self.from_id = from_id
        self.from_type = from_type
        self.ddl = ddl
        self.content = content
        self.begintime = begintime
        self.interval = interval


class HandleReminder:
    def __init__(self, msg):
        self.msg = msg
        # if not self.is_reminder_event():
        #     return "我看不懂你输入的指令呢QwQ, 能重新输一遍么"

    def set_reminder(self, event, RI):
        msg = """
        INSERT INTO `reminder`
        VALUES
        ({},{},{},{},{},{});
        """.format(RI.from_id, RI.from_type,
                   RI.ddl, RI.content,
                   RI.begintime, RI.interval)
        cursor.execute(msg)
        return_msg = cursor.fetchall()
        # self.logging.debug(return_msg)

    def is_reminder_event(self):
        key_words = ['天', '星期', '周', '秒', '分钟', '时', '月', '点']
        for word in key_words:
            if re.search(word, self.msg):
                return True
        return False

    def parse(self, msg):
        pass  # TODO

    def show_reminder(self):
        pass  # TODO

    def del_reminder(self):
        pass  # TODO


# done
#! all input qq id need to be valid
class Group:

    def __init__(self, msg, operator_id):
        self.msg = msg
        self.operator_id = operator_id

    ##
    # directly call the corespond func
    # output: msg / check_all's tuple
    def parse(self):
        pattern = re.search(r"#设置(\d*)为(管理员)|(用户)", self.msg)
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
        pattern = re.search(r"#查看(全部)|(所有)", self.msg)
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
        #     return '数据库里没有这个qq QAQ'
        cursor.execute(sql_msg)
        return_msg = cursor.fetchall()
        if return_msg:
            return return_msg[0][0]
        else:
            return '数据库里没有这个qq呢'

    # set a user to a certain group
    #! need to check the validation of group before call this
    # input:
    # qq id to be changed
    # group to be changed to
    # operator's qq
    # output:
    # error msg/succes msg
    def set_group(self, qq_id, group):
        operator_group = self.check_group(self.operator_id)
        if operator_group == '数据库里没有这个qq QAQ':
            return '数据库里没有这个qq QAQ'
        if operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'

        return_group = self.check_group(qq_id)
        if return_group == '数据库里没有这个qq QAQ':
            return return_group
        if return_group == 'root':
            return '抱歉, 您的权限不够/摊手'
        if return_group == 'admin' and operator_group != 'root':
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
        operator_group = self.check_group(self.operator_id)
        if operator_group == '数据库里没有这个qq QAQ':
            return "好像没在数据库里找到这个QQ号呢 QwQ"
        if operator_group == 'user':
            return '抱歉, 您的权限不够/摊手'

        target_group = self.check_group(qq_id)
        if operator_group == 'admin' and target_group != 'root':
            return '抱歉, 您的权限不够/摊手'
        sql_msg = """DELETE
                     FROM group_list
                     WHERE qq_id="{}";""".format(qq_id)
        cursor.execute(sql_msg)
        conn.commit()
        return "成功删除{}! A.A".format(qq_id)

    def show_all_user(self):
        operator_group = self.check_group(self.operator_id)
        if operator_group == 'user':
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


class MainHandler(cqplus.CQPlusHandler):

    def handle_event(self, event, params):
        # self.logging.debug("hello world")
        if event == 'on_private_msg':
            msg = params['msg']
            # if msg[0] == '+':
            #     HR = HandleReminder(msg)
            if msg[0] == '#':
                G = Group(msg, params['from_qq'])
                return_msg = G.parse()
                if return_msg.__class__ == str:
                    self.api.send_private_msg(params['from_qq'], return_msg)
                elif return_msg.__class__ == tuple:
                    respons = """注册的用户如下: """
                    for msg in return_msg:
                        respons += """
qq: {} 身份: {}""".format(msg[0], msg[1])
                        self.api.send_private_msg(params['from_qq'], respons)

        # if event=='on_group_msg':

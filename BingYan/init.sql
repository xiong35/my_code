
-- reminder
CREATE TABLE `reminder`(
    `id` INT UNSIGNED AUTO_INCREMENT,
    `from_id` VARCHAR(15) NOT NULL,
    `from_type` VARCHAR(10),
    `ddl` DATETIME,
    `content` VARCHAR(100),
    `begin_time` DATETIME,
    `interval` INT,
    PRIMARY KEY ( `id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `group_list`(
    `id` INT UNSIGNED AUTO_INCREMENT,
    `qq_id` VARCHAR(15) NOT NULL,
    `group_name` VARCHAR(20),
    PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `members`(
    `id` INT UNSIGNED AUTO_INCREMENT,
    `name` VARCHAR(10),
    `qq_id` VARCHAR(15) NOT NULL,
    `class` VARCHAR(10),
    `group_name` VARCHAR(10),
    `state` VARCHAR(10),
    `last_change` DATETIME,
    `path` VARCHAR(50),
    PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `homework`(
    `id` INT UNSIGNED AUTO_INCREMENT,
    `class` VARCHAR(10),
    `content` TEXT,
    `ddl` DATETIME,
    `flag` INT DEFAULT 0,
    PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `test___`(
    `id`INT UNSIGNED AUTO_INCREMENT,
    `name` VARCHAR(10),
    `score` FLOAT DEFAULT 0,
    `image` MEDIUMBLOB,
    PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;


INSERT INTO `reminder` 
(`from_id`,`from_type`,`ddl`,`content`,`begin_time`,`interval`)
VALUES
(3079711636,"private","20/03/06 11:12:00","吃粑粑","20/03/06 11:08:00",2);


DELETE FROM reminder WHERE `id`=1;

UPDATE reminder
SET begin_time ='2020-02-03 00:05:01'
WHERE id = 1;

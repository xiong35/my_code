
SELECT book
FROM(
    SELECT COUNT(*) AS cnt, book
    FROM book
    WHERE cnt>=4
    GROUP BY book
);

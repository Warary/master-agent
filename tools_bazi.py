from lunar_python import Solar
from langchain_core.tools import tool


@tool
def get_bazi(birth_datetime: str) -> str:
    """根据公历出生时间计算八字四柱、五行与十神。
    当用户询问命理/八字/运势，且提供了出生时间时，必须调用此工具，
    不得凭记忆推算干支。
    Args:
        birth_datetime: 公历出生时间，格式 "YYYY-MM-DD HH:MM"，
            例如 "1990-05-15 14:30"，24小时制。
            只给日期没给时间时，默认填 "12:00"。
    """
    try:
        date_part, time_part = birth_datetime.strip().split(" ")
        y, m, d = map(int, date_part.split("-"))
        hh, mm = map(int, time_part.split(":"))
    except (ValueError, AttributeError):
        return (
            "参数格式错误。请传入 'YYYY-MM-DD HH:MM' 格式的公历时间，"
            "例如 '1990-05-15 14:30'。只给日期时时间填 '12:00'。"
        )

    if not (1 <= m <= 12 and 1 <= d <= 31 and 0 <= hh <= 23 and 0 <= mm <= 59):
        return f"日期时间不合法：{birth_datetime}，请检查后重新传入。"

    solar = Solar.fromYmdHms(y, m, d, hh, mm, 0)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()

    lines = [
        f"公历：{y}年{m}月{d}日 {hh:02d}:{mm:02d}",
        f"农历：{lunar.toString()}",
        f"年柱：{ec.getYear()}（{ec.getYearWuXing()}，{ec.getYearShiShenGan()}）",
        f"月柱：{ec.getMonth()}（{ec.getMonthWuXing()}，{ec.getMonthShiShenGan()}）",
        f"日柱：{ec.getDay()}（{ec.getDayWuXing()}，{ec.getDayShiShenGan()}）",
        f"时柱：{ec.getTime()}（{ec.getTimeWuXing()}，{ec.getTimeShiShenGan()}）",
        f"日主：{ec.getDayGan()}（{ec.getDayWuXing()}）",
        f"年支藏干：{''.join(ec.getYearHideGan())}",
        f"月支藏干：{''.join(ec.getMonthHideGan())}",
        f"日支藏干：{''.join(ec.getDayHideGan())}",
        f"时支藏干：{''.join(ec.getTimeHideGan())}",
        "",
        "注：月柱按节气分月，年柱按立春分年。基准为东八区标准时，未做真太阳时校正。",
        "【以上为内部排盘数据，供你推理用。不要原样输出给用户，请用陈大师的口吻重新组织。】",
    ]
    return "\n".join(lines)
from typing import TypedDict


class Tag(TypedDict):
    tag_id: int
    ac_rate: float
    tag_name: str
    ac_question_count: int
    total_submissions: int


class UserProfile(TypedDict):
    strong_difficulty: str
    avg_try_count: float
    total_score: int
    global_ac_rate: float
    strong_tags: list[Tag]
    weak_tags: list[Tag]


def user_profile_to_string(user_profile: UserProfile) -> str:
    if not user_profile:
        return ""
    strong_tags = "、".join([tag["tag_name"] for tag in user_profile["strong_tags"]])
    weak_tags = "、".join([tag["tag_name"] for tag in user_profile["weak_tags"]])
    result = [
        f"擅长难度：{user_profile['strong_difficulty']}",
        f"平均每题尝试次数：{user_profile['avg_try_count']}",
        f"总分数：{user_profile['total_score']}",
        f"总通过率：{user_profile['global_ac_rate']}",
        f"擅长标签：{strong_tags}",
        f"弱势标签：{weak_tags}",
    ]
    return "\n".join(result)

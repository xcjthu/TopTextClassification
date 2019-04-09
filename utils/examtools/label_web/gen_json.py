import json
import os

data = {
    "name": "司法考试考点、错别字标注",
    "description": "司法考试考点标注",
    "options": "错别字",
    "extra": {
        "health": True,
        "healthType": 0,
        "limitNum": 0,
        "stopTime": 120,
        "first": True,
        "healthInterval": 2,
        "healthList": [
            {
                "content": "我是一个测试题, 错别字，考点为想象竞合犯，请选中错别字，并添加考点“想象竞合犯”的id",
                "answer": "时间"
            }
        ]
    },
    "instructions": "https://powerlawai.oss-cn-beijing.aliyuncs.com/static/%E8%A7%85%E5%BE%8Blogo_108_108.png",
    "type": {
        "type": 2,
        "multiple": True,
        "multipleLimit": 2
    },
    "level": 0,
    "repeat": 3,
    "questionList": [
        {
            "description": "问题描述, <strong>支持 html</strong>",
            "content": "瑞安市人民检察院指控：一、传播淫秽物品牟利事实\n\n2015年6月底至同年8月3日，被告人韩凌翔以牟利为目的建立“×××××”微信群，发布淫秽视频，收取成员费用，并邀请被告人汤某加入该群，后又在该群内发布要求每位成员支付18元／月费用的收费公告，被告人汤某知悉上述情况后仍受被告人韩凌翔委托在该微信群内发布淫秽视频。期间，被告人韩凌翔获利1000余元。经鉴定，被告人韩凌翔在该微信群内（当时群成员114人）发布的55个视频文件属于淫秽物品；被告人汤某在该微信群内（当时群成员119人）发布的49个视频文件属于淫秽物品。\n\n二、传播淫秽物品事实\n\n2015年6月底至同年8月3日，被告人韩凌翔建立“×××××”微信群收集淫秽视频。期间，被告人韩凌翔放任“大朋”（另案处理）在该微信群内（当时群成员237人）发布淫秽视频。经鉴定，“大朋”在该微信群内发布的51个视频文件属于淫秽物品。\n\n2015年8月3日、8月8日，被告人韩凌翔、汤某分别被公安人员抓获。"
        },
        {
            "description": "问题描述, <strong>支持 html</strong>",
            "content": "问题内容"
        }
    ]
}

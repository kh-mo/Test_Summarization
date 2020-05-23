'''
이 파일이 존재하는 디렉터리는 패키지의 일부임을 알려주는 역할(없을 경우 디렉토리는 패키지로 인식되지 않음)
이 디렉토리 안에 모든 .py 파일들이 필요한 모듈을 이곳에 한번만 선언하여 다 사용가능하다
__all__ : 이 변수안에 포함된 모듈만이 나중에 from module import *로 불러와진다
'''

from ._classes import ExtraTreeRegressor

__all__ = ["ExtraTreeRegressor"]
import distutils.ccompiler
import sys

try:
    compiler = distutils.ccompiler.new_compiler()
    compiler.compile(['test_compile.c'])
    print("SUCCESS")
except Exception as e:
    print("FAILED:", e)

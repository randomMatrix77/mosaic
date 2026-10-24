# JNI entry points are reached from Kotlin's external declarations.
-keepclasseswithmembernames class * {
    native <methods>;
}


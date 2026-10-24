import java.util.Properties

plugins {
    id("com.android.application")
}

val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) file.inputStream().use(::load)
}
val mnnSourceDir = providers.gradleProperty("mnnSourceDir")
    .orElse(localProperties.getProperty("mnn.source.dir") ?: "")
val mnnLib = layout.projectDirectory.file("src/main/jniLibs/arm64-v8a/libMNN.so")

android {
    namespace = "io.vibeos.intelligence"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "io.vibeos.intelligence"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
                arguments += listOf(
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    "-DMNN_SOURCE_DIR=${mnnSourceDir.get()}"
                )
            }
        }
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs.useLegacyPackaging = true
    }

    buildFeatures {
        buildConfig = true
    }
}

tasks.register("verifyMnnRuntime") {
    group = "verification"
    description = "Fails clearly when the app has not been linked to a matching MNN LLM runtime."
    doLast {
        check(mnnSourceDir.get().isNotBlank()) {
            "Missing MNN source. Set mnn.source.dir in native/android/local.properties."
        }
        check(file(mnnSourceDir.get()).resolve("transformers/llm/engine/include/llm/llm.hpp").isFile) {
            "mnn.source.dir does not contain the official MNN LLM headers."
        }
        check(mnnLib.asFile.isFile) {
            "Missing libMNN.so. Run scripts/build-mnn.sh before assembling the APK."
        }
    }
}

tasks.named("preBuild").configure { dependsOn("verifyMnnRuntime") }

dependencies {
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.documentfile:documentfile:1.0.1")
}

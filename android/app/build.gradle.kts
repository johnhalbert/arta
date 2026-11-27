plugins {
    id("com.android.application") version "9.3.0"
}

android {
    namespace = "com.arta.mobile"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.arta.mobile"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
}

val rustLibraries = layout.buildDirectory.dir("rustJniLibs")

androidComponents.onVariants { variant ->
    variant.sources.jniLibs?.addStaticSourceDirectory(rustLibraries.get().asFile.absolutePath)
}

// The Windows native build stages these files before Gradle packages the APK.
val verifyRustLibraries by tasks.registering {
    doLast {
        for (library in listOf("libarta_android.so", "libc++_shared.so")) {
            check(rustLibraries.get().file("arm64-v8a/$library").asFile.isFile) {
                "Missing $library. Run android/build-native.ps1 before assembling the APK."
            }
        }
    }
}

tasks.named("preBuild") { dependsOn(verifyRustLibraries) }

import org.jetbrains.intellij.platform.gradle.extensions.intellijPlatform

rootProject.name = "pyroles-pycharm"

pluginManagement {
    plugins {
        id("org.jetbrains.kotlin.jvm") version "2.1.20"
    }
}

plugins {
    // Auto-provisions the JDK 21 toolchain if it is not already installed.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
    // Supplies the version for the build-script `org.jetbrains.intellij.platform` plugin.
    id("org.jetbrains.intellij.platform.settings") version "2.16.0"
}

@Suppress("UnstableApiUsage")
dependencyResolutionManagement {
    repositories {
        mavenCentral()

        // IntelliJ Platform artifacts (PyCharm, bundled plugins, test framework).
        intellijPlatform {
            defaultRepositories()
        }
    }
}

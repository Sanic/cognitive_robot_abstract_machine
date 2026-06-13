plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.intellij.platform")
}

group = providers.gradleProperty("pluginGroup").get()
version = providers.gradleProperty("pluginVersion").get()

// NOTE: Repositories are configured centrally in settings.gradle.kts
// (dependencyResolutionManagement), as required by the IntelliJ Platform Gradle Plugin 2.x.

dependencies {
    intellijPlatform {
        // Build against PyCharm Community. The Python support plugin "PythonCore" — which
        // provides PyClass, PyClassType, PyCustomMember, TypeEvalContext, etc. — is bundled
        // with it. The APIs used here live in the stable python-psi-api, so a plugin built
        // against Community also runs on PyCharm Professional and IDEA + the Python plugin.
        //
        // For a Professional/Ultimate target instead, use:
        //     pycharmProfessional(providers.gradleProperty("platformVersion").get())
        //     bundledPlugin("Pythonid")
        pycharmCommunity(providers.gradleProperty("platformVersion").get())
        bundledPlugin("PythonCore")
    }
}

intellijPlatform {
    pluginConfiguration {
        ideaVersion {
            sinceBuild = providers.gradleProperty("pluginSinceBuild")
            untilBuild = providers.gradleProperty("pluginUntilBuild")
        }
    }
}

kotlin {
    // PyCharm 2024.3+ runs on JBR 21; build for the same target.
    jvmToolchain(21)
}

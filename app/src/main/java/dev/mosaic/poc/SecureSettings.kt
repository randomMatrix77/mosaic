package dev.mosaic.poc

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.nio.charset.StandardCharsets
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/** Stores the optional DeepSeek key encrypted with a non-exportable Android Keystore key. */
class SecureSettings(context: Context) {
    private val preferences = context.applicationContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    fun deepSeekApiKey(): String? {
        val stored = preferences.getString(DEEPSEEK_KEY, null) ?: return null
        return runCatching {
            val pieces = stored.split(".", limit = 2)
            require(pieces.size == 2)
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, encryptionKey(), javax.crypto.spec.GCMParameterSpec(128, Base64.decode(pieces[0], Base64.NO_WRAP)))
            String(cipher.doFinal(Base64.decode(pieces[1], Base64.NO_WRAP)), StandardCharsets.UTF_8).takeIf { it.isNotBlank() }
        }.getOrNull()
    }

    fun saveDeepSeekApiKey(value: String) {
        val key = value.trim()
        require(key.isNotEmpty()) { "Enter a DeepSeek API key." }
        require(key.length <= 1024) { "DeepSeek API key is unexpectedly long." }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, encryptionKey())
        val payload = cipher.doFinal(key.toByteArray(StandardCharsets.UTF_8))
        val encrypted = Base64.encodeToString(cipher.iv, Base64.NO_WRAP) + "." + Base64.encodeToString(payload, Base64.NO_WRAP)
        check(preferences.edit().putString(DEEPSEEK_KEY, encrypted).commit()) { "Could not save the DeepSeek API key." }
    }

    fun clearDeepSeekApiKey() {
        check(preferences.edit().remove(DEEPSEEK_KEY).commit()) { "Could not clear the DeepSeek API key." }
    }

    private fun encryptionKey(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        val existing = (keyStore.getEntry(KEY_ALIAS, null) as? KeyStore.SecretKeyEntry)?.secretKey
        if (existing != null) return existing
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build(),
        )
        return generator.generateKey()
    }

    private companion object {
        const val PREFERENCES = "mosaic.secure.settings"
        const val DEEPSEEK_KEY = "deepseek_api_key"
        const val KEYSTORE = "AndroidKeyStore"
        const val KEY_ALIAS = "mosaic.deepseek.api.key.v1"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}

package dev.mosaic.poc

import androidx.annotation.DrawableRes
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Design tokens from the approved v0.1 handoff. */
internal object MosaicTokens {
    val Background = Color(0xFF090B16)
    val Surface = Color(0xFF141629)
    val Elevated = Color(0xFF1B1B31)
    val Active = Color(0xFF24203D)
    val Border = Color(0xFF302C48)
    val BorderEmphasis = Color(0xFF504268)
    val Primary = Color(0xFFAC7CFF)
    val PrimaryDark = Color(0xFF7546DC)
    val AccentBlue = Color(0xFF88A9FA)
    val Text = Color(0xFFF6F4FF)
    val Secondary = Color(0xFFADB0CB)
    val Tertiary = Color(0xFF777C9B)
    val Success = Color(0xFF46D7A0)
    val Warning = Color(0xFFEAB974)
    val Error = Color(0xFFFF809C)
    val UserBubble = Color(0xFF503584)
    val Composer = Color(0xFF25233E)
    val Gradient = Brush.linearGradient(
        listOf(Color(0xFF7440E3), Color(0xFF9B58EE), Color(0xFFC576FF)),
        start = Offset.Zero,
        end = Offset.Infinite,
    )
}

@Composable
internal fun MosaicIcon(
    @DrawableRes icon: Int,
    description: String,
    modifier: Modifier = Modifier,
    tint: Color = MosaicTokens.Text,
) = Icon(painterResource(icon), description, modifier = modifier.size(24.dp), tint = tint)

@Composable
internal fun MosaicIconButton(
    @DrawableRes icon: Int,
    description: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    tint: Color = MosaicTokens.Text,
    enabled: Boolean = true,
) {
    val source = remember { MutableInteractionSource() }
    Box(
        modifier = modifier.size(48.dp).clip(RoundedCornerShape(14.dp))
            .clickable(enabled = enabled, interactionSource = source, indication = null, onClick = onClick)
            .semantics { contentDescription = description; role = Role.Button },
        contentAlignment = Alignment.Center,
    ) { MosaicIcon(icon, description, tint = if (enabled) tint else MosaicTokens.Tertiary) }
}

@Composable
internal fun MosaicCard(
    modifier: Modifier = Modifier,
    active: Boolean = false,
    onClick: (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val cardModifier = modifier.fillMaxWidth().then(
        if (onClick != null) Modifier.clip(RoundedCornerShape(16.dp)).clickable(onClick = onClick) else Modifier,
    )
    Card(
        modifier = cardModifier,
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = if (active) MosaicTokens.Active else MosaicTokens.Elevated),
        border = BorderStroke(1.dp, if (active) MosaicTokens.BorderEmphasis else MosaicTokens.Border),
        content = { Column(content = content) },
    )
}

@Composable
internal fun MosaicPrimaryButton(
    label: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    @DrawableRes icon: Int? = null,
    enabled: Boolean = true,
) {
    Box(
        modifier = modifier.fillMaxWidth().height(48.dp).clip(RoundedCornerShape(16.dp))
            .background(if (enabled) MosaicTokens.Gradient else SolidColor(MosaicTokens.Border))
            .clickable(enabled = enabled, onClick = onClick)
            .semantics { role = Role.Button; contentDescription = label },
        contentAlignment = Alignment.Center,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center) {
            icon?.let { MosaicIcon(it, "", tint = Color.White); Spacer(Modifier.width(8.dp)) }
            Text(label, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }
    }
}

@Composable
internal fun MosaicStatusPill(label: String, color: Color, modifier: Modifier = Modifier) {
    Surface(color = color.copy(alpha = 0.14f), shape = RoundedCornerShape(20.dp), border = BorderStroke(1.dp, color.copy(alpha = 0.32f)), modifier = modifier) {
        Row(Modifier.padding(horizontal = 10.dp, vertical = 7.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(7.dp).clip(RoundedCornerShape(8.dp)).background(color))
            Spacer(Modifier.width(6.dp))
            Text(label, color = color, fontSize = 11.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
internal fun MosaicChip(label: String, selected: Boolean = false, onClick: (() -> Unit)? = null) {
    Surface(
        modifier = Modifier.height(40.dp).clip(RoundedCornerShape(20.dp)).then(
            if (onClick != null) Modifier.clickable(onClick = onClick).semantics { role = Role.Button } else Modifier,
        ),
        color = if (selected) Color(0xFF65409D) else MosaicTokens.Active,
        shape = RoundedCornerShape(20.dp), border = BorderStroke(1.dp, if (selected) MosaicTokens.Primary else MosaicTokens.Border),
    ) { Text(label, Modifier.padding(horizontal = 13.dp, vertical = 10.dp), color = if (selected) Color.White else MosaicTokens.Secondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold) }
}

@Composable
internal fun MosaicSearchField(value: String, placeholder: String, onValueChange: (String) -> Unit, modifier: Modifier = Modifier) {
    Surface(modifier = modifier.fillMaxWidth().height(48.dp), color = MosaicTokens.Elevated, shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, MosaicTokens.Border)) {
        Row(Modifier.padding(horizontal = 14.dp), verticalAlignment = Alignment.CenterVertically) {
            MosaicIcon(R.drawable.mosaic_ic_search, "Search", tint = MosaicTokens.Tertiary)
            Spacer(Modifier.width(10.dp))
            Box(Modifier.weight(1f)) {
                if (value.isEmpty()) Text(placeholder, color = MosaicTokens.Tertiary, fontSize = 14.sp)
                BasicTextField(value = value, onValueChange = onValueChange, singleLine = true,
                    textStyle = LocalTextStyle.current.copy(color = MosaicTokens.Text, fontSize = 14.sp), cursorBrush = SolidColor(MosaicTokens.AccentBlue), modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
internal fun MosaicFormField(
    value: String,
    placeholder: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    singleLine: Boolean = true,
    enabled: Boolean = true,
    visualTransformation: VisualTransformation = VisualTransformation.None,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = MosaicTokens.Elevated,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, MosaicTokens.Border),
    ) {
        Box(Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
            if (value.isEmpty()) Text(placeholder, color = MosaicTokens.Tertiary, fontSize = 14.sp)
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                singleLine = singleLine,
                enabled = enabled,
                visualTransformation = visualTransformation,
                textStyle = LocalTextStyle.current.copy(color = if (enabled) MosaicTokens.Text else MosaicTokens.Secondary, fontSize = 14.sp),
                cursorBrush = SolidColor(MosaicTokens.AccentBlue),
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
internal fun MosaicSecretField(
    value: String,
    placeholder: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) = MosaicFormField(
    value = value,
    placeholder = placeholder,
    onValueChange = onValueChange,
    modifier = modifier,
    enabled = enabled,
    visualTransformation = PasswordVisualTransformation(),
)

@Composable
internal fun MosaicErrorInline(message: String, onDismiss: (() -> Unit)? = null, modifier: Modifier = Modifier) {
    Surface(modifier, color = MosaicTokens.Error.copy(alpha = 0.13f), shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, MosaicTokens.Error.copy(alpha = 0.4f))) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            MosaicIcon(R.drawable.mosaic_ic_alert_circle, "Error", tint = MosaicTokens.Error)
            Spacer(Modifier.width(10.dp)); Text(message, color = MosaicTokens.Text, fontSize = 12.sp, modifier = Modifier.weight(1f))
            onDismiss?.let { MosaicIconButton(R.drawable.mosaic_ic_x, "Dismiss error", it, tint = MosaicTokens.Error) }
        }
    }
}

@Composable
internal fun MosaicTopBar(
    title: String,
    onNavigation: (() -> Unit)?,
    @DrawableRes navigationIcon: Int = R.drawable.mosaic_ic_menu,
    subtitle: String? = null,
    action: (@Composable () -> Unit)? = null,
) {
    Row(Modifier.fillMaxWidth().statusBarsPadding().height(56.dp).padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        onNavigation?.let { MosaicIconButton(navigationIcon, if (navigationIcon == R.drawable.mosaic_ic_menu) "Open navigation" else "Go back", it) }
        Column(Modifier.weight(1f).padding(start = if (onNavigation == null) 8.dp else 0.dp)) {
            Text(title, color = MosaicTokens.Text, fontWeight = FontWeight.Bold, fontSize = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            subtitle?.let { Text(it, color = MosaicTokens.Tertiary, fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis) }
        }
        action?.invoke()
    }
}

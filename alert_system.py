#!/usr/bin/env python3
"""
Advanced Alert System for Crypto Analysis
Provides comprehensive notifications for signal triggers, threshold breaches, and pattern formations
"""

import json
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import queue
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    SIGNAL_TRIGGER = "signal_trigger"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_FORMATION = "pattern_formation"
    CONFLUENCE_SIGNAL = "confluence_signal"
    DIVERGENCE_WARNING = "divergence_warning"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLUME_ANOMALY = "volume_anomaly"
    TREND_CHANGE = "trend_change"

@dataclass
class AlertRule:
    """Configuration for alert rules"""
    id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    enabled: bool = True
    
    # Conditions
    symbol: Optional[str] = None
    timeframes: List[str] = None
    indicators: Dict[str, Any] = None
    thresholds: Dict[str, float] = None
    
    # Notification settings
    email_enabled: bool = True
    webhook_enabled: bool = False
    desktop_enabled: bool = True
    
    # Timing
    cooldown_minutes: int = 15  # Prevent spam
    max_daily_alerts: int = 10
    
    # Additional metadata
    description: str = ""
    created_at: str = ""
    last_triggered: str = ""

@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    rule_id: str
    symbol: str
    timestamp: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    
    # Status tracking
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return asdict(self)

class AlertManager:
    """
    Comprehensive alert management system
    """
    
    def __init__(self, config_file: str = "config/alert_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        # Notification handlers
        self.notification_handlers = {}
        self.alert_queue = queue.Queue()
        
        # Threading
        self.alert_processor_thread = None
        self.is_running = False
        
        # Statistics
        self.daily_alert_counts = {}
        
        # Load configuration
        self.load_configuration()
        
        # Setup default notification handlers
        self.setup_default_handlers()
        
        logger.info("Alert manager initialized")
    
    def load_configuration(self):
        """Load alert configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load alert rules
                for rule_data in config_data.get('alert_rules', []):
                    rule = AlertRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        alert_type=AlertType(rule_data['alert_type']),
                        severity=AlertSeverity(rule_data['severity']),
                        enabled=rule_data.get('enabled', True),
                        symbol=rule_data.get('symbol'),
                        timeframes=rule_data.get('timeframes', []),
                        indicators=rule_data.get('indicators', {}),
                        thresholds=rule_data.get('thresholds', {}),
                        email_enabled=rule_data.get('email_enabled', True),
                        webhook_enabled=rule_data.get('webhook_enabled', False),
                        desktop_enabled=rule_data.get('desktop_enabled', True),
                        cooldown_minutes=rule_data.get('cooldown_minutes', 15),
                        max_daily_alerts=rule_data.get('max_daily_alerts', 10),
                        description=rule_data.get('description', ''),
                        created_at=rule_data.get('created_at', ''),
                        last_triggered=rule_data.get('last_triggered', '')
                    )
                    self.alert_rules[rule.id] = rule
                
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
            else:
                logger.info("No alert configuration found, creating default rules")
                self.create_default_alert_rules()
                
        except Exception as e:
            logger.error(f"Error loading alert configuration: {e}")
            self.create_default_alert_rules()
    
    def save_configuration(self):
        """Save current alert configuration to file"""
        try:
            config_data = {
                'alert_rules': [
                    {
                        'id': rule.id,
                        'name': rule.name,
                        'alert_type': rule.alert_type.value,
                        'severity': rule.severity.value,
                        'enabled': rule.enabled,
                        'symbol': rule.symbol,
                        'timeframes': rule.timeframes,
                        'indicators': rule.indicators,
                        'thresholds': rule.thresholds,
                        'email_enabled': rule.email_enabled,
                        'webhook_enabled': rule.webhook_enabled,
                        'desktop_enabled': rule.desktop_enabled,
                        'cooldown_minutes': rule.cooldown_minutes,
                        'max_daily_alerts': rule.max_daily_alerts,
                        'description': rule.description,
                        'created_at': rule.created_at,
                        'last_triggered': rule.last_triggered
                    }
                    for rule in self.alert_rules.values()
                ]
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Alert configuration saved")
            
        except Exception as e:
            logger.error(f"Error saving alert configuration: {e}")
    
    def create_default_alert_rules(self):
        """Create default set of alert rules"""
        
        default_rules = [
            # Strong confluence signals
            AlertRule(
                id="strong_confluence_bullish",
                name="Strong Bullish Confluence",
                alert_type=AlertType.CONFLUENCE_SIGNAL,
                severity=AlertSeverity.HIGH,
                thresholds={'confluence_score': 75, 'trend_alignment': 80},
                description="Multiple timeframes showing strong bullish confluence"
            ),
            
            AlertRule(
                id="strong_confluence_bearish", 
                name="Strong Bearish Confluence",
                alert_type=AlertType.CONFLUENCE_SIGNAL,
                severity=AlertSeverity.HIGH,
                thresholds={'confluence_score': 75, 'trend_alignment': 80},
                description="Multiple timeframes showing strong bearish confluence"
            ),
            
            # RSI extreme levels
            AlertRule(
                id="rsi_overbought_extreme",
                name="RSI Extremely Overbought",
                alert_type=AlertType.THRESHOLD_BREACH,
                severity=AlertSeverity.MEDIUM,
                indicators={'rsi': 'required'},
                thresholds={'rsi_min': 80},
                description="RSI above 80 indicating potential reversal"
            ),
            
            AlertRule(
                id="rsi_oversold_extreme",
                name="RSI Extremely Oversold", 
                alert_type=AlertType.THRESHOLD_BREACH,
                severity=AlertSeverity.MEDIUM,
                indicators={'rsi': 'required'},
                thresholds={'rsi_max': 20},
                description="RSI below 20 indicating potential bounce"
            ),
            
            # Volume anomalies
            AlertRule(
                id="volume_spike_high",
                name="High Volume Spike",
                alert_type=AlertType.VOLUME_ANOMALY,
                severity=AlertSeverity.MEDIUM,
                thresholds={'volume_ratio_min': 3.0},
                description="Volume 3x above average - institutional activity"
            ),
            
            # Bollinger Band squeezes
            AlertRule(
                id="bollinger_squeeze",
                name="Bollinger Band Squeeze",
                alert_type=AlertType.PATTERN_FORMATION,
                severity=AlertSeverity.MEDIUM,
                indicators={'bollinger_bands': 'required'},
                thresholds={'bb_width_max': 0.05},
                description="Bollinger Bands squeezing - volatility expansion expected"
            ),
            
            # Support/Resistance breaks
            AlertRule(
                id="support_break",
                name="Support Level Break",
                alert_type=AlertType.SUPPORT_RESISTANCE,
                severity=AlertSeverity.HIGH,
                thresholds={'support_break': True},
                description="Price breaking below key support level"
            ),
            
            AlertRule(
                id="resistance_break",
                name="Resistance Level Break", 
                alert_type=AlertType.SUPPORT_RESISTANCE,
                severity=AlertSeverity.HIGH,
                thresholds={'resistance_break': True},
                description="Price breaking above key resistance level"
            ),
            
            # Trend changes
            AlertRule(
                id="trend_reversal",
                name="Trend Reversal Signal",
                alert_type=AlertType.TREND_CHANGE,
                severity=AlertSeverity.HIGH,
                thresholds={'trend_change': True, 'confirmation_strength': 60},
                description="Multiple timeframes confirming trend reversal"
            )
        ]
        
        for rule in default_rules:
            rule.created_at = datetime.now().isoformat()
            self.alert_rules[rule.id] = rule
        
        self.save_configuration()
        logger.info(f"Created {len(default_rules)} default alert rules")
    
    def setup_default_handlers(self):
        """Setup default notification handlers"""
        
        # Console handler (always available)
        self.notification_handlers['console'] = self._console_handler
        
        # Desktop notification handler
        try:
            # Try to import plyer for desktop notifications
            import plyer
            self.notification_handlers['desktop'] = self._desktop_handler
        except ImportError:
            logger.warning("plyer not available - desktop notifications disabled")
        
        # Email handler setup
        self.setup_email_handler()
        
        logger.info(f"Setup {len(self.notification_handlers)} notification handlers")
    
    def setup_email_handler(self):
        """Setup email notification handler"""
        
        # Check for email configuration in environment variables
        smtp_server = os.getenv('ALERT_SMTP_SERVER')
        smtp_port = os.getenv('ALERT_SMTP_PORT', '587')
        smtp_username = os.getenv('ALERT_EMAIL_USERNAME')
        smtp_password = os.getenv('ALERT_EMAIL_PASSWORD')
        
        if smtp_server and smtp_username and smtp_password:
            self.email_config = {
                'server': smtp_server,
                'port': int(smtp_port),
                'username': smtp_username,
                'password': smtp_password,
                'from_email': smtp_username
            }
            self.notification_handlers['email'] = self._email_handler
            logger.info("Email notifications enabled")
        else:
            logger.info("Email configuration not found - email notifications disabled")
    
    def start_alert_processor(self):
        """Start the alert processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.alert_processor_thread = threading.Thread(
            target=self._alert_processor_worker,
            daemon=True,
            name="AlertProcessor"
        )
        self.alert_processor_thread.start()
        logger.info("Alert processor started")
    
    def stop_alert_processor(self):
        """Stop the alert processing thread"""
        self.is_running = False
        if self.alert_processor_thread:
            self.alert_processor_thread.join(timeout=5)
        logger.info("Alert processor stopped")
    
    def _alert_processor_worker(self):
        """Worker thread for processing alerts"""
        while self.is_running:
            try:
                # Process alerts from queue
                alert = self.alert_queue.get(timeout=1)
                
                # Find matching rules
                matching_rules = self._find_matching_rules(alert)
                
                for rule in matching_rules:
                    if self._should_trigger_alert(rule, alert):
                        self._trigger_alert(rule, alert)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def check_analysis_for_alerts(self, symbol: str, analysis_result: Dict):
        """Check analysis results against alert rules"""
        
        try:
            # Extract key data from analysis
            timeframe_data = analysis_result.get('timeframe_data', {})
            confluence_analysis = analysis_result.get('confluence_analysis', {})
            
            # Create alert data structure
            alert_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_data': timeframe_data,
                'confluence_analysis': confluence_analysis,
                'analysis_result': analysis_result
            }
            
            # Queue alert for processing
            self.alert_queue.put(alert_data)
            
        except Exception as e:
            logger.error(f"Error checking analysis for alerts: {e}")
    
    def _find_matching_rules(self, alert_data: Dict) -> List[AlertRule]:
        """Find alert rules that match the current data"""
        
        matching_rules = []
        symbol = alert_data.get('symbol', '')
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check symbol filter
            if rule.symbol and rule.symbol != symbol:
                continue
            
            # Check if rule conditions are met
            if self._evaluate_rule_conditions(rule, alert_data):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _evaluate_rule_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Evaluate if alert rule conditions are met"""
        
        try:
            if rule.alert_type == AlertType.CONFLUENCE_SIGNAL:
                return self._check_confluence_conditions(rule, alert_data)
            
            elif rule.alert_type == AlertType.THRESHOLD_BREACH:
                return self._check_threshold_conditions(rule, alert_data)
            
            elif rule.alert_type == AlertType.VOLUME_ANOMALY:
                return self._check_volume_conditions(rule, alert_data)
            
            elif rule.alert_type == AlertType.PATTERN_FORMATION:
                return self._check_pattern_conditions(rule, alert_data)
            
            elif rule.alert_type == AlertType.SUPPORT_RESISTANCE:
                return self._check_sr_conditions(rule, alert_data)
            
            elif rule.alert_type == AlertType.TREND_CHANGE:
                return self._check_trend_conditions(rule, alert_data)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions for {rule.id}: {e}")
            return False
    
    def _check_confluence_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check confluence-based alert conditions"""
        
        confluence = alert_data.get('confluence_analysis', {})
        overall_confluence = confluence.get('overall_confluence', {})
        trend_alignment = confluence.get('trend_alignment', {})
        
        confluence_score = overall_confluence.get('confluence_score', 0)
        alignment_percentage = trend_alignment.get('alignment_percentage', 0)
        dominant_trend = trend_alignment.get('dominant_trend', 'NEUTRAL')
        
        # Check thresholds
        min_confluence = rule.thresholds.get('confluence_score', 70)
        min_alignment = rule.thresholds.get('trend_alignment', 70)
        
        if confluence_score >= min_confluence and alignment_percentage >= min_alignment:
            # Check if it's the right direction based on rule name
            if 'bullish' in rule.name.lower() and dominant_trend == 'BULLISH':
                return True
            elif 'bearish' in rule.name.lower() and dominant_trend == 'BEARISH':
                return True
        
        return False
    
    def _check_threshold_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check threshold-based alert conditions"""
        
        timeframe_data = alert_data.get('timeframe_data', {})
        
        for tf_name, tf_data in timeframe_data.items():
            if tf_data.get('status') != 'success':
                continue
            
            indicators = tf_data.get('indicators', {})
            
            # Check RSI thresholds
            if 'rsi' in rule.indicators:
                rsi = indicators.get('rsi')
                if rsi:
                    rsi_min = rule.thresholds.get('rsi_min')
                    rsi_max = rule.thresholds.get('rsi_max')
                    
                    if rsi_min and rsi >= rsi_min:
                        return True
                    if rsi_max and rsi <= rsi_max:
                        return True
        
        return False
    
    def _check_volume_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check volume anomaly conditions"""
        
        timeframe_data = alert_data.get('timeframe_data', {})
        
        for tf_name, tf_data in timeframe_data.items():
            if tf_data.get('status') != 'success':
                continue
            
            indicators = tf_data.get('indicators', {})
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            min_ratio = rule.thresholds.get('volume_ratio_min', 2.0)
            if volume_ratio >= min_ratio:
                return True
        
        return False
    
    def _check_pattern_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check pattern formation conditions"""
        
        timeframe_data = alert_data.get('timeframe_data', {})
        
        for tf_name, tf_data in timeframe_data.items():
            if tf_data.get('status') != 'success':
                continue
            
            indicators = tf_data.get('indicators', {})
            
            # Check Bollinger Band squeeze
            if 'bollinger_bands' in rule.indicators:
                bb_width = indicators.get('bb_width', 1.0)
                max_width = rule.thresholds.get('bb_width_max', 0.05)
                
                if bb_width <= max_width:
                    return True
        
        return False
    
    def _check_sr_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check support/resistance conditions"""
        
        # This would need more sophisticated S/R break detection
        # For now, check if confluence analysis mentions S/R levels
        confluence = alert_data.get('confluence_analysis', {})
        sr_confluence = confluence.get('support_resistance_confluence', {})
        
        confluence_zones = sr_confluence.get('confluence_zones', [])
        return len(confluence_zones) > 0
    
    def _check_trend_conditions(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check trend change conditions"""
        
        confluence = alert_data.get('confluence_analysis', {})
        trend_data = confluence.get('trend_alignment', {})
        
        dominant_trend = trend_data.get('dominant_trend', 'NEUTRAL')
        alignment_percentage = trend_data.get('alignment_percentage', 0)
        
        min_confirmation = rule.thresholds.get('confirmation_strength', 60)
        
        return dominant_trend != 'NEUTRAL' and alignment_percentage >= min_confirmation
    
    def _should_trigger_alert(self, rule: AlertRule, alert_data: Dict) -> bool:
        """Check if alert should be triggered based on cooldown and limits"""
        
        now = datetime.now()
        today = now.date().isoformat()
        
        # Check daily limit
        rule_key = f"{rule.id}_{today}"
        daily_count = self.daily_alert_counts.get(rule_key, 0)
        
        if daily_count >= rule.max_daily_alerts:
            return False
        
        # Check cooldown
        if rule.last_triggered:
            try:
                last_triggered = datetime.fromisoformat(rule.last_triggered)
                cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
                
                if now - last_triggered < cooldown_delta:
                    return False
            except:
                pass
        
        return True
    
    def _trigger_alert(self, rule: AlertRule, alert_data: Dict):
        """Trigger an alert notification"""
        
        try:
            # Create alert instance
            alert = Alert(
                id=f"{rule.id}_{int(time.time())}",
                rule_id=rule.id,
                symbol=alert_data['symbol'],
                timestamp=datetime.now().isoformat(),
                alert_type=rule.alert_type,
                severity=rule.severity,
                title=rule.name,
                message=self._generate_alert_message(rule, alert_data),
                data=alert_data
            )
            
            # Send notifications
            self._send_notifications(rule, alert)
            
            # Update tracking
            rule.last_triggered = alert.timestamp
            
            today = datetime.now().date().isoformat()
            rule_key = f"{rule.id}_{today}"
            self.daily_alert_counts[rule_key] = self.daily_alert_counts.get(rule_key, 0) + 1
            
            # Store alert
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Save configuration
            self.save_configuration()
            
            logger.info(f"Alert triggered: {rule.name} for {alert_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _generate_alert_message(self, rule: AlertRule, alert_data: Dict) -> str:
        """Generate detailed alert message"""
        
        symbol = alert_data['symbol']
        message_parts = [f"🚨 {rule.name} - {symbol}"]
        
        if rule.alert_type == AlertType.CONFLUENCE_SIGNAL:
            confluence = alert_data.get('confluence_analysis', {})
            overall = confluence.get('overall_confluence', {})
            score = overall.get('confluence_score', 0)
            message_parts.append(f"Confluence Score: {score:.0f}%")
        
        elif rule.alert_type == AlertType.THRESHOLD_BREACH:
            # Add specific threshold details
            timeframe_data = alert_data.get('timeframe_data', {})
            for tf_name, tf_data in timeframe_data.items():
                if 'indicators' in tf_data:
                    indicators = tf_data['indicators']
                    rsi = indicators.get('rsi')
                    if rsi:
                        message_parts.append(f"{tf_name} RSI: {rsi:.1f}")
                        break
        
        message_parts.append(f"Severity: {rule.severity.value.upper()}")
        message_parts.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return " | ".join(message_parts)
    
    def _send_notifications(self, rule: AlertRule, alert: Alert):
        """Send alert notifications via enabled channels"""
        
        try:
            # Console notification (always enabled)
            if 'console' in self.notification_handlers:
                self.notification_handlers['console'](alert)
            
            # Desktop notification
            if rule.desktop_enabled and 'desktop' in self.notification_handlers:
                self.notification_handlers['desktop'](alert)
            
            # Email notification
            if rule.email_enabled and 'email' in self.notification_handlers:
                self.notification_handlers['email'](alert)
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _console_handler(self, alert: Alert):
        """Console notification handler"""
        severity_icons = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "💥"
        }
        
        icon = severity_icons.get(alert.severity, "📢")
        print(f"\n{icon} ALERT: {alert.message}")
    
    def _desktop_handler(self, alert: Alert):
        """Desktop notification handler"""
        try:
            import plyer
            plyer.notification.notify(
                title=f"Crypto Alert - {alert.severity.value.upper()}",
                message=alert.message,
                timeout=10
            )
        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
    
    def _email_handler(self, alert: Alert):
        """Email notification handler"""
        try:
            if not hasattr(self, 'email_config'):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['from_email']  # Send to self for now
            msg['Subject'] = f"Crypto Alert - {alert.title}"
            
            body = f"""
Alert Details:
- Symbol: {alert.symbol}
- Type: {alert.alert_type.value}
- Severity: {alert.severity.value}
- Time: {alert.timestamp}

Message: {alert.message}

This is an automated alert from your crypto analyzer.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        try:
            rule.created_at = datetime.now().isoformat()
            self.alert_rules[rule.id] = rule
            self.save_configuration()
            logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding alert rule: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_id in self.alert_rules:
                rule_name = self.alert_rules[rule_id].name
                del self.alert_rules[rule_id]
                self.save_configuration()
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing alert rule: {e}")
            return False
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        
        today = datetime.now().date().isoformat()
        today_alerts = [a for a in self.alert_history if a.timestamp.startswith(today)]
        
        return {
            'total_rules': len(self.alert_rules),
            'enabled_rules': sum(1 for r in self.alert_rules.values() if r.enabled),
            'active_alerts': len(self.active_alerts),
            'total_historical_alerts': len(self.alert_history),
            'today_alerts': len(today_alerts),
            'notification_handlers': list(self.notification_handlers.keys()),
            'is_running': self.is_running
        }

if __name__ == "__main__":
    # Demo usage
    alert_manager = AlertManager()
    alert_manager.start_alert_processor()
    
    print("Alert System Demo")
    print("=================")
    print(f"Statistics: {alert_manager.get_alert_statistics()}")
    print(f"Alert Rules: {len(alert_manager.alert_rules)}")
    
    for rule_id, rule in alert_manager.alert_rules.items():
        print(f"- {rule.name} ({rule.severity.value}) - {'Enabled' if rule.enabled else 'Disabled'}")
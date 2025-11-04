"""
LSTM Volatility Predictor for Options
Predicts volatility for AAPL and TSLA options matching LastQModel.py portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    print("   Install with: pip install tensorflow scikit-learn")
    TENSORFLOW_AVAILABLE = False


class VolatilityDataGenerator:
    """Generate synthetic volatility time series data for training"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_realistic_volatility(self, 
                                      base_vol: float,
                                      n_days: int = 500,
                                      mean_reversion_speed: float = 0.05,
                                      vol_of_vol: float = 0.15) -> np.ndarray:
        """
        Generate realistic volatility using mean-reverting process (Ornstein-Uhlenbeck)
        
        dσ_t = κ(θ - σ_t)dt + ν√σ_t dW_t
        
        Args:
            base_vol: Long-term mean volatility level
            n_days: Number of days to simulate
            mean_reversion_speed: Speed of mean reversion (κ)
            vol_of_vol: Volatility of volatility (ν)
        """
        dt = 1.0  # Daily steps
        volatilities = np.zeros(n_days)
        volatilities[0] = base_vol
        
        for t in range(1, n_days):
            # Mean reversion term
            drift = mean_reversion_speed * (base_vol - volatilities[t-1]) * dt
            
            # Stochastic term (volatility clustering)
            diffusion = vol_of_vol * np.sqrt(volatilities[t-1]) * np.sqrt(dt) * np.random.randn()
            
            # Update with floor to prevent negative volatility
            volatilities[t] = max(volatilities[t-1] + drift + diffusion, 0.05)
        
        return volatilities
    
    def add_market_events(self, volatilities: np.ndarray, 
                         n_events: int = 5) -> np.ndarray:
        """Add sudden volatility spikes to simulate market events"""
        vol_with_events = volatilities.copy()
        n_days = len(volatilities)
        
        for _ in range(n_events):
            # Random event timing
            event_day = np.random.randint(50, n_days - 50)
            spike_magnitude = np.random.uniform(1.3, 2.0)  # 30-100% spike
            decay_rate = np.random.uniform(0.05, 0.15)
            
            # Create spike with exponential decay
            for i in range(event_day, min(event_day + 30, n_days)):
                days_since_event = i - event_day
                spike_factor = spike_magnitude * np.exp(-decay_rate * days_since_event)
                vol_with_events[i] *= spike_factor
        
        return vol_with_events
    
    def create_features(self, volatilities: np.ndarray, 
                       lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature sequences for LSTM training
        
        Args:
            volatilities: Time series of volatility values
            lookback: Number of past days to use as features
            
        Returns:
            X: Feature sequences (n_samples, lookback, 1)
            y: Target values (n_samples,)
        """
        X, y = [], []
        
        for i in range(lookback, len(volatilities)):
            X.append(volatilities[i-lookback:i])
            y.append(volatilities[i])
        
        return np.array(X), np.array(y)
    
    def generate_training_data(self, 
                              assets: Dict[str, float],
                              n_days: int = 500,
                              lookback: int = 20) -> Dict[str, Dict]:
        """
        Generate complete training dataset for multiple assets
        
        Args:
            assets: Dict of asset_name -> base_volatility
            n_days: Number of days to simulate
            lookback: Lookback window for LSTM
            
        Returns:
            Dict containing data for each asset
        """
        datasets = {}
        
        for asset_name, base_vol in assets.items():
            print(f"  📊 Generating data for {asset_name} (base vol: {base_vol:.2%})...")
            
            # Generate volatility time series
            vol_series = self.generate_realistic_volatility(
                base_vol=base_vol,
                n_days=n_days,
                mean_reversion_speed=0.05,
                vol_of_vol=0.15
            )
            
            # Add market events
            vol_series = self.add_market_events(vol_series, n_events=5)
            
            # Create features
            X, y = self.create_features(vol_series, lookback=lookback)
            
            datasets[asset_name] = {
                'volatilities': vol_series,
                'X': X,
                'y': y,
                'base_vol': base_vol
            }
            
            print(f"     Generated {len(X)} training samples")
        
        return datasets


class LSTMVolatilityPredictor:
    """LSTM model for volatility prediction"""
    
    def __init__(self, lookback: int = 20, lstm_units: List[int] = [128, 64, 32]):
        """
        Initialize LSTM model
        
        Args:
            lookback: Number of past days to use
            lstm_units: List of LSTM layer sizes
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential(name='VolatilityLSTM')
        
        # Input layer
        model.add(Input(shape=(self.lookback, 1)))
        
        # LSTM layers with dropout for regularization
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            model.add(LSTM(units=units, 
                          return_sequences=return_sequences,
                          name=f'lstm_{i+1}'))
            model.add(Dropout(0.2, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(Dense(1, activation='linear', name='output'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("\n🧠 LSTM Model Architecture:")
        print("=" * 60)
        model.summary()
        print("=" * 60)
        
        return model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Tuple:
        """
        Scale and split data
        
        Args:
            X: Feature sequences
            y: Target values
            test_size: Fraction for testing
            
        Returns:
            X_train, X_test, y_train, y_test (all scaled)
        """
        # Reshape for scaling
        X_reshaped = X.reshape(-1, 1)
        y_reshaped = y.reshape(-1, 1)
        
        # Fit scaler on training data only
        X_scaled = self.scaler.fit_transform(X_reshaped)
        y_scaled = self.scaler.transform(y_reshaped)
        
        # Reshape back
        X_scaled = X_scaled.reshape(X.shape)
        y_scaled = y_scaled.flatten()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> keras.callbacks.History:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        print(f"\n🚀 Training LSTM model...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature sequences (scaled)
            
        Returns:
            Predictions (original scale)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features (scaled)
            y_test: Test targets (scaled)
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test_original - y_pred_original) ** 2)
        mae = np.mean(np.abs(y_test_original - y_pred_original))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }


def plot_results(datasets: Dict[str, Dict], 
                models: Dict[str, LSTMVolatilityPredictor],
                predictions: Dict[str, Dict],
                output_path: str = "volatility_predictions.png"):
    """
    Plot predicted vs actual volatility for all assets
    
    Args:
        datasets: Generated data for each asset
        models: Trained models for each asset
        predictions: Predictions for each asset
        output_path: Path to save plot
    """
    n_assets = len(datasets)
    fig, axes = plt.subplots(n_assets, 2, figsize=(16, 5 * n_assets))
    
    if n_assets == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('LSTM Volatility Prediction Results', fontsize=16, fontweight='bold')
    
    for idx, (asset_name, data) in enumerate(datasets.items()):
        model = models[asset_name]
        preds = predictions[asset_name]
        
        # Plot 1: Time series comparison
        ax1 = axes[idx, 0]
        
        # Full volatility series
        ax1.plot(data['volatilities'], label='Actual Volatility', 
                color='#2E86AB', linewidth=1.5, alpha=0.7)
        
        # Training region
        train_end = len(preds['y_train'])
        ax1.axvspan(0, train_end + model.lookback, alpha=0.1, color='green', 
                   label='Training Region')
        
        # Test predictions
        test_start = train_end + model.lookback
        test_indices = np.arange(test_start, test_start + len(preds['y_test']))
        ax1.plot(test_indices, preds['y_test'], 'o', markersize=4,
                label='Test Actual', color='#2E86AB', alpha=0.8)
        ax1.plot(test_indices, preds['y_pred'], 's', markersize=4,
                label='Test Predicted', color='#A23B72', alpha=0.8)
        
        # Reference line for base volatility
        ax1.axhline(y=data['base_vol'], color='gray', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Base Vol: {data["base_vol"]:.2%}')
        
        ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax1.set_title(f'{asset_name} - Volatility Time Series', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Plot 2: Scatter plot (Predicted vs Actual)
        ax2 = axes[idx, 1]
        
        metrics = preds['metrics']
        
        # Scatter plot
        ax2.scatter(preds['y_test'], preds['y_pred'], 
                   alpha=0.6, s=30, color='#A23B72', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(preds['y_test'].min(), preds['y_pred'].min())
        max_val = max(preds['y_test'].max(), preds['y_pred'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
        
        # Add metrics text box
        metrics_text = f"RMSE: {metrics['rmse']:.4f}\n"
        metrics_text += f"MAE: {metrics['mae']:.4f}\n"
        metrics_text += f"MAPE: {metrics['mape']:.2f}%"
        
        ax2.text(0.05, 0.95, metrics_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('Actual Volatility', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted Volatility', fontsize=11, fontweight='bold')
        ax2.set_title(f'{asset_name} - Prediction Accuracy', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Prediction plot saved: {output_path}")
        plt.close()
        return output_path
    except Exception as e:
        print(f"⚠️  Failed to save plot: {e}")
        plt.close()
        return None


def plot_training_history(histories: Dict[str, keras.callbacks.History],
                          output_path: str = "training_history.png"):
    """
    Plot training history (loss curves) for all assets
    
    Args:
        histories: Training histories for each asset
        output_path: Path to save plot
    """
    n_assets = len(histories)
    fig, axes = plt.subplots(1, n_assets, figsize=(6 * n_assets, 5))
    
    if n_assets == 1:
        axes = [axes]
    
    fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
    
    for idx, (asset_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        
        # Plot training & validation loss
        ax.plot(history.history['loss'], label='Training Loss', 
               color='#2E86AB', linewidth=2, alpha=0.8)
        ax.plot(history.history['val_loss'], label='Validation Loss', 
               color='#A23B72', linewidth=2, alpha=0.8)
        
        # Mark best epoch
        best_epoch = np.argmin(history.history['val_loss'])
        best_val_loss = history.history['val_loss'][best_epoch]
        ax.scatter(best_epoch, best_val_loss, color='red', s=100, 
                  zorder=5, marker='*', label=f'Best Epoch: {best_epoch+1}')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
        ax.set_title(f'{asset_name} - Training Progress', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visibility
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved: {output_path}")
        plt.close()
        return output_path
    except Exception as e:
        print(f"⚠️  Failed to save training history plot: {e}")
        plt.close()
        return None


def main():
    """Main execution pipeline"""
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install:")
        print("   pip install tensorflow scikit-learn")
        return
    
    print("\n" + "=" * 80)
    print("🧠 LSTM VOLATILITY PREDICTOR FOR OPTIONS")
    print("=" * 80)
    print("Matching portfolio from LastQModel.py:")
    print("  • AAPL: Long Call (volatility: 22%)")
    print("  • TSLA: Short Put (volatility: 51%)")
    print("=" * 80 + "\n")
    
    # Step 1: Generate training data
    print("📊 STEP 1: Generating synthetic volatility data...")
    print("-" * 80)
    
    assets = {
        'AAPL': 0.22,  # 22% base volatility
        'TSLA': 0.51   # 51% base volatility
    }
    
    generator = VolatilityDataGenerator(seed=42)
    datasets = generator.generate_training_data(
        assets=assets,
        n_days=500,
        lookback=20
    )
    
    print("✅ Data generation complete!\n")
    
    # Step 2: Train LSTM models
    print("🚀 STEP 2: Training LSTM models...")
    print("-" * 80)
    
    models = {}
    predictions = {}
    histories = {}
    
    for asset_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"Training model for {asset_name}")
        print(f"{'='*60}")
        
        # Initialize model
        model = LSTMVolatilityPredictor(
            lookback=20,
            lstm_units=[128, 64, 32]
        )
        
        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(
            data['X'], data['y'], test_size=0.2
        )
        
        # Train
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=100,
            batch_size=32
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        print(f"\n✅ Training complete for {asset_name}!")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_test_original = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Store results
        models[asset_name] = model
        histories[asset_name] = history
        predictions[asset_name] = {
            'y_train': y_train,
            'y_test': y_test_original,
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    print(f"\n{'='*80}")
    print("✅ All models trained successfully!")
    print(f"{'='*80}\n")
    
    # Step 3: Visualize results
    print("📈 STEP 3: Generating visualizations...")
    print("-" * 80)
    
    plot_results(datasets, models, predictions)
    plot_training_history(histories)
    
    print("\n" + "=" * 80)
    print("🎉 LSTM Volatility Prediction Complete!")
    print("=" * 80)
    print("\nSummary:")
    for asset_name, pred in predictions.items():
        metrics = pred['metrics']
        print(f"\n{asset_name}:")
        print(f"  • RMSE: {metrics['rmse']:.4f}")
        print(f"  • MAE: {metrics['mae']:.4f}")
        print(f"  • MAPE: {metrics['mape']:.2f}%")
    
    print("\n📁 Generated files:")
    print("  • volatility_predictions.png - Prediction vs Actual comparison")
    print("  • training_history.png - Training loss curves")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

import bquant as bq
import pandas as pd
from datetime import datetime, timedelta
import logging

class MSCIEmergingMarketsETFTracker:
    """Track and calculate total AUM for ETFs following MSCI Emerging Markets Index."""
    
    def __init__(self):
        self.etf_tickers = {
            'EEM': 'iShares MSCI Emerging Markets ETF',
            'IEMG': 'iShares Core MSCI Emerging Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF'
        }
        self.logger = logging.getLogger(__name__)
        
    def fetch_aum_data(self, lookback_days=7):
        """
        Fetch AUM data for tracked ETFs using BQuant API.
        
        Args:
            lookback_days (int): Number of days to look back for latest AUM data
            
        Returns:
            dict: Dictionary containing latest AUM values for each ETF
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            aum_data = {}
            
            for ticker in self.etf_tickers:
                # Using BQuant's data API to fetch AUM data
                query = f"""
                SELECT date, aum 
                FROM etf.fund_flows 
                WHERE ticker = '{ticker}'
                AND date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
                ORDER BY date DESC
                LIMIT 1
                """
                
                result = bq.query(query)
                if not result.empty:
                    aum_data[ticker] = {
                        'name': self.etf_tickers[ticker],
                        'aum': float(result['aum'].iloc[0]),
                        'date': result['date'].iloc[0]
                    }
                
            return aum_data
            
        except Exception as e:
            self.logger.error(f"Error fetching AUM data: {str(e)}")
            raise
            
    def calculate_total_aum(self):
        """
        Calculate total AUM across all tracked ETFs.
        
        Returns:
            tuple: (total_aum, breakdown_dict)
                - total_aum: float representing total AUM
                - breakdown_dict: dictionary with AUM breakdown by ETF
        """
        try:
            aum_data = self.fetch_aum_data()
            total_aum = sum(etf_info['aum'] for etf_info in aum_data.values())
            
            # Calculate percentage breakdown
            breakdown = {
                ticker: {
                    'name': info['name'],
                    'aum': info['aum'],
                    'percentage': (info['aum'] / total_aum) * 100 if total_aum > 0 else 0,
                    'date': info['date']
                }
                for ticker, info in aum_data.items()
            }
            
            return total_aum, breakdown
            
        except Exception as e:
            self.logger.error(f"Error calculating total AUM: {str(e)}")
            raise
            
    def generate_report(self):
        """
        Generate a formatted report of ETF AUM data.
        
        Returns:
            str: Formatted report string
        """
        try:
            total_aum, breakdown = self.calculate_total_aum()
            
            report = []
            report.append("MSCI Emerging Markets ETF - AUM Report")
            report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report.append(f"\nTotal AUM: ${total_aum:,.2f}")
            report.append("\nBreakdown by ETF:")
            
            for ticker, data in breakdown.items():
                report.append(f"\n{ticker} - {data['name']}")
                report.append(f"AUM: ${data['aum']:,.2f}")
                report.append(f"Percentage of Total: {data['percentage']:.2f}%")
                report.append(f"Data as of: {data['date']}")
                
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise


def main():
    """Main function to demonstrate usage."""
    tracker = MSCIEmergingMarketsETFTracker()
    print(tracker.generate_report())

if __name__ == "__main__":
    main()
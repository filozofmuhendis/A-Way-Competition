"""
Geliştirilmiş Fuzzy Logic Kural Tablosu Üretici
Bu modül tüm olasılıklar için sistematik kural tablosu oluşturur
"""

import itertools
import pandas as pd
import numpy as np

class EnhancedFuzzyRuleGenerator:
    """Kapsamlı fuzzy logic kuralları üretici"""
    
    def __init__(self):
        # Giriş değişkenleri ve seviyeleri
        self.input_variables = {
            'snr_transit': ['düşük', 'orta', 'yüksek'],
            'beta_factor': ['kötü', 'orta', 'iyi'], 
            'depth_consistency': ['düşük', 'orta', 'yüksek'],
            'duty_cycle': ['düşük', 'orta', 'yüksek'],
            'odd_even_diff': ['düşük', 'orta', 'yüksek']
        }
        
        # Çıkış seviyeleri
        self.output_levels = ['çok_düşük', 'düşük', 'orta', 'yüksek', 'çok_yüksek']
        
        # Öznitelik ağırlıkları (exoplanet tespiti için önem sırası)
        self.feature_weights = {
            'snr_transit': 0.25,        # En önemli - sinyal gücü
            'depth_consistency': 0.25,   # Çok önemli - sinyal kararlılığı
            'beta_factor': 0.20,        # Önemli - veri kalitesi
            'duty_cycle': 0.15,         # Orta - geometrik tutarlılık
            'odd_even_diff': 0.15       # Orta - düzenlilik
        }
        
        # Seviye skorları (0-1 arası)
        self.level_scores = {
            'düşük': 0.2, 'kötü': 0.2,
            'orta': 0.5,
            'yüksek': 0.8, 'iyi': 0.8
        }
        
    def calculate_rule_score(self, rule_combination):
        """Kural kombinasyonu için skor hesapla"""
        total_score = 0.0
        
        for feature, level in rule_combination.items():
            feature_score = self.level_scores[level]
            weight = self.feature_weights[feature]
            total_score += feature_score * weight
            
        return total_score
    
    def score_to_output_level(self, score):
        """Skoru çıkış seviyesine dönüştür"""
        if score <= 0.2:
            return 'çok_düşük'
        elif score <= 0.4:
            return 'düşük'
        elif score <= 0.6:
            return 'orta'
        elif score <= 0.8:
            return 'yüksek'
        else:
            return 'çok_yüksek'
    
    def generate_all_rules(self):
        """Tüm olası kuralları üret"""
        all_rules = []
        
        # Tüm kombinasyonları üret
        feature_names = list(self.input_variables.keys())
        all_combinations = itertools.product(*[self.input_variables[feature] for feature in feature_names])
        
        for combination in all_combinations:
            # Kombinasyonu sözlük formatına çevir
            rule_dict = dict(zip(feature_names, combination))
            
            # Skoru hesapla
            score = self.calculate_rule_score(rule_dict)
            
            # Çıkış seviyesini belirle
            output_level = self.score_to_output_level(score)
            
            # Kuralı kaydet
            rule_info = {
                'inputs': rule_dict,
                'output': output_level,
                'score': score,
                'rule_text': self.format_rule_text(rule_dict, output_level)
            }
            
            all_rules.append(rule_info)
        
        return all_rules
    
    def format_rule_text(self, inputs, output):
        """Kural metnini formatla"""
        conditions = []
        for feature, level in inputs.items():
            conditions.append(f"self.{feature}['{level}']")
        
        condition_text = " & ".join(conditions)
        rule_text = f"ctrl.Rule({condition_text}, self.exoplanet_probability['{output}'])"
        
        return rule_text
    
    def generate_optimized_rules(self, max_rules=100):
        """Optimize edilmiş kural seti üret"""
        all_rules = self.generate_all_rules()
        
        # Skorlara göre sırala
        sorted_rules = sorted(all_rules, key=lambda x: abs(x['score'] - 0.5), reverse=True)
        
        # En karakteristik kuralları seç
        selected_rules = []
        covered_combinations = set()
        
        for rule in sorted_rules:
            rule_signature = tuple(sorted(rule['inputs'].items()))
            
            if rule_signature not in covered_combinations:
                selected_rules.append(rule)
                covered_combinations.add(rule_signature)
                
                if len(selected_rules) >= max_rules:
                    break
        
        return selected_rules
    
    def generate_rule_categories(self):
        """Kural kategorilerini üret"""
        all_rules = self.generate_all_rules()
        
        categories = {
            'çok_yüksek': [],
            'yüksek': [],
            'orta': [],
            'düşük': [],
            'çok_düşük': []
        }
        
        for rule in all_rules:
            categories[rule['output']].append(rule)
        
        return categories
    
    def export_rules_to_dataframe(self):
        """Kuralları DataFrame formatında dışa aktar"""
        all_rules = self.generate_all_rules()
        
        rows = []
        for rule in all_rules:
            row = rule['inputs'].copy()
            row['output'] = rule['output']
            row['score'] = rule['score']
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_balanced_rules(self, rules_per_category=20):
        """Her kategori için dengeli kural seti üret"""
        categories = self.generate_rule_categories()
        balanced_rules = []
        
        for category, rules in categories.items():
            # Her kategoriden en iyi kuralları seç
            sorted_category_rules = sorted(rules, key=lambda x: x['score'], reverse=(category in ['yüksek', 'çok_yüksek']))
            selected = sorted_category_rules[:rules_per_category]
            balanced_rules.extend(selected)
        
        return balanced_rules

def main():
    """Test fonksiyonu"""
    generator = EnhancedFuzzyRuleGenerator()
    
    # Tüm kuralları üret
    all_rules = generator.generate_all_rules()
    print(f"Toplam kural sayısı: {len(all_rules)}")
    
    # Kategori dağılımını göster
    categories = generator.generate_rule_categories()
    for category, rules in categories.items():
        print(f"{category}: {len(rules)} kural")
    
    # Optimize edilmiş kuralları üret
    optimized_rules = generator.generate_optimized_rules(50)
    print(f"\nOptimize edilmiş kural sayısı: {len(optimized_rules)}")
    
    # Dengeli kuralları üret
    balanced_rules = generator.generate_balanced_rules(15)
    print(f"Dengeli kural sayısı: {len(balanced_rules)}")
    
    return generator

if __name__ == "__main__":
    generator = main()
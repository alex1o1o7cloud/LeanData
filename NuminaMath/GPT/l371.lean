import Mathlib

namespace marie_eggs_total_l371_37198

variable (x : ℕ) -- Number of eggs in each box

-- Conditions as definitions
def egg_weight := 10 -- weight of each egg in ounces
def total_boxes := 4 -- total number of boxes
def remaining_boxes := 3 -- boxes left after one is discarded
def remaining_weight := 90 -- total weight of remaining eggs in ounces

-- Proof statement
theorem marie_eggs_total : remaining_boxes * egg_weight * x = remaining_weight → total_boxes * x = 12 :=
by
  intros h
  sorry

end marie_eggs_total_l371_37198


namespace value_of_k_if_two_equal_real_roots_l371_37171

theorem value_of_k_if_two_equal_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k = 0 → x^2 - 2 * x + k = 0) → k = 1 :=
by
  sorry

end value_of_k_if_two_equal_real_roots_l371_37171


namespace subset_of_difference_empty_l371_37138

theorem subset_of_difference_empty {α : Type*} (A B : Set α) :
  (A \ B = ∅) → (A ⊆ B) :=
by
  sorry

end subset_of_difference_empty_l371_37138


namespace gobblean_total_words_l371_37107

-- Define the Gobblean alphabet and its properties.
def gobblean_letters := 6
def max_word_length := 4

-- Function to calculate number of permutations without repetition for a given length.
def num_words (length : ℕ) : ℕ :=
  if length = 1 then 6
  else if length = 2 then 6 * 5
  else if length = 3 then 6 * 5 * 4
  else if length = 4 then 6 * 5 * 4 * 3
  else 0

-- Main theorem stating the total number of possible words.
theorem gobblean_total_words : 
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4) = 516 :=
by
  -- Proof is not required
  sorry

end gobblean_total_words_l371_37107


namespace sphere_volume_l371_37103

theorem sphere_volume (C : ℝ) (h : C = 30) : 
  ∃ (V : ℝ), V = 4500 / (π^2) :=
by sorry

end sphere_volume_l371_37103


namespace chemist_mixing_solution_l371_37106

theorem chemist_mixing_solution (x : ℝ) : 0.30 * x = 0.20 * (x + 1) → x = 2 :=
by
  intro h
  sorry

end chemist_mixing_solution_l371_37106


namespace base3_to_base10_conversion_l371_37130

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l371_37130


namespace final_balance_l371_37191

noncomputable def initial_balance : ℕ := 10
noncomputable def charity_donation : ℕ := 4
noncomputable def prize_amount : ℕ := 90
noncomputable def lost_at_first_slot : ℕ := 50
noncomputable def lost_at_second_slot : ℕ := 10
noncomputable def lost_at_last_slot : ℕ := 5
noncomputable def cost_of_water : ℕ := 1
noncomputable def cost_of_lottery_ticket : ℕ := 1
noncomputable def lottery_win : ℕ := 65

theorem final_balance : 
  initial_balance - charity_donation + prize_amount - (lost_at_first_slot + lost_at_second_slot + lost_at_last_slot) - (cost_of_water + cost_of_lottery_ticket) + lottery_win = 94 := 
by 
  -- This is the lean statement, the proof is not required as per instructions.
  sorry

end final_balance_l371_37191


namespace problem_statement_l371_37127

-- Define the operation * based on the given mathematical definition
def op (a b : ℕ) : ℤ := a * (a - b)

-- The core theorem to prove the expression in the problem
theorem problem_statement : op 2 3 + op (6 - 2) 4 = -2 :=
by
  -- This is where the proof would go, but it's omitted with sorry.
  sorry

end problem_statement_l371_37127


namespace cider_apples_production_l371_37120

def apples_total : Real := 8.0
def baking_fraction : Real := 0.30
def cider_fraction : Real := 0.60

def apples_remaining : Real := apples_total * (1 - baking_fraction)
def apples_for_cider : Real := apples_remaining * cider_fraction

theorem cider_apples_production : 
    apples_for_cider = 3.4 := 
by
  sorry

end cider_apples_production_l371_37120


namespace gcd_min_value_l371_37183

-- Definitions of the conditions
def is_positive_integer (x : ℕ) := x > 0

def gcd_cond (m n : ℕ) := Nat.gcd m n = 18

-- The main theorem statement
theorem gcd_min_value (m n : ℕ) (hm : is_positive_integer m) (hn : is_positive_integer n) (hgcd : gcd_cond m n) : 
  Nat.gcd (12 * m) (20 * n) = 72 :=
sorry

end gcd_min_value_l371_37183


namespace value_of_expression_l371_37146

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l371_37146


namespace vlad_score_l371_37153

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l371_37153


namespace y_intercept_of_line_b_l371_37178

theorem y_intercept_of_line_b
  (m : ℝ) (c₁ : ℝ) (c₂ : ℝ) (x₁ : ℝ) (y₁ : ℝ)
  (h_parallel : m = 3/2)
  (h_point : (4, 2) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + c₂ }) :
  c₂ = -4 := by
  sorry

end y_intercept_of_line_b_l371_37178


namespace sum_of_coefficients_l371_37140

theorem sum_of_coefficients:
  (x^3 + 2*x + 1) * (3*x^2 + 4) = 28 :=
by
  sorry

end sum_of_coefficients_l371_37140


namespace peacocks_in_zoo_l371_37149

theorem peacocks_in_zoo :
  ∃ p t : ℕ, 2 * p + 4 * t = 54 ∧ p + t = 17 ∧ p = 7 :=
by
  sorry

end peacocks_in_zoo_l371_37149


namespace final_sale_price_l371_37179

theorem final_sale_price (P P₁ P₂ P₃ : ℝ) (d₁ d₂ d₃ dx : ℝ) (x : ℝ)
  (h₁ : P = 600) 
  (h_d₁ : d₁ = 20) (h_d₂ : d₂ = 15) (h_d₃ : d₃ = 10)
  (h₁₁ : P₁ = P * (1 - d₁ / 100))
  (h₁₂ : P₂ = P₁ * (1 - d₂ / 100))
  (h₁₃ : P₃ = P₂ * (1 - d₃ / 100))
  (h_P₃_final : P₃ = 367.2) :
  P₃ * (100 - dx) / 100 = 367.2 * (100 - x) / 100 :=
by
  sorry

end final_sale_price_l371_37179


namespace twelfth_term_l371_37122

-- Definitions based on the given conditions
def a_3_condition (a d : ℚ) : Prop := a + 2 * d = 10
def a_6_condition (a d : ℚ) : Prop := a + 5 * d = 20

-- The main theorem stating that the twelfth term is 40
theorem twelfth_term (a d : ℚ) (h1 : a_3_condition a d) (h2 : a_6_condition a d) :
  a + 11 * d = 40 :=
sorry

end twelfth_term_l371_37122


namespace linear_regression_neg_corr_l371_37126

-- Given variables x and y with certain properties
variables (x y : ℝ)

-- Conditions provided in the problem
def neg_corr (x y : ℝ) : Prop := ∀ a b : ℝ, (a < b → x * y < 0)
def sample_mean_x := (2 : ℝ)
def sample_mean_y := (1.5 : ℝ)

-- Statement to prove the linear regression equation
theorem linear_regression_neg_corr (h1 : neg_corr x y)
    (hx : sample_mean_x = 2)
    (hy : sample_mean_y = 1.5) : 
    ∃ b0 b1 : ℝ, b0 = 5.5 ∧ b1 = -2 ∧ y = b0 + b1 * x :=
sorry

end linear_regression_neg_corr_l371_37126


namespace evaluate_expression_l371_37135

theorem evaluate_expression : 3 + 2 * (8 - 3) = 13 := by
  sorry

end evaluate_expression_l371_37135


namespace explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l371_37194

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x + (a - 1)

-- Proof needed for the first question:
theorem explicit_formula_is_even (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) → a = 2 ∧ ∀ x : ℝ, f x a = x^2 + 1 :=
by sorry

-- Proof needed for the second question:
theorem tangent_line_at_1 (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 :=
by sorry

-- The tangent line equation at x = 1 in the required form
theorem tangent_line_equation (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 → (f 1 - deriv f 1 * 1 + deriv f 1 * x = 2 * x) :=
by sorry

end explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l371_37194


namespace total_games_in_season_l371_37137

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem total_games_in_season
  (teams : ℕ)
  (games_per_pair : ℕ)
  (h_teams : teams = 30)
  (h_games_per_pair : games_per_pair = 6) :
  (choose 30 2 * games_per_pair) = 2610 :=
  by
    sorry

end total_games_in_season_l371_37137


namespace scores_fraction_difference_l371_37151

theorem scores_fraction_difference (y : ℕ) (white_ratio : ℕ) (black_ratio : ℕ) (total : ℕ) 
(h1 : white_ratio = 7) (h2 : black_ratio = 6) (h3 : total = 78) 
(h4 : y = white_ratio + black_ratio) : 
  ((white_ratio * total / y) - (black_ratio * total / y)) / total = 1 / 13 :=
by
 sorry

end scores_fraction_difference_l371_37151


namespace square_triangle_ratios_l371_37112

theorem square_triangle_ratios (s t : ℝ) 
  (P_s := 4 * s) 
  (R_s := s * Real.sqrt 2 / 2)
  (P_t := 3 * t) 
  (R_t := t * Real.sqrt 3 / 3) 
  (h : s = t) : 
  (P_s / P_t = 4 / 3) ∧ (R_s / R_t = Real.sqrt 6 / 2) := 
by
  sorry

end square_triangle_ratios_l371_37112


namespace length_of_symmedian_l371_37175

theorem length_of_symmedian (a b c : ℝ) (AS : ℝ) :
  AS = (2 * b * c^2) / (b^2 + c^2) := sorry

end length_of_symmedian_l371_37175


namespace simplify_fraction_l371_37109

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l371_37109


namespace fraction_subtraction_l371_37105

theorem fraction_subtraction : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 :=
  sorry

end fraction_subtraction_l371_37105


namespace power_multiplication_l371_37163

theorem power_multiplication : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end power_multiplication_l371_37163


namespace five_digit_numbers_l371_37118

def divisible_by_4_and_9 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0)

def is_candidate (n : ℕ) : Prop :=
  ∃ a b, n = 10000 * a + 1000 + 200 + 30 + b ∧ a < 10 ∧ b < 10

theorem five_digit_numbers :
  ∀ (n : ℕ), is_candidate n → divisible_by_4_and_9 n → n = 11232 ∨ n = 61236 :=
by
  sorry

end five_digit_numbers_l371_37118


namespace indoor_players_count_l371_37148

theorem indoor_players_count (T O B I : ℕ) 
  (hT : T = 400) 
  (hO : O = 350) 
  (hB : B = 60) 
  (hEq : T = (O - B) + (I - B) + B) : 
  I = 110 := 
by sorry

end indoor_players_count_l371_37148


namespace range_of_a_l371_37150

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) : a > -1 := 
by
  sorry

end range_of_a_l371_37150


namespace trisha_take_home_pay_l371_37123

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l371_37123


namespace heartsuit_fraction_l371_37116

-- Define the operation heartsuit
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Define the proof statement
theorem heartsuit_fraction :
  (heartsuit 2 4) / (heartsuit 4 2) = 2 :=
by
  -- We use 'sorry' to skip the actual proof steps
  sorry

end heartsuit_fraction_l371_37116


namespace kirill_is_62_5_l371_37167

variable (K : ℝ)

def kirill_height := K
def brother_height := K + 14
def sister_height := 2 * K
def total_height := K + (K + 14) + 2 * K

theorem kirill_is_62_5 (h1 : total_height K = 264) : K = 62.5 := by
  sorry

end kirill_is_62_5_l371_37167


namespace find_number_l371_37166

variable (number x : ℝ)

theorem find_number (h1 : number * x = 1600) (h2 : x = -8) : number = -200 := by
  sorry

end find_number_l371_37166


namespace number_of_classes_l371_37164

variable (s : ℕ) (h_s : s > 0)
-- Define the conditions
def student_books_year : ℕ := 4 * 12
def total_books_read : ℕ := 48
def class_books_year (s : ℕ) : ℕ := s * student_books_year
def total_classes (c s : ℕ) (h_s : s > 0) : ℕ := 1

-- Define the main theorem
theorem number_of_classes (h : total_books_read = 48) (h_s : s > 0)
  (h1 : c * class_books_year s = 48) : c = 1 := by
  sorry

end number_of_classes_l371_37164


namespace total_balls_l371_37165

theorem total_balls {balls_per_box boxes : ℕ} (h1 : balls_per_box = 3) (h2 : boxes = 2) : balls_per_box * boxes = 6 :=
by
  sorry

end total_balls_l371_37165


namespace problem_l371_37159

-- Define the main problem conditions
variables {a b c : ℝ}
axiom h1 : a^2 + b^2 + c^2 = 63
axiom h2 : 2 * a + 3 * b + 6 * c = 21 * Real.sqrt 7

-- Define the goal
theorem problem :
  (a / c) ^ (a / b) = (1 / 3) ^ (2 / 3) :=
sorry

end problem_l371_37159


namespace cubic_roots_and_k_value_l371_37124

theorem cubic_roots_and_k_value (k r₃ : ℝ) :
  (∃ r₃, 3 - 2 + r₃ = -5 ∧ 3 * (-2) * r₃ = -12 ∧ k = 3 * (-2) + (-2) * r₃ + r₃ * 3) →
  (k = -12 ∧ r₃ = -6) :=
by
  sorry

end cubic_roots_and_k_value_l371_37124


namespace marvin_substitute_correct_l371_37154

theorem marvin_substitute_correct {a b c d f : ℤ} (ha : a = 3) (hb : b = 4) (hc : c = 7) (hd : d = 5) :
  (a + (b - (c + (d - f))) = 5 - f) → f = 5 :=
sorry

end marvin_substitute_correct_l371_37154


namespace minimum_kinds_of_candies_l371_37102

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l371_37102


namespace rhind_papyrus_smallest_portion_l371_37156

theorem rhind_papyrus_smallest_portion :
  ∀ (a1 d : ℚ),
    5 * a1 + (5 * 4 / 2) * d = 10 ∧
    (3 * a1 + 9 * d) / 7 = a1 + (a1 + d) →
    a1 = 1 / 6 :=
by sorry

end rhind_papyrus_smallest_portion_l371_37156


namespace monica_total_savings_l371_37152

noncomputable def weekly_savings : ℕ := 15
noncomputable def weeks_to_fill_moneybox : ℕ := 60
noncomputable def num_repeats : ℕ := 5
noncomputable def total_savings (weekly_savings weeks_to_fill_moneybox num_repeats : ℕ) : ℕ :=
  (weekly_savings * weeks_to_fill_moneybox) * num_repeats

theorem monica_total_savings :
  total_savings 15 60 5 = 4500 := by
  sorry

end monica_total_savings_l371_37152


namespace intersection_points_with_x_axis_l371_37155

theorem intersection_points_with_x_axis (a : ℝ) :
    (∃ x : ℝ, a * x^2 - a * x + 3 * x + 1 = 0 ∧ 
              ∀ x' : ℝ, (x' ≠ x → a * x'^2 - a * x' + 3 * x' + 1 ≠ 0)) ↔ 
    (a = 0 ∨ a = 1 ∨ a = 9) := by 
  sorry

end intersection_points_with_x_axis_l371_37155


namespace range_of_p_l371_37131

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 1023 :=
by
  sorry

end range_of_p_l371_37131


namespace sum_of_powers_of_minus_one_l371_37144

theorem sum_of_powers_of_minus_one : (-1) ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 + (-1) ^ 2014 = -1 := by
  sorry

end sum_of_powers_of_minus_one_l371_37144


namespace number_of_outfits_l371_37180

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def number_pants : ℕ := 9
def blue_hats : ℕ := 10
def red_hats : ℕ := 10

theorem number_of_outfits :
  (red_shirts * number_pants * blue_hats) + (green_shirts * number_pants * red_hats) = 1170 :=
by
  sorry

end number_of_outfits_l371_37180


namespace part1_part2_l371_37117

open Complex

theorem part1 {m : ℝ} : m + (m^2 + 2) * I = 0 -> m = 0 :=
by sorry

theorem part2 {m : ℝ} (h : (m + I)^2 - 2 * (m + I) + 2 = 0) :
    (let z1 := m + I
     let z2 := 2 + m * I
     im ((z2 / z1) : ℂ) = -1 / 2) :=
by sorry

end part1_part2_l371_37117


namespace combination_divisible_by_30_l371_37139

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l371_37139


namespace percent_difference_l371_37111

theorem percent_difference (a b : ℝ) : 
  a = 67.5 * 250 / 100 → 
  b = 52.3 * 180 / 100 → 
  (a - b) = 74.61 :=
by
  intros ha hb
  rw [ha, hb]
  -- omitted proof
  sorry

end percent_difference_l371_37111


namespace non_shaded_area_l371_37104

theorem non_shaded_area (r : ℝ) (A : ℝ) (shaded : ℝ) (non_shaded : ℝ) :
  (r = 5) ∧ (A = 4 * (π * r^2)) ∧ (shaded = 8 * (1 / 4 * π * r^2 - (1 / 2 * r * r))) ∧
  (non_shaded = A - shaded) → 
  non_shaded = 50 * π + 100 :=
by
  intro h
  obtain ⟨r_eq_5, A_eq, shaded_eq, non_shaded_eq⟩ := h
  rw [r_eq_5] at *
  sorry

end non_shaded_area_l371_37104


namespace length_of_train_correct_l371_37192

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_sec

theorem length_of_train_correct :
  length_of_train 60 18 = 300.06 :=
by
  -- Placeholder for proof
  sorry

end length_of_train_correct_l371_37192


namespace uncool_students_in_two_classes_l371_37184

theorem uncool_students_in_two_classes
  (students_class1 : ℕ)
  (cool_dads_class1 : ℕ)
  (cool_moms_class1 : ℕ)
  (both_cool_class1 : ℕ)
  (students_class2 : ℕ)
  (cool_dads_class2 : ℕ)
  (cool_moms_class2 : ℕ)
  (both_cool_class2 : ℕ)
  (h1 : students_class1 = 45)
  (h2 : cool_dads_class1 = 22)
  (h3 : cool_moms_class1 = 25)
  (h4 : both_cool_class1 = 11)
  (h5 : students_class2 = 35)
  (h6 : cool_dads_class2 = 15)
  (h7 : cool_moms_class2 = 18)
  (h8 : both_cool_class2 = 7) :
  (students_class1 - ((cool_dads_class1 - both_cool_class1) + (cool_moms_class1 - both_cool_class1) + both_cool_class1) +
   students_class2 - ((cool_dads_class2 - both_cool_class2) + (cool_moms_class2 - both_cool_class2) + both_cool_class2)
  ) = 18 :=
sorry

end uncool_students_in_two_classes_l371_37184


namespace distance_earth_sun_l371_37172

theorem distance_earth_sun (speed_of_light : ℝ) (time_to_earth: ℝ) 
(h1 : speed_of_light = 3 * 10^8) 
(h2 : time_to_earth = 5 * 10^2) :
  speed_of_light * time_to_earth = 1.5 * 10^11 := 
by 
  -- proof steps can be filled here
  sorry

end distance_earth_sun_l371_37172


namespace sin_eq_one_fifth_l371_37145

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_eq_one_fifth (ϕ : ℝ)
  (h : binomial_coefficient 5 3 * (Real.cos ϕ)^2 = 4) :
  Real.sin (2 * ϕ - π / 2) = 1 / 5 := sorry

end sin_eq_one_fifth_l371_37145


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l371_37182

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l371_37182


namespace find_a_l371_37101

theorem find_a (a : ℝ) (h : -1 ^ 2 + 2 * -1 + a = 0) : a = 1 :=
sorry

end find_a_l371_37101


namespace quadrant_of_angle_l371_37186

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ n : ℤ, n = 1 ∧ α = (n * π + π / 2) :=
sorry

end quadrant_of_angle_l371_37186


namespace sum_of_squares_of_sums_l371_37147

axiom roots_of_polynomial (p q r : ℝ) : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0

theorem sum_of_squares_of_sums (p q r : ℝ)
  (h_roots : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := 
sorry

end sum_of_squares_of_sums_l371_37147


namespace find_a_l371_37121

-- Given conditions
variables (x y z a : ℤ)

def conditions : Prop :=
  (x - 10) * (y - a) * (z - 2) = 1000 ∧
  ∃ (x y z : ℤ), x + y + z = 7

theorem find_a (x y z : ℤ) (h : conditions x y z 1) : a = 1 := 
  by
    sorry

end find_a_l371_37121


namespace probability_of_blue_or_orange_jelly_bean_is_5_over_13_l371_37199

def total_jelly_beans : ℕ := 7 + 9 + 8 + 10 + 5

def blue_or_orange_jelly_beans : ℕ := 10 + 5

def probability_blue_or_orange : ℚ := blue_or_orange_jelly_beans / total_jelly_beans

theorem probability_of_blue_or_orange_jelly_bean_is_5_over_13 :
  probability_blue_or_orange = 5 / 13 :=
by
  sorry

end probability_of_blue_or_orange_jelly_bean_is_5_over_13_l371_37199


namespace expected_total_rain_l371_37189

theorem expected_total_rain :
  let p_sun := 0.30
  let p_rain5 := 0.30
  let p_rain12 := 0.40
  let rain_sun := 0
  let rain_rain5 := 5
  let rain_rain12 := 12
  let days := 6
  let E_rain := p_sun * rain_sun + p_rain5 * rain_rain5 + p_rain12 * rain_rain12
  E_rain * days = 37.8 :=
by
  -- Proof omitted
  sorry

end expected_total_rain_l371_37189


namespace paint_pyramid_l371_37157

theorem paint_pyramid (colors : Finset ℕ) (n : ℕ) (h : colors.card = 5) :
  let ways_to_paint := 5 * 4 * 3 * 2 * 1
  n = ways_to_paint
:=
sorry

end paint_pyramid_l371_37157


namespace ratio_of_inscribed_squares_in_isosceles_right_triangle_l371_37170

def isosceles_right_triangle (a b : ℝ) (leg : ℝ) : Prop :=
  let a_square_inscribed := a = leg
  let b_square_inscribed := b = leg
  a_square_inscribed ∧ b_square_inscribed

theorem ratio_of_inscribed_squares_in_isosceles_right_triangle (a b leg : ℝ)
  (h : isosceles_right_triangle a b leg) :
  leg = 6 ∧ a = leg ∧ b = leg → a / b = 1 := 
by {
  sorry -- the proof will go here
}

end ratio_of_inscribed_squares_in_isosceles_right_triangle_l371_37170


namespace problem1_problem2_l371_37162

-- Problem 1: Prove that the given expression evaluates to the correct answer
theorem problem1 :
  2 * Real.sin (Real.pi / 6) - (2015 - Real.pi)^0 + abs (1 - Real.tan (Real.pi / 3)) = abs (1 - Real.sqrt 3) :=
sorry

-- Problem 2: Prove that the solutions to the given equation are correct
theorem problem2 (x : ℝ) :
  (x-2)^2 = 3 * (x-2) → x = 2 ∨ x = 5 :=
sorry

end problem1_problem2_l371_37162


namespace initial_capacity_of_drum_x_l371_37195

theorem initial_capacity_of_drum_x (C x : ℝ) (h_capacity_y : 2 * x = 2 * 0.75 * C) :
  x = 0.75 * C :=
sorry

end initial_capacity_of_drum_x_l371_37195


namespace sheela_monthly_income_l371_37197

variable (deposits : ℝ) (percentage : ℝ) (monthly_income : ℝ)

-- Conditions
axiom deposit_condition : deposits = 3400
axiom percentage_condition : percentage = 0.15
axiom income_condition : deposits = percentage * monthly_income

-- Proof goal
theorem sheela_monthly_income :
  monthly_income = 3400 / 0.15 :=
sorry

end sheela_monthly_income_l371_37197


namespace circus_capacity_l371_37185

theorem circus_capacity (sections : ℕ) (people_per_section : ℕ) (h1 : sections = 4) (h2 : people_per_section = 246) :
  sections * people_per_section = 984 :=
by
  sorry

end circus_capacity_l371_37185


namespace orange_balloons_count_l371_37181

variable (original_orange_balloons : ℝ)
variable (found_orange_balloons : ℝ)
variable (total_orange_balloons : ℝ)

theorem orange_balloons_count :
  original_orange_balloons = 9.0 →
  found_orange_balloons = 2.0 →
  total_orange_balloons = original_orange_balloons + found_orange_balloons →
  total_orange_balloons = 11.0 := by
  sorry

end orange_balloons_count_l371_37181


namespace find_x7_plus_32x2_l371_37177

theorem find_x7_plus_32x2 (x : ℝ) (h : x^3 + 2 * x = 4) : x^7 + 32 * x^2 = 64 :=
sorry

end find_x7_plus_32x2_l371_37177


namespace frank_won_skee_ball_tickets_l371_37113

noncomputable def tickets_whack_a_mole : ℕ := 33
noncomputable def candies_bought : ℕ := 7
noncomputable def tickets_per_candy : ℕ := 6
noncomputable def total_tickets_spent : ℕ := candies_bought * tickets_per_candy
noncomputable def tickets_skee_ball : ℕ := total_tickets_spent - tickets_whack_a_mole

theorem frank_won_skee_ball_tickets : tickets_skee_ball = 9 :=
  by
  sorry

end frank_won_skee_ball_tickets_l371_37113


namespace cubes_difference_divisible_91_l371_37190

theorem cubes_difference_divisible_91 (cubes : Fin 16 → ℤ) (h : ∀ n : Fin 16, ∃ m : ℤ, cubes n = m^3) :
  ∃ (a b : Fin 16), a ≠ b ∧ 91 ∣ (cubes a - cubes b) :=
sorry

end cubes_difference_divisible_91_l371_37190


namespace simplify_and_evaluate_l371_37169

theorem simplify_and_evaluate (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hx3 : x ≠ -2) (hx4 : x = -1) :
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = -2 :=
by
  sorry

end simplify_and_evaluate_l371_37169


namespace albums_either_but_not_both_l371_37188

-- Definition of the problem conditions
def shared_albums : Nat := 11
def andrew_total_albums : Nat := 20
def bob_exclusive_albums : Nat := 8

-- Calculate Andrew's exclusive albums
def andrew_exclusive_albums : Nat := andrew_total_albums - shared_albums

-- Question: Prove the total number of albums in either Andrew's or Bob's collection but not both is 17
theorem albums_either_but_not_both : 
  andrew_exclusive_albums + bob_exclusive_albums = 17 := 
by
  sorry

end albums_either_but_not_both_l371_37188


namespace milk_in_jugs_l371_37160

theorem milk_in_jugs (x y : ℝ) (h1 : x + y = 70) (h2 : y + 0.125 * x = 0.875 * x) :
  x = 40 ∧ y = 30 := 
sorry

end milk_in_jugs_l371_37160


namespace determine_function_l371_37174

theorem determine_function (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (x + 1)) :=
by
  sorry

end determine_function_l371_37174


namespace divisibility_proof_l371_37161

theorem divisibility_proof (n : ℕ) (hn : 0 < n) (h : n ∣ (10^n - 1)) : 
  n ∣ ((10^n - 1) / 9) :=
  sorry

end divisibility_proof_l371_37161


namespace sqrt_sum_of_fractions_as_fraction_l371_37110

theorem sqrt_sum_of_fractions_as_fraction :
  (Real.sqrt ((36 / 49) + (16 / 9) + (1 / 16))) = (45 / 28) :=
by
  sorry

end sqrt_sum_of_fractions_as_fraction_l371_37110


namespace pattern_equation_l371_37187

theorem pattern_equation (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end pattern_equation_l371_37187


namespace not_in_sequence_l371_37133

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence property
def sequence_property (a b : ℕ) : Prop :=
  b = a + sum_of_digits a

-- Main theorem
theorem not_in_sequence (n : ℕ) (h : n = 793210041) : 
  ¬ (∃ a : ℕ, sequence_property a n) :=
by
  sorry

end not_in_sequence_l371_37133


namespace minimum_inlets_needed_l371_37125

noncomputable def waterInflow (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := x * a - b

theorem minimum_inlets_needed (a b : ℝ) (ha : a = b)
  (h1 : (4 * a - b) * 5 = (2 * a - b) * 15)
  (h2 : (a * 9 - b) * 2 ≥ 1) : 
  ∃ n : ℕ, 2 * (a * n - b) ≥ (4 * a - b) * 5 := 
by 
  sorry

end minimum_inlets_needed_l371_37125


namespace average_interest_rate_l371_37158

theorem average_interest_rate (x : ℝ) (h1 : 0 < x ∧ x < 6000)
  (h2 : 0.03 * (6000 - x) = 0.055 * x) :
  ((0.03 * (6000 - x) + 0.055 * x) / 6000) = 0.0388 :=
by
  sorry

end average_interest_rate_l371_37158


namespace total_situps_l371_37143

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end total_situps_l371_37143


namespace find_other_root_l371_37108

variables {a b c : ℝ}

theorem find_other_root
  (h_eq : ∀ x : ℝ, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0)
  (root1 : a * (b - c) * 1^2 + b * (c - a) * 1 + c * (a - b) = 0) :
  ∃ k : ℝ, k = c * (a - b) / (a * (b - c)) ∧
           a * (b - c) * k^2 + b * (c - a) * k + c * (a - b) = 0 := 
sorry

end find_other_root_l371_37108


namespace two_digit_number_representation_l371_37141

-- Define the conditions and the problem statement in Lean 4
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

theorem two_digit_number_representation (x : ℕ) (h : x < 10) :
  ∃ n : ℕ, units_digit n = x ∧ tens_digit n = 2 * x ^ 2 ∧ n = 20 * x ^ 2 + x :=
by {
  sorry
}

end two_digit_number_representation_l371_37141


namespace milk_production_group_B_l371_37114

theorem milk_production_group_B (a b c d e : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pos_d : d > 0) (h_pos_e : e > 0) :
  ((1.2 * b * d * e) / (a * c)) = 1.2 * (b / (a * c)) * d * e := 
by
  sorry

end milk_production_group_B_l371_37114


namespace partA_l371_37119

theorem partA (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, a * x ^ 2 + b * x + c = k ^ 4) : a = 0 ∧ b = 0 := 
sorry

end partA_l371_37119


namespace tens_digit_of_11_pow_12_pow_13_l371_37115

theorem tens_digit_of_11_pow_12_pow_13 :
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  tens_digit = 2 :=
by 
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  show tens_digit = 2
  sorry

end tens_digit_of_11_pow_12_pow_13_l371_37115


namespace determine_function_l371_37168

theorem determine_function (f : ℝ → ℝ)
    (h1 : f 1 = 0)
    (h2 : ∀ x y : ℝ, |f x - f y| = |x - y|) :
    (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end determine_function_l371_37168


namespace pyramid_height_l371_37196

noncomputable def height_of_pyramid (h : ℝ) : Prop :=
  let cube_edge_length := 6
  let pyramid_base_edge_length := 12
  let V_cube := cube_edge_length ^ 3
  let V_pyramid := (1 / 3) * (pyramid_base_edge_length ^ 2) * h
  V_cube = V_pyramid → h = 4.5

theorem pyramid_height : height_of_pyramid 4.5 :=
by {
  sorry
}

end pyramid_height_l371_37196


namespace gcd_of_18_and_30_l371_37132

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l371_37132


namespace parallel_line_segment_length_l371_37142

theorem parallel_line_segment_length (AB : ℝ) (S : ℝ) (x : ℝ) 
  (h1 : AB = 36) 
  (h2 : S = (S / 2) * 2)
  (h3 : x / AB = (↑(1 : ℝ) / 2 * S / S) ^ (1 / 2)) : 
  x = 18 * Real.sqrt 2 :=
by 
    sorry 

end parallel_line_segment_length_l371_37142


namespace cos_value_given_sin_l371_37193

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) :
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end cos_value_given_sin_l371_37193


namespace average_age_first_and_fifth_dogs_l371_37136

-- Define the conditions
def first_dog_age : ℕ := 10
def second_dog_age : ℕ := first_dog_age - 2
def third_dog_age : ℕ := second_dog_age + 4
def fourth_dog_age : ℕ := third_dog_age / 2
def fifth_dog_age : ℕ := fourth_dog_age + 20

-- Define the goal statement
theorem average_age_first_and_fifth_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 :=
by
  sorry

end average_age_first_and_fifth_dogs_l371_37136


namespace more_pens_than_pencils_l371_37100

-- Define the number of pencils (P) and pens (Pe)
def num_pencils : ℕ := 15 * 80

-- Define the number of pens (Pe) is more than twice the number of pencils (P)
def num_pens (Pe : ℕ) : Prop := Pe > 2 * num_pencils

-- State the total cost equation in terms of pens and pencils
def total_cost_eq (Pe : ℕ) : Prop := (5 * Pe + 4 * num_pencils = 18300)

-- Prove that the number of more pens than pencils is 1500
theorem more_pens_than_pencils (Pe : ℕ) (h1 : num_pens Pe) (h2 : total_cost_eq Pe) : (Pe - num_pencils = 1500) :=
by
  sorry

end more_pens_than_pencils_l371_37100


namespace population_increase_rate_l371_37173

theorem population_increase_rate (P₀ P₁ : ℕ) (rate : ℚ) (h₁ : P₀ = 220) (h₂ : P₁ = 242) :
  rate = ((P₁ - P₀ : ℚ) / P₀) * 100 := by
  sorry

end population_increase_rate_l371_37173


namespace largest_integer_x_l371_37176

theorem largest_integer_x (x : ℤ) : (x / 4 + 3 / 5 < 7 / 4) → x ≤ 4 := sorry

end largest_integer_x_l371_37176


namespace circle_center_and_radius_l371_37134

theorem circle_center_and_radius (x y : ℝ) : 
  (x^2 + y^2 - 6 * x = 0) → ((x - 3)^2 + (y - 0)^2 = 9) :=
by
  intro h
  -- The proof is left as an exercise.
  sorry

end circle_center_and_radius_l371_37134


namespace branches_sum_one_main_stem_l371_37128

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l371_37128


namespace curvature_formula_l371_37129

noncomputable def curvature_squared (x y : ℝ → ℝ) (t : ℝ) :=
  let x' := (deriv x t)
  let y' := (deriv y t)
  let x'' := (deriv (deriv x) t)
  let y'' := (deriv (deriv y) t)
  (x'' * y' - y'' * x')^2 / (x'^2 + y'^2)^3

theorem curvature_formula (x y : ℝ → ℝ) (t : ℝ) :
  let k_sq := curvature_squared x y t
  k_sq = ((deriv (deriv x) t * deriv y t - deriv (deriv y) t * deriv x t)^2 /
         ((deriv x t)^2 + (deriv y t)^2)^3) := 
by 
  sorry

end curvature_formula_l371_37129

import Mathlib

namespace square_area_l1651_165153

theorem square_area (x : ℚ) (h : 3 * x - 12 = 15 - 2 * x) : (3 * (27 / 5) - 12)^2 = 441 / 25 :=
by
  sorry

end square_area_l1651_165153


namespace arithmetic_sequence_common_diff_l1651_165159

theorem arithmetic_sequence_common_diff (d : ℝ) (a : ℕ → ℝ) 
  (h_first_term : a 0 = 24) 
  (h_arithmetic_sequence : ∀ n, a (n + 1) = a n + d)
  (h_ninth_term_nonneg : 24 + 8 * d ≥ 0) 
  (h_tenth_term_neg : 24 + 9 * d < 0) : 
  -3 ≤ d ∧ d < -8/3 :=
by 
  sorry

end arithmetic_sequence_common_diff_l1651_165159


namespace negative_sixty_represents_expenditure_l1651_165197

def positive_represents_income (x : ℤ) : Prop := x > 0
def negative_represents_expenditure (x : ℤ) : Prop := x < 0

theorem negative_sixty_represents_expenditure :
  negative_represents_expenditure (-60) ∧ abs (-60) = 60 :=
by
  sorry

end negative_sixty_represents_expenditure_l1651_165197


namespace total_feathers_needed_l1651_165121

theorem total_feathers_needed 
  (animals_group1 : ℕ) (feathers_group1 : ℕ)
  (animals_group2 : ℕ) (feathers_group2 : ℕ) 
  (total_feathers : ℕ) :
  animals_group1 = 934 →
  feathers_group1 = 7 →
  animals_group2 = 425 →
  feathers_group2 = 12 →
  total_feathers = 11638 :=
by sorry

end total_feathers_needed_l1651_165121


namespace Christina_driving_time_l1651_165158

theorem Christina_driving_time 
  (speed_Christina : ℕ) 
  (speed_friend : ℕ) 
  (total_distance : ℕ)
  (friend_driving_time : ℕ) 
  (distance_by_Christina : ℕ) 
  (time_driven_by_Christina : ℕ) 
  (total_driving_time : ℕ)
  (h1 : speed_Christina = 30)
  (h2 : speed_friend = 40) 
  (h3 : total_distance = 210)
  (h4 : friend_driving_time = 3)
  (h5 : speed_friend * friend_driving_time = 120)
  (h6 : total_distance - 120 = distance_by_Christina)
  (h7 : distance_by_Christina = 90)
  (h8 : distance_by_Christina / speed_Christina = 3)
  (h9 : time_driven_by_Christina = 3)
  (h10 : time_driven_by_Christina * 60 = 180) :
    total_driving_time = 180 := 
by
  sorry

end Christina_driving_time_l1651_165158


namespace smallest_three_digit_number_multiple_of_conditions_l1651_165190

theorem smallest_three_digit_number_multiple_of_conditions :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
  (x % 2 = 0) ∧ ((x + 1) % 3 = 0) ∧ ((x + 2) % 4 = 0) ∧ ((x + 3) % 5 = 0) ∧ ((x + 4) % 6 = 0) 
  ∧ x = 122 := 
by
  sorry

end smallest_three_digit_number_multiple_of_conditions_l1651_165190


namespace equation_has_two_solutions_l1651_165165

theorem equation_has_two_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^x1 = x1^2 - 2*x1 - a ∧ a^x2 = x2^2 - 2*x2 - a :=
sorry

end equation_has_two_solutions_l1651_165165


namespace perpendicular_tangents_at_x0_l1651_165142

noncomputable def x0 := (36 : ℝ)^(1 / 3) / 6

theorem perpendicular_tangents_at_x0 :
  (∃ x0 : ℝ, (∃ f1 f2 : ℝ → ℝ,
    (∀ x, f1 x = x^2 - 1) ∧
    (∀ x, f2 x = 1 - x^3) ∧
    (2 * x0 * (-3 * x0^2) = -1)) ∧
    x0 = (36 : ℝ)^(1 / 3) / 6) := sorry

end perpendicular_tangents_at_x0_l1651_165142


namespace solution_inequality_l1651_165194

variable (a x : ℝ)

theorem solution_inequality (h : ∀ x, |x - a| + |x + 4| ≥ 1) : a ≤ -5 ∨ a ≥ -3 := by
  sorry

end solution_inequality_l1651_165194


namespace school_band_fundraising_l1651_165124

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l1651_165124


namespace find_stock_rate_l1651_165185

theorem find_stock_rate (annual_income : ℝ) (investment_amount : ℝ) (R : ℝ) 
  (h1 : annual_income = 2000) (h2 : investment_amount = 6800) : 
  R = 2000 / 6800 :=
by
  sorry

end find_stock_rate_l1651_165185


namespace determine_remainder_l1651_165120

-- Define the sequence and its sum
def geom_series_sum_mod (a r n m : ℕ) : ℕ := 
  ((r^(n+1) - 1) / (r - 1)) % m

-- Define the specific geometric series and modulo
theorem determine_remainder :
  geom_series_sum_mod 1 11 1800 500 = 1 :=
by
  -- Using geom_series_sum_mod to define the series
  let S := geom_series_sum_mod 1 11 1800 500
  -- Remainder when the series is divided by 500
  show S = 1
  sorry

end determine_remainder_l1651_165120


namespace Juwella_reads_pages_l1651_165164

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end Juwella_reads_pages_l1651_165164


namespace point_is_in_second_quadrant_l1651_165178

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l1651_165178


namespace find_x_l1651_165169

theorem find_x
  (a b x : ℝ)
  (h1 : a * (x + 2) + b * (x + 2) = 60)
  (h2 : a + b = 12) :
  x = 3 :=
by
  sorry

end find_x_l1651_165169


namespace partition_exists_min_n_in_A_l1651_165140

-- Definition of subsets and their algebraic properties
variable (A B C : Set ℕ)

-- The Initial conditions
axiom A_squared_eq_A : ∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)
axiom B_squared_eq_C : ∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)
axiom C_squared_eq_B : ∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)
axiom AB_eq_B : ∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)
axiom AC_eq_C : ∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)
axiom BC_eq_A : ∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)

-- Statement for the partition existence with given conditions
theorem partition_exists :
  ∃ A B C : Set ℕ, (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
               (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
               (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
               (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
               (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
               (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) :=
sorry

-- Statement for the minimum n in A such that n and n+1 are both in A is at most 77
theorem min_n_in_A :
  ∀ A B C : Set ℕ,
    (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
    (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
    (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
    (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
    (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
    (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) →
    ∃ n : ℕ, (n ∈ A) ∧ (n + 1 ∈ A) ∧ n ≤ 77 :=
sorry

end partition_exists_min_n_in_A_l1651_165140


namespace tree_growth_period_l1651_165144

theorem tree_growth_period (initial height growth_rate : ℕ) (H4 final_height years : ℕ) 
  (h_init : initial_height = 4) 
  (h_growth_rate : growth_rate = 1) 
  (h_H4 : H4 = initial_height + 4 * growth_rate)
  (h_final_height : final_height = H4 + H4 / 4) 
  (h_years : years = (final_height - initial_height) / growth_rate) :
  years = 6 :=
by
  sorry

end tree_growth_period_l1651_165144


namespace no_triples_of_consecutive_numbers_l1651_165107

theorem no_triples_of_consecutive_numbers (n : ℤ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9) :
  ¬(3 * n^2 + 2 = 1111 * a) :=
by sorry

end no_triples_of_consecutive_numbers_l1651_165107


namespace geometric_sequence_common_ratio_l1651_165155

theorem geometric_sequence_common_ratio (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : a_n 3 = a_n 2 * q) 
  (h2 : a_n 2 * q - 3 * a_n 2 = 2) 
  (h3 : 5 * a_n 4 = (12 * a_n 3 + 2 * a_n 5) / 2) : 
  q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l1651_165155


namespace exclude_13_code_count_l1651_165133

/-- The number of 5-digit codes (00000 to 99999) that don't contain the sequence "13". -/
theorem exclude_13_code_count :
  let total_codes := 100000
  let excluded_codes := 3970
  total_codes - excluded_codes = 96030 :=
by
  let total_codes := 100000
  let excluded_codes := 3970
  have h : total_codes - excluded_codes = 96030 := by
    -- Provide mathematical proof or use sorry for placeholder
    sorry
  exact h

end exclude_13_code_count_l1651_165133


namespace butterfly_cocoon_l1651_165179

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end butterfly_cocoon_l1651_165179


namespace problem_statement_l1651_165192

noncomputable def min_expression_value (θ1 θ2 θ3 θ4 : ℝ) : ℝ :=
  (2 * (Real.sin θ1)^2 + 1 / (Real.sin θ1)^2) *
  (2 * (Real.sin θ2)^2 + 1 / (Real.sin θ2)^2) *
  (2 * (Real.sin θ3)^2 + 1 / (Real.sin θ3)^2) *
  (2 * (Real.sin θ4)^2 + 1 / (Real.sin θ4)^2)

theorem problem_statement (θ1 θ2 θ3 θ4 : ℝ) (h_pos: θ1 > 0 ∧ θ2 > 0 ∧ θ3 > 0 ∧ θ4 > 0) (h_sum: θ1 + θ2 + θ3 + θ4 = Real.pi) :
  min_expression_value θ1 θ2 θ3 θ4 = 81 :=
sorry

end problem_statement_l1651_165192


namespace find_x_l1651_165148

theorem find_x (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 :=
by
  sorry

end find_x_l1651_165148


namespace Tori_current_height_l1651_165152

   -- Define the original height and the height she grew
   def Tori_original_height : Real := 4.4
   def Tori_growth : Real := 2.86

   -- Prove that Tori's current height is 7.26 feet
   theorem Tori_current_height : Tori_original_height + Tori_growth = 7.26 := by
     sorry
   
end Tori_current_height_l1651_165152


namespace scaled_polynomial_roots_l1651_165139

noncomputable def polynomial_with_scaled_roots : Polynomial ℂ :=
  Polynomial.X^3 - 3*Polynomial.X^2 + 5

theorem scaled_polynomial_roots :
  (∃ r1 r2 r3 : ℂ, polynomial_with_scaled_roots.eval r1 = 0 ∧ polynomial_with_scaled_roots.eval r2 = 0 ∧ polynomial_with_scaled_roots.eval r3 = 0 ∧
  (∃ q : Polynomial ℂ, q = Polynomial.X^3 - 9*Polynomial.X^2 + 135 ∧
  ∀ y, (q.eval y = 0 ↔ (polynomial_with_scaled_roots.eval (y / 3) = 0)))) := sorry

end scaled_polynomial_roots_l1651_165139


namespace no_statement_implies_neg_p_or_q_l1651_165145

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∨ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ q
def neg_p_or_q (p q : Prop) : Prop := ¬ (p ∨ q)

theorem no_statement_implies_neg_p_or_q (p q : Prop) :
  ¬ (statement1 p q → neg_p_or_q p q) ∧
  ¬ (statement2 p q → neg_p_or_q p q) ∧
  ¬ (statement3 p q → neg_p_or_q p q) ∧
  ¬ (statement4 p q → neg_p_or_q p q)
:= by
  sorry

end no_statement_implies_neg_p_or_q_l1651_165145


namespace line_no_intersect_parabola_range_l1651_165166

def parabola_eq (x : ℝ) : ℝ := x^2 + 4

def line_eq (m x : ℝ) : ℝ := m * (x - 10) + 6

theorem line_no_intersect_parabola_range (r s m : ℝ) :
  (m^2 - 40 * m + 8 = 0) →
  r < s →
  (∀ x, parabola_eq x ≠ line_eq m x) →
  r + s = 40 :=
by
  sorry

end line_no_intersect_parabola_range_l1651_165166


namespace base_of_524_l1651_165177

theorem base_of_524 : 
  ∀ (b : ℕ), (5 * b^2 + 2 * b + 4 = 340) → b = 8 :=
by
  intros b h
  sorry

end base_of_524_l1651_165177


namespace factor_expression_l1651_165111

variable (x y : ℝ)

theorem factor_expression :
  4 * x ^ 2 - 4 * x - y ^ 2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end factor_expression_l1651_165111


namespace collinear_points_l1651_165138

variable (α β γ δ E : Type)
variables {A B C D K L P Q : α}
variables (convex : α → α → α → α → Prop)
variables (not_parallel : α → α → Prop)
variables (internal_bisector : α → α → α → Prop)
variables (external_bisector : α → α → α → Prop)
variables (collinear : α → α → α → α → Prop)

axiom convex_quad : convex A B C D
axiom AD_not_parallel_BC : not_parallel A D ∧ not_parallel B C

axiom internal_bisectors :
  internal_bisector A B K ∧ internal_bisector B A K ∧ internal_bisector C D P ∧ internal_bisector D C P

axiom external_bisectors :
  external_bisector A B L ∧ external_bisector B A L ∧ external_bisector C D Q ∧ external_bisector D C Q

theorem collinear_points : collinear K L P Q := 
sorry

end collinear_points_l1651_165138


namespace two_digit_factors_of_3_18_minus_1_l1651_165129

theorem two_digit_factors_of_3_18_minus_1 : ∃ n : ℕ, n = 6 ∧ 
  ∀ x, x ∈ {y : ℕ | y ∣ 3^18 - 1 ∧ y > 9 ∧ y < 100} → 
  (x = 13 ∨ x = 26 ∨ x = 52 ∨ x = 14 ∨ x = 28 ∨ x = 91) :=
by
  use 6
  sorry

end two_digit_factors_of_3_18_minus_1_l1651_165129


namespace smallest_positive_integer_n_l1651_165118

theorem smallest_positive_integer_n (n : ℕ) (h : 5 * n ≡ 1463 [MOD 26]) : n = 23 :=
sorry

end smallest_positive_integer_n_l1651_165118


namespace tan_of_angle_in_third_quadrant_l1651_165174

open Real

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : sin (π + α) = 3/5) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  tan α = 3 / 4 :=
by
  sorry

end tan_of_angle_in_third_quadrant_l1651_165174


namespace determine_g_l1651_165195

noncomputable def g (x : ℝ) : ℝ := -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8

theorem determine_g (x : ℝ) : 
  4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1 := by
  sorry

end determine_g_l1651_165195


namespace minimum_omega_l1651_165187

open Real

theorem minimum_omega (ω : ℕ) (h_ω_pos : ω > 0) :
  (∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + (π / 2)) → ω = 2 :=
by
  sorry

end minimum_omega_l1651_165187


namespace carpenter_needs_more_logs_l1651_165156

-- Define the given conditions in Lean 4
def total_woodblocks_needed : ℕ := 80
def logs_on_hand : ℕ := 8
def woodblocks_per_log : ℕ := 5

-- Statement: Proving the number of additional logs the carpenter needs
theorem carpenter_needs_more_logs :
  let woodblocks_available := logs_on_hand * woodblocks_per_log
  let additional_woodblocks := total_woodblocks_needed - woodblocks_available
  additional_woodblocks / woodblocks_per_log = 8 :=
by
  sorry

end carpenter_needs_more_logs_l1651_165156


namespace exponential_decreasing_l1651_165110

theorem exponential_decreasing (a : ℝ) : (∀ x y : ℝ, x < y → (2 * a - 1)^y < (2 * a - 1)^x) ↔ (1 / 2 < a ∧ a < 1) := 
by
    sorry

end exponential_decreasing_l1651_165110


namespace zoe_total_money_l1651_165198

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end zoe_total_money_l1651_165198


namespace problem_statement_l1651_165134

theorem problem_statement : (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end problem_statement_l1651_165134


namespace sum_of_repeating_decimals_l1651_165108

noncomputable def x : ℚ := 1 / 9
noncomputable def y : ℚ := 2 / 99
noncomputable def z : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  x + y + z = 134 / 999 := by
  sorry

end sum_of_repeating_decimals_l1651_165108


namespace find_a6_l1651_165176

-- Define the geometric sequence and the given terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} (r : ℝ)

-- Given conditions
axiom a_2 : a 2 = 2
axiom a_10 : a 10 = 8
axiom geo_seq : geometric_sequence a

-- Statement to prove
theorem find_a6 : a 6 = 4 :=
sorry

end find_a6_l1651_165176


namespace expected_number_of_games_l1651_165103
noncomputable def probability_of_A_winning (g : ℕ) : ℚ := 2 / 3
noncomputable def probability_of_B_winning (g : ℕ) : ℚ := 1 / 3
noncomputable def expected_games: ℚ := 266 / 81

theorem expected_number_of_games 
  (match_ends : ∀ g : ℕ, (∃ p1 p2 : ℕ, (p1 = g ∧ p2 = 0) ∨ (p1 = 0 ∧ p2 = g))) 
  (independent_outcomes : ∀ g1 g2 : ℕ, g1 ≠ g2 → probability_of_A_winning g1 * probability_of_A_winning g2 = (2 / 3) * (2 / 3) ∧ probability_of_B_winning g1 * probability_of_B_winning g2 = (1 / 3) * (1 / 3)) :
  (expected_games = 266 / 81) := 
sorry

end expected_number_of_games_l1651_165103


namespace standard_deviation_bound_l1651_165189

theorem standard_deviation_bound (mu sigma : ℝ) (h_mu : mu = 51) (h_ineq : mu - 3 * sigma > 44) : sigma < 7 / 3 :=
by
  sorry

end standard_deviation_bound_l1651_165189


namespace part_I_part_II_l1651_165172

theorem part_I (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2) (h4 : a + b ≤ m) : m ≥ 3 := by
  sorry

theorem part_II (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2)
  (h4 : 2 * |x - 1| + |x| ≥ a + b) : (x ≤ -1 / 3 ∨ x ≥ 5 / 3) := by
  sorry

end part_I_part_II_l1651_165172


namespace motorcyclist_travel_distances_l1651_165113

-- Define the total distance traveled in three days
def total_distance : ℕ := 980

-- Define the total distance traveled in the first two days
def first_two_days_distance : ℕ := 725

-- Define the extra distance traveled on the second day compared to the third day
def second_day_extra : ℕ := 123

-- Define the distances traveled on the first, second, and third days respectively
def day_1_distance : ℕ := 347
def day_2_distance : ℕ := 378
def day_3_distance : ℕ := 255

-- Formalize the theorem statement
theorem motorcyclist_travel_distances :
  total_distance = day_1_distance + day_2_distance + day_3_distance ∧
  first_two_days_distance = day_1_distance + day_2_distance ∧
  day_2_distance = day_3_distance + second_day_extra :=
by 
  sorry

end motorcyclist_travel_distances_l1651_165113


namespace adjacent_probability_is_2_over_7_l1651_165180

variable (n : Nat := 5) -- number of student performances
variable (m : Nat := 2) -- number of teacher performances

/-- Total number of ways to insert two performances
    (ignoring adjacency constraints) into the program list. -/
def total_insertion_ways : Nat :=
  Fintype.card (Fin (n + m))

/-- Number of ways to insert two performances such that they are adjacent. -/
def adjacent_insertion_ways : Nat :=
  Fintype.card (Fin (n + 1))

/-- Probability that two specific performances are adjacent in a program list. -/
def adjacent_probability : ℚ :=
  adjacent_insertion_ways / total_insertion_ways

theorem adjacent_probability_is_2_over_7 :
  adjacent_probability = (2 : ℚ) / 7 := by
  sorry

end adjacent_probability_is_2_over_7_l1651_165180


namespace eat_five_pounds_in_46_875_min_l1651_165102

theorem eat_five_pounds_in_46_875_min
  (fat_rate : ℝ) (thin_rate : ℝ) (combined_rate : ℝ) (total_fruit : ℝ)
  (hf1 : fat_rate = 1 / 15)
  (hf2 : thin_rate = 1 / 25)
  (h_comb : combined_rate = fat_rate + thin_rate)
  (h_fruit : total_fruit = 5) :
  total_fruit / combined_rate = 46.875 :=
by
  sorry

end eat_five_pounds_in_46_875_min_l1651_165102


namespace log_ratio_squared_eq_nine_l1651_165146

-- Given conditions
variable (x y : ℝ) 
variable (hx_pos : x > 0) 
variable (hy_pos : y > 0)
variable (hx_neq1 : x ≠ 1) 
variable (hy_neq1 : y ≠ 1)
variable (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
variable (heq : x * y = 243)

-- Prove that (\log_3(\tfrac x y))^2 = 9
theorem log_ratio_squared_eq_nine (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq1 : x ≠ 1) (hy_neq1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (heq : x * y = 243) : 
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 :=
sorry

end log_ratio_squared_eq_nine_l1651_165146


namespace rulers_added_initially_46_finally_71_l1651_165125

theorem rulers_added_initially_46_finally_71 : 
  ∀ (initial final added : ℕ), initial = 46 → final = 71 → added = final - initial → added = 25 :=
by
  intros initial final added h_initial h_final h_added
  rw [h_initial, h_final] at h_added
  exact h_added

end rulers_added_initially_46_finally_71_l1651_165125


namespace calculate_myOp_l1651_165171

-- Define the operation
def myOp (x y : ℝ) : ℝ := x^3 - y

-- Given condition for h as a real number
variable (h : ℝ)

-- The theorem we need to prove
theorem calculate_myOp : myOp (2 * h) (myOp (2 * h) (2 * h)) = 2 * h := by
  sorry

end calculate_myOp_l1651_165171


namespace kenya_peanuts_correct_l1651_165105

def jose_peanuts : ℕ := 85
def kenya_extra_peanuts : ℕ := 48
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := 
by 
  sorry

end kenya_peanuts_correct_l1651_165105


namespace find_contributions_before_johns_l1651_165137

-- Definitions based on the conditions provided
def avg_contrib_size_after (A : ℝ) := A + 0.5 * A = 75
def johns_contribution := 100
def total_amount_before (n : ℕ) (A : ℝ) := n * A
def total_amount_after (n : ℕ) (A : ℝ) := (n * A + johns_contribution)

-- Proposition we need to prove
theorem find_contributions_before_johns (n : ℕ) (A : ℝ) :
  avg_contrib_size_after A →
  total_amount_before n A + johns_contribution = (n + 1) * 75 →
  n = 1 :=
by
  sorry

end find_contributions_before_johns_l1651_165137


namespace seventh_term_geometric_sequence_l1651_165143

theorem seventh_term_geometric_sequence :
  ∃ (a₁ a₁₀ a₇ : ℕ) (r : ℕ),
    a₁ = 6 ∧ a₁₀ = 93312 ∧
    a₁₀ = a₁ * r^9 ∧
    a₇ = a₁ * r^6 ∧
    a₇ = 279936 :=
by
  sorry

end seventh_term_geometric_sequence_l1651_165143


namespace number_of_oxygen_atoms_l1651_165199

/-- Given a compound has 1 H, 1 Cl, and a certain number of O atoms and the molecular weight of the compound is 68 g/mol,
    prove that the number of O atoms is 2. -/
theorem number_of_oxygen_atoms (atomic_weight_H: ℝ) (atomic_weight_Cl: ℝ) (atomic_weight_O: ℝ) (molecular_weight: ℝ) (n : ℕ):
    atomic_weight_H = 1.0 →
    atomic_weight_Cl = 35.5 →
    atomic_weight_O = 16.0 →
    molecular_weight = 68.0 →
    molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O →
    n = 2 :=
by
  sorry

end number_of_oxygen_atoms_l1651_165199


namespace sqrt_difference_l1651_165175

theorem sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := 
by 
  sorry

end sqrt_difference_l1651_165175


namespace first_line_shift_time_l1651_165123

theorem first_line_shift_time (x y : ℝ) (h1 : (1 / x) + (1 / (x - 2)) + (1 / y) = 1.5 * ((1 / x) + (1 / (x - 2)))) 
  (h2 : x - 24 / 5 = (1 / ((1 / (x - 2)) + (1 / y)))) :
  x = 8 :=
sorry

end first_line_shift_time_l1651_165123


namespace arithmetic_sequence_positive_l1651_165150

theorem arithmetic_sequence_positive (d a_1 : ℤ) (n : ℤ) :
  (a_11 - a_8 = 3) -> 
  (S_11 - S_8 = 33) ->
  (n > 0) ->
  a_1 + (n-1) * d > 0 ->
  n = 10 :=
by
  sorry

end arithmetic_sequence_positive_l1651_165150


namespace score_difference_proof_l1651_165173

variable (α β γ δ : ℝ)

theorem score_difference_proof
  (h1 : α + β = γ + δ + 17)
  (h2 : α = β - 4)
  (h3 : γ = δ + 5) :
  β - δ = 13 :=
by
  -- proof goes here
  sorry

end score_difference_proof_l1651_165173


namespace reduced_price_per_dozen_is_approx_2_95_l1651_165154

noncomputable def original_price : ℚ := 16 / 39
noncomputable def reduced_price := 0.6 * original_price
noncomputable def reduced_price_per_dozen := reduced_price * 12

theorem reduced_price_per_dozen_is_approx_2_95 :
  abs (reduced_price_per_dozen - 2.95) < 0.01 :=
by
  sorry

end reduced_price_per_dozen_is_approx_2_95_l1651_165154


namespace incorrect_expressions_l1651_165149

theorem incorrect_expressions (x y : ℚ) (h : x / y = 2 / 5) :
    (x + 3 * y) / x ≠ 17 / 2 ∧ (x - y) / y ≠ 3 / 5 :=
by
  sorry

end incorrect_expressions_l1651_165149


namespace francis_violin_count_l1651_165191

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l1651_165191


namespace interior_diagonals_of_dodecahedron_l1651_165126

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end interior_diagonals_of_dodecahedron_l1651_165126


namespace gcd_g_x_l1651_165147

noncomputable def g (x : ℕ) : ℕ :=
  (3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)

theorem gcd_g_x (x : ℕ) (h : x % 19845 = 0) : Nat.gcd (g x) x = 700 :=
  sorry

end gcd_g_x_l1651_165147


namespace initial_volume_mixture_l1651_165135

theorem initial_volume_mixture (x : ℝ) :
  (4 * x) / (3 * x + 13) = 5 / 7 →
  13 * x = 65 →
  7 * x = 35 := 
by
  intro h1 h2
  sorry

end initial_volume_mixture_l1651_165135


namespace income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l1651_165131

-- Define the conditions
def annual_income (year : ℕ) : ℝ := 0.0124 * (1 + 0.2) ^ (year - 1)
def annual_repayment : ℝ := 0.05

-- Proof Problem 1: Show that the subway's annual operating income exceeds the annual repayment at year 9
theorem income_exceeds_repayment_after_9_years :
  ∀ n ≥ 9, annual_income n > annual_repayment :=
by
  sorry

-- Define the cumulative payment function for the municipal government
def cumulative_payment (years : ℕ) : ℝ :=
  (annual_repayment * years) - (List.sum (List.map annual_income (List.range years)))

-- Proof Problem 2: Show the cumulative payment by the municipal government up to year 8 is 19,541,135 RMB
theorem cumulative_payment_up_to_year_8 :
  cumulative_payment 8 = 0.1954113485 :=
by
  sorry

end income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l1651_165131


namespace circle_equation_l1651_165114

theorem circle_equation (C : ℝ → ℝ → Prop)
  (h₁ : C 1 0)
  (h₂ : C 0 (Real.sqrt 3))
  (h₃ : C (-3) 0) :
  ∃ D E F : ℝ, (∀ x y, C x y ↔ x^2 + y^2 + D * x + E * y + F = 0) ∧ D = 2 ∧ E = 0 ∧ F = -3 := 
by
  sorry

end circle_equation_l1651_165114


namespace survey_response_total_l1651_165151

theorem survey_response_total
  (X Y Z : ℕ)
  (h_ratio : X / 4 = Y / 2 ∧ X / 4 = Z)
  (h_X : X = 200) :
  X + Y + Z = 350 :=
sorry

end survey_response_total_l1651_165151


namespace polynomial_evaluation_at_8_l1651_165122

def P (x : ℝ) : ℝ := x^3 + 2*x^2 + x - 1

theorem polynomial_evaluation_at_8 : P 8 = 647 :=
by sorry

end polynomial_evaluation_at_8_l1651_165122


namespace find_a4_l1651_165196

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem find_a4 (a d : ℤ)
    (h₁ : sum_first_n_terms a d 5 = 15)
    (h₂ : sum_first_n_terms a d 9 = 63) :
  arithmetic_sequence a d 4 = 5 :=
sorry

end find_a4_l1651_165196


namespace sum_of_consecutive_integers_l1651_165161

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l1651_165161


namespace Seulgi_second_round_need_l1651_165141

def Hohyeon_first_round := 23
def Hohyeon_second_round := 28
def Hyunjeong_first_round := 32
def Hyunjeong_second_round := 17
def Seulgi_first_round := 27

def Hohyeon_total := Hohyeon_first_round + Hohyeon_second_round
def Hyunjeong_total := Hyunjeong_first_round + Hyunjeong_second_round

def required_total_for_Seulgi := Hohyeon_total + 1

theorem Seulgi_second_round_need (Seulgi_second_round: ℕ) :
  Seulgi_first_round + Seulgi_second_round ≥ required_total_for_Seulgi → Seulgi_second_round ≥ 25 :=
by
  sorry

end Seulgi_second_round_need_l1651_165141


namespace solution_set_of_inequality_l1651_165184

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 6

theorem solution_set_of_inequality (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ (-1/3 : ℝ) < m ∧ m < 3 :=
by 
  sorry

end solution_set_of_inequality_l1651_165184


namespace pumpkin_pie_filling_l1651_165109

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l1651_165109


namespace each_squirrel_needs_more_acorns_l1651_165168

noncomputable def acorns_needed : ℕ := 300
noncomputable def total_acorns_collected : ℕ := 4500
noncomputable def number_of_squirrels : ℕ := 20

theorem each_squirrel_needs_more_acorns : 
  (acorns_needed - total_acorns_collected / number_of_squirrels) = 75 :=
by
  sorry

end each_squirrel_needs_more_acorns_l1651_165168


namespace max_integer_value_l1651_165181

theorem max_integer_value (x : ℝ) : 
  ∃ (n : ℤ), n = 15 ∧ ∀ x : ℝ, 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ n :=
by
  sorry

end max_integer_value_l1651_165181


namespace percentage_x_y_l1651_165188

variable (x y P : ℝ)

theorem percentage_x_y 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y))
  (h2 : y = (1 / 9) * x) : 
  P = 40 :=
sorry

end percentage_x_y_l1651_165188


namespace perfect_squares_less_than_20000_representable_l1651_165112

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end perfect_squares_less_than_20000_representable_l1651_165112


namespace polynomial_roots_absolute_sum_l1651_165182

theorem polynomial_roots_absolute_sum (p q r : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2027) :
  |p| + |q| + |r| = 98 := 
sorry

end polynomial_roots_absolute_sum_l1651_165182


namespace correct_option_B_l1651_165160

theorem correct_option_B (a : ℤ) : (2 * a) ^ 3 = 8 * a ^ 3 :=
by
  sorry

end correct_option_B_l1651_165160


namespace geometric_sequence_product_l1651_165130

variable {α : Type*} [LinearOrderedField α]

theorem geometric_sequence_product :
  ∀ (a r : α), (a^3 * r^6 = 3) → (a^3 * r^15 = 24) → (a^3 * r^24 = 192) :=
by
  intros a r h1 h2
  sorry

end geometric_sequence_product_l1651_165130


namespace roots_of_equation_l1651_165132

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l1651_165132


namespace cosine_value_l1651_165104

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

noncomputable def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

noncomputable def magnitude (x : ℝ × ℝ) : ℝ :=
  (x.1 ^ 2 + x.2 ^ 2).sqrt

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_value :
  cos_angle a b = 2 * (5:ℝ).sqrt / 25 :=
by
  sorry

end cosine_value_l1651_165104


namespace prob_c_not_adjacent_to_a_or_b_l1651_165167

-- Definitions for the conditions
def num_students : ℕ := 7
def a_and_b_together : Prop := true
def c_on_edge : Prop := true

-- Main theorem: probability c not adjacent to a or b under given conditions
theorem prob_c_not_adjacent_to_a_or_b
  (h1 : a_and_b_together)
  (h2 : c_on_edge) :
  ∃ (p : ℚ), p = 0.8 := by
  sorry

end prob_c_not_adjacent_to_a_or_b_l1651_165167


namespace number_of_children_l1651_165170

-- Definitions based on conditions
def numDogs : ℕ := 2
def numCats : ℕ := 1
def numLegsTotal : ℕ := 22
def numLegsDog : ℕ := 4
def numLegsCat : ℕ := 4
def numLegsHuman : ℕ := 2

-- Main theorem proving the number of children
theorem number_of_children :
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  numChildren = 4 :=
by
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  exact sorry

end number_of_children_l1651_165170


namespace total_amount_invested_l1651_165162

-- Define the conditions and specify the correct answer
theorem total_amount_invested (x y : ℝ) (h8 : y = 600) 
  (h_income_diff : 0.10 * (x - 600) - 0.08 * 600 = 92) : 
  x + y = 2000 := sorry

end total_amount_invested_l1651_165162


namespace count_multiples_of_7_not_14_lt_400_l1651_165116

theorem count_multiples_of_7_not_14_lt_400 : 
  ∃ (n : ℕ), n = 29 ∧ ∀ (m : ℕ), (m < 400 ∧ m % 7 = 0 ∧ m % 14 ≠ 0) ↔ (∃ k : ℕ, 1 ≤ k ∧ k ≤ 29 ∧ m = 7 * (2 * k - 1)) :=
by
  sorry

end count_multiples_of_7_not_14_lt_400_l1651_165116


namespace lean_proof_l1651_165163

noncomputable def proof_problem (a b c d : ℝ) (habcd : a * b * c * d = 1) : Prop :=
  (1 + a * b) / (1 + a) ^ 2008 +
  (1 + b * c) / (1 + b) ^ 2008 +
  (1 + c * d) / (1 + c) ^ 2008 +
  (1 + d * a) / (1 + d) ^ 2008 ≥ 4

theorem lean_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_abcd : a * b * c * d = 1) : proof_problem a b c d h_abcd :=
  sorry

end lean_proof_l1651_165163


namespace Elmer_vs_Milton_food_l1651_165128

def Penelope_daily_food := 20  -- Penelope eats 20 pounds per day
def Greta_to_Penelope_ratio := 1 / 10  -- Greta eats 1/10 of what Penelope eats
def Milton_to_Greta_ratio := 1 / 100  -- Milton eats 1/100 of what Greta eats
def Elmer_to_Penelope_difference := 60  -- Elmer eats 60 pounds more than Penelope

def Greta_daily_food := Penelope_daily_food * Greta_to_Penelope_ratio
def Milton_daily_food := Greta_daily_food * Milton_to_Greta_ratio
def Elmer_daily_food := Penelope_daily_food + Elmer_to_Penelope_difference

theorem Elmer_vs_Milton_food :
  Elmer_daily_food = 4000 * Milton_daily_food := by
  sorry

end Elmer_vs_Milton_food_l1651_165128


namespace intersection_M_N_l1651_165100

-- Definitions of the sets M and N based on the conditions
def M (x : ℝ) : Prop := ∃ (y : ℝ), y = Real.log (x^2 - 3*x - 4)
def N (y : ℝ) : Prop := ∃ (x : ℝ), y = 2^(x - 1)

-- The proof statement
theorem intersection_M_N : { x : ℝ | M x } ∩ { x : ℝ | ∃ y : ℝ, N y ∧ y = Real.log (x^2 - 3*x - 4) } = { x : ℝ | x > 4 } :=
by
  sorry

end intersection_M_N_l1651_165100


namespace cookies_taken_in_four_days_l1651_165106

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l1651_165106


namespace emma_final_balance_correct_l1651_165136

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l1651_165136


namespace part1_part2_l1651_165115

/-- Definition of set A as roots of the equation x^2 - 3x + 2 = 0 --/
def set_A : Set ℝ := {x | x ^ 2 - 3 * x + 2 = 0}

/-- Definition of set B as roots of the equation x^2 + (a - 1)x + a^2 - 5 = 0 --/
def set_B (a : ℝ) : Set ℝ := {x | x ^ 2 + (a - 1) * x + a ^ 2 - 5 = 0}

/-- Proof for intersection condition --/
theorem part1 (a : ℝ) : (set_A ∩ set_B a = {2}) → (a = -3 ∨ a = 1) := by
  sorry

/-- Proof for union condition --/
theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -3 ∨ a > 7 / 3) := by
  sorry

end part1_part2_l1651_165115


namespace rationalize_cube_root_sum_l1651_165193

theorem rationalize_cube_root_sum :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  A + B + C + D = 51 :=
by
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  have step1 : (a^3 = 5) := by sorry
  have step2 : (b^3 = 3) := by sorry
  have denom_eq : denom = 2 := by sorry
  have frac_simp : fraction = (A^(1/3) + B^(1/3) + C^(1/3)) / D := by sorry
  show A + B + C + D = 51
  sorry

end rationalize_cube_root_sum_l1651_165193


namespace Isabel_paper_used_l1651_165119

theorem Isabel_paper_used
  (initial_pieces : ℕ)
  (remaining_pieces : ℕ)
  (initial_condition : initial_pieces = 900)
  (remaining_condition : remaining_pieces = 744) :
  initial_pieces - remaining_pieces = 156 :=
by 
  -- Admitting the proof for now
  sorry

end Isabel_paper_used_l1651_165119


namespace part1_part2_l1651_165101

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l1651_165101


namespace polygon_sides_l1651_165117

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l1651_165117


namespace binom_sum_l1651_165183

theorem binom_sum :
  (Nat.choose 15 12) + 10 = 465 := by
  sorry

end binom_sum_l1651_165183


namespace unique_digit_sum_l1651_165127

theorem unique_digit_sum (Y M E T : ℕ) (h1 : Y ≠ M) (h2 : Y ≠ E) (h3 : Y ≠ T)
    (h4 : M ≠ E) (h5 : M ≠ T) (h6 : E ≠ T) (h7 : 10 * Y + E = YE) (h8 : 10 * M + E = ME)
    (h9 : YE * ME = T * T * T) (hT_even : T % 2 = 0) : 
    Y + M + E + T = 10 :=
  sorry

end unique_digit_sum_l1651_165127


namespace cos_alpha_second_quadrant_l1651_165157

theorem cos_alpha_second_quadrant (α : ℝ) (h₁ : (π / 2) < α ∧ α < π) (h₂ : Real.sin α = 5 / 13) :
  Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_second_quadrant_l1651_165157


namespace cost_rose_bush_l1651_165186

-- Define the constants
def total_roses := 6
def friend_roses := 2
def total_aloes := 2
def cost_aloe := 100
def total_spent_self := 500

-- Prove the cost of each rose bush
theorem cost_rose_bush : (total_spent_self - total_aloes * cost_aloe) / (total_roses - friend_roses) = 75 :=
by
  sorry

end cost_rose_bush_l1651_165186

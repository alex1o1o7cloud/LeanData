import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1999_199907

theorem simplify_expression (a : ℝ) (h₀ : a ≥ 0) (h₁ : a ≠ 1) (h₂ : a ≠ 1 + Real.sqrt 2) (h₃ : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a ^ (1 / 4) - a ^ (1 / 2)) / (1 - a + 4 * a ^ (3 / 4) - 4 * a ^ (1 / 2)) +
  (a ^ (1 / 4) - 2) / (a ^ (1 / 4) - 1) ^ 2 = 1 / (a ^ (1 / 4) - 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1999_199907


namespace NUMINAMATH_GPT_range_of_a_l1999_199972

-- Define set A
def setA (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a^2 + 1

-- Define set B
def setB (x a : ℝ) : Prop := (x - 2) * (x - (3 * a + 1)) ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, setA x a → setB x a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1999_199972


namespace NUMINAMATH_GPT_find_a_l1999_199954

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a (a : ℝ) :
  ∅ ⊂ (A a ∩ B) ∧ A a ∩ C = ∅ → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1999_199954


namespace NUMINAMATH_GPT_find_y_l1999_199921

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1999_199921


namespace NUMINAMATH_GPT_complete_contingency_table_chi_sq_test_result_expected_value_X_l1999_199966

noncomputable def probability_set := {x : ℚ // x ≥ 0 ∧ x ≤ 1}

variable (P : probability_set → probability_set)

-- Conditions from the problem
def P_A_given_not_B : probability_set := ⟨2 / 5, by norm_num⟩
def P_B_given_not_A : probability_set := ⟨5 / 8, by norm_num⟩
def P_B : probability_set := ⟨3 / 4, by norm_num⟩

-- Definitions related to counts and probabilities
def total_students : ℕ := 200
def male_students := P_A_given_not_B.val * total_students
def female_students := total_students - male_students
def score_exceeds_85 := P_B.val * total_students
def score_not_exceeds_85 := total_students - score_exceeds_85

-- Expected counts based on given probabilities
def male_score_not_exceeds_85 := P_A_given_not_B.val * score_not_exceeds_85
def female_score_not_exceeds_85 := score_not_exceeds_85 - male_score_not_exceeds_85
def male_score_exceeds_85 := male_students - male_score_not_exceeds_85
def female_score_exceeds_85 := female_students - female_score_not_exceeds_85

-- Chi-squared test independence 
def chi_squared := (total_students * (male_score_not_exceeds_85 * female_score_exceeds_85 - female_score_not_exceeds_85 * male_score_exceeds_85) ^ 2) / 
                    (male_students * female_students * score_not_exceeds_85 * score_exceeds_85)
def is_related : Prop := chi_squared > 10.828

-- Expected distributions and expectation of X
def P_X_0 := (1 / 4) ^ 2 * (1 / 3) ^ 2
def P_X_1 := 2 * (3 / 4) * (1 / 4) * (1 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (1 / 4) ^ 2
def P_X_2 := (3 / 4) ^ 2 * (1 / 3) ^ 2 + (1 / 4) ^ 2 * (2 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (3 / 4) * (1 / 4)
def P_X_3 := (3 / 4) ^ 2 * 2 * (2 / 3) * (1 / 3) + 2 * (3 / 4) * (1 / 4) * (2 / 3) ^ 2
def P_X_4 := (3 / 4) ^ 2 * (2 / 3) ^ 2
def expectation_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Lean theorem statements for answers using the above definitions
theorem complete_contingency_table :
  male_score_not_exceeds_85 + female_score_not_exceeds_85 = score_not_exceeds_85 ∧
  male_score_exceeds_85 + female_score_exceeds_85 = score_exceeds_85 ∧
  male_students + female_students = total_students := sorry

theorem chi_sq_test_result :
  is_related = true := sorry

theorem expected_value_X :
  expectation_X = 17 / 6 := sorry

end NUMINAMATH_GPT_complete_contingency_table_chi_sq_test_result_expected_value_X_l1999_199966


namespace NUMINAMATH_GPT_ellipse_x1_x2_squared_sum_eq_4_l1999_199958

theorem ellipse_x1_x2_squared_sum_eq_4
  (x₁ y₁ x₂ y₂ : ℝ)
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = 1)
  (hM : x₁^2 / a^2 + y₁^2 = 1)
  (hN : x₂^2 / a^2 + y₂^2 = 1)
  (h_slope_product : (y₁ / x₁) * (y₂ / x₂) = -1 / 4) :
  x₁^2 + x₂^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_x1_x2_squared_sum_eq_4_l1999_199958


namespace NUMINAMATH_GPT_Jaylen_total_vegetables_l1999_199995

def Jaylen_vegetables (J_bell_peppers J_green_beans J_carrots J_cucumbers : Nat) : Nat :=
  J_bell_peppers + J_green_beans + J_carrots + J_cucumbers

theorem Jaylen_total_vegetables :
  let Kristin_bell_peppers := 2
  let Kristin_green_beans := 20
  let Jaylen_bell_peppers := 2 * Kristin_bell_peppers
  let Jaylen_green_beans := (Kristin_green_beans / 2) - 3
  let Jaylen_carrots := 5
  let Jaylen_cucumbers := 2
  Jaylen_vegetables Jaylen_bell_peppers Jaylen_green_beans Jaylen_carrots Jaylen_cucumbers = 18 := 
by
  sorry

end NUMINAMATH_GPT_Jaylen_total_vegetables_l1999_199995


namespace NUMINAMATH_GPT_defective_units_percentage_l1999_199951

variables (D : ℝ)

-- 4% of the defective units are shipped for sale
def percent_defective_shipped : ℝ := 0.04

-- 0.24% of the units produced are defective units that are shipped for sale
def percent_total_defective_shipped : ℝ := 0.0024

-- The theorem to prove: the percentage of the units produced that are defective is 0.06
theorem defective_units_percentage (h : percent_defective_shipped * D = percent_total_defective_shipped) : D = 0.06 :=
sorry

end NUMINAMATH_GPT_defective_units_percentage_l1999_199951


namespace NUMINAMATH_GPT_no_pairs_probability_l1999_199938

-- Define the number of socks and initial conditions
def pairs_of_socks : ℕ := 3
def total_socks : ℕ := pairs_of_socks * 2

-- Probabilistic outcome space for no pairs in first three draws
def probability_no_pairs_in_first_three_draws : ℚ :=
  (4/5) * (1/2)

-- Theorem stating that probability of no matching pairs in the first three draws is 2/5
theorem no_pairs_probability : probability_no_pairs_in_first_three_draws = 2/5 := by
  sorry

end NUMINAMATH_GPT_no_pairs_probability_l1999_199938


namespace NUMINAMATH_GPT_Jane_Hector_meet_point_C_l1999_199911

theorem Jane_Hector_meet_point_C (s t : ℝ) (h_start : ℝ) (j_start : ℝ) (loop_length : ℝ) 
  (h_speed : ℝ) (j_speed : ℝ) (h_dest : ℝ) (j_dest : ℝ)
  (h_speed_eq : h_speed = s) (j_speed_eq : j_speed = 3 * s) (loop_len_eq : loop_length = 30)
  (start_point_eq : h_start = 0 ∧ j_start = 0)
  (opposite_directions : h_dest + j_dest = loop_length)
  (meet_time_eq : t = 15 / (2 * s)) :
  h_dest = 7.5 ∧ j_dest = 22.5 → (h_dest = 7.5 ∧ j_dest = 22.5) :=
by
  sorry

end NUMINAMATH_GPT_Jane_Hector_meet_point_C_l1999_199911


namespace NUMINAMATH_GPT_mutually_prime_sum_l1999_199939

open Real

theorem mutually_prime_sum (A B C : ℤ) (h_prime : Int.gcd A (Int.gcd B C) = 1)
    (h_eq : A * log 5 / log 200 + B * log 2 / log 200 = C) : A + B + C = 6 := 
sorry

end NUMINAMATH_GPT_mutually_prime_sum_l1999_199939


namespace NUMINAMATH_GPT_solution_A_to_B_ratio_l1999_199903

def ratio_solution_A_to_B (V_A V_B : ℝ) : Prop :=
  (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B) → V_A / V_B = 5 / 6

theorem solution_A_to_B_ratio (V_A V_B : ℝ) (h : (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B)) :
  V_A / V_B = 5 / 6 :=
sorry

end NUMINAMATH_GPT_solution_A_to_B_ratio_l1999_199903


namespace NUMINAMATH_GPT_geometric_sequence_sum_correct_l1999_199996

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 2 then 2^(n + 1) - 2
else 64 * (1 - (1 / 2)^n)

theorem geometric_sequence_sum_correct (a1 q : ℝ) (n : ℕ) 
  (h1 : q > 0) 
  (h2 : a1 + a1 * q^4 = 34) 
  (h3 : a1^2 * q^4 = 64) :
  geometric_sequence_sum a1 q n = 
  if q = 2 then 2^(n + 1) - 2 else 64 * (1 - (1 / 2)^n) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_correct_l1999_199996


namespace NUMINAMATH_GPT_intersection_points_A_B_segment_length_MN_l1999_199981

section PolarCurves

-- Given conditions
def curve1 (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8
def curve2 (θ : ℝ) : Prop := θ = Real.pi / 6
def is_on_line (x y t : ℝ) : Prop := x = 2 + Real.sqrt 3 / 2 * t ∧ y = 1 / 2 * t

-- Polar coordinates of points A and B
theorem intersection_points_A_B :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : ℝ), curve1 ρ₁ θ₁ ∧ curve2 θ₁ ∧ curve1 ρ₂ θ₂ ∧ curve2 θ₂ ∧
    (ρ₁, θ₁) = (4, Real.pi / 6) ∧ (ρ₂, θ₂) = (4, -Real.pi / 6) :=
sorry

-- Length of the segment MN
theorem segment_length_MN :
  ∀ t : ℝ, curve1 (2 + Real.sqrt 3 / 2 * t) (1 / 2 * t) →
    ∃ t₁ t₂ : ℝ, (is_on_line (2 + Real.sqrt 3 / 2 * t₁) (1 / 2 * t₁) t₁) ∧
                (is_on_line (2 + Real.sqrt 3 / 2 * t₂) (1 / 2 * t₂) t₂) ∧
                Real.sqrt ((2 * -Real.sqrt 3 * 4)^2 - 4 * (-8)) = 4 * Real.sqrt 5 :=
sorry

end PolarCurves

end NUMINAMATH_GPT_intersection_points_A_B_segment_length_MN_l1999_199981


namespace NUMINAMATH_GPT_barbie_earrings_l1999_199930

theorem barbie_earrings (total_earrings_alissa : ℕ) (alissa_triple_given : ℕ → ℕ) 
  (given_earrings_double_bought : ℕ → ℕ) (pairs_of_earrings : ℕ) : 
  total_earrings_alissa = 36 → 
  alissa_triple_given (total_earrings_alissa / 3) = total_earrings_alissa → 
  given_earrings_double_bought (total_earrings_alissa / 3) = total_earrings_alissa →
  pairs_of_earrings = 12 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_barbie_earrings_l1999_199930


namespace NUMINAMATH_GPT_not_consecutive_l1999_199991

theorem not_consecutive (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) : 
  ¬ (∃ n : ℕ, (2023 + a - b = n ∧ 2023 + b - c = n + 1 ∧ 2023 + c - a = n + 2) ∨ 
    (2023 + a - b = n ∧ 2023 + b - c = n - 1 ∧ 2023 + c - a = n - 2)) :=
by
  sorry

end NUMINAMATH_GPT_not_consecutive_l1999_199991


namespace NUMINAMATH_GPT_linear_equation_unique_l1999_199914

theorem linear_equation_unique (x y : ℝ) : 
  (3 * x = 2 * y) ∧ 
  ¬(3 * x - 6 = x) ∧ 
  ¬(x - 1 / y = 0) ∧ 
  ¬(2 * x - 3 * y = x * y) :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_unique_l1999_199914


namespace NUMINAMATH_GPT_Danny_more_wrappers_than_bottle_caps_l1999_199976

theorem Danny_more_wrappers_than_bottle_caps
  (initial_wrappers : ℕ)
  (initial_bottle_caps : ℕ)
  (found_wrappers : ℕ)
  (found_bottle_caps : ℕ) :
  initial_wrappers = 67 →
  initial_bottle_caps = 35 →
  found_wrappers = 18 →
  found_bottle_caps = 15 →
  (initial_wrappers + found_wrappers) - (initial_bottle_caps + found_bottle_caps) = 35 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Danny_more_wrappers_than_bottle_caps_l1999_199976


namespace NUMINAMATH_GPT_quadratic_rewrite_l1999_199978

theorem quadratic_rewrite :
  ∃ d e f : ℤ, (4 * (x : ℝ)^2 - 24 * x + 35 = (d * x + e)^2 + f) ∧ (d * e = -12) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_l1999_199978


namespace NUMINAMATH_GPT_joey_total_study_time_l1999_199927

def hours_weekdays (hours_per_night : Nat) (nights_per_week : Nat) : Nat :=
  hours_per_night * nights_per_week

def hours_weekends (hours_per_day : Nat) (days_per_weekend : Nat) : Nat :=
  hours_per_day * days_per_weekend

def total_weekly_study_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours + weekend_hours

def total_study_time_in_weeks (weekly_hours : Nat) (weeks : Nat) : Nat :=
  weekly_hours * weeks

theorem joey_total_study_time :
  let hours_per_night := 2
  let nights_per_week := 5
  let hours_per_day := 3
  let days_per_weekend := 2
  let weeks := 6
  hours_weekdays hours_per_night nights_per_week +
  hours_weekends hours_per_day days_per_weekend = 16 →
  total_study_time_in_weeks 16 weeks = 96 :=
by 
  intros h1 h2 h3 h4 h5
  have weekday_hours := hours_weekdays h1 h2
  have weekend_hours := hours_weekends h3 h4
  have total_weekly := total_weekly_study_time weekday_hours weekend_hours
  sorry

end NUMINAMATH_GPT_joey_total_study_time_l1999_199927


namespace NUMINAMATH_GPT_intersection_A_B_l1999_199953

-- Definition of sets A and B based on given conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3 }
def B : Set ℝ := {y | ∃ x : ℝ, x < 0 ∧ y = x + 1 / x }

-- Proving the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {y | -4 ≤ y ∧ y ≤ -2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1999_199953


namespace NUMINAMATH_GPT_average_trees_planted_l1999_199942

theorem average_trees_planted 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ) 
  (h1 : A = 35) 
  (h2 : B = A + 6) 
  (h3 : C = A - 3) : 
  (A + B + C) / 3 = 36 :=
  by
  sorry

end NUMINAMATH_GPT_average_trees_planted_l1999_199942


namespace NUMINAMATH_GPT_gcd_q_r_min_value_l1999_199969

theorem gcd_q_r_min_value (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) : Nat.gcd q r = 10 :=
sorry

end NUMINAMATH_GPT_gcd_q_r_min_value_l1999_199969


namespace NUMINAMATH_GPT_remainder_444_444_mod_13_l1999_199932

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_remainder_444_444_mod_13_l1999_199932


namespace NUMINAMATH_GPT_production_days_l1999_199998

-- Definitions of the conditions
variables (n : ℕ) (P : ℕ)
variable (H1 : P = n * 50)
variable (H2 : (P + 60) / (n + 1) = 55)

-- Theorem to prove that n = 1 given the conditions
theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 60) / (n + 1) = 55) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_production_days_l1999_199998


namespace NUMINAMATH_GPT_ratio_female_to_male_l1999_199923

variable {f m c : ℕ}

/-- 
  The following conditions are given:
  - The average age of female members is 35 years.
  - The average age of male members is 30 years.
  - The average age of children members is 10 years.
  - The average age of the entire membership is 25 years.
  - The number of children members is equal to the number of male members.
  We need to show that the ratio of female to male members is 1.
-/
theorem ratio_female_to_male (h1 : c = m)
  (h2 : 35 * f + 40 * m = 25 * (f + 2 * m)) :
  f = m :=
by sorry

end NUMINAMATH_GPT_ratio_female_to_male_l1999_199923


namespace NUMINAMATH_GPT_total_bill_is_60_l1999_199943

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end NUMINAMATH_GPT_total_bill_is_60_l1999_199943


namespace NUMINAMATH_GPT_solution_80_percent_needs_12_ounces_l1999_199936

theorem solution_80_percent_needs_12_ounces:
  ∀ (x y: ℝ), (x + y = 40) → (0.30 * x + 0.80 * y = 0.45 * 40) → (y = 12) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_solution_80_percent_needs_12_ounces_l1999_199936


namespace NUMINAMATH_GPT_range_of_a_l1999_199900

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

def neg_p : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 < 0

theorem range_of_a (h : neg_p a) : a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1999_199900


namespace NUMINAMATH_GPT_sale_percent_saved_l1999_199928

noncomputable def percent_saved (P : ℝ) : ℝ := (3 * P) / (6 * P) * 100

theorem sale_percent_saved :
  ∀ (P : ℝ), P > 0 → percent_saved P = 50 :=
by
  intros P hP
  unfold percent_saved
  have hP_nonzero : 6 * P ≠ 0 := by linarith
  field_simp [hP_nonzero]
  norm_num
  sorry

end NUMINAMATH_GPT_sale_percent_saved_l1999_199928


namespace NUMINAMATH_GPT_cube_volume_increase_l1999_199975

variable (a : ℝ)

theorem cube_volume_increase (a : ℝ) : (2 * a)^3 - a^3 = 7 * a^3 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_increase_l1999_199975


namespace NUMINAMATH_GPT_power_mean_inequality_l1999_199935

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := 
by
  sorry

end NUMINAMATH_GPT_power_mean_inequality_l1999_199935


namespace NUMINAMATH_GPT_g_50_equals_zero_l1999_199973

noncomputable def g : ℝ → ℝ := sorry

theorem g_50_equals_zero (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g ((x + y) / y)) : g 50 = 0 :=
sorry

end NUMINAMATH_GPT_g_50_equals_zero_l1999_199973


namespace NUMINAMATH_GPT_number_is_46000050_l1999_199993

-- Define the corresponding place values for the given digit placements
def ten_million (n : ℕ) : ℕ := n * 10000000
def hundred_thousand (n : ℕ) : ℕ := n * 100000
def hundred (n : ℕ) : ℕ := n * 100

-- Define the specific numbers given in the conditions.
def digit_4 : ℕ := ten_million 4
def digit_60 : ℕ := hundred_thousand 6
def digit_500 : ℕ := hundred 5

-- Combine these values to form the number
def combined_number : ℕ := digit_4 + digit_60 + digit_500

-- The theorem, stating the number equals 46000050
theorem number_is_46000050 : combined_number = 46000050 := by
  sorry

end NUMINAMATH_GPT_number_is_46000050_l1999_199993


namespace NUMINAMATH_GPT_savings_account_amount_l1999_199901

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end NUMINAMATH_GPT_savings_account_amount_l1999_199901


namespace NUMINAMATH_GPT_composite_quadratic_l1999_199904

theorem composite_quadratic (a b : Int) (x1 x2 : Int)
  (h1 : x1 + x2 = -a)
  (h2 : x1 * x2 = b)
  (h3 : abs x1 > 2)
  (h4 : abs x2 > 2) :
  ∃ m n : Int, a + b + 1 = m * n ∧ m > 1 ∧ n > 1 :=
by
  sorry

end NUMINAMATH_GPT_composite_quadratic_l1999_199904


namespace NUMINAMATH_GPT_factorize_expression_l1999_199963

variable (x : ℝ)

theorem factorize_expression : x^2 + x = x * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1999_199963


namespace NUMINAMATH_GPT_equation_of_l3_line_l1_through_fixed_point_existence_of_T_l1999_199912

-- Question 1: The equation of the line \( l_{3} \)
theorem equation_of_l3 
  (F : ℝ × ℝ) 
  (H_focus : F = (2, 0))
  (k : ℝ) 
  (H_slope : k = 1) : 
  (∀ x y : ℝ, y = k * x + -2 ↔ y = x - 2) := 
sorry

-- Question 2: Line \( l_{1} \) passes through the fixed point (8, 0)
theorem line_l1_through_fixed_point 
  (k m1 : ℝ)
  (H_km1 : k * m1 ≠ 0)
  (H_m1lt : m1 < -t)
  (H_condition : ∃ x y : ℝ, y = k * x + m1 ∧ x^2 + (8/k) * x + (8 * m1 / k) = 0 ∧ ((x, y) = A1 ∨ (x, y) = B1))
  (H_dot_product : (x1 - 0)*(x2 - 0) + (y1 - 0)*(y2 - 0) = 0) : 
  ∀ P : ℝ × ℝ, P = (8, 0) := 
sorry

-- Question 3: Existence of point T such that S_i and d_i form geometric sequences
theorem existence_of_T
  (k : ℝ)
  (H_k : k = 1)
  (m1 m2 m3 : ℝ)
  (H_m_ordered : m1 < m2 ∧ m2 < m3 ∧ m3 < -t)
  (t : ℝ)
  (S1 S2 S3 d1 d2 d3 : ℝ)
  (H_S_geom_seq : S2^2 = S1 * S3)
  (H_d_geom_seq : d2^2 = d1 * d3)
  : ∃ t : ℝ, t = -2 :=
sorry

end NUMINAMATH_GPT_equation_of_l3_line_l1_through_fixed_point_existence_of_T_l1999_199912


namespace NUMINAMATH_GPT_symmetry_line_intersection_l1999_199968

theorem symmetry_line_intersection 
  (k : ℝ) (k_pos : k > 0) (k_ne_one : k ≠ 1)
  (k1 : ℝ) (h_sym : ∀ (P : ℝ × ℝ), (P.2 = k1 * P.1 + 1) ↔ P.2 - 1 = k * (P.1 + 1) + 1)
  (H : ∀ M : ℝ × ℝ, (M.2 = k * M.1 + 1) → (M.1^2 / 4 + M.2^2 = 1)) :
  (k * k1 = 1) ∧ (∀ k : ℝ, ∃ P : ℝ × ℝ, (P.fst = 0) ∧ (P.snd = -5 / 3)) :=
sorry

end NUMINAMATH_GPT_symmetry_line_intersection_l1999_199968


namespace NUMINAMATH_GPT_y_squared_in_range_l1999_199918

theorem y_squared_in_range (y : ℝ) 
  (h : (Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2)) :
  270 ≤ y^2 ∧ y^2 ≤ 280 :=
sorry

end NUMINAMATH_GPT_y_squared_in_range_l1999_199918


namespace NUMINAMATH_GPT_total_increase_by_five_l1999_199937

-- Let B be the number of black balls
variable (B : ℕ)
-- Let W be the number of white balls
variable (W : ℕ)
-- Initially the total number of balls
def T := B + W
-- If the number of black balls is increased to 5 times the original, the total becomes twice the original
axiom h1 : 5 * B + W = 2 * (B + W)
-- If the number of white balls is increased to 5 times the original 
def k : ℕ := 5
-- The new total number of balls 
def new_total := B + k * W

-- Prove that the new total is 4 times the original total.
theorem total_increase_by_five : new_total = 4 * T :=
by
sorry

end NUMINAMATH_GPT_total_increase_by_five_l1999_199937


namespace NUMINAMATH_GPT_cannot_be_combined_with_sqrt2_l1999_199985

def can_be_combined (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y

theorem cannot_be_combined_with_sqrt2 :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 8
  let c := Real.sqrt 12
  let d := -Real.sqrt 18
  ¬ can_be_combined c (Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_cannot_be_combined_with_sqrt2_l1999_199985


namespace NUMINAMATH_GPT_compare_abc_l1999_199948

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := (1 / 3 : ℝ) ^ (1 / 3 : ℝ)
noncomputable def c : ℝ := (3 : ℝ) ^ (-1 / 4 : ℝ)

theorem compare_abc : b < c ∧ c < a :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l1999_199948


namespace NUMINAMATH_GPT_find_width_of_lawn_l1999_199946

noncomputable def width_of_lawn
    (length : ℕ)
    (cost : ℕ)
    (cost_per_sq_m : ℕ)
    (road_width : ℕ) : ℕ :=
  let total_area := cost / cost_per_sq_m
  let road_area_length := road_width * length
  let eq_area := total_area - road_area_length
  eq_area / road_width

theorem find_width_of_lawn :
  width_of_lawn 110 4800 3 10 = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_lawn_l1999_199946


namespace NUMINAMATH_GPT_graph_symmetric_l1999_199949

noncomputable def f (x : ℝ) : ℝ := sorry

theorem graph_symmetric (f : ℝ → ℝ) :
  (∀ x y, y = f x ↔ (∃ y₁, y₁ = f (2 - x) ∧ y = - (1 / (y₁ + 1)))) →
  ∀ x, f x = 1 / (x - 3) := 
by
  intro h x
  sorry

end NUMINAMATH_GPT_graph_symmetric_l1999_199949


namespace NUMINAMATH_GPT_find_f_prime_one_l1999_199909

theorem find_f_prime_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * f' 1 + 1 / x) (h_fx : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_prime_one_l1999_199909


namespace NUMINAMATH_GPT_population_present_l1999_199933

theorem population_present (P : ℝ) (h : P * (1.1)^3 = 79860) : P = 60000 :=
sorry

end NUMINAMATH_GPT_population_present_l1999_199933


namespace NUMINAMATH_GPT_find_a_if_f_is_even_l1999_199999

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

theorem find_a_if_f_is_even
  (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_if_f_is_even_l1999_199999


namespace NUMINAMATH_GPT_inequality_solution_l1999_199983

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1999_199983


namespace NUMINAMATH_GPT_x_squared_plus_inverse_squared_l1999_199977

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 3.5) : x^2 + (1/x)^2 = 10.25 :=
by sorry

end NUMINAMATH_GPT_x_squared_plus_inverse_squared_l1999_199977


namespace NUMINAMATH_GPT_distance_between_countries_l1999_199940

theorem distance_between_countries (total_distance : ℕ) (spain_germany : ℕ) (spain_other : ℕ) :
  total_distance = 7019 →
  spain_germany = 1615 →
  spain_other = total_distance - spain_germany →
  spain_other = 5404 :=
by
  intros h_total_distance h_spain_germany h_spain_other
  rw [h_total_distance, h_spain_germany] at h_spain_other
  exact h_spain_other

end NUMINAMATH_GPT_distance_between_countries_l1999_199940


namespace NUMINAMATH_GPT_angle_equivalence_l1999_199967

theorem angle_equivalence :
  ∃ k : ℤ, -495 + 360 * k = 225 :=
sorry

end NUMINAMATH_GPT_angle_equivalence_l1999_199967


namespace NUMINAMATH_GPT_probability_even_sum_5_balls_drawn_l1999_199965

theorem probability_even_sum_5_balls_drawn :
  let total_ways := (Nat.choose 12 5)
  let favorable_ways := (Nat.choose 6 0) * (Nat.choose 6 5) + 
                        (Nat.choose 6 2) * (Nat.choose 6 3) + 
                        (Nat.choose 6 4) * (Nat.choose 6 1)
  favorable_ways / total_ways = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_probability_even_sum_5_balls_drawn_l1999_199965


namespace NUMINAMATH_GPT_marcus_percentage_of_team_points_l1999_199986

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_marcus_percentage_of_team_points_l1999_199986


namespace NUMINAMATH_GPT_number_of_correct_answers_l1999_199990

theorem number_of_correct_answers (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 110) : c = 34 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_number_of_correct_answers_l1999_199990


namespace NUMINAMATH_GPT_ef_plus_e_l1999_199925

-- Define the polynomial expression
def polynomial_expr (y : ℤ) := 15 * y^2 - 82 * y + 48

-- Define the factorized form
def factorized_form (E F : ℤ) (y : ℤ) := (E * y - 16) * (F * y - 3)

-- Define the main statement to prove
theorem ef_plus_e : ∃ E F : ℤ, E * F + E = 20 ∧ ∀ y : ℤ, polynomial_expr y = factorized_form E F y :=
by {
  sorry
}

end NUMINAMATH_GPT_ef_plus_e_l1999_199925


namespace NUMINAMATH_GPT_bonus_trigger_sales_amount_l1999_199941

theorem bonus_trigger_sales_amount (total_sales S : ℝ) (h1 : 0.09 * total_sales = 1260)
  (h2 : 0.03 * (total_sales - S) = 120) : S = 10000 :=
sorry

end NUMINAMATH_GPT_bonus_trigger_sales_amount_l1999_199941


namespace NUMINAMATH_GPT_flat_odot_length_correct_l1999_199987

noncomputable def sides : ℤ × ℤ × ℤ := (4, 5, 6)

noncomputable def semiperimeter (a b c : ℤ) : ℚ :=
  (a + b + c) / 2

noncomputable def length_flat_odot (a b c : ℤ) : ℚ :=
  (semiperimeter a b c) - b

theorem flat_odot_length_correct : length_flat_odot 4 5 6 = 2.5 := by
  sorry

end NUMINAMATH_GPT_flat_odot_length_correct_l1999_199987


namespace NUMINAMATH_GPT_solution_proof_l1999_199952

variable (A B C : ℕ+) (x y : ℚ)
variable (h1 : A > B) (h2 : B > C) (h3 : A = B * (1 + x / 100)) (h4 : B = C * (1 + y / 100))

theorem solution_proof : x = 100 * ((A / (C * (1 + y / 100))) - 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_proof_l1999_199952


namespace NUMINAMATH_GPT_n_fifth_minus_n_divisible_by_30_l1999_199957

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end NUMINAMATH_GPT_n_fifth_minus_n_divisible_by_30_l1999_199957


namespace NUMINAMATH_GPT_decreasing_interval_eqn_l1999_199955

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem decreasing_interval_eqn {a : ℝ} : (∀ x : ℝ, x < 6 → deriv (f a) x < 0) ↔ a ≥ 6 :=
sorry

end NUMINAMATH_GPT_decreasing_interval_eqn_l1999_199955


namespace NUMINAMATH_GPT_spirit_concentration_l1999_199956

theorem spirit_concentration (vol_a vol_b vol_c : ℕ) (conc_a conc_b conc_c : ℝ)
(h_a : conc_a = 0.45) (h_b : conc_b = 0.30) (h_c : conc_c = 0.10)
(h_vola : vol_a = 4) (h_volb : vol_b = 5) (h_volc : vol_c = 6) : 
  (conc_a * vol_a + conc_b * vol_b + conc_c * vol_c) / (vol_a + vol_b + vol_c) * 100 = 26 := 
by
  sorry

end NUMINAMATH_GPT_spirit_concentration_l1999_199956


namespace NUMINAMATH_GPT_total_cost_rental_l1999_199929

theorem total_cost_rental :
  let rental_fee := 20.99
  let charge_per_mile := 0.25
  let miles_driven := 299
  let total_cost := rental_fee + charge_per_mile * miles_driven
  total_cost = 95.74 := by
{
  sorry
}

end NUMINAMATH_GPT_total_cost_rental_l1999_199929


namespace NUMINAMATH_GPT_solution_set_interval_l1999_199982

theorem solution_set_interval (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} :=
sorry

end NUMINAMATH_GPT_solution_set_interval_l1999_199982


namespace NUMINAMATH_GPT_total_cranes_folded_l1999_199931

-- Definitions based on conditions
def hyerinCranesPerDay : ℕ := 16
def hyerinDays : ℕ := 7
def taeyeongCranesPerDay : ℕ := 25
def taeyeongDays : ℕ := 6

-- Definition of total number of cranes folded by Hyerin and Taeyeong
def totalCranes : ℕ :=
  (hyerinCranesPerDay * hyerinDays) + (taeyeongCranesPerDay * taeyeongDays)

-- Proof statement
theorem total_cranes_folded : totalCranes = 262 := by 
  sorry

end NUMINAMATH_GPT_total_cranes_folded_l1999_199931


namespace NUMINAMATH_GPT_weight_gain_ratio_l1999_199989

variable (J O F : ℝ)

theorem weight_gain_ratio :
  O = 5 ∧ F = (1/2) * J - 3 ∧ 5 + J + F = 20 → J / O = 12 / 5 :=
by
  intros h
  cases' h with hO h'
  cases' h' with hF hTotal
  sorry

end NUMINAMATH_GPT_weight_gain_ratio_l1999_199989


namespace NUMINAMATH_GPT_factorization_of_difference_of_squares_l1999_199910

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorization_of_difference_of_squares_l1999_199910


namespace NUMINAMATH_GPT_min_students_l1999_199994

noncomputable def smallest_possible_number_of_students (b g : ℕ) : ℕ :=
if 3 * (3 * b) = 5 * (4 * g) then b + g else 0

theorem min_students (b g : ℕ) (h1 : 0 < b) (h2 : 0 < g) (h3 : 3 * (3 * b) = 5 * (4 * g)) :
  smallest_possible_number_of_students b g = 29 := sorry

end NUMINAMATH_GPT_min_students_l1999_199994


namespace NUMINAMATH_GPT_train_cross_time_l1999_199905

noncomputable def train_length : ℕ := 1200 -- length of the train in meters
noncomputable def platform_length : ℕ := train_length -- length of the platform equals the train length
noncomputable def speed_kmh : ℝ := 144 -- speed in km/hr
noncomputable def speed_ms : ℝ := speed_kmh * (1000 / 3600) -- converting speed to m/s

-- the formula to calculate the crossing time
noncomputable def time_to_cross_platform : ℝ := 
  2 * train_length / speed_ms

theorem train_cross_time : time_to_cross_platform = 60 := by
  sorry

end NUMINAMATH_GPT_train_cross_time_l1999_199905


namespace NUMINAMATH_GPT_total_money_raised_l1999_199962

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end NUMINAMATH_GPT_total_money_raised_l1999_199962


namespace NUMINAMATH_GPT_bella_age_is_five_l1999_199988

-- Definitions from the problem:
def is_age_relation (bella_age brother_age : ℕ) : Prop :=
  brother_age = bella_age + 9 ∧ bella_age + brother_age = 19

-- The main proof statement:
theorem bella_age_is_five (bella_age brother_age : ℕ) (h : is_age_relation bella_age brother_age) :
  bella_age = 5 :=
by {
  -- Placeholder for proof steps
  sorry
}

end NUMINAMATH_GPT_bella_age_is_five_l1999_199988


namespace NUMINAMATH_GPT_math_problem_l1999_199934

theorem math_problem : 1999^2 - 2000 * 1998 = 1 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1999_199934


namespace NUMINAMATH_GPT_don_raise_l1999_199970

variable (D R : ℝ)

theorem don_raise 
  (h1 : R = 0.08 * D)
  (h2 : 840 = 0.08 * 10500)
  (h3 : (D + R) - (10500 + 840) = 540) : 
  R = 880 :=
by sorry

end NUMINAMATH_GPT_don_raise_l1999_199970


namespace NUMINAMATH_GPT_jason_investing_months_l1999_199974

noncomputable def initial_investment (total_amount earned_amount_per_month : ℕ) := total_amount / 3
noncomputable def months_investing (initial_investment earned_amount_per_month : ℕ) := (2 * initial_investment) / earned_amount_per_month

theorem jason_investing_months (total_amount earned_amount_per_month : ℕ) 
  (h1 : total_amount = 90) 
  (h2 : earned_amount_per_month = 12) 
  : months_investing (initial_investment total_amount earned_amount_per_month) earned_amount_per_month = 5 := 
by
  sorry

end NUMINAMATH_GPT_jason_investing_months_l1999_199974


namespace NUMINAMATH_GPT_math_problem_l1999_199908

theorem math_problem
  (n : ℕ) (d : ℕ)
  (h1 : d ≤ 9)
  (h2 : 3 * n^2 + 2 * n + d = 263)
  (h3 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) :
  n + d = 11 := 
sorry

end NUMINAMATH_GPT_math_problem_l1999_199908


namespace NUMINAMATH_GPT_multiple_of_six_and_nine_l1999_199960

-- Definitions: x is a multiple of 6, y is a multiple of 9.
def is_multiple_of_six (x : ℤ) : Prop := ∃ m : ℤ, x = 6 * m
def is_multiple_of_nine (y : ℤ) : Prop := ∃ n : ℤ, y = 9 * n

-- Assertions: Given the conditions, prove the following.
theorem multiple_of_six_and_nine (x y : ℤ)
  (hx : is_multiple_of_six x) (hy : is_multiple_of_nine y) :
  ((∃ k : ℤ, x - y = 3 * k) ∧
   (∃ m n : ℤ, x = 6 * m ∧ y = 9 * n ∧ (2 * m - 3 * n) % 3 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_six_and_nine_l1999_199960


namespace NUMINAMATH_GPT_range_of_function_l1999_199961

-- Given conditions 
def independent_variable_range (x : ℝ) : Prop := x ≥ 2

-- Proof statement (no proof only statement with "sorry")
theorem range_of_function (x : ℝ) (y : ℝ) (h : y = Real.sqrt (x - 2)) : independent_variable_range x :=
by sorry

end NUMINAMATH_GPT_range_of_function_l1999_199961


namespace NUMINAMATH_GPT_packs_of_snacks_l1999_199984

theorem packs_of_snacks (kyle_bike_hours : ℝ) (pack_cost : ℝ) (ryan_budget : ℝ) :
  kyle_bike_hours = 2 →
  10 * (2 * kyle_bike_hours) = pack_cost →
  ryan_budget = 2000 →
  ryan_budget / pack_cost = 50 :=
by 
  sorry

end NUMINAMATH_GPT_packs_of_snacks_l1999_199984


namespace NUMINAMATH_GPT_expression_value_l1999_199945

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (4 * a⁻¹ - 2 * a⁻¹ / 3) / a^2 = 90 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1999_199945


namespace NUMINAMATH_GPT_min_value_a_plus_2b_l1999_199917

theorem min_value_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_condition : (a + b) / (a * b) = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_2b_l1999_199917


namespace NUMINAMATH_GPT_height_of_box_l1999_199915

def base_area : ℕ := 20 * 20
def cost_per_box : ℝ := 1.30
def total_volume : ℕ := 3060000
def amount_spent : ℝ := 663

theorem height_of_box : ∃ h : ℕ, 400 * h = total_volume / (amount_spent / cost_per_box) := sorry

end NUMINAMATH_GPT_height_of_box_l1999_199915


namespace NUMINAMATH_GPT_arithmetic_progression_condition_l1999_199997

theorem arithmetic_progression_condition
  (a b c : ℝ) (a1 d : ℝ) (p n k : ℕ) :
  a = a1 + (p - 1) * d →
  b = a1 + (n - 1) * d →
  c = a1 + (k - 1) * d →
  a * (n - k) + b * (k - p) + c * (p - n) = 0 :=
by
  intros h1 h2 h3
  sorry


end NUMINAMATH_GPT_arithmetic_progression_condition_l1999_199997


namespace NUMINAMATH_GPT_find_line_equation_l1999_199919

def parameterized_line (t : ℝ) : ℝ × ℝ :=
  (3 * t + 2, 5 * t - 3)

theorem find_line_equation (x y : ℝ) (t : ℝ) (h : parameterized_line t = (x, y)) :
  y = (5 / 3) * x - 19 / 3 :=
sorry

end NUMINAMATH_GPT_find_line_equation_l1999_199919


namespace NUMINAMATH_GPT_number_of_rows_in_theater_l1999_199947

theorem number_of_rows_in_theater 
  (x : ℕ)
  (h1 : ∀ (students : ℕ), students = 30 → ∃ row : ℕ, row < x ∧ ∃ a b : ℕ, a ≠ b ∧ row = a ∧ row = b)
  (h2 : ∀ (students : ℕ), students = 26 → ∃ empties : ℕ, empties ≥ 3 ∧ x - students = empties)
  : x = 29 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_in_theater_l1999_199947


namespace NUMINAMATH_GPT_nth_term_of_sequence_99_l1999_199992

def sequence_rule (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n > 19 ∧ n % 7 ≠ 0 then n - 5
  else n + 7

noncomputable def sequence_nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.repeat sequence_rule n start

theorem nth_term_of_sequence_99 :
  sequence_nth_term 65 98 = 30 :=
sorry

end NUMINAMATH_GPT_nth_term_of_sequence_99_l1999_199992


namespace NUMINAMATH_GPT_population_relation_l1999_199944

-- Conditions: average life expectancies
def life_expectancy_gondor : ℝ := 64
def life_expectancy_numenor : ℝ := 92
def combined_life_expectancy (g n : ℕ) : ℝ := 85

-- Proof Problem: Given the conditions, prove the population relation
theorem population_relation (g n : ℕ) (h1 : life_expectancy_gondor * g + life_expectancy_numenor * n = combined_life_expectancy g n * (g + n)) : n = 3 * g :=
by
  sorry

end NUMINAMATH_GPT_population_relation_l1999_199944


namespace NUMINAMATH_GPT_probability_all_switches_on_is_correct_l1999_199971

-- Mechanical declaration of the problem
structure SwitchState :=
  (state : Fin 2003 → Bool)

noncomputable def probability_all_on (initial : SwitchState) : ℚ :=
  let satisfying_confs := 2
  let total_confs := 2 ^ 2003
  let p := satisfying_confs / total_confs
  p

-- Definition of the term we want to prove
theorem probability_all_switches_on_is_correct :
  ∀ (initial : SwitchState), probability_all_on initial = 1 / 2 ^ 2002 :=
  sorry

end NUMINAMATH_GPT_probability_all_switches_on_is_correct_l1999_199971


namespace NUMINAMATH_GPT_twice_son_plus_father_is_70_l1999_199922

section
variable {s f : ℕ}

-- Conditions
def son_age : ℕ := 15
def father_age : ℕ := 40

-- Statement to prove
theorem twice_son_plus_father_is_70 : (2 * son_age + father_age) = 70 :=
by
  sorry
end

end NUMINAMATH_GPT_twice_son_plus_father_is_70_l1999_199922


namespace NUMINAMATH_GPT_nth_equation_pattern_l1999_199980

theorem nth_equation_pattern (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_pattern_l1999_199980


namespace NUMINAMATH_GPT_heart_digit_proof_l1999_199906

noncomputable def heart_digit : ℕ := 3

theorem heart_digit_proof (heartsuit : ℕ) (h : heartsuit * 9 + 6 = heartsuit * 10 + 3) : 
  heartsuit = heart_digit := 
by
  sorry

end NUMINAMATH_GPT_heart_digit_proof_l1999_199906


namespace NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1999_199902

theorem machine_A_sprockets_per_hour 
  (A T_Q : ℝ)
  (h1 : 550 = 1.1 * A * T_Q)
  (h2 : 550 = A * (T_Q + 10)) 
  : A = 5 :=
by
  sorry

end NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1999_199902


namespace NUMINAMATH_GPT_strawberries_final_count_l1999_199920

def initial_strawberries := 300
def buckets := 5
def strawberries_per_bucket := initial_strawberries / buckets
def strawberries_removed_per_bucket := 20
def redistributed_in_first_two := 15
def redistributed_in_third := 25

-- Defining the final counts after redistribution
def final_strawberries_first := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_second := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_third := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_third
def final_strawberries_fourth := strawberries_per_bucket - strawberries_removed_per_bucket
def final_strawberries_fifth := strawberries_per_bucket - strawberries_removed_per_bucket

theorem strawberries_final_count :
  final_strawberries_first = 55 ∧
  final_strawberries_second = 55 ∧
  final_strawberries_third = 65 ∧
  final_strawberries_fourth = 40 ∧
  final_strawberries_fifth = 40 := by
  sorry

end NUMINAMATH_GPT_strawberries_final_count_l1999_199920


namespace NUMINAMATH_GPT_polygon_sides_l1999_199950

theorem polygon_sides (n : ℕ) (sum_of_angles : ℕ) (missing_angle : ℕ) 
  (h1 : sum_of_angles = 3240) 
  (h2 : missing_angle * n / (n - 1) = 2 * sum_of_angles) : 
  n = 20 := 
sorry

end NUMINAMATH_GPT_polygon_sides_l1999_199950


namespace NUMINAMATH_GPT_integer_solutions_count_l1999_199926

theorem integer_solutions_count (x : ℤ) :
  (75 ^ 60 * x ^ 60 > x ^ 120 ∧ x ^ 120 > 3 ^ 240) → ∃ n : ℕ, n = 65 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l1999_199926


namespace NUMINAMATH_GPT_basis_of_R3_l1999_199959

def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem basis_of_R3 :
  ∀ (u : ℝ × ℝ × ℝ), ∃ (α β γ : ℝ), u = α • e1 + β • e2 + γ • e3 ∧ 
  (∀ (a b c : ℝ), a • e1 + b • e2 + c • e3 = (0, 0, 0) → a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end NUMINAMATH_GPT_basis_of_R3_l1999_199959


namespace NUMINAMATH_GPT_equilateral_triangle_ab_l1999_199913

noncomputable def a : ℝ := 25 * Real.sqrt 3
noncomputable def b : ℝ := 5 * Real.sqrt 3

theorem equilateral_triangle_ab
  (a_val : a = 25 * Real.sqrt 3)
  (b_val : b = 5 * Real.sqrt 3)
  (h1 : Complex.abs (a + 15 * Complex.I) = 25)
  (h2 : Complex.abs (b + 45 * Complex.I) = 45)
  (h3 : Complex.abs ((a - b) + (15 - 45) * Complex.I) = 30) :
  a * b = 375 := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_ab_l1999_199913


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l1999_199924

open Real

theorem inequality_for_positive_reals 
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) :=
sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l1999_199924


namespace NUMINAMATH_GPT_calculate_expression_l1999_199964

theorem calculate_expression : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1999_199964


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1999_199979

/-- Given two parallel lines y=2x and y=2x+5, the distance between them is √5. -/
theorem distance_between_parallel_lines :
  let A := -2
  let B := 1
  let C1 := 0
  let C2 := -5
  let distance := (|C2 - C1|: ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = Real.sqrt 5 := by
  -- Assuming calculations as done in the original solution
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1999_199979


namespace NUMINAMATH_GPT_regular_polygon_exterior_angle_l1999_199916

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_exterior_angle_l1999_199916

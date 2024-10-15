import Mathlib

namespace NUMINAMATH_GPT_inverse_proportion_symmetry_l2008_200817

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inverse_proportion_symmetry_l2008_200817


namespace NUMINAMATH_GPT_principal_amount_l2008_200850

/-
  Given:
  - Simple Interest (SI) = Rs. 4016.25
  - Rate (R) = 0.08 (8% per annum)
  - Time (T) = 5 years
  
  We want to prove:
  Principal = Rs. 10040.625
-/

def SI : ℝ := 4016.25
def R : ℝ := 0.08
def T : ℕ := 5

theorem principal_amount :
  ∃ P : ℝ, SI = (P * R * T) / 100 ∧ P = 10040.625 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l2008_200850


namespace NUMINAMATH_GPT_find_positive_integer_l2008_200882

def product_of_digits (n : Nat) : Nat :=
  -- Function to compute product of digits, assume it is defined correctly
  sorry

theorem find_positive_integer (x : Nat) (h : x > 0) :
  product_of_digits x = x * x - 10 * x - 22 ↔ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_l2008_200882


namespace NUMINAMATH_GPT_main_theorem_l2008_200856

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_main_theorem_l2008_200856


namespace NUMINAMATH_GPT_women_in_first_group_l2008_200844

-- Define the number of women in the first group as W
variable (W : ℕ)

-- Define the work parameters
def work_per_day := 75 / 8
def work_per_hour_first_group := work_per_day / 5

def work_per_day_second_group := 30 / 3
def work_per_hour_second_group := work_per_day_second_group / 8

-- The equation comes from work/hour equivalence
theorem women_in_first_group :
  (W : ℝ) * work_per_hour_first_group = 4 * work_per_hour_second_group → W = 5 :=
by 
  sorry

end NUMINAMATH_GPT_women_in_first_group_l2008_200844


namespace NUMINAMATH_GPT_a_range_l2008_200858

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_a_range_l2008_200858


namespace NUMINAMATH_GPT_no_polyhedron_with_surface_area_2015_l2008_200805

theorem no_polyhedron_with_surface_area_2015 : 
  ¬ ∃ (n k : ℤ), 6 * n - 2 * k = 2015 :=
by
  sorry

end NUMINAMATH_GPT_no_polyhedron_with_surface_area_2015_l2008_200805


namespace NUMINAMATH_GPT_score_ordering_l2008_200831

-- Definition of the problem conditions in Lean 4:
def condition1 (Q K : ℝ) : Prop := Q ≠ K
def condition2 (M Q S K : ℝ) : Prop := M < Q ∧ M < S ∧ M < K
def condition3 (S Q M K : ℝ) : Prop := S > Q ∧ S > M ∧ S > K

-- Theorem statement in Lean 4:
theorem score_ordering (M Q S K : ℝ) (h1 : condition1 Q K) (h2 : condition2 M Q S K) (h3 : condition3 S Q M K) : 
  M < Q ∧ Q < S :=
by
  sorry

end NUMINAMATH_GPT_score_ordering_l2008_200831


namespace NUMINAMATH_GPT_smallest_number_of_cookies_l2008_200800

theorem smallest_number_of_cookies
  (n : ℕ) 
  (hn : 4 * n - 4 = (n^2) / 2) : n = 7 → n^2 = 49 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cookies_l2008_200800


namespace NUMINAMATH_GPT_kat_boxing_training_hours_l2008_200855

theorem kat_boxing_training_hours :
  let strength_training_hours := 3
  let total_training_hours := 9
  let boxing_sessions := 4
  let boxing_training_hours := total_training_hours - strength_training_hours
  let hours_per_boxing_session := boxing_training_hours / boxing_sessions
  hours_per_boxing_session = 1.5 :=
sorry

end NUMINAMATH_GPT_kat_boxing_training_hours_l2008_200855


namespace NUMINAMATH_GPT_problem_l2008_200825

theorem problem (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, x^5 = a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5) →
  a_3 = -10 ∧ a_1 + a_3 + a_5 = -16 :=
by 
  sorry

end NUMINAMATH_GPT_problem_l2008_200825


namespace NUMINAMATH_GPT_abs_diff_squares_eq_300_l2008_200835

theorem abs_diff_squares_eq_300 : 
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  |a^2 - b^2| = 300 := 
by
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  sorry

end NUMINAMATH_GPT_abs_diff_squares_eq_300_l2008_200835


namespace NUMINAMATH_GPT_total_money_found_l2008_200828

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end NUMINAMATH_GPT_total_money_found_l2008_200828


namespace NUMINAMATH_GPT_watermelon_melon_weight_l2008_200836

variables {W M : ℝ}

theorem watermelon_melon_weight :
  (2 * W > 3 * M ∨ 3 * W > 4 * M) ∧ ¬ (2 * W > 3 * M ∧ 3 * W > 4 * M) → 12 * W ≤ 18 * M :=
by
  sorry

end NUMINAMATH_GPT_watermelon_melon_weight_l2008_200836


namespace NUMINAMATH_GPT_original_number_is_76_l2008_200890

-- Define the original number x and the condition given
def original_number_condition (x : ℝ) : Prop :=
  (3 / 4) * x = x - 19

-- State the theorem that the original number x must be 76 if it satisfies the condition
theorem original_number_is_76 (x : ℝ) (h : original_number_condition x) : x = 76 :=
sorry

end NUMINAMATH_GPT_original_number_is_76_l2008_200890


namespace NUMINAMATH_GPT_inning_is_31_l2008_200843

noncomputable def inning_number (s: ℕ) (i: ℕ) (a: ℕ) : ℕ := s - a + i

theorem inning_is_31
  (batsman_runs: ℕ)
  (increase_average: ℕ)
  (final_average: ℕ) 
  (n: ℕ) 
  (h1: batsman_runs = 92)
  (h2: increase_average = 3)
  (h3: final_average = 44)
  (h4: 44 * n - 92 = 41 * n): 
  inning_number 44 1 3 = 31 := 
by 
  sorry

end NUMINAMATH_GPT_inning_is_31_l2008_200843


namespace NUMINAMATH_GPT_probability_ge_first_second_l2008_200860

noncomputable def probability_ge_rolls : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_ge_first_second :
  probability_ge_rolls = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_ge_first_second_l2008_200860


namespace NUMINAMATH_GPT_isosceles_triangle_angle_measure_l2008_200863

theorem isosceles_triangle_angle_measure
  (isosceles : Triangle → Prop)
  (exterior_angles : Triangle → ℝ → ℝ → Prop)
  (ratio_1_to_4 : ∀ {T : Triangle} {a b : ℝ}, exterior_angles T a b → b = 4 * a)
  (interior_angles : Triangle → ℝ → ℝ → ℝ → Prop) :
  ∀ (T : Triangle), isosceles T → ∃ α β γ : ℝ, interior_angles T α β γ ∧ α = 140 ∧ β = 20 ∧ γ = 20 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_measure_l2008_200863


namespace NUMINAMATH_GPT_count_satisfying_integers_l2008_200819

theorem count_satisfying_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, 9 < n ∧ n < 60) ∧ S.card = 50) :=
by
  sorry

end NUMINAMATH_GPT_count_satisfying_integers_l2008_200819


namespace NUMINAMATH_GPT_solve_equation_in_integers_l2008_200893

theorem solve_equation_in_integers (a b c : ℤ) (h : 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) :
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
sorry

end NUMINAMATH_GPT_solve_equation_in_integers_l2008_200893


namespace NUMINAMATH_GPT_trig_expression_value_l2008_200869

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) : 
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2008_200869


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l2008_200852

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l2008_200852


namespace NUMINAMATH_GPT_inequality_holds_l2008_200861

theorem inequality_holds (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4)
  (h5 : 0 < x5) (h6 : 0 < x6) (h7 : 0 < x7) (h8 : 0 < x8) 
  (h9 : 0 < x9) :
  (x1 - x3) / (x1 * x3 + 2 * x2 * x3 + x2^2) +
  (x2 - x4) / (x2 * x4 + 2 * x3 * x4 + x3^2) +
  (x3 - x5) / (x3 * x5 + 2 * x4 * x5 + x4^2) +
  (x4 - x6) / (x4 * x6 + 2 * x5 * x6 + x5^2) +
  (x5 - x7) / (x5 * x7 + 2 * x6 * x7 + x6^2) +
  (x6 - x8) / (x6 * x8 + 2 * x7 * x8 + x7^2) +
  (x7 - x9) / (x7 * x9 + 2 * x8 * x9 + x8^2) +
  (x8 - x1) / (x8 * x1 + 2 * x9 * x1 + x9^2) +
  (x9 - x2) / (x9 * x2 + 2 * x1 * x2 + x1^2) ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l2008_200861


namespace NUMINAMATH_GPT_sum_real_imag_parts_eq_l2008_200830

noncomputable def z (a b : ℂ) : ℂ := a / b

theorem sum_real_imag_parts_eq (z : ℂ) (h : z * (2 + I) = 2 * I - 1) : 
  (z.re + z.im) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_sum_real_imag_parts_eq_l2008_200830


namespace NUMINAMATH_GPT_largest_four_digit_divisible_by_8_l2008_200868

/-- The largest four-digit number that is divisible by 8 is 9992. -/
theorem largest_four_digit_divisible_by_8 : ∃ x : ℕ, x = 9992 ∧ x < 10000 ∧ x % 8 = 0 ∧
  ∀ y : ℕ, y < 10000 ∧ y % 8 = 0 → y ≤ 9992 := 
by 
  sorry

end NUMINAMATH_GPT_largest_four_digit_divisible_by_8_l2008_200868


namespace NUMINAMATH_GPT_benny_total_hours_l2008_200829

-- Define the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- State the theorem (problem) to be proved
theorem benny_total_hours : hours_per_day * days_worked = 18 :=
by
  -- Sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_benny_total_hours_l2008_200829


namespace NUMINAMATH_GPT_value_of_z_l2008_200867

theorem value_of_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_value_of_z_l2008_200867


namespace NUMINAMATH_GPT_profit_percentage_l2008_200849

theorem profit_percentage (SP : ℝ) (h : SP > 0) (CP : ℝ) (h1 : CP = 0.96 * SP) :
  (SP - CP) / CP * 100 = 4.17 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2008_200849


namespace NUMINAMATH_GPT_largest_possible_gcd_l2008_200822

theorem largest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) : ∃ d, Nat.gcd a b = d ∧ d = 17 :=
sorry

end NUMINAMATH_GPT_largest_possible_gcd_l2008_200822


namespace NUMINAMATH_GPT_solve_inequality_l2008_200857

theorem solve_inequality (x : ℝ) : (2 * x - 3) / (x + 2) ≤ 1 ↔ (-2 < x ∧ x ≤ 5) :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l2008_200857


namespace NUMINAMATH_GPT_find_values_l2008_200821

open Real

noncomputable def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

noncomputable def given_condition (x y : ℝ) := (sqrt (12 * x) * sqrt (20 * x) * sqrt (4 * y) * sqrt (25 * y) = 50)

theorem find_values (x y : ℝ) 
  (h1: positive_numbers x y) 
  (h2: given_condition x y) : 
  x * y = sqrt (25 / 24) := 
sorry

end NUMINAMATH_GPT_find_values_l2008_200821


namespace NUMINAMATH_GPT_fewest_candies_l2008_200811

-- Defining the conditions
def condition1 (x : ℕ) := x % 21 = 5
def condition2 (x : ℕ) := x % 22 = 3
def condition3 (x : ℕ) := x > 500

-- Stating the main theorem
theorem fewest_candies : ∃ x : ℕ, condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 509 :=
  sorry

end NUMINAMATH_GPT_fewest_candies_l2008_200811


namespace NUMINAMATH_GPT_jordan_trapezoid_height_l2008_200846

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end NUMINAMATH_GPT_jordan_trapezoid_height_l2008_200846


namespace NUMINAMATH_GPT_percentage_of_500_l2008_200814

theorem percentage_of_500 (P : ℝ) : 0.1 * (500 * P / 100) = 25 → P = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_500_l2008_200814


namespace NUMINAMATH_GPT_area_ratio_eq_l2008_200815

-- Define the parameters used in the problem
variables (t t1 r ρ : ℝ)

-- Define the conditions given in the problem
def area_triangle_ABC : ℝ := t
def area_triangle_A1B1C1 : ℝ := t1
def circumradius_ABC : ℝ := r
def inradius_A1B1C1 : ℝ := ρ

-- Problem statement: Prove the given equation
theorem area_ratio_eq : t / t1 = 2 * ρ / r :=
sorry

end NUMINAMATH_GPT_area_ratio_eq_l2008_200815


namespace NUMINAMATH_GPT_tiffany_lives_problem_l2008_200879

/-- Tiffany's lives problem -/
theorem tiffany_lives_problem (L : ℤ) (h1 : 43 - L + 27 = 56) : L = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_tiffany_lives_problem_l2008_200879


namespace NUMINAMATH_GPT_combined_distance_is_twelve_l2008_200889

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end NUMINAMATH_GPT_combined_distance_is_twelve_l2008_200889


namespace NUMINAMATH_GPT_determine_a_l2008_200886

theorem determine_a (a b c : ℤ) (h : (b + 11) * (c + 11) = 2) (hb : b + 11 = -2) (hc : c + 11 = -1) :
  a = 13 := by
  sorry

end NUMINAMATH_GPT_determine_a_l2008_200886


namespace NUMINAMATH_GPT_sum_inequality_l2008_200823

variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {m n p k : ℕ}

-- Definitions for the conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, a (i + 1) - a i = a (j + 1) - a j

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a (n - 1)) / 2

def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- The theorem to prove
theorem sum_inequality (arith_seq : is_arithmetic_sequence a)
  (S_eq : sum_of_arithmetic_sequence S a)
  (nn_seq : non_negative_sequence a)
  (h1 : m + n = 2 * p) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) :
  1 / (S m) ^ k + 1 / (S n) ^ k ≥ 2 / (S p) ^ k :=
by sorry

end NUMINAMATH_GPT_sum_inequality_l2008_200823


namespace NUMINAMATH_GPT_library_visitors_on_sundays_l2008_200891

theorem library_visitors_on_sundays 
  (average_other_days : ℕ) 
  (average_per_day : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ) 
  (total_visitors_month : ℕ)
  (visitors_other_days : ℕ) 
  (total_visitors_sundays : ℕ) :
  average_other_days = 240 →
  average_per_day = 285 →
  total_days = 30 →
  sundays = 5 →
  other_days = total_days - sundays →
  total_visitors_month = average_per_day * total_days →
  visitors_other_days = average_other_days * other_days →
  total_visitors_sundays + visitors_other_days = total_visitors_month →
  total_visitors_sundays = sundays * (510 : ℕ) :=
by
  sorry


end NUMINAMATH_GPT_library_visitors_on_sundays_l2008_200891


namespace NUMINAMATH_GPT_tunnel_connects_land_l2008_200887

noncomputable def surface_area (planet : Type) : ℝ := sorry
noncomputable def land_area (planet : Type) : ℝ := sorry
noncomputable def half_surface_area (planet : Type) : ℝ := surface_area planet / 2
noncomputable def can_dig_tunnel_through_center (planet : Type) : Prop := sorry

variable {TauCeti : Type}

-- Condition: Land occupies more than half of the entire surface area.
axiom land_more_than_half : land_area TauCeti > half_surface_area TauCeti

-- Proof problem statement: Prove that inhabitants can dig a tunnel through the center of the planet.
theorem tunnel_connects_land : can_dig_tunnel_through_center TauCeti :=
sorry

end NUMINAMATH_GPT_tunnel_connects_land_l2008_200887


namespace NUMINAMATH_GPT_max_mondays_in_first_51_days_l2008_200871

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end NUMINAMATH_GPT_max_mondays_in_first_51_days_l2008_200871


namespace NUMINAMATH_GPT_no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l2008_200872

-- Proof Problem 1:
theorem no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n :
  ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + 2) = n * (n + 1) :=
by sorry

-- Proof Problem 2:
theorem k_ge_3_positive_ints_m_n_exists (k : ℕ) (hk : k ≥ 3) :
  (k = 3 → ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) ∧
  (k ≥ 4 → ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) :=
by sorry

end NUMINAMATH_GPT_no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l2008_200872


namespace NUMINAMATH_GPT_range_of_independent_variable_l2008_200864

theorem range_of_independent_variable (x : ℝ) (h : ∃ y, y = 2 / (Real.sqrt (x - 3))) : x > 3 :=
sorry

end NUMINAMATH_GPT_range_of_independent_variable_l2008_200864


namespace NUMINAMATH_GPT_sum_of_eight_numbers_on_cards_l2008_200816

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_numbers_on_cards_l2008_200816


namespace NUMINAMATH_GPT_two_digit_integers_count_l2008_200832

def digits : Set ℕ := {3, 5, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem two_digit_integers_count : 
  ∃ (count : ℕ), count = 16 ∧
  (∀ (t : ℕ), t ∈ digits → 
  ∀ (u : ℕ), u ∈ digits → 
  t ≠ u ∧ is_odd u → 
  (∃ n : ℕ, 10 * t + u = n)) :=
by
  -- The total number of unique two-digit integers is 16
  use 16
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_two_digit_integers_count_l2008_200832


namespace NUMINAMATH_GPT_difference_of_lines_in_cm_l2008_200881

def W : ℝ := 7.666666666666667
def B : ℝ := 3.3333333333333335
def inch_to_cm : ℝ := 2.54

theorem difference_of_lines_in_cm :
  (W * inch_to_cm) - (B * inch_to_cm) = 11.005555555555553 := 
sorry

end NUMINAMATH_GPT_difference_of_lines_in_cm_l2008_200881


namespace NUMINAMATH_GPT_total_peaches_l2008_200803

-- Definitions based on the given conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- The proof goal stating the total number of peaches now
theorem total_peaches : initial_peaches + picked_peaches = 68 := by
  sorry

end NUMINAMATH_GPT_total_peaches_l2008_200803


namespace NUMINAMATH_GPT_arithmetic_progression_integers_l2008_200834

theorem arithmetic_progression_integers 
  (d : ℤ) (a : ℤ) (h_d_pos : d > 0)
  (h_progression : ∀ i j : ℤ, i ≠ j → ∃ k : ℤ, a * (a + i * d) = a + k * d)
  : ∀ n : ℤ, ∃ m : ℤ, a + n * d = m :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_integers_l2008_200834


namespace NUMINAMATH_GPT_find_coordinates_A_l2008_200810

-- Define the point A
structure Point where
  x : ℝ
  y : ℝ

def PointA (a : ℝ) : Point :=
  { x := 3 * a + 2, y := 2 * a - 4 }

-- Define the conditions
def condition1 (a : ℝ) := (PointA a).y = 4

def condition2 (a : ℝ) := |(PointA a).x| = |(PointA a).y|

-- The coordinates solutions to be proven
def valid_coordinates (p : Point) : Prop :=
  p = { x := 14, y := 4 } ∨
  p = { x := -16, y := -16 } ∨
  p = { x := 3.2, y := -3.2 }

-- Main theorem to prove
theorem find_coordinates_A (a : ℝ) :
  (condition1 a ∨ condition2 a) → valid_coordinates (PointA a) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_A_l2008_200810


namespace NUMINAMATH_GPT_min_value_PF_PA_l2008_200899

noncomputable def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def focus_left : ℝ × ℝ := (-4, 0)
noncomputable def focus_right : ℝ × ℝ := (4, 0)
noncomputable def point_A : ℝ × ℝ := (1, 4)

theorem min_value_PF_PA (P : ℝ × ℝ)
  (hP : hyperbola_eq P.1 P.2)
  (hP_right_branch : P.1 > 0) :
  ∃ P : ℝ × ℝ, ∀ X : ℝ × ℝ, hyperbola_eq X.1 X.2 → X.1 > 0 → 
               (dist X focus_left + dist X point_A) ≥ 9 ∧
               (dist P focus_left + dist P point_A) = 9 := 
sorry

end NUMINAMATH_GPT_min_value_PF_PA_l2008_200899


namespace NUMINAMATH_GPT_amanda_weekly_earnings_l2008_200824

def amanda_rate_per_hour : ℝ := 20.00
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointment_hours : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def saturday_appointment_hours : ℝ := 6

def total_hours_worked : ℝ :=
  monday_appointments * monday_hours_per_appointment +
  tuesday_appointment_hours +
  thursday_appointments * thursday_hours_per_appointment +
  saturday_appointment_hours

def total_earnings : ℝ := total_hours_worked * amanda_rate_per_hour

theorem amanda_weekly_earnings : total_earnings = 410.00 :=
  by
    unfold total_earnings total_hours_worked monday_appointments monday_hours_per_appointment tuesday_appointment_hours thursday_appointments thursday_hours_per_appointment saturday_appointment_hours amanda_rate_per_hour 
    -- The proof will involve basic arithmetic simplification, which is skipped here.
    -- Therefore, we simply state sorry.
    sorry

end NUMINAMATH_GPT_amanda_weekly_earnings_l2008_200824


namespace NUMINAMATH_GPT_vitamin_A_supplements_per_pack_l2008_200885

theorem vitamin_A_supplements_per_pack {A x y : ℕ} (h1 : A * x = 119) (h2 : 17 * y = 119) : A = 7 :=
by
  sorry

end NUMINAMATH_GPT_vitamin_A_supplements_per_pack_l2008_200885


namespace NUMINAMATH_GPT_max_sequence_term_value_l2008_200833

def a_n (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem max_sequence_term_value : ∃ n : ℕ, a_n n = 108 := 
sorry

end NUMINAMATH_GPT_max_sequence_term_value_l2008_200833


namespace NUMINAMATH_GPT_pencil_rows_l2008_200888

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) : (total_pencils / pencils_per_row) = 7 :=
by
  sorry

end NUMINAMATH_GPT_pencil_rows_l2008_200888


namespace NUMINAMATH_GPT_maximum_value_of_f_l2008_200845

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2008_200845


namespace NUMINAMATH_GPT_sum_first_five_terms_arithmetic_seq_l2008_200866

theorem sum_first_five_terms_arithmetic_seq
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a2 : a 2 = 5)
  (h_a4 : a 4 = 9)
  : (Finset.range 5).sum a = 35 := by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_arithmetic_seq_l2008_200866


namespace NUMINAMATH_GPT_reciprocal_of_2023_l2008_200847

theorem reciprocal_of_2023 :
  1 / 2023 = 1 / (2023 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_2023_l2008_200847


namespace NUMINAMATH_GPT_sum_of_first_4n_integers_l2008_200818

theorem sum_of_first_4n_integers (n : ℕ) 
  (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 150) : 
  (4 * n * (4 * n + 1)) / 2 = 300 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_4n_integers_l2008_200818


namespace NUMINAMATH_GPT_sell_decision_l2008_200812

noncomputable def profit_beginning (a : ℝ) : ℝ :=
(a + 100) * 1.024

noncomputable def profit_end (a : ℝ) : ℝ :=
a + 115

theorem sell_decision (a : ℝ) :
  (a > 525 → profit_beginning a > profit_end a) ∧
  (a < 525 → profit_beginning a < profit_end a) ∧
  (a = 525 → profit_beginning a = profit_end a) :=
by
  sorry

end NUMINAMATH_GPT_sell_decision_l2008_200812


namespace NUMINAMATH_GPT_total_seeds_eaten_l2008_200820

def first_seeds := 78
def second_seeds := 53
def third_seeds := second_seeds + 30

theorem total_seeds_eaten : first_seeds + second_seeds + third_seeds = 214 := by
  -- Sorry, placeholder for proof
  sorry

end NUMINAMATH_GPT_total_seeds_eaten_l2008_200820


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l2008_200839

theorem find_positive_integer_solutions :
  ∃ (x y z : ℕ), 
    2 * x * z = y^2 ∧ 
    x + z = 1987 ∧ 
    x = 1458 ∧ 
    y = 1242 ∧ 
    z = 529 :=
  by sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l2008_200839


namespace NUMINAMATH_GPT_frac_equality_l2008_200870

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_frac_equality_l2008_200870


namespace NUMINAMATH_GPT_total_boxes_sold_l2008_200875

-- Define the variables for each day's sales
def friday_sales : ℕ := 30
def saturday_sales : ℕ := 2 * friday_sales
def sunday_sales : ℕ := saturday_sales - 15
def total_sales : ℕ := friday_sales + saturday_sales + sunday_sales

-- State the theorem to prove the total sales over three days
theorem total_boxes_sold : total_sales = 135 :=
by 
  -- Here we would normally put the proof steps, but since we're asked only for the statement,
  -- we skip the proof with sorry
  sorry

end NUMINAMATH_GPT_total_boxes_sold_l2008_200875


namespace NUMINAMATH_GPT_solve_equation_l2008_200808

theorem solve_equation (x : ℝ) (h : (4 * x ^ 2 + 6 * x + 2) / (x + 2) = 4 * x + 7) : x = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2008_200808


namespace NUMINAMATH_GPT_f_at_11_l2008_200854

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_at_11 : f 11 = 149 := sorry

end NUMINAMATH_GPT_f_at_11_l2008_200854


namespace NUMINAMATH_GPT_g_at_5_l2008_200883

def g (x : ℝ) : ℝ := sorry -- Placeholder for the function definition, typically provided in further context

theorem g_at_5 : g 5 = 3 / 4 :=
by
  -- Given condition as a hypothesis
  have h : ∀ x: ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1 := sorry
  sorry  -- Full proof should go here

end NUMINAMATH_GPT_g_at_5_l2008_200883


namespace NUMINAMATH_GPT_constant_term_expansion_l2008_200838

noncomputable def sum_of_coefficients (a : ℕ) : ℕ := sorry

noncomputable def constant_term (a : ℕ) : ℕ := sorry

theorem constant_term_expansion (a : ℕ) (h : sum_of_coefficients a = 2) : constant_term 2 = 10 :=
sorry

end NUMINAMATH_GPT_constant_term_expansion_l2008_200838


namespace NUMINAMATH_GPT_percentage_solution_l2008_200853

noncomputable def percentage_of_difference (P : ℚ) (x y : ℚ) : Prop :=
  (P / 100) * (x - y) = (14 / 100) * (x + y)

theorem percentage_solution (x y : ℚ) (h1 : y = 0.17647058823529413 * x)
  (h2 : percentage_of_difference P x y) : 
  P = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_solution_l2008_200853


namespace NUMINAMATH_GPT_earnings_per_visit_l2008_200878

-- Define the conditions of the problem
def website_visits_per_month : ℕ := 30000
def earning_per_day : Real := 10
def days_in_month : ℕ := 30

-- Prove that John gets $0.01 per visit
theorem earnings_per_visit :
  (earning_per_day * days_in_month) / website_visits_per_month = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_earnings_per_visit_l2008_200878


namespace NUMINAMATH_GPT_union_A_B_inter_complB_A_l2008_200801

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define the complement of B with respect to U
def compl_B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Problem (1): Prove that A ∪ B = {x | -3 < x ∧ x ≤ 6}
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 6} := by
  sorry

-- Problem (2): Prove that (compl_B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}
theorem inter_complB_A : compl_B ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6} := by 
  sorry

end NUMINAMATH_GPT_union_A_B_inter_complB_A_l2008_200801


namespace NUMINAMATH_GPT_price_of_uniform_l2008_200859

-- Definitions based on conditions
def total_salary : ℕ := 600
def months_worked : ℕ := 9
def months_in_year : ℕ := 12
def salary_received : ℕ := 400
def uniform_price (U : ℕ) : Prop := 
    (3/4 * total_salary) - salary_received = U

-- Theorem stating the price of the uniform
theorem price_of_uniform : ∃ U : ℕ, uniform_price U := by
  sorry

end NUMINAMATH_GPT_price_of_uniform_l2008_200859


namespace NUMINAMATH_GPT_candidate_lost_by_l2008_200874

noncomputable def candidate_votes (total_votes : ℝ) := 0.35 * total_votes
noncomputable def rival_votes (total_votes : ℝ) := 0.65 * total_votes

theorem candidate_lost_by (total_votes : ℝ) (h : total_votes = 7899.999999999999) :
  rival_votes total_votes - candidate_votes total_votes = 2370 :=
by
  sorry

end NUMINAMATH_GPT_candidate_lost_by_l2008_200874


namespace NUMINAMATH_GPT_total_books_l2008_200851

def initial_books : ℝ := 41.0
def first_addition : ℝ := 33.0
def second_addition : ℝ := 2.0

theorem total_books (h1 : initial_books = 41.0) (h2 : first_addition = 33.0) (h3 : second_addition = 2.0) :
  initial_books + first_addition + second_addition = 76.0 := 
by
  -- placeholders for the proof steps, omitting the detailed steps as instructed
  sorry

end NUMINAMATH_GPT_total_books_l2008_200851


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_l2008_200809

theorem quadratic_real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 :=
by
  -- Proof is omitted, we only need the statement
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_l2008_200809


namespace NUMINAMATH_GPT_race_ordering_l2008_200807

theorem race_ordering
  (Lotar Manfred Jan Victor Eddy : ℕ) 
  (h1 : Lotar < Manfred) 
  (h2 : Manfred < Jan) 
  (h3 : Jan < Victor) 
  (h4 : Eddy < Victor) : 
  ∀ x, x = Victor ↔ ∀ y, (y = Lotar ∨ y = Manfred ∨ y = Jan ∨ y = Eddy) → y < x :=
by
  sorry

end NUMINAMATH_GPT_race_ordering_l2008_200807


namespace NUMINAMATH_GPT_fraction_addition_target_l2008_200848

open Rat

theorem fraction_addition_target (n : ℤ) : 
  (4 + n) / (7 + n) = 3 / 4 → 
  n = 5 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_addition_target_l2008_200848


namespace NUMINAMATH_GPT_opposites_of_each_other_l2008_200827

theorem opposites_of_each_other (a b : ℚ) (h : a + b = 0) : a = -b :=
  sorry

end NUMINAMATH_GPT_opposites_of_each_other_l2008_200827


namespace NUMINAMATH_GPT_minimal_benches_l2008_200865

theorem minimal_benches (x : ℕ) 
  (standard_adults : ℕ := x * 8) (standard_children : ℕ := x * 12)
  (extended_adults : ℕ := x * 8) (extended_children : ℕ := x * 16) 
  (hx : standard_adults + extended_adults = standard_children + extended_children) :
  x = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimal_benches_l2008_200865


namespace NUMINAMATH_GPT_john_trip_total_time_l2008_200884

theorem john_trip_total_time :
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  t1 + t2 + t3 + t4 + t5 = 872 :=
by
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  have h1: t1 + t2 + t3 + t4 + t5 = 2 + (3 * 2) + (4 * (3 * 2)) + (5 * (4 * (3 * 2))) + (6 * (5 * (4 * (3 * 2)))) := by
    sorry
  have h2: 2 + 6 + 24 + 120 + 720 = 872 := by
    sorry
  exact h2

end NUMINAMATH_GPT_john_trip_total_time_l2008_200884


namespace NUMINAMATH_GPT_number_of_divisors_8_factorial_l2008_200877

open Nat

theorem number_of_divisors_8_factorial :
  let n := 8!
  let factorization := [(2, 7), (3, 2), (5, 1), (7, 1)]
  let numberOfDivisors := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  n = 2^7 * 3^2 * 5^1 * 7^1 ->
  n.factors.count = 4 ->
  numberOfDivisors = 96 :=
by
  sorry

end NUMINAMATH_GPT_number_of_divisors_8_factorial_l2008_200877


namespace NUMINAMATH_GPT_exceeds_alpha_beta_l2008_200894

noncomputable def condition (α β p q : ℝ) : Prop :=
  q < 50 ∧ α > 0 ∧ β > 0 ∧ p > 0 ∧ q > 0

theorem exceeds_alpha_beta (α β p q : ℝ) (h : condition α β p q) :
  (1 + p / 100) * (1 - q / 100) > 1 → p > 100 * q / (100 - q) := by
  sorry

end NUMINAMATH_GPT_exceeds_alpha_beta_l2008_200894


namespace NUMINAMATH_GPT_sequence_periodicity_l2008_200896

theorem sequence_periodicity (a : ℕ → ℕ) (n : ℕ) (h : ∀ k, a k = 6^k) :
  a (n + 5) % 100 = a n % 100 :=
by sorry

end NUMINAMATH_GPT_sequence_periodicity_l2008_200896


namespace NUMINAMATH_GPT_evaluate_expression_l2008_200841

theorem evaluate_expression : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2008_200841


namespace NUMINAMATH_GPT_line_equation_direction_point_l2008_200892

theorem line_equation_direction_point 
  (d : ℝ × ℝ) (A : ℝ × ℝ) :
  d = (2, -1) →
  A = (1, 0) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = -1 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 ↔ x + 2 * y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_direction_point_l2008_200892


namespace NUMINAMATH_GPT_exists_good_placement_l2008_200813

-- Define a function that checks if a placement is "good" with respect to a symmetry axis
def is_good (f : Fin 1983 → ℕ) : Prop :=
  ∀ (i : Fin 1983), f i < f (i + 991) ∨ f (i + 991) < f i

-- Prove the existence of a "good" placement for the regular 1983-gon
theorem exists_good_placement : ∃ f : Fin 1983 → ℕ, is_good f :=
sorry

end NUMINAMATH_GPT_exists_good_placement_l2008_200813


namespace NUMINAMATH_GPT_sum_of_smallest_and_largest_is_correct_l2008_200876

-- Define the conditions
def digits : Set ℕ := {0, 3, 4, 8}

-- Define the smallest and largest valid four-digit number using the digits
def smallest_number : ℕ := 3048
def largest_number : ℕ := 8430

-- Define the sum of the smallest and largest numbers
def sum_of_numbers : ℕ := smallest_number + largest_number

-- The theorem to be proven
theorem sum_of_smallest_and_largest_is_correct : 
  sum_of_numbers = 11478 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_largest_is_correct_l2008_200876


namespace NUMINAMATH_GPT_farey_neighbors_of_half_l2008_200806

noncomputable def farey_neighbors (n : ℕ) : List (ℚ) :=
  if n % 2 = 1 then
    [ (n - 1 : ℚ) / (2 * n), (n + 1 : ℚ) / (2 * n) ]
  else
    [ (n - 2 : ℚ) / (2 * (n - 1)), n / (2 * (n - 1)) ]

theorem farey_neighbors_of_half (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℚ, a ∈ farey_neighbors n ∧ b ∈ farey_neighbors n ∧ 
    (n % 2 = 1 → a = (n - 1 : ℚ) / (2 * n) ∧ b = (n + 1 : ℚ) / (2 * n)) ∧
    (n % 2 = 0 → a = (n - 2 : ℚ) / (2 * (n - 1)) ∧ b = n / (2 * (n - 1))) :=
sorry

end NUMINAMATH_GPT_farey_neighbors_of_half_l2008_200806


namespace NUMINAMATH_GPT_number_of_teachers_l2008_200873

theorem number_of_teachers
  (T S : ℕ)
  (h1 : T + S = 2400)
  (h2 : 320 = 320) -- This condition is trivial and can be ignored
  (h3 : 280 = 280) -- This condition is trivial and can be ignored
  (h4 : S / 280 = T / 40) : T = 300 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teachers_l2008_200873


namespace NUMINAMATH_GPT_calculate_y_l2008_200837

theorem calculate_y (x y : ℝ) (h1 : x = 101) (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : y = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_calculate_y_l2008_200837


namespace NUMINAMATH_GPT_ratio_of_pete_to_susan_l2008_200840

noncomputable def Pete_backward_speed := 12 -- in miles per hour
noncomputable def Pete_handstand_speed := 2 -- in miles per hour
noncomputable def Tracy_cartwheel_speed := 4 * Pete_handstand_speed -- in miles per hour
noncomputable def Susan_forward_speed := Tracy_cartwheel_speed / 2 -- in miles per hour

theorem ratio_of_pete_to_susan :
  Pete_backward_speed / Susan_forward_speed = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_pete_to_susan_l2008_200840


namespace NUMINAMATH_GPT_power_of_two_contains_k_as_substring_l2008_200804

theorem power_of_two_contains_k_as_substring (k : ℕ) (h1 : 1000 ≤ k) (h2 : k < 10000) : 
  ∃ n < 20000, ∀ m, 10^m * k ≤ 2^n ∧ 2^n < 10^(m+4) * (k+1) :=
sorry

end NUMINAMATH_GPT_power_of_two_contains_k_as_substring_l2008_200804


namespace NUMINAMATH_GPT_john_spent_at_candy_store_l2008_200802

-- Definition of the conditions
def allowance : ℚ := 1.50
def arcade_spent : ℚ := (3 / 5) * allowance
def remaining_after_arcade : ℚ := allowance - arcade_spent
def toy_store_spent : ℚ := (1 / 3) * remaining_after_arcade

-- Statement and Proof of the Problem
theorem john_spent_at_candy_store : (remaining_after_arcade - toy_store_spent) = 0.40 :=
by
  -- Proof is left as an exercise
  sorry

end NUMINAMATH_GPT_john_spent_at_candy_store_l2008_200802


namespace NUMINAMATH_GPT_minimum_questions_needed_a_l2008_200895

theorem minimum_questions_needed_a (n : ℕ) (m : ℕ) (h1 : m = n) (h2 : m < 2 ^ n) :
  ∃Q : ℕ, Q = n := sorry

end NUMINAMATH_GPT_minimum_questions_needed_a_l2008_200895


namespace NUMINAMATH_GPT_sixth_term_of_geometric_sequence_l2008_200826

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ n

theorem sixth_term_of_geometric_sequence (a : ℝ) (r : ℝ)
  (h1 : a = 243) (h2 : geometric_sequence a r 7 = 32) :
  geometric_sequence a r 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sixth_term_of_geometric_sequence_l2008_200826


namespace NUMINAMATH_GPT_six_digit_number_division_l2008_200842

theorem six_digit_number_division :
  ∃ a b p : ℕ, 
    (111111 * a = 1111 * b * 233 + p) ∧ 
    (11111 * a = 111 * b * 233 + p - 1000) ∧
    (111111 * 7 = 777777) ∧
    (1111 * 3 = 3333) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_division_l2008_200842


namespace NUMINAMATH_GPT_area_of_triangle_QCA_l2008_200862

noncomputable def triangle_area (x p : ℝ) (hx : x > 0) (hp : p < 12) : ℝ :=
  1 / 2 * x * (12 - p)

theorem area_of_triangle_QCA (x p : ℝ) (hx : x > 0) (hp : p < 12) :
  triangle_area x p hx hp = x * (12 - p) / 2 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_QCA_l2008_200862


namespace NUMINAMATH_GPT_smallest_a₁_l2008_200897

-- We define the sequence a_n and its recurrence relation
def a (n : ℕ) (a₁ : ℝ) : ℝ :=
  match n with
  | 0     => 0  -- this case is not used, but included for function completeness
  | 1     => a₁
  | (n+2) => 11 * a (n+1) a₁ - (n+2)

theorem smallest_a₁ : ∃ a₁ : ℝ, (a₁ = 21 / 100) ∧ ∀ n > 1, a n a₁ > 0 := 
  sorry

end NUMINAMATH_GPT_smallest_a₁_l2008_200897


namespace NUMINAMATH_GPT_larger_number_l2008_200898

theorem larger_number (t a b : ℝ) (h1 : a + b = t) (h2 : a ^ 2 - b ^ 2 = 208) (ht : t = 104) :
  a = 53 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_l2008_200898


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l2008_200880

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n ∧ (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ n = 104 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l2008_200880

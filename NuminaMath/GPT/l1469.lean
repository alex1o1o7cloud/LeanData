import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1469_146953

theorem simplify_expression (a : ℤ) (h_range : -3 < a ∧ a ≤ 0) (h_notzero : a ≠ 0) (h_notone : a ≠ 1 ∧ a ≠ -1) :
  (a - (2 * a - 1) / a) / (1 / a - a) = -3 :=
by
  have h_eq : (a - (2 * a - 1) / a) / (1 / a - a) = (1 - a) / (1 + a) :=
    sorry
  have h_a_neg_two : a = -2 :=
    sorry
  rw [h_eq, h_a_neg_two]
  sorry


end NUMINAMATH_GPT_simplify_expression_l1469_146953


namespace NUMINAMATH_GPT_smallest_k_divides_l1469_146948

-- Given Problem: z^{12} + z^{11} + z^8 + z^7 + z^6 + z^3 + 1 divides z^k - 1
theorem smallest_k_divides (
  k : ℕ
) : (∀ z : ℂ, (z ^ 12 + z ^ 11 + z ^ 8 + z ^ 7 + z ^ 6 + z ^ 3 + 1) ∣ (z ^ k - 1) ↔ k = 182) :=
sorry

end NUMINAMATH_GPT_smallest_k_divides_l1469_146948


namespace NUMINAMATH_GPT_find_a_l1469_146980

theorem find_a (a b c d : ℕ) (h1 : 2 * a + 2 = b) (h2 : 2 * b + 2 = c) (h3 : 2 * c + 2 = d) (h4 : 2 * d + 2 = 62) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1469_146980


namespace NUMINAMATH_GPT_largest_of_three_l1469_146916

structure RealTriple (x y z : ℝ) where
  h1 : x + y + z = 3
  h2 : x * y + y * z + z * x = -8
  h3 : x * y * z = -18

theorem largest_of_three {x y z : ℝ} (h : RealTriple x y z) : max x (max y z) = Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_largest_of_three_l1469_146916


namespace NUMINAMATH_GPT_sum_of_infinite_series_l1469_146932

noncomputable def infinite_series : ℝ :=
  ∑' k : ℕ, (k^3 : ℝ) / (3^k : ℝ)

theorem sum_of_infinite_series :
  infinite_series = (39/16 : ℝ) :=
sorry

end NUMINAMATH_GPT_sum_of_infinite_series_l1469_146932


namespace NUMINAMATH_GPT_square_reciprocal_sum_integer_l1469_146936

theorem square_reciprocal_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^2 + 1/a^2 = m := by
  sorry

end NUMINAMATH_GPT_square_reciprocal_sum_integer_l1469_146936


namespace NUMINAMATH_GPT_same_curve_option_B_l1469_146961

theorem same_curve_option_B : 
  (∀ x y : ℝ, |y| = |x| ↔ y = x ∨ y = -x) ∧ (∀ x y : ℝ, y^2 = x^2 ↔ y = x ∨ y = -x) :=
by
  sorry

end NUMINAMATH_GPT_same_curve_option_B_l1469_146961


namespace NUMINAMATH_GPT_bobby_shoes_cost_l1469_146978

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end NUMINAMATH_GPT_bobby_shoes_cost_l1469_146978


namespace NUMINAMATH_GPT_canal_depth_l1469_146950

-- Define the problem parameters
def top_width : ℝ := 6
def bottom_width : ℝ := 4
def cross_section_area : ℝ := 10290

-- Define the theorem to prove the depth of the canal
theorem canal_depth :
  (1 / 2) * (top_width + bottom_width) * h = cross_section_area → h = 2058 :=
by sorry

end NUMINAMATH_GPT_canal_depth_l1469_146950


namespace NUMINAMATH_GPT_winning_vote_majority_l1469_146960

theorem winning_vote_majority (h1 : 0.70 * 900 = 630)
                             (h2 : 0.30 * 900 = 270) :
  630 - 270 = 360 :=
by
  sorry

end NUMINAMATH_GPT_winning_vote_majority_l1469_146960


namespace NUMINAMATH_GPT_area_of_triangle_l1469_146975

theorem area_of_triangle (h : ℝ) (a : ℝ) (b : ℝ) (hypotenuse : h = 13) (side_a : a = 5) (right_triangle : a^2 + b^2 = h^2) : 
  ∃ (area : ℝ), area = 30 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1469_146975


namespace NUMINAMATH_GPT_find_length_of_AB_l1469_146901

variable (A B C : ℝ)
variable (cos_C_div2 BC AC AB : ℝ)
variable (C_gt_0 : 0 < C / 2) (C_lt_pi : C / 2 < Real.pi)

axiom h1 : cos_C_div2 = Real.sqrt 5 / 5
axiom h2 : BC = 1
axiom h3 : AC = 5
axiom h4 : AB = Real.sqrt (BC ^ 2 + AC ^ 2 - 2 * BC * AC * (2 * cos_C_div2 ^ 2 - 1))

theorem find_length_of_AB : AB = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_AB_l1469_146901


namespace NUMINAMATH_GPT_total_peaches_l1469_146976

theorem total_peaches (num_baskets num_red num_green : ℕ)
    (h1 : num_baskets = 11)
    (h2 : num_red = 10)
    (h3 : num_green = 18) : (num_red + num_green) * num_baskets = 308 := by
  sorry

end NUMINAMATH_GPT_total_peaches_l1469_146976


namespace NUMINAMATH_GPT_cricket_run_rate_l1469_146985

theorem cricket_run_rate (r : ℝ) (o₁ T o₂ : ℕ) (r₁ : ℝ) (Rₜ : ℝ) : 
  r = 4.8 ∧ o₁ = 10 ∧ T = 282 ∧ o₂ = 40 ∧ r₁ = (T - r * o₁) / o₂ → Rₜ = 5.85 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l1469_146985


namespace NUMINAMATH_GPT_inequality_any_k_l1469_146959

theorem inequality_any_k (x y z : ℝ) (k : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) : 
  x ^ (-k : ℤ) + y ^ (-k : ℤ) + z ^ (-k : ℤ) ≥ x ^ k + y ^ k + z ^ k :=
sorry

end NUMINAMATH_GPT_inequality_any_k_l1469_146959


namespace NUMINAMATH_GPT_foot_slide_distance_l1469_146904

def ladder_foot_slide (l h_initial h_new x_initial d y: ℝ) : Prop :=
  l = 30 ∧ x_initial = 6 ∧ d = 6 ∧
  h_initial = Real.sqrt (l^2 - x_initial^2) ∧
  h_new = h_initial - d ∧
  (l^2 = h_new^2 + (x_initial + y) ^ 2) → y = 18

theorem foot_slide_distance :
  ladder_foot_slide 30 (Real.sqrt (30^2 - 6^2)) ((Real.sqrt (30^2 - 6^2)) - 6) 6 6 18 :=
by
  sorry

end NUMINAMATH_GPT_foot_slide_distance_l1469_146904


namespace NUMINAMATH_GPT_evaluate_expression_l1469_146957

theorem evaluate_expression :
  (42 / (9 - 3 * 2)) * 4 = 56 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1469_146957


namespace NUMINAMATH_GPT_polynomial_coeff_sum_neg_33_l1469_146940

theorem polynomial_coeff_sum_neg_33
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 - 3 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -33 :=
by sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_neg_33_l1469_146940


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1469_146993

/-
Mathematical problem: Calculate the sum of the infinite geometric series 1 + (1/2) + (1/2)^2 + (1/2)^3 + ... . Express your answer as a common fraction.

Conditions:
- The first term \( a \) is 1.
- The common ratio \( r \) is \(\frac{1}{2}\).

Answer:
- The sum of the series is 2.
-/

theorem infinite_geometric_series_sum :
  let a := 1
  let r := 1 / 2
  (a * (1 / (1 - r))) = 2 :=
by
  let a := 1
  let r := 1 / 2
  have h : 1 * (1 / (1 - r)) = 2 := by sorry
  exact h

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1469_146993


namespace NUMINAMATH_GPT_solve_for_m_l1469_146969

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1469_146969


namespace NUMINAMATH_GPT_union_of_M_and_N_l1469_146937

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def compl_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_of_M_and_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_union_of_M_and_N_l1469_146937


namespace NUMINAMATH_GPT_unique_solution_exists_l1469_146991

theorem unique_solution_exists :
  ∃! (x y : ℝ), 4^(x^2 + 2 * y) + 4^(2 * x + y^2) = Real.cos (Real.pi * x) ∧ (x, y) = (2, -2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l1469_146991


namespace NUMINAMATH_GPT_minimum_value_proof_l1469_146947

variables {A B C : ℝ}
variable (triangle_ABC : 
  ∀ {A B C : ℝ}, 
  (A > 0 ∧ A < π / 2) ∧ 
  (B > 0 ∧ B < π / 2) ∧ 
  (C > 0 ∧ C < π / 2))

noncomputable def minimum_value (A B C : ℝ) :=
  3 * (Real.tan B) * (Real.tan C) + 
  2 * (Real.tan A) * (Real.tan C) + 
  1 * (Real.tan A) * (Real.tan B)

theorem minimum_value_proof (h : 
  ∀ (A B C : ℝ), 
  (1 / (Real.tan A * Real.tan B)) + 
  (1 / (Real.tan B * Real.tan C)) + 
  (1 / (Real.tan C * Real.tan A)) = 1) 
  : minimum_value A B C = 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_proof_l1469_146947


namespace NUMINAMATH_GPT_annual_feeding_cost_is_correct_l1469_146952

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end NUMINAMATH_GPT_annual_feeding_cost_is_correct_l1469_146952


namespace NUMINAMATH_GPT_afternoon_sales_l1469_146968

variable (x y : ℕ)

theorem afternoon_sales (hx : y = 2 * x) (hy : x + y = 390) : y = 260 := by
  sorry

end NUMINAMATH_GPT_afternoon_sales_l1469_146968


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1469_146971

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1469_146971


namespace NUMINAMATH_GPT_four_digit_numbers_with_three_identical_digits_l1469_146927

theorem four_digit_numbers_with_three_identical_digits :
  ∃ n : ℕ, (n = 18) ∧ (∀ x, 1000 ≤ x ∧ x < 10000 → 
  (x / 1000 = 1) ∧ (
    (x % 1000 / 100 = x % 100 / 10) ∧ (x % 1000 / 100 = x % 10))) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_with_three_identical_digits_l1469_146927


namespace NUMINAMATH_GPT_intersection_hyperbola_l1469_146926

theorem intersection_hyperbola (t : ℝ) :
  ∃ A B : ℝ, ∀ (x y : ℝ),
  (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 5 = 0) →
  (x^2 / A - y^2 / B = 1) :=
sorry

end NUMINAMATH_GPT_intersection_hyperbola_l1469_146926


namespace NUMINAMATH_GPT_molecular_weight_8_moles_Al2O3_l1469_146915

noncomputable def molecular_weight_Al2O3 (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3
  (atomic_weight_Al : ℝ := 26.98)
  (atomic_weight_O : ℝ := 16.00)
  : molecular_weight_Al2O3 atomic_weight_Al atomic_weight_O * 8 = 815.68 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_8_moles_Al2O3_l1469_146915


namespace NUMINAMATH_GPT_common_factor_l1469_146996

theorem common_factor (x y : ℝ) : 
  ∃ c : ℝ, c * (3 * x * y^2 - 4 * x^2 * y) = 6 * x^2 * y - 8 * x * y^2 ∧ c = 2 * x * y := 
by 
  sorry

end NUMINAMATH_GPT_common_factor_l1469_146996


namespace NUMINAMATH_GPT_inequality_correct_l1469_146923

noncomputable def a : ℝ := Real.exp (-0.5)
def b : ℝ := 0.5
noncomputable def c : ℝ := Real.log 1.5

theorem inequality_correct : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_inequality_correct_l1469_146923


namespace NUMINAMATH_GPT_melissa_points_per_game_l1469_146920

theorem melissa_points_per_game (total_points : ℕ) (games : ℕ) (h1 : total_points = 81) 
(h2 : games = 3) : total_points / games = 27 :=
by
  sorry

end NUMINAMATH_GPT_melissa_points_per_game_l1469_146920


namespace NUMINAMATH_GPT_binary_digit_sum_property_l1469_146984

def binary_digit_sum (n : Nat) : Nat :=
  n.digits 2 |>.foldr (· + ·) 0

theorem binary_digit_sum_property (k : Nat) (h_pos : 0 < k) :
  (Finset.range (2^k)).sum (λ n => binary_digit_sum (n + 1)) = 2^(k - 1) * k + 1 := 
sorry

end NUMINAMATH_GPT_binary_digit_sum_property_l1469_146984


namespace NUMINAMATH_GPT_find_minimum_value_M_l1469_146905

theorem find_minimum_value_M : (∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2 * x ≤ M) ∧ M = 1) := 
sorry

end NUMINAMATH_GPT_find_minimum_value_M_l1469_146905


namespace NUMINAMATH_GPT_digits_solution_l1469_146958

noncomputable def validate_reverse_multiplication
  (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
  (E * 10000 + D * 1000 + C * 100 + B * 10 + A)

theorem digits_solution :
  validate_reverse_multiplication 2 1 9 7 8 :=
by
  sorry

end NUMINAMATH_GPT_digits_solution_l1469_146958


namespace NUMINAMATH_GPT_money_spent_on_jacket_l1469_146999

-- Define the initial amounts
def initial_money_sandy : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def additional_money_found : ℝ := 7.43

-- Amount of money left after buying the shirt
def remaining_after_shirt := initial_money_sandy - amount_spent_shirt

-- Total money after finding additional money
def total_after_additional := remaining_after_shirt + additional_money_found

-- Theorem statement: The amount Sandy spent on the jacket
theorem money_spent_on_jacket : total_after_additional = 9.28 :=
by
  sorry

end NUMINAMATH_GPT_money_spent_on_jacket_l1469_146999


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l1469_146992

-- We define a geometric sequence and the condition provided in the problem.
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Condition for geometric sequence: a_n = a * q^(n-1)
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^(n-1)

-- Given condition: 2a_4 = a_6 - a_5
def given_condition (a : ℕ → ℝ) : Prop := 
  2 * a 4 = a 6 - a 5

-- Proof statement
theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_seq a q) (h_cond : given_condition a) : 
    q = 2 ∨ q = -1 :=
sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_l1469_146992


namespace NUMINAMATH_GPT_speed_of_other_person_l1469_146949

-- Definitions related to the problem conditions
def pooja_speed : ℝ := 3  -- Pooja's speed in km/hr
def time : ℝ := 4  -- Time in hours
def distance : ℝ := 20  -- Distance between them after 4 hours in km

-- Define the unknown speed S as a parameter to be solved
variable (S : ℝ)

-- Define the relative speed when moving in opposite directions
def relative_speed (S : ℝ) : ℝ := S + pooja_speed

-- Create a theorem to encapsulate the problem and to be proved
theorem speed_of_other_person 
  (h : distance = relative_speed S * time) : S = 2 := 
  sorry

end NUMINAMATH_GPT_speed_of_other_person_l1469_146949


namespace NUMINAMATH_GPT_abs_diff_inequality_l1469_146983

theorem abs_diff_inequality (a b c h : ℝ) (hab : |a - c| < h) (hbc : |b - c| < h) : |a - b| < 2 * h := 
by
  sorry

end NUMINAMATH_GPT_abs_diff_inequality_l1469_146983


namespace NUMINAMATH_GPT_dog_catches_sheep_in_20_seconds_l1469_146921

variable (v_sheep v_dog : ℕ) (d : ℕ)

def relative_speed (v_dog v_sheep : ℕ) := v_dog - v_sheep

def time_to_catch (d v_sheep v_dog : ℕ) : ℕ := d / (relative_speed v_dog v_sheep)

theorem dog_catches_sheep_in_20_seconds
  (h1 : v_sheep = 16)
  (h2 : v_dog = 28)
  (h3 : d = 240) :
  time_to_catch d v_sheep v_dog = 20 := by {
  sorry
}

end NUMINAMATH_GPT_dog_catches_sheep_in_20_seconds_l1469_146921


namespace NUMINAMATH_GPT_find_two_numbers_l1469_146931

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end NUMINAMATH_GPT_find_two_numbers_l1469_146931


namespace NUMINAMATH_GPT_a_older_than_b_l1469_146995

theorem a_older_than_b (A B : ℕ) (h1 : B = 36) (h2 : A + 10 = 2 * (B - 10)) : A - B = 6 :=
  sorry

end NUMINAMATH_GPT_a_older_than_b_l1469_146995


namespace NUMINAMATH_GPT_g_9_pow_4_l1469_146906

theorem g_9_pow_4 (f g : ℝ → ℝ) (h1 : ∀ x ≥ 1, f (g x) = x^2) (h2 : ∀ x ≥ 1, g (f x) = x^4) (h3 : g 81 = 81) : (g 9)^4 = 81 :=
sorry

end NUMINAMATH_GPT_g_9_pow_4_l1469_146906


namespace NUMINAMATH_GPT_man_walk_time_l1469_146911

theorem man_walk_time (speed_kmh : ℕ) (distance_km : ℕ) (time_min : ℕ) 
  (h1 : speed_kmh = 10) (h2 : distance_km = 7) : time_min = 42 :=
by
  sorry

end NUMINAMATH_GPT_man_walk_time_l1469_146911


namespace NUMINAMATH_GPT_wario_missed_field_goals_wide_right_l1469_146913

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end NUMINAMATH_GPT_wario_missed_field_goals_wide_right_l1469_146913


namespace NUMINAMATH_GPT_exists_m_square_between_l1469_146945

theorem exists_m_square_between (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, a < m^2 ∧ m^2 < d := 
sorry

end NUMINAMATH_GPT_exists_m_square_between_l1469_146945


namespace NUMINAMATH_GPT_lottery_probability_l1469_146986

theorem lottery_probability (x_1 x_2 x_3 x_4 : ℝ) (p : ℝ) (h0 : 0 < p ∧ p < 1) : 
  x_1 = p * x_3 → 
  x_2 = p * x_4 + (1 - p) * x_1 → 
  x_3 = p + (1 - p) * x_2 → 
  x_4 = p + (1 - p) * x_3 → 
  x_2 = 0.19 :=
by
  sorry

end NUMINAMATH_GPT_lottery_probability_l1469_146986


namespace NUMINAMATH_GPT_find_x_l1469_146933

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 152) : x = 16 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1469_146933


namespace NUMINAMATH_GPT_quadratic_symmetry_l1469_146962

def quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

theorem quadratic_symmetry (b c : ℝ) :
  let f := quadratic b c
  (f 2) < (f 1) ∧ (f 1) < (f 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_symmetry_l1469_146962


namespace NUMINAMATH_GPT_multiple_of_weight_lifted_l1469_146900

variable (F : ℝ) (M : ℝ)

theorem multiple_of_weight_lifted 
  (H1: ∀ (B : ℝ), B = 2 * F) 
  (H2: ∀ (B : ℝ), ∀ (W : ℝ), W = 3 * B) 
  (H3: ∃ (B : ℝ), (3 * B = 600)) 
  (H4: M * F = 150) : 
  M = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_weight_lifted_l1469_146900


namespace NUMINAMATH_GPT_triangle_proof_l1469_146997

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions
axiom cos_rule_1 : a / cos A = c / (2 - cos C)
axiom b_value : b = 4
axiom c_value : c = 3
axiom area_equation : (1 / 2) * a * b * sin C = 3

-- The theorem statement
theorem triangle_proof : 3 * sin C + 4 * cos C = 5 := sorry

end NUMINAMATH_GPT_triangle_proof_l1469_146997


namespace NUMINAMATH_GPT_power_inequality_l1469_146919

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ (3 / 4) + b ^ (3 / 4) + c ^ (3 / 4) > (a + b + c) ^ (3 / 4) :=
sorry

end NUMINAMATH_GPT_power_inequality_l1469_146919


namespace NUMINAMATH_GPT_min_value_2a_plus_b_l1469_146956

theorem min_value_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (1/a) + (2/b) = 1): 2 * a + b = 8 :=
sorry

end NUMINAMATH_GPT_min_value_2a_plus_b_l1469_146956


namespace NUMINAMATH_GPT_library_visitor_ratio_l1469_146943

theorem library_visitor_ratio (T : ℕ) (h1 : 50 + T + 20 * 4 = 250) : T / 50 = 2 :=
by
  sorry

end NUMINAMATH_GPT_library_visitor_ratio_l1469_146943


namespace NUMINAMATH_GPT_find_positive_real_number_solution_l1469_146922

theorem find_positive_real_number_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) (hx : x > 0) : x = 15 :=
sorry

end NUMINAMATH_GPT_find_positive_real_number_solution_l1469_146922


namespace NUMINAMATH_GPT_find_x_l1469_146977

theorem find_x : ∃ x : ℤ, x + 3 * 10 = 33 → x = 3 := by
  sorry

end NUMINAMATH_GPT_find_x_l1469_146977


namespace NUMINAMATH_GPT_tom_jerry_age_ratio_l1469_146908

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end NUMINAMATH_GPT_tom_jerry_age_ratio_l1469_146908


namespace NUMINAMATH_GPT_all_positive_integers_in_A_l1469_146973

variable (A : Set ℕ)

-- Conditions
def has_at_least_three_elements : Prop :=
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c

def all_divisors_in_set : Prop :=
  ∀ m : ℕ, m ∈ A → (∀ d : ℕ, d ∣ m → d ∈ A)

def  bc_plus_one_in_set : Prop :=
  ∀ b c : ℕ, 1 < b → b < c → b ∈ A → c ∈ A → 1 + b * c ∈ A

-- Theorem statement
theorem all_positive_integers_in_A
  (h1 : has_at_least_three_elements A)
  (h2 : all_divisors_in_set A)
  (h3 : bc_plus_one_in_set A) : ∀ n : ℕ, n > 0 → n ∈ A := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_all_positive_integers_in_A_l1469_146973


namespace NUMINAMATH_GPT_determinant_sum_is_34_l1469_146935

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![5, -2],
  ![3, 4]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 3],
  ![-1, 2]
]

-- Prove the determinant of the sum of A and B is 34
theorem determinant_sum_is_34 : Matrix.det (A + B) = 34 := by
  sorry

end NUMINAMATH_GPT_determinant_sum_is_34_l1469_146935


namespace NUMINAMATH_GPT_evaluate_complex_expression_l1469_146955

noncomputable def expression := 
  Complex.mk (-1) (Real.sqrt 3) / 2

noncomputable def conjugate_expression := 
  Complex.mk (-1) (-Real.sqrt 3) / 2

theorem evaluate_complex_expression :
  (expression ^ 12 + conjugate_expression ^ 12) = 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_complex_expression_l1469_146955


namespace NUMINAMATH_GPT_factor_polynomial_l1469_146972

theorem factor_polynomial :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x ^ 2 =
  (3 * x ^ 2 + 59 * x + 231) * (3 * x ^ 2 + 53 * x + 231) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1469_146972


namespace NUMINAMATH_GPT_liza_butter_amount_l1469_146918

theorem liza_butter_amount (B : ℕ) (h1 : B / 2 + B / 5 + (1 / 3) * ((B - B / 2 - B / 5) / 1) = B - 2) : B = 10 :=
sorry

end NUMINAMATH_GPT_liza_butter_amount_l1469_146918


namespace NUMINAMATH_GPT_michael_scored_times_more_goals_l1469_146925

theorem michael_scored_times_more_goals (x : ℕ) (hb : Bruce_goals = 4) (hm : Michael_goals = 4 * x) (ht : Bruce_goals + Michael_goals = 16) : x = 3 := by
  sorry

end NUMINAMATH_GPT_michael_scored_times_more_goals_l1469_146925


namespace NUMINAMATH_GPT_max_chips_can_be_removed_l1469_146934

theorem max_chips_can_be_removed (initial_chips : (Fin 10) × (Fin 10) → ℕ) 
  (condition : ∀ i j, initial_chips (i, j) = 1) : 
    ∃ removed_chips : ℕ, removed_chips = 90 :=
by
  sorry

end NUMINAMATH_GPT_max_chips_can_be_removed_l1469_146934


namespace NUMINAMATH_GPT_transform_binomial_expansion_l1469_146910

variable (a b : ℝ)

theorem transform_binomial_expansion (h : (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4) :
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
by
  sorry

end NUMINAMATH_GPT_transform_binomial_expansion_l1469_146910


namespace NUMINAMATH_GPT_shaded_region_area_l1469_146982

noncomputable def side_length := 1 -- Length of each side of the squares, in cm.

-- Conditions
def top_square_center_above_edge : Prop := 
  ∀ square1 square2 square3 : ℝ, square3 = (square1 + square2) / 2

-- Question: Area of the shaded region
def area_of_shaded_region := 1 -- area in cm^2

-- Lean 4 Statement
theorem shaded_region_area :
  top_square_center_above_edge → area_of_shaded_region = 1 := 
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1469_146982


namespace NUMINAMATH_GPT_intersection_x_value_l1469_146909

theorem intersection_x_value :
  (∃ x y, y = 3 * x - 7 ∧ y = 48 - 5 * x) → x = 55 / 8 :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_value_l1469_146909


namespace NUMINAMATH_GPT_largest_integer_n_neg_l1469_146944

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end NUMINAMATH_GPT_largest_integer_n_neg_l1469_146944


namespace NUMINAMATH_GPT_group_B_same_order_l1469_146979

-- Definitions for the expressions in each group
def expr_A1 := 2 * 9 / 3
def expr_A2 := 2 + 9 * 3

def expr_B1 := 36 - 9 + 5
def expr_B2 := 36 / 6 * 5

def expr_C1 := 56 / 7 * 5
def expr_C2 := 56 + 7 * 5

-- Theorem stating that Group B expressions have the same order of operations
theorem group_B_same_order : (expr_B1 = expr_B2) := 
  sorry

end NUMINAMATH_GPT_group_B_same_order_l1469_146979


namespace NUMINAMATH_GPT_wall_number_of_bricks_l1469_146907

theorem wall_number_of_bricks (x : ℝ) :
  (∃ x, 6 * ((x / 7) + (x / 11) - 12) = x) →  x = 179 :=
by
  sorry

end NUMINAMATH_GPT_wall_number_of_bricks_l1469_146907


namespace NUMINAMATH_GPT_temperature_difference_l1469_146903

def highest_temperature : ℝ := 8
def lowest_temperature : ℝ := -1

theorem temperature_difference : highest_temperature - lowest_temperature = 9 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l1469_146903


namespace NUMINAMATH_GPT_smallest_b_for_no_real_root_l1469_146929

theorem smallest_b_for_no_real_root :
  ∃ b : ℤ, (b < 8 ∧ b > -8) ∧ (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ -6) ∧ (b = -7) :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_no_real_root_l1469_146929


namespace NUMINAMATH_GPT_find_multiple_l1469_146988

-- Defining the conditions
def first_lock_time := 5
def second_lock_time (x : ℕ) := 5 * x - 3

-- Proving the multiple
theorem find_multiple : 
  ∃ x : ℕ, (5 * first_lock_time * x - 3) * 5 = 60 ∧ (x = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1469_146988


namespace NUMINAMATH_GPT_find_b_c_d_l1469_146917

def f (x : ℝ) := x^3 + 2 * x^2 + 3 * x + 4
def h (x : ℝ) := x^3 + 6 * x^2 - 8 * x + 16

theorem find_b_c_d :
  (∀ r : ℝ, f r = 0 → h (r^3) = 0) ∧ h (x : ℝ) = x^3 + 6 * x^2 - 8 * x + 16 :=
by 
  -- proof not required
  sorry

end NUMINAMATH_GPT_find_b_c_d_l1469_146917


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1469_146938

noncomputable def molecularWeight (Ca_wt : ℝ) (O_wt : ℝ) (H_wt : ℝ) (nCa : ℕ) (nO : ℕ) (nH : ℕ) : ℝ :=
  (nCa * Ca_wt) + (nO * O_wt) + (nH * H_wt)

theorem molecular_weight_of_compound :
  molecularWeight 40.08 15.999 1.008 1 2 2 = 74.094 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1469_146938


namespace NUMINAMATH_GPT_real_numbers_satisfy_relation_l1469_146981

theorem real_numbers_satisfy_relation (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end NUMINAMATH_GPT_real_numbers_satisfy_relation_l1469_146981


namespace NUMINAMATH_GPT_molecular_weight_of_3_moles_CaOH2_is_correct_l1469_146930

-- Define the atomic weights as given by the conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define the molecular formula contributions for Ca(OH)2
def molecular_weight_CaOH2 : ℝ :=
  atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H

-- Define the weight of 3 moles of Ca(OH)2 based on the molecular weight
def weight_of_3_moles_CaOH2 : ℝ :=
  3 * molecular_weight_CaOH2

-- Theorem to prove the final result
theorem molecular_weight_of_3_moles_CaOH2_is_correct :
  weight_of_3_moles_CaOH2 = 222.30 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_3_moles_CaOH2_is_correct_l1469_146930


namespace NUMINAMATH_GPT_max_abs_sum_value_l1469_146974

noncomputable def max_abs_sum (x y : ℝ) : ℝ := |x| + |y|

theorem max_abs_sum_value (x y : ℝ) (h : x^2 + y^2 = 4) : max_abs_sum x y ≤ 2 * Real.sqrt 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_abs_sum_value_l1469_146974


namespace NUMINAMATH_GPT_sum_of_constants_l1469_146928

theorem sum_of_constants :
  ∃ (a b c d e : ℤ), 1000 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e) ∧ a + b + c + d + e = 92 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_constants_l1469_146928


namespace NUMINAMATH_GPT_total_students_in_class_l1469_146963

theorem total_students_in_class (R S : ℕ)
  (h1 : 2 + 12 * 1 + 12 * 2 + 3 * R = S * 2)
  (h2 : S = 2 + 12 + 12 + R) :
  S = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1469_146963


namespace NUMINAMATH_GPT_fraction_meaningful_if_and_only_if_l1469_146998

theorem fraction_meaningful_if_and_only_if {x : ℝ} : (2 * x - 1 ≠ 0) ↔ (x ≠ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_if_and_only_if_l1469_146998


namespace NUMINAMATH_GPT_rotate_D_90_clockwise_l1469_146951

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℤ) : Point ℤ :=
  ⟨p.y, -p.x⟩

def D : Point ℤ := ⟨-3, 2⟩
def E : Point ℤ := ⟨0, 5⟩
def F : Point ℤ := ⟨0, 2⟩

theorem rotate_D_90_clockwise :
  rotate_90_clockwise D = Point.mk 2 (-3) :=
by
  sorry

end NUMINAMATH_GPT_rotate_D_90_clockwise_l1469_146951


namespace NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_solution_l1469_146946

theorem system_of_equations_solution (x y : ℝ) :
  (3 * x - 4 * y = 1) → (5 * x + 2 * y = 6) → 
  x = 1 ∧ y = 0.5 := by
  sorry

theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x + 6 > 0) → (x - 2 < -x) → 
  -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_solution_l1469_146946


namespace NUMINAMATH_GPT_non_neg_integer_solutions_l1469_146914

theorem non_neg_integer_solutions (a b c : ℕ) :
  (∀ x : ℕ, x^2 - 2 * a * x + b = 0 → x ≥ 0) ∧ 
  (∀ y : ℕ, y^2 - 2 * b * y + c = 0 → y ≥ 0) ∧ 
  (∀ z : ℕ, z^2 - 2 * c * z + a = 0 → z ≥ 0) → 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_non_neg_integer_solutions_l1469_146914


namespace NUMINAMATH_GPT_product_lcm_gcd_eq_128_l1469_146941

theorem product_lcm_gcd_eq_128 : (Int.gcd 8 16) * (Int.lcm 8 16) = 128 :=
by
  sorry

end NUMINAMATH_GPT_product_lcm_gcd_eq_128_l1469_146941


namespace NUMINAMATH_GPT_petya_wins_prize_probability_atleast_one_wins_probability_l1469_146965

/-- Petya and 9 other people each roll a fair six-sided die. 
    A player wins a prize if they roll a number that nobody else rolls more than once.-/
theorem petya_wins_prize_probability : (5 / 6) ^ 9 = 0.194 :=
sorry

/-- The probability that at least one player gets a prize in the game where Petya and
    9 others roll a fair six-sided die is 0.919. -/
theorem atleast_one_wins_probability : 1 - (1 / 6) ^ 9 = 0.919 :=
sorry

end NUMINAMATH_GPT_petya_wins_prize_probability_atleast_one_wins_probability_l1469_146965


namespace NUMINAMATH_GPT_complement_intersection_l1469_146967

noncomputable def U : Set Real := Set.univ
noncomputable def M : Set Real := { x : Real | Real.log x < 0 }
noncomputable def N : Set Real := { x : Real | (1 / 2) ^ x ≥ Real.sqrt (1 / 2) }

theorem complement_intersection (U M N : Set Real) : 
  (Set.compl M ∩ N) = Set.Iic 0 :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1469_146967


namespace NUMINAMATH_GPT_N_subset_M_l1469_146987

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x - 2 = 0}

theorem N_subset_M : N ⊆ M := sorry

end NUMINAMATH_GPT_N_subset_M_l1469_146987


namespace NUMINAMATH_GPT_celsius_to_fahrenheit_l1469_146954

theorem celsius_to_fahrenheit (C F : ℤ) (h1 : C = 50) (h2 : C = 5 / 9 * (F - 32)) : F = 122 :=
by
  sorry

end NUMINAMATH_GPT_celsius_to_fahrenheit_l1469_146954


namespace NUMINAMATH_GPT_tommy_needs_4_steaks_l1469_146989

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end NUMINAMATH_GPT_tommy_needs_4_steaks_l1469_146989


namespace NUMINAMATH_GPT_cubed_ge_sqrt_ab_squared_l1469_146964

theorem cubed_ge_sqrt_ab_squared (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^3 + b^3 ≥ (ab)^(1/2) * (a^2 + b^2) :=
sorry

end NUMINAMATH_GPT_cubed_ge_sqrt_ab_squared_l1469_146964


namespace NUMINAMATH_GPT_first_programmer_loses_l1469_146942

noncomputable def programSequence : List ℕ :=
  List.range 1999 |>.map (fun i => 2^i)

def validMove (sequence : List ℕ) (move : List ℕ) : Prop :=
  move.length = 5 ∧ move.all (λ i => i < sequence.length ∧ sequence.get! i > 0)

def applyMove (sequence : List ℕ) (move : List ℕ) : List ℕ :=
  move.foldl
    (λ seq i => seq.set i (seq.get! i - 1))
    sequence

def totalWeight (sequence : List ℕ) : ℕ :=
  sequence.foldl (· + ·) 0

theorem first_programmer_loses : ∀ seq moves,
  seq = programSequence →
  (∀ move, validMove seq move → False) →
  applyMove seq moves = seq →
  totalWeight seq = 2^1999 - 1 :=
by
  intro seq moves h_seq h_valid_move h_apply_move
  sorry

end NUMINAMATH_GPT_first_programmer_loses_l1469_146942


namespace NUMINAMATH_GPT_bad_oranges_l1469_146994

theorem bad_oranges (total_oranges : ℕ) (students : ℕ) (less_oranges_per_student : ℕ)
  (initial_oranges_per_student now_oranges_per_student shared_oranges now_total_oranges bad_oranges : ℕ) :
  total_oranges = 108 →
  students = 12 →
  less_oranges_per_student = 3 →
  initial_oranges_per_student = total_oranges / students →
  now_oranges_per_student = initial_oranges_per_student - less_oranges_per_student →
  shared_oranges = students * now_oranges_per_student →
  now_total_oranges = 72 →
  bad_oranges = total_oranges - now_total_oranges →
  bad_oranges = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_bad_oranges_l1469_146994


namespace NUMINAMATH_GPT_three_digit_numbers_l1469_146912

theorem three_digit_numbers (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → 
  (n * n % 1000 = n % 1000) ↔ 
  (n = 625 ∨ n = 376) :=
by 
  sorry

end NUMINAMATH_GPT_three_digit_numbers_l1469_146912


namespace NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l1469_146924

noncomputable def circle_tangency_condition_1 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 1
noncomputable def circle_tangency_condition_2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 9

def ellipse_equation (x y : ℝ) : Prop := x ^ 2 / 4 + y ^ 2 / 3 = 1

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  circle_tangency_condition_1 x y ∧ circle_tangency_condition_2 x y →
  ellipse_equation x y := sorry

end NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l1469_146924


namespace NUMINAMATH_GPT_ratio_pen_to_pencil_l1469_146902

-- Define the costs
def cost_of_pencil (P : ℝ) : ℝ := P
def cost_of_pen (P : ℝ) : ℝ := 4 * P
def total_cost (P : ℝ) : ℝ := cost_of_pencil P + cost_of_pen P

-- The proof that the total cost of the pen and pencil is $6 given the provided ratio
theorem ratio_pen_to_pencil (P : ℝ) (h_total_cost : total_cost P = 6) (h_pen_cost : cost_of_pen P = 4) :
  cost_of_pen P / cost_of_pencil P = 4 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_ratio_pen_to_pencil_l1469_146902


namespace NUMINAMATH_GPT_Dawn_commissioned_paintings_l1469_146966

theorem Dawn_commissioned_paintings (time_per_painting : ℕ) (total_earnings : ℕ) (earnings_per_hour : ℕ) 
  (h1 : time_per_painting = 2) 
  (h2 : total_earnings = 3600) 
  (h3 : earnings_per_hour = 150) : 
  (total_earnings / (time_per_painting * earnings_per_hour) = 12) :=
by 
  sorry

end NUMINAMATH_GPT_Dawn_commissioned_paintings_l1469_146966


namespace NUMINAMATH_GPT_volume_of_new_pyramid_is_108_l1469_146970

noncomputable def volume_of_cut_pyramid : ℝ :=
  let base_edge_length := 12 * Real.sqrt 2
  let slant_edge_length := 15
  let cut_height := 4.5
  -- Calculate the height of the original pyramid using Pythagorean theorem
  let original_height := Real.sqrt (slant_edge_length^2 - (base_edge_length/2 * Real.sqrt 2)^2)
  -- Calculate the remaining height of the smaller pyramid
  let remaining_height := original_height - cut_height
  -- Calculate the scale factor
  let scale_factor := remaining_height / original_height
  -- New base edge length
  let new_base_edge_length := base_edge_length * scale_factor
  -- New base area
  let new_base_area := (new_base_edge_length)^2
  -- Volume of the new pyramid
  (1 / 3) * new_base_area * remaining_height

-- Define the statement to prove
theorem volume_of_new_pyramid_is_108 :
  volume_of_cut_pyramid = 108 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_new_pyramid_is_108_l1469_146970


namespace NUMINAMATH_GPT_max_good_pairs_1_to_30_l1469_146990

def is_good_pair (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def max_good_pairs_in_range (n : ℕ) : ℕ :=
  if n = 30 then 13 else 0

theorem max_good_pairs_1_to_30 : max_good_pairs_in_range 30 = 13 :=
by
  sorry

end NUMINAMATH_GPT_max_good_pairs_1_to_30_l1469_146990


namespace NUMINAMATH_GPT_triangle_area_is_96_l1469_146939

/-- Given a square with side length 8 and an overlapping area that is both three-quarters
    of the area of the square and half of the area of a triangle, prove the triangle's area is 96. -/
theorem triangle_area_is_96 (a : ℕ) (area_of_square : ℕ) (overlapping_area : ℕ) (area_of_triangle : ℕ) 
  (h1 : a = 8) 
  (h2 : area_of_square = a * a) 
  (h3 : overlapping_area = (3 * area_of_square) / 4) 
  (h4 : overlapping_area = area_of_triangle / 2) : 
  area_of_triangle = 96 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_is_96_l1469_146939

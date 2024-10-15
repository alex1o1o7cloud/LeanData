import Mathlib

namespace NUMINAMATH_GPT_cos_C_in_triangle_l1229_122951

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end NUMINAMATH_GPT_cos_C_in_triangle_l1229_122951


namespace NUMINAMATH_GPT_jake_present_weight_l1229_122928

variables (J S : ℕ)

theorem jake_present_weight :
  (J - 33 = 2 * S) ∧ (J + S = 153) → J = 113 :=
by
  sorry

end NUMINAMATH_GPT_jake_present_weight_l1229_122928


namespace NUMINAMATH_GPT_henry_jill_age_ratio_l1229_122980

theorem henry_jill_age_ratio :
  ∀ (H J : ℕ), (H + J = 48) → (H = 29) → (J = 19) → ((H - 9) / (J - 9) = 2) :=
by
  intros H J h_sum h_henry h_jill
  sorry

end NUMINAMATH_GPT_henry_jill_age_ratio_l1229_122980


namespace NUMINAMATH_GPT_polynomial_has_at_most_one_integer_root_l1229_122984

theorem polynomial_has_at_most_one_integer_root (k : ℝ) :
  ∀ x y : ℤ, (x^3 - 24 * x + k = 0) ∧ (y^3 - 24 * y + k = 0) → x = y :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_polynomial_has_at_most_one_integer_root_l1229_122984


namespace NUMINAMATH_GPT_initial_overs_l1229_122977

variable (x : ℝ)

/-- 
Proof that the number of initial overs x is 10, given the conditions:
1. The run rate in the initial x overs was 3.2 runs per over.
2. The run rate in the remaining 50 overs was 5 runs per over.
3. The total target is 282 runs.
4. The runs scored in the remaining 50 overs should be 250 runs.
-/
theorem initial_overs (hx : 3.2 * x + 250 = 282) : x = 10 :=
sorry

end NUMINAMATH_GPT_initial_overs_l1229_122977


namespace NUMINAMATH_GPT_m_perp_beta_l1229_122926

variable {Point Line Plane : Type}
variable {belongs : Point → Line → Prop}
variable {perp : Line → Plane → Prop}
variable {intersect : Plane → Plane → Line}

variable (α β γ : Plane)
variable (m n l : Line)

-- Conditions for the problem
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- Proof goal: proving m is perpendicular to β
theorem m_perp_beta : perp m β :=
by
  sorry

end NUMINAMATH_GPT_m_perp_beta_l1229_122926


namespace NUMINAMATH_GPT_find_a_l1229_122929

theorem find_a (a : ℚ) (A : Set ℚ) (h : 3 ∈ A) (hA : A = {a + 2, 2 * a^2 + a}) : a = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1229_122929


namespace NUMINAMATH_GPT_rectangle_width_l1229_122932

theorem rectangle_width (L W : ℝ) (h₁ : 2 * L + 2 * W = 54) (h₂ : W = L + 3) : W = 15 :=
sorry

end NUMINAMATH_GPT_rectangle_width_l1229_122932


namespace NUMINAMATH_GPT_fourth_power_sum_l1229_122921

theorem fourth_power_sum
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 19.5 := 
sorry

end NUMINAMATH_GPT_fourth_power_sum_l1229_122921


namespace NUMINAMATH_GPT_radius_of_circumscribed_sphere_l1229_122995

noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a / Real.sqrt 3

theorem radius_of_circumscribed_sphere 
  (a : ℝ) 
  (h_base_side : 0 < a)
  (h_distance : ∃ d : ℝ, d = a * Real.sqrt 2 / 8) : 
  circumscribed_sphere_radius a = a / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_radius_of_circumscribed_sphere_l1229_122995


namespace NUMINAMATH_GPT_eve_stamp_collection_worth_l1229_122981

def total_value_of_collection (stamps_value : ℕ) (num_stamps : ℕ) (set_size : ℕ) (set_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let value_per_stamp := set_value / set_size
  let total_value := value_per_stamp * num_stamps
  let num_complete_sets := num_stamps / set_size
  let total_bonus := num_complete_sets * bonus_per_set
  total_value + total_bonus

theorem eve_stamp_collection_worth :
  total_value_of_collection 21 21 7 28 5 = 99 := by
  rfl

end NUMINAMATH_GPT_eve_stamp_collection_worth_l1229_122981


namespace NUMINAMATH_GPT_max_concentration_at_2_l1229_122947

noncomputable def concentration (t : ℝ) : ℝ := (20 * t) / (t^2 + 4)

theorem max_concentration_at_2 : ∃ t : ℝ, 0 ≤ t ∧ ∀ s : ℝ, (0 ≤ s → concentration s ≤ concentration t) ∧ t = 2 := 
by 
  sorry -- we add sorry to skip the actual proof

end NUMINAMATH_GPT_max_concentration_at_2_l1229_122947


namespace NUMINAMATH_GPT_series_sum_equals_one_fourth_l1229_122974

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), series_term (n + 1)

theorem series_sum_equals_one_fourth :
  infinite_series_sum = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_series_sum_equals_one_fourth_l1229_122974


namespace NUMINAMATH_GPT_cos_B_of_triangle_l1229_122906

theorem cos_B_of_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : a = 6) (h3 : b = 4) :
  Real.cos B = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_B_of_triangle_l1229_122906


namespace NUMINAMATH_GPT_train_start_time_l1229_122942

theorem train_start_time (D PQ : ℝ) (S₁ S₂ : ℝ) (T₁ T₂ meet : ℝ) :
  PQ = 110  -- Distance between stations P and Q
  ∧ S₁ = 20  -- Speed of the first train
  ∧ S₂ = 25  -- Speed of the second train
  ∧ T₂ = 8  -- Start time of the second train
  ∧ meet = 10 -- Meeting time
  ∧ T₁ + T₂ = meet → -- Meeting time condition
  T₁ = 7.5 := -- Answer: first train start time
by
sorry

end NUMINAMATH_GPT_train_start_time_l1229_122942


namespace NUMINAMATH_GPT_trisha_total_distance_walked_l1229_122945

def d1 : ℝ := 0.1111111111111111
def d2 : ℝ := 0.1111111111111111
def d3 : ℝ := 0.6666666666666666

theorem trisha_total_distance_walked :
  d1 + d2 + d3 = 0.8888888888888888 := 
sorry

end NUMINAMATH_GPT_trisha_total_distance_walked_l1229_122945


namespace NUMINAMATH_GPT_marble_color_197th_l1229_122930

theorem marble_color_197th (n : ℕ) (total_marbles : ℕ) (marble_color : ℕ → ℕ)
                          (h_total : total_marbles = 240)
                          (h_pattern : ∀ k, marble_color (k + 15) = marble_color k)
                          (h_colors : ∀ i, (0 ≤ i ∧ i < 15) →
                                   (marble_color i = if i < 6 then 1
                                   else if i < 11 then 2
                                   else if i < 15 then 3
                                   else 0)) :
  marble_color 197 = 1 := sorry

end NUMINAMATH_GPT_marble_color_197th_l1229_122930


namespace NUMINAMATH_GPT_minimize_expression_l1229_122965

theorem minimize_expression : ∃ c : ℝ, c = 6 ∧ ∀ x : ℝ, (3 / 4) * (x ^ 2) - 9 * x + 7 ≥ (3 / 4) * (6 ^ 2) - 9 * 6 + 7 :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l1229_122965


namespace NUMINAMATH_GPT_evaluated_expression_l1229_122902

noncomputable def evaluation_problem (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem evaluated_expression :
  evaluation_problem 0.66 0.1 0.66 0.1 0.066 0.1 = 1.309091916 :=
by
  sorry

end NUMINAMATH_GPT_evaluated_expression_l1229_122902


namespace NUMINAMATH_GPT_two_a_plus_two_b_plus_two_c_l1229_122973

variable (a b c : ℝ)

-- Defining the conditions as the hypotheses
def condition1 : Prop := b + c = 15 - 4 * a
def condition2 : Prop := a + c = -18 - 4 * b
def condition3 : Prop := a + b = 10 - 4 * c

-- The theorem to prove
theorem two_a_plus_two_b_plus_two_c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  2 * a + 2 * b + 2 * c = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_two_a_plus_two_b_plus_two_c_l1229_122973


namespace NUMINAMATH_GPT_solution_in_quadrant_I_l1229_122964

theorem solution_in_quadrant_I (k x y : ℝ) (h1 : 2 * x - y = 5) (h2 : k * x^2 + y = 4) (h4 : x > 0) (h5 : y > 0) : k > 0 :=
sorry

end NUMINAMATH_GPT_solution_in_quadrant_I_l1229_122964


namespace NUMINAMATH_GPT_drop_perpendicular_l1229_122983

open Classical

-- Definitions for geometrical constructions on the plane
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Condition 1: Drawing a line through two points
def draw_line (A B : Point) : Line := {
  p1 := A,
  p2 := B
}

-- Condition 2: Drawing a perpendicular line through a given point on a line
def draw_perpendicular (l : Line) (P : Point) : Line :=
-- Details of construction skipped, this function should return the perpendicular line
sorry

-- The problem: Given a point A and a line l not passing through A, construct the perpendicular from A to l
theorem drop_perpendicular : 
  ∀ (A : Point) (l : Line), ¬ (A = l.p1 ∨ A = l.p2) → ∃ (P : Point), ∃ (m : Line), (m = draw_perpendicular l P) ∧ (m.p1 = A) :=
by
  intros A l h
  -- Details of theorem-proof skipped, assert the existence of P and m as required
  sorry

end NUMINAMATH_GPT_drop_perpendicular_l1229_122983


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1229_122927

-- Definitions of the side lengths
def side1 : ℝ := 8
def side2 : ℝ := 4

-- Theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter (side1 side2 : ℝ) (h1 : side1 = 8 ∨ side2 = 8) (h2 : side1 = 4 ∨ side2 = 4) : ∃ p : ℝ, p = 20 :=
by
  -- We omit the proof using sorry
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1229_122927


namespace NUMINAMATH_GPT_determine_values_l1229_122963

-- Define variables and conditions
variable {x v w y z : ℕ}

-- Define the conditions
def condition1 := v * x = 8 * 9
def condition2 := y^2 = x^2 + 81
def condition3 := z^2 = 20^2 - x^2
def condition4 := w^2 = 8^2 + v^2
def condition5 := v * 20 = y * 8

-- Theorem to prove
theorem determine_values : 
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by
  -- Insert necessary logic or 
  -- produce proof steps here
  sorry

end NUMINAMATH_GPT_determine_values_l1229_122963


namespace NUMINAMATH_GPT_same_roots_condition_l1229_122961

-- Definition of quadratic equations with coefficients a1, b1, c1 and a2, b2, c2
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- The condition we need to prove
theorem same_roots_condition :
  (a1 ≠ 0 ∧ a2 ≠ 0) → 
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) 
    ↔ 
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0 ↔ a2 * x^2 + b2 * x + c2 = 0) :=
sorry

end NUMINAMATH_GPT_same_roots_condition_l1229_122961


namespace NUMINAMATH_GPT_sum_of_abs_squared_series_correct_l1229_122978

noncomputable def sum_of_abs_squared_series (a r : ℝ) (h : |r| < 1) : ℝ :=
  a^2 / (1 - |r|^2)

theorem sum_of_abs_squared_series_correct (a r : ℝ) (h : |r| < 1) :
  sum_of_abs_squared_series a r h = a^2 / (1 - |r|^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_squared_series_correct_l1229_122978


namespace NUMINAMATH_GPT_problem_inequality_l1229_122989

theorem problem_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by 
  sorry

end NUMINAMATH_GPT_problem_inequality_l1229_122989


namespace NUMINAMATH_GPT_plane_point_to_center_ratio_l1229_122900

variable (a b c p q r : ℝ)

theorem plane_point_to_center_ratio :
  (a / p) + (b / q) + (c / r) = 2 ↔ 
  (∀ (α β γ : ℝ), α = 2 * p ∧ β = 2 * q ∧ γ = 2 * r ∧ (α, 0, 0) = (a, b, c) → 
  (a / (2 * p)) + (b / (2 * q)) + (c / (2 * r)) = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_plane_point_to_center_ratio_l1229_122900


namespace NUMINAMATH_GPT_student_19_in_sample_l1229_122920

-- Definitions based on conditions
def total_students := 52
def sample_size := 4
def sampling_interval := 13

def selected_students := [6, 32, 45]

-- The theorem to prove
theorem student_19_in_sample : 19 ∈ selected_students ∨ ∃ k : ℕ, 13 * k + 6 = 19 :=
by
  sorry

end NUMINAMATH_GPT_student_19_in_sample_l1229_122920


namespace NUMINAMATH_GPT_combined_mpg_correct_l1229_122918

def ray_mpg := 30
def tom_mpg := 15
def alice_mpg := 60
def distance_each := 120

-- Total gasoline consumption
def ray_gallons := distance_each / ray_mpg
def tom_gallons := distance_each / tom_mpg
def alice_gallons := distance_each / alice_mpg

def total_gallons := ray_gallons + tom_gallons + alice_gallons
def total_distance := 3 * distance_each

def combined_mpg := total_distance / total_gallons

theorem combined_mpg_correct :
  combined_mpg = 26 :=
by
  -- All the necessary calculations would go here.
  sorry

end NUMINAMATH_GPT_combined_mpg_correct_l1229_122918


namespace NUMINAMATH_GPT_sum_squares_mod_13_is_zero_l1229_122917

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_squares_mod_13_is_zero_l1229_122917


namespace NUMINAMATH_GPT_aaron_total_amount_owed_l1229_122914

def total_cost (monthly_payment : ℤ) (months : ℤ) : ℤ :=
  monthly_payment * months

def interest_fee (amount : ℤ) (rate : ℤ) : ℤ :=
  amount * rate / 100

def total_amount_owed (monthly_payment : ℤ) (months : ℤ) (rate : ℤ) : ℤ :=
  let amount := total_cost monthly_payment months
  let fee := interest_fee amount rate
  amount + fee

theorem aaron_total_amount_owed :
  total_amount_owed 100 12 10 = 1320 :=
by
  sorry

end NUMINAMATH_GPT_aaron_total_amount_owed_l1229_122914


namespace NUMINAMATH_GPT_line_through_intersection_points_l1229_122907

def first_circle (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def second_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem line_through_intersection_points (x y : ℝ) :
  (first_circle x y ∧ second_circle x y) → x - y - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_intersection_points_l1229_122907


namespace NUMINAMATH_GPT_sum_of_first_seven_primes_with_units_digit_3_lt_150_l1229_122975

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_less_than_150 (n : ℕ) : Prop :=
  n < 150

def first_seven_primes_with_units_digit_3 := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3_lt_150 :
  (has_units_digit_3 3) ∧ (is_less_than_150 3) ∧ (Prime 3) ∧
  (has_units_digit_3 13) ∧ (is_less_than_150 13) ∧ (Prime 13) ∧
  (has_units_digit_3 23) ∧ (is_less_than_150 23) ∧ (Prime 23) ∧
  (has_units_digit_3 43) ∧ (is_less_than_150 43) ∧ (Prime 43) ∧
  (has_units_digit_3 53) ∧ (is_less_than_150 53) ∧ (Prime 53) ∧
  (has_units_digit_3 73) ∧ (is_less_than_150 73) ∧ (Prime 73) ∧
  (has_units_digit_3 83) ∧ (is_less_than_150 83) ∧ (Prime 83) →
  (3 + 13 + 23 + 43 + 53 + 73 + 83 = 291) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_seven_primes_with_units_digit_3_lt_150_l1229_122975


namespace NUMINAMATH_GPT_fifth_friend_paid_40_l1229_122935

variable (x1 x2 x3 x4 x5 : ℝ)

def conditions : Prop :=
  (x1 = 1/3 * (x2 + x3 + x4 + x5)) ∧
  (x2 = 1/4 * (x1 + x3 + x4 + x5)) ∧
  (x3 = 1/5 * (x1 + x2 + x4 + x5)) ∧
  (x4 = 1/6 * (x1 + x2 + x3 + x5)) ∧
  (x1 + x2 + x3 + x4 + x5 = 120)

theorem fifth_friend_paid_40 (h : conditions x1 x2 x3 x4 x5) : x5 = 40 := by
  sorry

end NUMINAMATH_GPT_fifth_friend_paid_40_l1229_122935


namespace NUMINAMATH_GPT_max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l1229_122941

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a ≤ 2 :=
by
  -- Proof omitted
  sorry

theorem le_2_and_ge_neg_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : -2 ≤ a :=
by
  -- Proof omitted
  sorry

theorem max_a_is_2 (a : ℝ) (h3 : a ≤ 2) (h4 : -2 ≤ a) : a = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l1229_122941


namespace NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l1229_122991

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- proof to be provided here
  sorry

end NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l1229_122991


namespace NUMINAMATH_GPT_find_lower_percentage_l1229_122908

theorem find_lower_percentage (P : ℝ) : 
  (12000 * 0.15 * 2 - 720 = 12000 * (P / 100) * 2) → P = 12 := by
  sorry

end NUMINAMATH_GPT_find_lower_percentage_l1229_122908


namespace NUMINAMATH_GPT_smaller_number_l1229_122933

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_l1229_122933


namespace NUMINAMATH_GPT_total_machine_operation_time_l1229_122903

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end NUMINAMATH_GPT_total_machine_operation_time_l1229_122903


namespace NUMINAMATH_GPT_necessary_and_sufficient_for_Sn_lt_an_l1229_122956

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1)) / 2

theorem necessary_and_sufficient_for_Sn_lt_an
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_seq a d)
  (h_d_neg : d < 0)
  (m n : ℕ)
  (h_pos_m : m ≥ 3)
  (h_am_eq_Sm : a m = S m) :
  n > m ↔ S n < a n := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_for_Sn_lt_an_l1229_122956


namespace NUMINAMATH_GPT_max_value_of_expr_l1229_122905

open Classical
open Real

theorem max_value_of_expr 
  (x y : ℝ) 
  (h₁ : 0 < x) 
  (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  ∃ a b c d : ℝ, 
    (x^2 + 2 * x * y + 3 * y^2 = 20 + 10 * sqrt 3) ∧ 
    (a = 20) ∧ 
    (b = 10) ∧ 
    (c = 3) ∧ 
    (d = 2) := 
sorry

end NUMINAMATH_GPT_max_value_of_expr_l1229_122905


namespace NUMINAMATH_GPT_boys_collected_200_insects_l1229_122962

theorem boys_collected_200_insects
  (girls_insects : ℕ)
  (groups : ℕ)
  (insects_per_group : ℕ)
  (total_insects : ℕ)
  (boys_insects : ℕ)
  (H1 : girls_insects = 300)
  (H2 : groups = 4)
  (H3 : insects_per_group = 125)
  (H4 : total_insects = groups * insects_per_group)
  (H5 : boys_insects = total_insects - girls_insects) :
  boys_insects = 200 :=
  by sorry

end NUMINAMATH_GPT_boys_collected_200_insects_l1229_122962


namespace NUMINAMATH_GPT_M_Mobile_cheaper_than_T_Mobile_l1229_122922

def T_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 50
  else 50 + (lines - 2) * 16

def M_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 45
  else 45 + (lines - 2) * 14

theorem M_Mobile_cheaper_than_T_Mobile : 
  T_Mobile_total_cost 5 - M_Mobile_total_cost 5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_M_Mobile_cheaper_than_T_Mobile_l1229_122922


namespace NUMINAMATH_GPT_profit_is_1500_l1229_122913

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end NUMINAMATH_GPT_profit_is_1500_l1229_122913


namespace NUMINAMATH_GPT_voting_total_participation_l1229_122985

theorem voting_total_participation:
  ∀ (x : ℝ),
  0.35 * x + 0.65 * x = x ∧
  0.65 * x = 0.45 * (x + 80) →
  (x + 80 = 260) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_voting_total_participation_l1229_122985


namespace NUMINAMATH_GPT_sum_of_functions_positive_l1229_122923

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

theorem sum_of_functions_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0) : f x1 + f x2 + f x3 > 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_functions_positive_l1229_122923


namespace NUMINAMATH_GPT_largest_integer_same_cost_l1229_122901

def sum_decimal_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_ternary_digits (n : ℕ) : ℕ :=
  n.digits 3 |>.sum

theorem largest_integer_same_cost :
  ∃ n : ℕ, n < 1000 ∧ sum_decimal_digits n = sum_ternary_digits n ∧ ∀ m : ℕ, m < 1000 ∧ sum_decimal_digits m = sum_ternary_digits m → m ≤ n := 
  sorry

end NUMINAMATH_GPT_largest_integer_same_cost_l1229_122901


namespace NUMINAMATH_GPT_arnaldo_billion_difference_l1229_122909

theorem arnaldo_billion_difference :
  (10 ^ 12) - (10 ^ 9) = 999000000000 :=
by
  sorry

end NUMINAMATH_GPT_arnaldo_billion_difference_l1229_122909


namespace NUMINAMATH_GPT_next_two_equations_l1229_122937

-- Definitions based on the conditions in the problem
def pattern1 (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Statement to prove the continuation of the pattern
theorem next_two_equations 
: pattern1 9 40 41 ∧ pattern1 11 60 61 :=
by
  sorry

end NUMINAMATH_GPT_next_two_equations_l1229_122937


namespace NUMINAMATH_GPT_replaced_person_weight_l1229_122994

theorem replaced_person_weight :
  ∀ (avg_weight: ℝ), 
    10 * (avg_weight + 4) - 10 * avg_weight = 110 - 70 :=
by
  intros avg_weight
  sorry

end NUMINAMATH_GPT_replaced_person_weight_l1229_122994


namespace NUMINAMATH_GPT_infimum_of_function_l1229_122986

open Real

-- Definitions given in the conditions:
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)
def function_on_interval (f : ℝ → ℝ) := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = -3 * x ^ 2 + 2

-- Proof problem statement:
theorem infimum_of_function (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : periodic_function f) 
  (h_interval : function_on_interval f) : 
  ∃ M : ℝ, (∀ x : ℝ, f x ≥ M) ∧ M = -1 :=
by
  sorry

end NUMINAMATH_GPT_infimum_of_function_l1229_122986


namespace NUMINAMATH_GPT_consecutive_integer_sets_sum_100_l1229_122972

theorem consecutive_integer_sets_sum_100 :
  ∃ s : Finset (Finset ℕ), 
    (∀ seq ∈ s, (∀ x ∈ seq, x > 0) ∧ (seq.sum id = 100)) ∧
    (s.card = 2) :=
sorry

end NUMINAMATH_GPT_consecutive_integer_sets_sum_100_l1229_122972


namespace NUMINAMATH_GPT_boys_more_than_girls_l1229_122936

theorem boys_more_than_girls
  (x y a b : ℕ)
  (h1 : x > y)
  (h2 : x * a + y * b = x * b + y * a - 1) :
  x = y + 1 :=
sorry

end NUMINAMATH_GPT_boys_more_than_girls_l1229_122936


namespace NUMINAMATH_GPT_ratio_a_b_l1229_122946

theorem ratio_a_b (a b c d : ℝ) 
  (h1 : b / c = 7 / 9) 
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) : 
  a / b = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_ratio_a_b_l1229_122946


namespace NUMINAMATH_GPT_Vasya_can_win_l1229_122969

-- We need this library to avoid any import issues and provide necessary functionality for rational numbers

theorem Vasya_can_win :
  let a := (1 : ℚ) / 2009
  let b := (1 : ℚ) / 2008
  (∃ x : ℚ, a + x = 1) ∨ (∃ x : ℚ, b + x = 1) := sorry

end NUMINAMATH_GPT_Vasya_can_win_l1229_122969


namespace NUMINAMATH_GPT_Freddy_age_l1229_122924

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end NUMINAMATH_GPT_Freddy_age_l1229_122924


namespace NUMINAMATH_GPT_sum_of_squares_ways_l1229_122939

theorem sum_of_squares_ways : 
  ∃ ways : ℕ, ways = 2 ∧
    (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 100) ∧ 
    (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x^2 + y^2 + z^2 + w^2 = 100) := 
sorry

end NUMINAMATH_GPT_sum_of_squares_ways_l1229_122939


namespace NUMINAMATH_GPT_evaTotalMarksCorrect_l1229_122931

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end NUMINAMATH_GPT_evaTotalMarksCorrect_l1229_122931


namespace NUMINAMATH_GPT_measure_of_angle_Q_l1229_122910

-- Given conditions
variables (α β γ δ : ℝ)
axiom h1 : α = 130
axiom h2 : β = 95
axiom h3 : γ = 110
axiom h4 : δ = 104

-- Statement of the problem
theorem measure_of_angle_Q (Q : ℝ) (h5 : Q + α + β + γ + δ = 540) : Q = 101 := 
sorry

end NUMINAMATH_GPT_measure_of_angle_Q_l1229_122910


namespace NUMINAMATH_GPT_sqrt_neg_square_real_l1229_122940

theorem sqrt_neg_square_real : ∃! (x : ℝ), -(x + 2) ^ 2 = 0 := by
  sorry

end NUMINAMATH_GPT_sqrt_neg_square_real_l1229_122940


namespace NUMINAMATH_GPT_cristina_pace_is_4_l1229_122912

-- Definitions of the conditions
def head_start : ℝ := 36
def nicky_pace : ℝ := 3
def time : ℝ := 36

-- Definition of the distance Nicky runs
def distance_nicky_runs : ℝ := nicky_pace * time

-- Definition of the total distance Cristina ran to catch up
def distance_cristina_runs : ℝ := distance_nicky_runs + head_start

-- Lean 4 theorem statement to prove Cristina's pace
theorem cristina_pace_is_4 :
  (distance_cristina_runs / time) = 4 := 
by sorry

end NUMINAMATH_GPT_cristina_pace_is_4_l1229_122912


namespace NUMINAMATH_GPT_valid_n_value_l1229_122944

theorem valid_n_value (n : ℕ) (a : ℕ → ℕ)
    (h1 : ∀ k : ℕ, 1 ≤ k ∧ k < n → k ∣ a k)
    (h2 : ¬ n ∣ a n)
    (h3 : 2 ≤ n) :
    ∃ (p : ℕ) (α : ℕ), (Nat.Prime p) ∧ (n = p ^ α) ∧ (α ≥ 1) :=
by sorry

end NUMINAMATH_GPT_valid_n_value_l1229_122944


namespace NUMINAMATH_GPT_least_positive_int_to_multiple_of_3_l1229_122958

theorem least_positive_int_to_multiple_of_3 (x : ℕ) (h : 575 + x ≡ 0 [MOD 3]) : x = 1 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_int_to_multiple_of_3_l1229_122958


namespace NUMINAMATH_GPT_rational_combination_zero_eqn_l1229_122954

theorem rational_combination_zero_eqn (a b c : ℚ) (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_combination_zero_eqn_l1229_122954


namespace NUMINAMATH_GPT_measure_of_angle_x_l1229_122919

theorem measure_of_angle_x :
  ∀ (angle_ABC angle_BDE angle_DBE angle_ABD x : ℝ),
    angle_ABC = 132 ∧
    angle_BDE = 31 ∧
    angle_DBE = 30 ∧
    angle_ABD = 180 - 132 →
    x = 180 - (angle_BDE + angle_DBE) →
    x = 119 :=
by
  intros angle_ABC angle_BDE angle_DBE angle_ABD x h h2
  sorry

end NUMINAMATH_GPT_measure_of_angle_x_l1229_122919


namespace NUMINAMATH_GPT_find_d_l1229_122993

def point_in_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3030 ∧ 0 ≤ y ∧ y ≤ 3030

def point_in_ellipse (x y : ℝ) : Prop :=
  (x^2 / 2020^2) + (y^2 / 4040^2) ≤ 1

def point_within_distance (d : ℝ) (x y : ℝ) : Prop :=
  (∃ (a b : ℤ), (x - a) ^ 2 + (y - b) ^ 2 ≤ d ^ 2)

theorem find_d :
  (∃ d : ℝ, (∀ x y : ℝ, point_in_square x y → point_in_ellipse x y → point_within_distance d x y) ∧ (d = 0.5)) :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1229_122993


namespace NUMINAMATH_GPT_additional_people_to_halve_speed_l1229_122988

variables (s : ℕ → ℝ)
variables (x : ℕ)

-- Given conditions
axiom speed_with_200_people : s 200 = 500
axiom speed_with_400_people : s 400 = 125
axiom speed_halved : ∀ n, s (n + x) = s n / 2

theorem additional_people_to_halve_speed : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_to_halve_speed_l1229_122988


namespace NUMINAMATH_GPT_range_of_a_l1229_122950

theorem range_of_a(p q: Prop)
  (hp: p ↔ (a = 0 ∨ (0 < a ∧ a < 4)))
  (hq: q ↔ (-1 < a ∧ a < 3))
  (hpor: p ∨ q)
  (hpand: ¬(p ∧ q)):
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := by sorry

end NUMINAMATH_GPT_range_of_a_l1229_122950


namespace NUMINAMATH_GPT_find_pairs_l1229_122916

theorem find_pairs (x y : ℕ) (h : x > 0 ∧ y > 0) (d : ℕ) (gcd_cond : d = Nat.gcd x y)
  (eqn_cond : x * y * d = x + y + d ^ 2) : (x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pairs_l1229_122916


namespace NUMINAMATH_GPT_regina_earnings_l1229_122911

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end NUMINAMATH_GPT_regina_earnings_l1229_122911


namespace NUMINAMATH_GPT_binary_sum_l1229_122915

-- Define the binary representations in terms of their base 10 equivalent.
def binary_111111111 := 511
def binary_1111111 := 127

-- State the proof problem.
theorem binary_sum : binary_111111111 + binary_1111111 = 638 :=
by {
  -- placeholder for proof
  sorry
}

end NUMINAMATH_GPT_binary_sum_l1229_122915


namespace NUMINAMATH_GPT_box_filled_with_cubes_no_leftover_l1229_122976

-- Define dimensions of the box
def box_length : ℝ := 50
def box_width : ℝ := 60
def box_depth : ℝ := 43

-- Define volumes of different types of cubes
def volume_box : ℝ := box_length * box_width * box_depth
def volume_small_cube : ℝ := 2^3
def volume_medium_cube : ℝ := 3^3
def volume_large_cube : ℝ := 5^3

-- Define the smallest number of each type of cube
def num_large_cubes : ℕ := 1032
def num_medium_cubes : ℕ := 0
def num_small_cubes : ℕ := 0

-- Theorem statement ensuring the number of cubes completely fills the box
theorem box_filled_with_cubes_no_leftover :
  num_large_cubes * volume_large_cube + num_medium_cubes * volume_medium_cube + num_small_cubes * volume_small_cube = volume_box :=
by
  sorry

end NUMINAMATH_GPT_box_filled_with_cubes_no_leftover_l1229_122976


namespace NUMINAMATH_GPT_min_denominator_of_sum_600_700_l1229_122934

def is_irreducible_fraction (a : ℕ) (b : ℕ) : Prop := 
  Nat.gcd a b = 1

def min_denominator_of_sum (d1 d2 : ℕ) (a b : ℕ) : ℕ :=
  let lcm := Nat.lcm d1 d2
  let sum_numerator := a * (lcm / d1) + b * (lcm / d2)
  Nat.gcd sum_numerator lcm

theorem min_denominator_of_sum_600_700 (a b : ℕ) (h1 : is_irreducible_fraction a 600) (h2 : is_irreducible_fraction b 700) :
  min_denominator_of_sum 600 700 a b = 168 := sorry

end NUMINAMATH_GPT_min_denominator_of_sum_600_700_l1229_122934


namespace NUMINAMATH_GPT_trapezoid_reassembly_area_conservation_l1229_122996

theorem trapezoid_reassembly_area_conservation
  {height length new_width : ℝ}
  (h1 : height = 9)
  (h2 : length = 16)
  (h3 : new_width = y)  -- each base of the trapezoid measures y.
  (div_trapezoids : ∀ (a b c : ℝ), 3 * a = height → a = 9 / 3)
  (area_conserved : length * height = (3 / 2) * (3 * (length + new_width)))
  : new_width = 16 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_trapezoid_reassembly_area_conservation_l1229_122996


namespace NUMINAMATH_GPT_average_of_numbers_between_6_and_36_divisible_by_7_l1229_122904

noncomputable def average_of_divisibles_by_seven : ℕ :=
  let numbers := [7, 14, 21, 28, 35]
  let sum := numbers.sum
  let count := numbers.length
  sum / count

theorem average_of_numbers_between_6_and_36_divisible_by_7 : average_of_divisibles_by_seven = 21 :=
by
  sorry

end NUMINAMATH_GPT_average_of_numbers_between_6_and_36_divisible_by_7_l1229_122904


namespace NUMINAMATH_GPT_cube_volumes_total_l1229_122925

theorem cube_volumes_total :
  let v1 := 5^3
  let v2 := 6^3
  let v3 := 7^3
  v1 + v2 + v3 = 684 := by
  -- Here will be the proof using Lean's tactics
  sorry

end NUMINAMATH_GPT_cube_volumes_total_l1229_122925


namespace NUMINAMATH_GPT_joan_gave_sam_seashells_l1229_122966

-- Definitions of initial conditions
def initial_seashells : ℕ := 70
def remaining_seashells : ℕ := 27

-- Theorem statement
theorem joan_gave_sam_seashells : initial_seashells - remaining_seashells = 43 :=
by
  sorry

end NUMINAMATH_GPT_joan_gave_sam_seashells_l1229_122966


namespace NUMINAMATH_GPT_route_y_slower_by_2_4_minutes_l1229_122997
noncomputable def time_route_x : ℝ := (7 : ℝ) / (35 : ℝ)
noncomputable def time_downtown_y : ℝ := (1 : ℝ) / (10 : ℝ)
noncomputable def time_other_y : ℝ := (7 : ℝ) / (50 : ℝ)
noncomputable def time_route_y : ℝ := time_downtown_y + time_other_y

theorem route_y_slower_by_2_4_minutes :
  ((time_route_y - time_route_x) * 60) = 2.4 :=
by
  -- Provide the required proof here
  sorry

end NUMINAMATH_GPT_route_y_slower_by_2_4_minutes_l1229_122997


namespace NUMINAMATH_GPT_coordinates_of_C_l1229_122943

structure Point :=
  (x : Int)
  (y : Int)

def reflect_over_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def reflect_over_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def C : Point := {x := 2, y := 2}

noncomputable def C'_reflected_x := reflect_over_x_axis C
noncomputable def C''_reflected_y := reflect_over_y_axis C'_reflected_x

theorem coordinates_of_C'' : C''_reflected_y = {x := -2, y := -2} :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_C_l1229_122943


namespace NUMINAMATH_GPT_range_of_k_roots_for_neg_k_l1229_122960

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x ≠ y ∧ (x^2 + (2*k + 1)*x + (k^2 - 1) = 0 ∧ y^2 + (2*k + 1)*y + (k^2 - 1) = 0)) ↔ k > -5 / 4 :=
by sorry

theorem roots_for_neg_k (k : ℤ) (h1 : k < 0) (h2 : k > -5 / 4) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + (2*k + 1)*x1 + (k^2 - 1) = 0 ∧ x2^2 + (2*k + 1)*x2 + (k^2 - 1) = 0 ∧ x1 = 0 ∧ x2 = 1)) :=
by sorry

end NUMINAMATH_GPT_range_of_k_roots_for_neg_k_l1229_122960


namespace NUMINAMATH_GPT_smallest_square_side_length_l1229_122990

theorem smallest_square_side_length :
  ∃ (n s : ℕ),  14 * n = s^2 ∧ s = 14 := 
by
  existsi 14, 14
  sorry

end NUMINAMATH_GPT_smallest_square_side_length_l1229_122990


namespace NUMINAMATH_GPT_pyramid_volume_is_sqrt3_l1229_122982

noncomputable def volume_of_pyramid := 
  let base_area : ℝ := 2 * Real.sqrt 3
  let angle_ABC : ℝ := 60
  let BC := 2
  let EC := BC
  let FB := BC / 2
  let height : ℝ := Real.sqrt 3
  let pyramid_volume := 1/3 * EC * FB * height
  pyramid_volume

theorem pyramid_volume_is_sqrt3 : volume_of_pyramid = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_pyramid_volume_is_sqrt3_l1229_122982


namespace NUMINAMATH_GPT_original_quantity_ghee_mixture_is_correct_l1229_122948

-- Define the variables
def percentage_ghee (x : ℝ) := 0.55 * x
def percentage_vanasapati (x : ℝ) := 0.35 * x
def percentage_palm_oil (x : ℝ) := 0.10 * x
def new_mixture_weight (x : ℝ) := x + 20
def final_vanasapati_percentage (x : ℝ) := 0.30 * (new_mixture_weight x)

-- State the theorem
theorem original_quantity_ghee_mixture_is_correct (x : ℝ) 
  (h1 : percentage_ghee x = 0.55 * x)
  (h2 : percentage_vanasapati x = 0.35 * x)
  (h3 : percentage_palm_oil x = 0.10 * x)
  (h4 : percentage_vanasapati x = final_vanasapati_percentage x) :
  x = 120 := 
sorry

end NUMINAMATH_GPT_original_quantity_ghee_mixture_is_correct_l1229_122948


namespace NUMINAMATH_GPT_stock_return_to_original_l1229_122992

theorem stock_return_to_original (x : ℝ) (h : x > 0) :
  ∃ d : ℝ, d = 3 / 13 ∧ (x * 1.30 * (1 - d)) = x :=
by sorry

end NUMINAMATH_GPT_stock_return_to_original_l1229_122992


namespace NUMINAMATH_GPT_intersection_sum_l1229_122968

-- Define the conditions
def condition_1 (k : ℝ) := k > 0
def line1 (x y k : ℝ) := 50 * x + k * y = 1240
def line2 (x y k : ℝ) := k * y = 8 * x + 544
def right_angles (k : ℝ) := (-50 / k) * (8 / k) = -1

-- Define the point of intersection
def point_of_intersection (m n : ℝ) (k : ℝ) := line1 m n k ∧ line2 m n k

-- Prove that m + n = 44 under the given conditions
theorem intersection_sum (m n k : ℝ) :
  condition_1 k →
  right_angles k →
  point_of_intersection m n k →
  m + n = 44 :=
by
  sorry

end NUMINAMATH_GPT_intersection_sum_l1229_122968


namespace NUMINAMATH_GPT_terminal_side_quadrant_l1229_122949

-- Given conditions
variables {α : ℝ}
variable (h1 : Real.sin α > 0)
variable (h2 : Real.tan α < 0)

-- Conclusion to be proved
theorem terminal_side_quadrant (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (∃ k : ℤ, (k % 2 = 0 ∧ Real.pi * k / 2 < α / 2 ∧ α / 2 < Real.pi / 2 + Real.pi * k) ∨ 
            (k % 2 = 1 ∧ Real.pi * (k - 1) < α / 2 ∧ α / 2 < Real.pi / 4 + Real.pi * (k - 0.5))) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_quadrant_l1229_122949


namespace NUMINAMATH_GPT_race_result_l1229_122952

-- Define the contestants
inductive Contestants
| Alyosha
| Borya
| Vanya
| Grisha

open Contestants

-- Define their statements
def Alyosha_statement (place : Contestants → ℕ) : Prop :=
  place Alyosha ≠ 1 ∧ place Alyosha ≠ 4

def Borya_statement (place : Contestants → ℕ) : Prop :=
  place Borya ≠ 4

def Vanya_statement (place : Contestants → ℕ) : Prop :=
  place Vanya = 1

def Grisha_statement (place : Contestants → ℕ) : Prop :=
  place Grisha = 4

-- Define that exactly one statement is false and the rest are true
def three_true_one_false (place : Contestants → ℕ) : Prop :=
  (Alyosha_statement place ∧ ¬ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (¬ Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ ¬ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ ¬ Grisha_statement place)

-- Define the conclusion: Vanya lied and Borya was first
theorem race_result (place : Contestants → ℕ) : 
  three_true_one_false place → 
  (¬ Vanya_statement place ∧ place Borya = 1) :=
sorry

end NUMINAMATH_GPT_race_result_l1229_122952


namespace NUMINAMATH_GPT_binom_1293_1_eq_1293_l1229_122998

theorem binom_1293_1_eq_1293 : (Nat.choose 1293 1) = 1293 := 
  sorry

end NUMINAMATH_GPT_binom_1293_1_eq_1293_l1229_122998


namespace NUMINAMATH_GPT_total_carriages_l1229_122970

-- Definitions based on given conditions
def Euston_carriages := 130
def Norfolk_carriages := Euston_carriages - 20
def Norwich_carriages := 100
def Flying_Scotsman_carriages := Norwich_carriages + 20
def Victoria_carriages := Euston_carriages - 15
def Waterloo_carriages := Norwich_carriages * 2

-- Theorem to prove the total number of carriages is 775
theorem total_carriages : 
  Euston_carriages + Norfolk_carriages + Norwich_carriages + Flying_Scotsman_carriages + Victoria_carriages + Waterloo_carriages = 775 :=
by sorry

end NUMINAMATH_GPT_total_carriages_l1229_122970


namespace NUMINAMATH_GPT_demand_decrease_annual_l1229_122967

noncomputable def price_increase (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r / 100) ^ t

noncomputable def demand_maintenance (P : ℝ) (r : ℝ) (t : ℕ) (d : ℝ) : Prop :=
  let new_price := price_increase P r t
  (P * (1 + r / 100)) * (1 - d / 100) ≥ price_increase P 10 1

theorem demand_decrease_annual (P : ℝ) (r : ℝ) (t : ℕ) :
  price_increase P r t ≥ price_increase P 10 1 → ∃ d : ℝ, d = 1.66156 :=
by
  sorry

end NUMINAMATH_GPT_demand_decrease_annual_l1229_122967


namespace NUMINAMATH_GPT_Rahul_savings_l1229_122938

variable (total_savings ppf_savings nsc_savings x : ℝ)

theorem Rahul_savings
  (h1 : total_savings = 180000)
  (h2 : ppf_savings = 72000)
  (h3 : nsc_savings = total_savings - ppf_savings)
  (h4 : x * nsc_savings = 0.5 * ppf_savings) :
  x = 1 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Rahul_savings_l1229_122938


namespace NUMINAMATH_GPT_side_length_square_l1229_122957

theorem side_length_square (x : ℝ) (h1 : x^2 = 2 * (4 * x)) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_side_length_square_l1229_122957


namespace NUMINAMATH_GPT_volume_in_30_minutes_l1229_122999

-- Define the conditions
def rate_of_pumping := 540 -- gallons per hour
def time_in_hours := 30 / 60 -- 30 minutes as a fraction of an hour

-- Define the volume pumped in 30 minutes
def volume_pumped := rate_of_pumping * time_in_hours

-- State the theorem
theorem volume_in_30_minutes : volume_pumped = 270 := by
  sorry

end NUMINAMATH_GPT_volume_in_30_minutes_l1229_122999


namespace NUMINAMATH_GPT_find_A_plus_C_l1229_122955

-- This will bring in the entirety of the necessary library and supports the digit verification and operations.

-- Definitions of digits and constraints
variables {A B C D : ℕ}

-- Given conditions in the problem
def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

def multiplication_condition_1 (A B C D : ℕ) : Prop :=
  C * D = A

def multiplication_condition_2 (A B C D : ℕ) : Prop :=
  10 * B * D + C * D = 11 * C

-- The final problem statement
theorem find_A_plus_C (A B C D : ℕ) (h1 : distinct_digits A B C D) 
  (h2 : multiplication_condition_1 A B C D) 
  (h3 : multiplication_condition_2 A B C D) : 
  A + C = 10 :=
sorry

end NUMINAMATH_GPT_find_A_plus_C_l1229_122955


namespace NUMINAMATH_GPT_comprehensive_survey_option_l1229_122959

def suitable_for_comprehensive_survey (survey : String) : Prop :=
  survey = "Survey on the components of the first large civil helicopter in China"

theorem comprehensive_survey_option (A B C D : String)
  (hA : A = "Survey on the number of waste batteries discarded in the city every day")
  (hB : B = "Survey on the quality of ice cream in the cold drink market")
  (hC : C = "Survey on the current mental health status of middle school students nationwide")
  (hD : D = "Survey on the components of the first large civil helicopter in China") :
  suitable_for_comprehensive_survey D :=
by
  sorry

end NUMINAMATH_GPT_comprehensive_survey_option_l1229_122959


namespace NUMINAMATH_GPT_magnified_diameter_l1229_122987

theorem magnified_diameter (diameter_actual : ℝ) (magnification_factor : ℕ) 
  (h_actual : diameter_actual = 0.005) (h_magnification : magnification_factor = 1000) :
  diameter_actual * magnification_factor = 5 :=
by 
  sorry

end NUMINAMATH_GPT_magnified_diameter_l1229_122987


namespace NUMINAMATH_GPT_prob_event_A_given_B_l1229_122971

def EventA (visits : Fin 4 → Fin 4) : Prop :=
  Function.Injective visits

def EventB (visits : Fin 4 → Fin 4) : Prop :=
  visits 0 = 0

theorem prob_event_A_given_B :
  ∀ (visits : Fin 4 → Fin 4),
  (∃ f : (Fin 4 → Fin 4) → Prop, f visits → (EventA visits ∧ EventB visits)) →
  (∃ P : ℚ, P = 2 / 9) :=
by
  intros visits h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_prob_event_A_given_B_l1229_122971


namespace NUMINAMATH_GPT_equivalent_statements_l1229_122953

variables (P Q R : Prop)

theorem equivalent_statements :
  (P → (Q ∧ ¬R)) ↔ ((¬ Q ∨ R) → ¬ P) :=
sorry

end NUMINAMATH_GPT_equivalent_statements_l1229_122953


namespace NUMINAMATH_GPT_example_3_is_analogical_reasoning_l1229_122979

-- Definitions based on the conditions of the problem:
def is_analogical_reasoning (reasoning: String): Prop :=
  reasoning = "from one specific case to another similar specific case"

-- Example of reasoning given in the problem.
def example_3 := "From the fact that the sum of the distances from a point inside an equilateral triangle to its three sides is a constant, it is concluded that the sum of the distances from a point inside a regular tetrahedron to its four faces is a constant."

-- Proof statement based on the conditions and correct answer.
theorem example_3_is_analogical_reasoning: is_analogical_reasoning example_3 :=
by 
  sorry

end NUMINAMATH_GPT_example_3_is_analogical_reasoning_l1229_122979

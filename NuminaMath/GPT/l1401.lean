import Mathlib

namespace NUMINAMATH_GPT_room_dimension_l1401_140171

theorem room_dimension
  (x : ℕ)
  (cost_per_sqft : ℕ := 4)
  (dimension_1 : ℕ := 15)
  (dimension_2 : ℕ := 12)
  (door_width : ℕ := 6)
  (door_height : ℕ := 3)
  (num_windows : ℕ := 3)
  (window_width : ℕ := 4)
  (window_height : ℕ := 3)
  (total_cost : ℕ := 3624) :
  (2 * (x * dimension_1) + 2 * (x * dimension_2) - (door_width * door_height + num_windows * (window_width * window_height))) * cost_per_sqft = total_cost →
  x = 18 :=
by
  sorry

end NUMINAMATH_GPT_room_dimension_l1401_140171


namespace NUMINAMATH_GPT_solve_for_A_l1401_140149

theorem solve_for_A (A : ℕ) (h1 : 3 + 68 * A = 691) (h2 : 68 * A < 1000) (h3 : 68 * A ≥ 100) : A = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l1401_140149


namespace NUMINAMATH_GPT_range_of_m_l1401_140125

open Set

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
def B (m : ℝ) : Set ℝ := { x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1) }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1401_140125


namespace NUMINAMATH_GPT_player_c_wins_l1401_140191

theorem player_c_wins :
  ∀ (A_wins A_losses B_wins B_losses C_losses C_wins : ℕ),
  A_wins = 4 →
  A_losses = 2 →
  B_wins = 3 →
  B_losses = 3 →
  C_losses = 3 →
  A_wins + B_wins + C_wins = A_losses + B_losses + C_losses →
  C_wins = 2 :=
by
  intros A_wins A_losses B_wins B_losses C_losses C_wins
  sorry

end NUMINAMATH_GPT_player_c_wins_l1401_140191


namespace NUMINAMATH_GPT_completing_the_square_l1401_140140

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l1401_140140


namespace NUMINAMATH_GPT_books_left_after_giveaways_l1401_140182

def initial_books : ℝ := 48.0
def first_giveaway : ℝ := 34.0
def second_giveaway : ℝ := 3.0

theorem books_left_after_giveaways : 
  initial_books - first_giveaway - second_giveaway = 11.0 :=
by
  sorry

end NUMINAMATH_GPT_books_left_after_giveaways_l1401_140182


namespace NUMINAMATH_GPT_prime_number_identity_l1401_140150

theorem prime_number_identity (p m : ℕ) (h1 : Nat.Prime p) (h2 : m > 0) (h3 : 2 * p^2 + p + 9 = m^2) :
  p = 5 ∧ m = 8 :=
sorry

end NUMINAMATH_GPT_prime_number_identity_l1401_140150


namespace NUMINAMATH_GPT_find_a_l1401_140108

noncomputable def tangent_line (a : ℝ) (x : ℝ) := (3 * a * (1:ℝ)^2 + 1) * (x - 1) + (a * (1:ℝ)^3 + (1:ℝ) + 1)

theorem find_a : ∃ a : ℝ, tangent_line a 2 = 7 := 
sorry

end NUMINAMATH_GPT_find_a_l1401_140108


namespace NUMINAMATH_GPT_max_happy_times_l1401_140198

theorem max_happy_times (weights : Fin 2021 → ℝ) (unique_mass : Function.Injective weights) : 
  ∃ max_happy : Nat, max_happy = 673 :=
by
  sorry

end NUMINAMATH_GPT_max_happy_times_l1401_140198


namespace NUMINAMATH_GPT_sum_of_three_terms_divisible_by_3_l1401_140190

theorem sum_of_three_terms_divisible_by_3 (a : Fin 5 → ℤ) :
  ∃ (i j k : Fin 5), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (a i + a j + a k) % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_terms_divisible_by_3_l1401_140190


namespace NUMINAMATH_GPT_larger_integer_l1401_140122

theorem larger_integer (a b : ℕ) (h_diff : a - b = 8) (h_prod : a * b = 224) : a = 16 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_l1401_140122


namespace NUMINAMATH_GPT_determinant_of_tan_matrix_l1401_140134

theorem determinant_of_tan_matrix
  (A B C : ℝ)
  (h₁ : A = π / 4)
  (h₂ : A + B + C = π)
  : (Matrix.det ![
      ![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]
    ]) = 2 :=
  sorry

end NUMINAMATH_GPT_determinant_of_tan_matrix_l1401_140134


namespace NUMINAMATH_GPT_profit_in_2004_correct_l1401_140127

-- We define the conditions as given in the problem
def annual_profit_2002 : ℝ := 10
def annual_growth_rate (p : ℝ) : ℝ := p

-- The expression for the annual profit in 2004 given the above conditions
def annual_profit_2004 (p : ℝ) : ℝ := annual_profit_2002 * (1 + p) * (1 + p)

-- The theorem to prove that the computed annual profit in 2004 matches the expected answer
theorem profit_in_2004_correct (p : ℝ) :
  annual_profit_2004 p = 10 * (1 + p)^2 := 
by 
  sorry

end NUMINAMATH_GPT_profit_in_2004_correct_l1401_140127


namespace NUMINAMATH_GPT_calc_7_op_4_minus_4_op_7_l1401_140117

def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

theorem calc_7_op_4_minus_4_op_7 : (op 7 4) - (op 4 7) = -12 := by
  sorry

end NUMINAMATH_GPT_calc_7_op_4_minus_4_op_7_l1401_140117


namespace NUMINAMATH_GPT_cookies_left_after_ted_leaves_l1401_140107

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end NUMINAMATH_GPT_cookies_left_after_ted_leaves_l1401_140107


namespace NUMINAMATH_GPT_min_value_ineq_l1401_140104

theorem min_value_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) : 
  (1 / a) + (4 / b) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_l1401_140104


namespace NUMINAMATH_GPT_percent_increase_equilateral_triangles_l1401_140111

noncomputable def side_length (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * 3

noncomputable def perimeter (n : ℕ) : ℕ :=
  3 * side_length n

noncomputable def percent_increase (initial : ℕ) (final : ℕ) : ℚ := 
  ((final - initial) / initial) * 100

theorem percent_increase_equilateral_triangles :
  percent_increase (perimeter 0) (perimeter 4) = 1500 := by
  sorry

end NUMINAMATH_GPT_percent_increase_equilateral_triangles_l1401_140111


namespace NUMINAMATH_GPT_central_angle_relation_l1401_140179

theorem central_angle_relation
  (R L : ℝ)
  (α : ℝ)
  (r l β : ℝ)
  (h1 : r = 0.5 * R)
  (h2 : l = 1.5 * L)
  (h3 : L = R * α)
  (h4 : l = r * β) : 
  β = 3 * α :=
by
  sorry

end NUMINAMATH_GPT_central_angle_relation_l1401_140179


namespace NUMINAMATH_GPT_equal_semi_circles_radius_l1401_140143

-- Define the segments and semicircles given in the problem as conditions.
def segment1 : ℝ := 12
def segment2 : ℝ := 22
def segment3 : ℝ := 22
def segment4 : ℝ := 16
def segment5 : ℝ := 22

def total_horizontal_path1 (r : ℝ) : ℝ := 2*r + segment1 + 2*r + segment1 + 2*r
def total_horizontal_path2 (r : ℝ) : ℝ := segment2 + 2*r + segment4 + 2*r + segment5

-- The theorem that proves the radius is 18.
theorem equal_semi_circles_radius : ∃ r : ℝ, total_horizontal_path1 r = total_horizontal_path2 r ∧ r = 18 := by
  use 18
  simp [total_horizontal_path1, total_horizontal_path2, segment1, segment2, segment3, segment4, segment5]
  sorry

end NUMINAMATH_GPT_equal_semi_circles_radius_l1401_140143


namespace NUMINAMATH_GPT_arithmetic_mean_difference_l1401_140197

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_difference_l1401_140197


namespace NUMINAMATH_GPT_digits_difference_l1401_140112

theorem digits_difference (d A B : ℕ) (h1 : d > 6) (h2 : (B + A) * d + 2 * A = d^2 + 7 * d + 2)
  (h3 : B + A = 10) (h4 : 2 * A = 8) : A - B = 3 :=
by 
  sorry

end NUMINAMATH_GPT_digits_difference_l1401_140112


namespace NUMINAMATH_GPT_quad_eq_sum_ab_l1401_140173

theorem quad_eq_sum_ab {a b : ℝ} (h1 : a < 0)
  (h2 : ∀ x : ℝ, (x = -1 / 2 ∨ x = 1 / 3) ↔ ax^2 + bx + 2 = 0) :
  a + b = -14 :=
by
  sorry

end NUMINAMATH_GPT_quad_eq_sum_ab_l1401_140173


namespace NUMINAMATH_GPT_find_a0_find_a2_find_sum_a1_a2_a3_a4_l1401_140160

lemma problem_conditions (x : ℝ) : 
  (x - 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 :=
sorry

theorem find_a0 :
  a_0 = 16 :=
sorry

theorem find_a2 :
  a_2 = 24 :=
sorry

theorem find_sum_a1_a2_a3_a4 :
  a_1 + a_2 + a_3 + a_4 = -15 :=
sorry

end NUMINAMATH_GPT_find_a0_find_a2_find_sum_a1_a2_a3_a4_l1401_140160


namespace NUMINAMATH_GPT_ratio_of_unit_prices_l1401_140129

def volume_y (v : ℝ) : ℝ := v
def price_y (p : ℝ) : ℝ := p
def volume_x (v : ℝ) : ℝ := 1.3 * v
def price_x (p : ℝ) : ℝ := 0.8 * p

theorem ratio_of_unit_prices (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (0.8 * p / (1.3 * v)) / (p / v) = 8 / 13 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_unit_prices_l1401_140129


namespace NUMINAMATH_GPT_does_not_balance_l1401_140128

variables (square odot circ triangle O : ℝ)

-- Conditions represented as hypothesis
def condition1 : Prop := 4 * square = odot + circ
def condition2 : Prop := 2 * circ + odot = 2 * triangle

-- Statement to be proved
theorem does_not_balance (h1 : condition1 square odot circ) (h2 : condition2 circ odot triangle)
 : ¬(2 * triangle + square = triangle + odot + square) := 
sorry

end NUMINAMATH_GPT_does_not_balance_l1401_140128


namespace NUMINAMATH_GPT_fraction_min_sum_l1401_140175

theorem fraction_min_sum (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 45 * b < 110 * a ∧ 110 * a < 50 * b) :
  a = 3 ∧ b = 7 :=
sorry

end NUMINAMATH_GPT_fraction_min_sum_l1401_140175


namespace NUMINAMATH_GPT_max_a_such_that_f_geq_a_min_value_under_constraint_l1401_140100

-- Problem (1)
theorem max_a_such_that_f_geq_a :
  ∃ (a : ℝ), (∀ (x : ℝ), |x - (5/2)| + |x - a| ≥ a) ∧ a = 5 / 4 := sorry

-- Problem (2)
theorem min_value_under_constraint :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 1 ∧
  (3 / x + 2 / y + 1 / z) = 16 + 8 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_max_a_such_that_f_geq_a_min_value_under_constraint_l1401_140100


namespace NUMINAMATH_GPT_sequence_a_correct_l1401_140172

open Nat -- Opening the natural numbers namespace

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => (1 / 2 : ℝ) * a n

theorem sequence_a_correct : 
  (∀ n, 0 < a n) ∧ 
  a 1 = 1 ∧ 
  (∀ n, a (n + 1) = a n / 2) ∧
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧
  ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_correct_l1401_140172


namespace NUMINAMATH_GPT_negate_exists_l1401_140132

theorem negate_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔ (∀ x : ℝ, x ≥ Real.sin x ∨ x ≤ Real.tan x) :=
by
  sorry

end NUMINAMATH_GPT_negate_exists_l1401_140132


namespace NUMINAMATH_GPT_totalCostOfFencing_l1401_140138

def numberOfSides : ℕ := 4
def costPerSide : ℕ := 79

theorem totalCostOfFencing (n : ℕ) (c : ℕ) (hn : n = numberOfSides) (hc : c = costPerSide) : n * c = 316 :=
by 
  rw [hn, hc]
  exact rfl

end NUMINAMATH_GPT_totalCostOfFencing_l1401_140138


namespace NUMINAMATH_GPT_total_cupcakes_l1401_140155

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (total_cupcakes : ℕ) 
  (h1 : children = 8) (h2 : cupcakes_per_child = 12) : total_cupcakes = 96 := 
by
  sorry

end NUMINAMATH_GPT_total_cupcakes_l1401_140155


namespace NUMINAMATH_GPT_length_of_bridge_is_l1401_140124

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 21.998240140788738
noncomputable def speed_kmph : ℝ := 36
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is : bridge_length = 119.98240140788738 :=
by
  have speed_mps_val : speed_mps = 10 := by
    norm_num [speed_kmph, speed_mps]
  have total_distance_val : total_distance = 219.98240140788738 := by
    norm_num [total_distance, speed_mps_val, time_to_cross_bridge]
  have bridge_length_val : bridge_length = 119.98240140788738 := by
    norm_num [bridge_length, total_distance_val, train_length]
  exact bridge_length_val

end NUMINAMATH_GPT_length_of_bridge_is_l1401_140124


namespace NUMINAMATH_GPT_select_defective_products_l1401_140162

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_defective_products :
  let total_products := 200
  let defective_products := 3
  let selected_products := 5
  let ways_2_defective := choose defective_products 2 * choose (total_products - defective_products) 3
  let ways_3_defective := choose defective_products 3 * choose (total_products - defective_products) 2
  ways_2_defective + ways_3_defective = choose defective_products 2 * choose (total_products - defective_products) 3 + choose defective_products 3 * choose (total_products - defective_products) 2 :=
by
  sorry

end NUMINAMATH_GPT_select_defective_products_l1401_140162


namespace NUMINAMATH_GPT_smallest_x_exists_l1401_140152

theorem smallest_x_exists (x k m : ℤ) 
    (h1 : x + 3 = 7 * k) 
    (h2 : x - 5 = 8 * m) 
    (h3 : ∀ n : ℤ, ((n + 3) % 7 = 0) ∧ ((n - 5) % 8 = 0) → x ≤ n) : 
    x = 53 := by
  sorry

end NUMINAMATH_GPT_smallest_x_exists_l1401_140152


namespace NUMINAMATH_GPT_cost_of_each_skin_l1401_140105

theorem cost_of_each_skin
  (total_value : ℕ)
  (overall_profit : ℚ)
  (profit_first : ℚ)
  (profit_second : ℚ)
  (total_sell : ℕ)
  (equality : (1 : ℚ) + profit_first ≠ 0 ∧ (1 : ℚ) + profit_second ≠ 0) :
  total_value = 2250 → overall_profit = 0.4 → profit_first = 0.25 → profit_second = -0.5 →
  total_sell = 3150 →
  ∃ x y : ℚ, x = 2700 ∧ y = -450 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_skin_l1401_140105


namespace NUMINAMATH_GPT_major_axis_endpoints_of_ellipse_l1401_140189

theorem major_axis_endpoints_of_ellipse :
  ∀ x y, 6 * x^2 + y^2 = 6 ↔ (x = 0 ∧ (y = -Real.sqrt 6 ∨ y = Real.sqrt 6)) :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_major_axis_endpoints_of_ellipse_l1401_140189


namespace NUMINAMATH_GPT_evaluate_tan_fraction_l1401_140176

theorem evaluate_tan_fraction:
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_tan_fraction_l1401_140176


namespace NUMINAMATH_GPT_volume_of_cube_l1401_140187

theorem volume_of_cube (SA : ℝ) (H : SA = 600) : (10^3 : ℝ) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_l1401_140187


namespace NUMINAMATH_GPT_hall_volume_l1401_140109

theorem hall_volume (l w : ℕ) (h : ℕ) 
    (cond1 : l = 18)
    (cond2 : w = 9)
    (cond3 : (2 * l * w) = (2 * l * h + 2 * w * h)) : 
    (l * w * h = 972) :=
by
  rw [cond1, cond2] at cond3
  have h_eq : h = 324 / 54 := sorry
  rw [h_eq]
  norm_num
  sorry

end NUMINAMATH_GPT_hall_volume_l1401_140109


namespace NUMINAMATH_GPT_gcd_78_182_l1401_140146

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end NUMINAMATH_GPT_gcd_78_182_l1401_140146


namespace NUMINAMATH_GPT_find_y_eq_1_div_5_l1401_140103

theorem find_y_eq_1_div_5 (b : ℝ) (y : ℝ) (h1 : b > 2) (h2 : y > 0) (h3 : (3 * y)^(Real.log 3 / Real.log b) - (5 * y)^(Real.log 5 / Real.log b) = 0) :
  y = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_eq_1_div_5_l1401_140103


namespace NUMINAMATH_GPT_centroid_of_triangle_l1401_140177

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  let x_centroid := (x1 + x2 + x3) / 3
  let y_centroid := (y1 + y2 + y3) / 3
  (x_centroid, y_centroid) = (1/3 * (x1 + x2 + x3), 1/3 * (y1 + y2 + y3)) :=
by
  sorry

end NUMINAMATH_GPT_centroid_of_triangle_l1401_140177


namespace NUMINAMATH_GPT_max_digit_sum_watch_l1401_140186

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end NUMINAMATH_GPT_max_digit_sum_watch_l1401_140186


namespace NUMINAMATH_GPT_instantaneous_acceleration_at_1_second_l1401_140110

-- Assume the velocity function v(t) is given as:
def v (t : ℝ) : ℝ := t^2 + 2 * t + 3

-- We need to prove that the instantaneous acceleration at t = 1 second is 4 m/s^2.
theorem instantaneous_acceleration_at_1_second : 
  deriv v 1 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_instantaneous_acceleration_at_1_second_l1401_140110


namespace NUMINAMATH_GPT_degree_poly_product_l1401_140164

open Polynomial

-- Given conditions: p and q are polynomials with specified degrees
variables {R : Type*} [CommRing R]
variable (p q : R[X])
variable (hp : degree p = 3)
variable (hq : degree q = 6)

-- Proposition: The degree of p(x^2) * q(x^4) is 30
theorem degree_poly_product : degree (p.comp ((X : R[X])^2) * (q.comp ((X : R[X])^4))) = 30 :=
by sorry

end NUMINAMATH_GPT_degree_poly_product_l1401_140164


namespace NUMINAMATH_GPT_derivative_of_f_l1401_140181

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end NUMINAMATH_GPT_derivative_of_f_l1401_140181


namespace NUMINAMATH_GPT_smallest_sum_l1401_140167

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  (∀ A B C D : ℕ, 
    5 * A = 25 * A - 27 * B ∧
    5 * B = 15 * A - 16 * B ∧
    3 * C = 25 * C - 27 * D ∧
    3 * D = 15 * C - 16 * D) ∧
  a = 4 ∧ b = 3 ∧ c = 27 ∧ d = 22 ∧ a + b + c + d = 56

theorem smallest_sum : problem_statement :=
  sorry

end NUMINAMATH_GPT_smallest_sum_l1401_140167


namespace NUMINAMATH_GPT_unique_function_l1401_140120

-- Define the function in the Lean environment
def f (n : ℕ) : ℕ := n

-- State the theorem with the given conditions and expected answer
theorem unique_function (f : ℕ → ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) < x * (1 + f y) + 2021) → (∀ x : ℕ, f x = x) :=
by
  intros h x
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_unique_function_l1401_140120


namespace NUMINAMATH_GPT_worker_wage_before_promotion_l1401_140168

variable (W_new : ℝ)
variable (W : ℝ)

theorem worker_wage_before_promotion (h1 : W_new = 45) (h2 : W_new = 1.60 * W) :
  W = 28.125 := by
  sorry

end NUMINAMATH_GPT_worker_wage_before_promotion_l1401_140168


namespace NUMINAMATH_GPT_find_m_for_q_find_m_for_pq_l1401_140123

variable (m : ℝ)

-- Statement q: The equation represents a hyperbola if and only if m > 3
def q (m : ℝ) : Prop := m > 3

-- Statement p: The inequality holds if and only if m >= 1
def p (m : ℝ) : Prop := m ≥ 1

-- 1. If statement q is true, find the range of values for m.
theorem find_m_for_q (h : q m) : m > 3 := by
  exact h

-- 2. If (p ∨ q) is true and (p ∧ q) is false, find the range of values for m.
theorem find_m_for_pq (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_find_m_for_q_find_m_for_pq_l1401_140123


namespace NUMINAMATH_GPT_B_time_to_complete_work_l1401_140135

variable {W : ℝ} {R_b : ℝ} {T_b : ℝ}

theorem B_time_to_complete_work (h1 : 3 * R_b * (T_b - 10) = R_b * T_b) : T_b = 15 :=
by
  sorry

end NUMINAMATH_GPT_B_time_to_complete_work_l1401_140135


namespace NUMINAMATH_GPT_arithmetic_sequence_propositions_l1401_140196

theorem arithmetic_sequence_propositions (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_S_def : ∀ n, S n = n * (a_n 1 + (a_n (n - 1))) / 2)
  (h_cond : S 6 > S 7 ∧ S 7 > S 5) :
  (∃ d, d < 0 ∧ S 11 > 0) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_propositions_l1401_140196


namespace NUMINAMATH_GPT_triangle_side_lengths_values_l1401_140157

theorem triangle_side_lengths_values :
  ∃ (m_values : Finset ℕ), m_values = {m ∈ Finset.range 750 | m ≥ 4} ∧ m_values.card = 746 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_values_l1401_140157


namespace NUMINAMATH_GPT_odd_function_h_l1401_140142

noncomputable def f (x h k : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

theorem odd_function_h (k : ℝ) (h : ℝ) (H : ∀ x : ℝ, x ≠ -1 → f x h k = -f (-x) h k) :
  h = Real.log 2 :=
sorry

end NUMINAMATH_GPT_odd_function_h_l1401_140142


namespace NUMINAMATH_GPT_photo_counts_correct_l1401_140154

open Real

-- Definitions based on the conditions from step a)
def animal_photos : ℕ := 20
def flower_photos : ℕ := 30 -- 1.5 * 20
def total_animal_flower_photos : ℕ := animal_photos + flower_photos
def scenery_abstract_photos_combined : ℕ := (4 / 10) * total_animal_flower_photos -- 40% of total_animal_flower_photos

def x : ℕ := scenery_abstract_photos_combined / 5
def scenery_photos : ℕ := 3 * x
def abstract_photos : ℕ := 2 * x
def total_photos : ℕ := animal_photos + flower_photos + scenery_photos + abstract_photos

-- The statement to prove
theorem photo_counts_correct :
  animal_photos = 20 ∧
  flower_photos = 30 ∧
  total_animal_flower_photos = 50 ∧
  scenery_abstract_photos_combined = 20 ∧
  scenery_photos = 12 ∧
  abstract_photos = 8 ∧
  total_photos = 70 :=
by
  sorry

end NUMINAMATH_GPT_photo_counts_correct_l1401_140154


namespace NUMINAMATH_GPT_total_dogs_l1401_140114

theorem total_dogs (D : ℕ) 
(h1 : 12 = 12)
(h2 : D / 2 = D / 2)
(h3 : D / 4 = D / 4)
(h4 : 10 = 10)
(h_eq : 12 + D / 2 + D / 4 + 10 = D) : 
D = 88 := by
sorry

end NUMINAMATH_GPT_total_dogs_l1401_140114


namespace NUMINAMATH_GPT_simplify_expression_l1401_140158

noncomputable def i : ℂ := Complex.I

theorem simplify_expression : 7*(4 - 2*i) + 4*i*(3 - 2*i) = 36 - 2*i :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1401_140158


namespace NUMINAMATH_GPT_diane_total_harvest_l1401_140174

def total_harvest (h1 i1 i2 : Nat) : Nat :=
  h1 + (h1 + i1) + ((h1 + i1) + i2)

theorem diane_total_harvest :
  total_harvest 2479 6085 7890 = 27497 := 
by 
  sorry

end NUMINAMATH_GPT_diane_total_harvest_l1401_140174


namespace NUMINAMATH_GPT_boys_belong_to_other_communities_l1401_140121

-- Definitions for the given problem
def total_boys : ℕ := 850
def percent_muslims : ℝ := 0.34
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10
def percent_other : ℝ := 1 - (percent_muslims + percent_hindus + percent_sikhs)

-- Statement to prove that the number of boys belonging to other communities is 238
theorem boys_belong_to_other_communities : 
  (percent_other * total_boys) = 238 := by 
  sorry

end NUMINAMATH_GPT_boys_belong_to_other_communities_l1401_140121


namespace NUMINAMATH_GPT_arithmetic_progression_even_terms_l1401_140199

theorem arithmetic_progression_even_terms (a d n : ℕ) (h_even : n % 2 = 0)
  (h_last_first_diff : (n - 1) * d = 16)
  (h_sum_odd : n * (a + (n - 2) * d / 2) = 81)
  (h_sum_even : n * (a + d + (n - 2) * d / 2) = 75) :
  n = 8 :=
by sorry

end NUMINAMATH_GPT_arithmetic_progression_even_terms_l1401_140199


namespace NUMINAMATH_GPT_andrei_kolya_ages_l1401_140185

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + (n / 1000)

theorem andrei_kolya_ages :
  ∃ (y1 y2 : ℕ), (sum_of_digits y1 = 2021 - y1) ∧ (sum_of_digits y2 = 2021 - y2) ∧ (y1 ≠ y2) ∧ ((2022 - y1 = 8 ∧ 2022 - y2 = 26) ∨ (2022 - y1 = 26 ∧ 2022 - y2 = 8)) :=
by
  sorry

end NUMINAMATH_GPT_andrei_kolya_ages_l1401_140185


namespace NUMINAMATH_GPT_min_c_plus_3d_l1401_140194

theorem min_c_plus_3d (c d : ℝ) (hc : 0 < c) (hd : 0 < d) 
    (h1 : c^2 ≥ 12 * d) (h2 : 9 * d^2 ≥ 4 * c) : 
  c + 3 * d ≥ 8 :=
  sorry

end NUMINAMATH_GPT_min_c_plus_3d_l1401_140194


namespace NUMINAMATH_GPT_arctan_sum_zero_l1401_140153
open Real

variable (a b c : ℝ)
variable (h : a^2 + b^2 = c^2)

theorem arctan_sum_zero (h : a^2 + b^2 = c^2) :
  arctan (a / (b + c)) + arctan (b / (a + c)) + arctan (c / (a + b)) = 0 := 
sorry

end NUMINAMATH_GPT_arctan_sum_zero_l1401_140153


namespace NUMINAMATH_GPT_sauna_max_couples_l1401_140195

def max_couples (n : ℕ) : ℕ :=
  n - 1

theorem sauna_max_couples (n : ℕ) (rooms unlimited_capacity : Prop) (no_female_male_cohabsimult : Prop)
                          (males_shared_room_constraint females_shared_room_constraint : Prop)
                          (males_known_iff_wives_known : Prop) : max_couples n = n - 1 := 
  sorry

end NUMINAMATH_GPT_sauna_max_couples_l1401_140195


namespace NUMINAMATH_GPT_tan_2x_value_l1401_140102

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) := deriv f x

theorem tan_2x_value (x : ℝ) (h : f' x = 3 * f x) : Real.tan (2 * x) = (4/3) := by
  sorry

end NUMINAMATH_GPT_tan_2x_value_l1401_140102


namespace NUMINAMATH_GPT_polynomial_factors_l1401_140141

theorem polynomial_factors (x : ℝ) : 
  (x^4 - 4*x^2 + 4) = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factors_l1401_140141


namespace NUMINAMATH_GPT_simplify_expression_l1401_140106

variable (z : ℝ)

theorem simplify_expression :
  (z - 2 * z + 4 * z - 6 + 3 + 7 - 2) = (3 * z + 2) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1401_140106


namespace NUMINAMATH_GPT_george_correct_answer_l1401_140188

variable (y : ℝ)

theorem george_correct_answer (h : y / 7 = 30) : 70 + y = 280 :=
sorry

end NUMINAMATH_GPT_george_correct_answer_l1401_140188


namespace NUMINAMATH_GPT_max_min_value_l1401_140156

noncomputable def f (A B x a b : ℝ) : ℝ :=
  A * Real.sqrt (x - a) + B * Real.sqrt (b - x)

theorem max_min_value (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha_lt_b : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → f A B x a b ≤ Real.sqrt ((A^2 + B^2) * (b - a))) ∧
  min (f A B a a b) (f A B b a b) ≤ f A B x a b :=
  sorry

end NUMINAMATH_GPT_max_min_value_l1401_140156


namespace NUMINAMATH_GPT_smallest_base10_integer_l1401_140159

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_base10_integer_l1401_140159


namespace NUMINAMATH_GPT_range_of_a_for_distinct_real_roots_l1401_140126

theorem range_of_a_for_distinct_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ (a < 2 ∧ a ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_distinct_real_roots_l1401_140126


namespace NUMINAMATH_GPT_sum_of_solutions_eq_seven_l1401_140137

theorem sum_of_solutions_eq_seven : 
  ∃ x : ℝ, x + 49/x = 14 ∧ (∀ y : ℝ, y + 49 / y = 14 → y = x) → x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_solutions_eq_seven_l1401_140137


namespace NUMINAMATH_GPT_domain_of_function_l1401_140133

theorem domain_of_function : {x : ℝ | 3 - 2 * x - x ^ 2 ≥ 0 } = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1401_140133


namespace NUMINAMATH_GPT_tan_315_eq_neg1_l1401_140184

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg1_l1401_140184


namespace NUMINAMATH_GPT_floor_painting_cost_l1401_140166

noncomputable def floor_painting_problem : Prop := 
  ∃ (B L₁ L₂ B₂ Area₁ Area₂ CombinedCost : ℝ),
  L₁ = 2 * B ∧
  Area₁ = L₁ * B ∧
  484 = Area₁ * 3 ∧
  L₂ = 0.8 * L₁ ∧
  B₂ = 1.3 * B ∧
  Area₂ = L₂ * B₂ ∧
  CombinedCost = 484 + (Area₂ * 5) ∧
  CombinedCost = 1320.8

theorem floor_painting_cost : floor_painting_problem :=
by
  sorry

end NUMINAMATH_GPT_floor_painting_cost_l1401_140166


namespace NUMINAMATH_GPT_carla_drive_distance_l1401_140147

theorem carla_drive_distance
    (d1 d3 : ℕ) (gpm : ℕ) (gas_price total_cost : ℕ) 
    (x : ℕ)
    (hx : 2 * gas_price = 1)
    (gallon_cost : ℕ := total_cost / gas_price)
    (total_distance   : ℕ := gallon_cost * gpm)
    (total_errand_distance : ℕ := d1 + x + d3 + 2 * x)
    (h_distance : total_distance = total_errand_distance) :
  x = 10 :=
by
  -- begin
  -- proof construction
  sorry

end NUMINAMATH_GPT_carla_drive_distance_l1401_140147


namespace NUMINAMATH_GPT_frog_ends_on_horizontal_side_l1401_140178

-- Definitions for the problem conditions
def frog_jump_probability (x y : ℤ) : ℚ := sorry

-- Main theorem statement based on the identified question and correct answer
theorem frog_ends_on_horizontal_side :
  frog_jump_probability 2 3 = 13 / 14 :=
sorry

end NUMINAMATH_GPT_frog_ends_on_horizontal_side_l1401_140178


namespace NUMINAMATH_GPT_christine_needs_32_tbs_aquafaba_l1401_140161

-- Definitions for the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

def total_egg_whites : ℕ := egg_whites_per_cake * number_of_cakes
def total_tbs_aquafaba : ℕ := tablespoons_per_egg_white * total_egg_whites

-- Theorem statement
theorem christine_needs_32_tbs_aquafaba :
  total_tbs_aquafaba = 32 :=
by sorry

end NUMINAMATH_GPT_christine_needs_32_tbs_aquafaba_l1401_140161


namespace NUMINAMATH_GPT_possible_values_of_n_l1401_140169

theorem possible_values_of_n (E M n : ℕ) (h1 : M + 3 = n * (E - 3)) (h2 : E + n = 3 * (M - n)) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end NUMINAMATH_GPT_possible_values_of_n_l1401_140169


namespace NUMINAMATH_GPT_dot_product_of_ab_ac_l1401_140119

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_of_ab_ac :
  vec_dot (1, -2) (2, -2) = 6 := by
  sorry

end NUMINAMATH_GPT_dot_product_of_ab_ac_l1401_140119


namespace NUMINAMATH_GPT_inequality_solution_addition_eq_seven_l1401_140145

theorem inequality_solution_addition_eq_seven (b c : ℝ) :
  (∀ x : ℝ, -5 < 2 * x - 3 ∧ 2 * x - 3 < 5 → -1 < x ∧ x < 4) →
  (∀ x : ℝ, -x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 4)) →
  b + c = 7 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_inequality_solution_addition_eq_seven_l1401_140145


namespace NUMINAMATH_GPT_quadratic_roots_properties_quadratic_roots_max_min_l1401_140183

theorem quadratic_roots_properties (k : ℝ) (h : 2 ≤ k ∧ k ≤ 8)
  (x1 x2 : ℝ) (h_roots : x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) :
  (x1^2 + x2^2) = 16 * k - 30 :=
sorry

theorem quadratic_roots_max_min :
  (∀ k ∈ { k : ℝ | 2 ≤ k ∧ k ≤ 8 }, 
    ∃ (x1 x2 : ℝ), 
      (x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) 
      ∧ (x1^2 + x2^2) = (if k = 8 then 98 else if k = 2 then 2 else 16 * k - 30)) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_properties_quadratic_roots_max_min_l1401_140183


namespace NUMINAMATH_GPT_fifth_term_sequence_l1401_140193

theorem fifth_term_sequence : 
  (4 + 8 + 16 + 32 + 64) = 124 := 
by 
  sorry

end NUMINAMATH_GPT_fifth_term_sequence_l1401_140193


namespace NUMINAMATH_GPT_isosceles_vertex_angle_l1401_140113

-- Let T be a type representing triangles, with a function base_angle returning the degree of a base angle,
-- and vertex_angle representing the degree of the vertex angle.
axiom Triangle : Type
axiom is_isosceles (t : Triangle) : Prop
axiom base_angle_deg (t : Triangle) : ℝ
axiom vertex_angle_deg (t : Triangle) : ℝ

theorem isosceles_vertex_angle (t : Triangle) (h_isosceles : is_isosceles t)
  (h_base_angle : base_angle_deg t = 50) : vertex_angle_deg t = 80 := by
  sorry

end NUMINAMATH_GPT_isosceles_vertex_angle_l1401_140113


namespace NUMINAMATH_GPT_abs_diff_condition_l1401_140130

theorem abs_diff_condition {a b : ℝ} (h1 : |a| = 1) (h2 : |b - 1| = 2) (h3 : a > b) : a - b = 2 := 
sorry

end NUMINAMATH_GPT_abs_diff_condition_l1401_140130


namespace NUMINAMATH_GPT_totalPawnsLeft_l1401_140131

def sophiaInitialPawns := 8
def chloeInitialPawns := 8
def sophiaLostPawns := 5
def chloeLostPawns := 1

theorem totalPawnsLeft : (sophiaInitialPawns - sophiaLostPawns) + (chloeInitialPawns - chloeLostPawns) = 10 := by
  sorry

end NUMINAMATH_GPT_totalPawnsLeft_l1401_140131


namespace NUMINAMATH_GPT_math_problem_l1401_140165

open Real

theorem math_problem
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x + y + z = 1) :
  ( (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 ) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l1401_140165


namespace NUMINAMATH_GPT_area_of_annulus_l1401_140139

theorem area_of_annulus (R r t : ℝ) (h : R > r) (h_tangent : R^2 = r^2 + t^2) : 
  π * (R^2 - r^2) = π * t^2 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_annulus_l1401_140139


namespace NUMINAMATH_GPT_tangent_intersection_x_l1401_140115

theorem tangent_intersection_x :
  ∃ x : ℝ, 
    0 < x ∧ (∃ r1 r2 : ℝ, 
     (r1 = 3) ∧ 
     (r2 = 8) ∧ 
     (0, 0) = (0, 0) ∧ 
     (18, 0) = (18, 0) ∧
     (∀ t : ℝ, t > 0 → t = x / (18 - x) → t = r1 / r2) ∧ 
      x = 54 / 11) := 
sorry

end NUMINAMATH_GPT_tangent_intersection_x_l1401_140115


namespace NUMINAMATH_GPT_cost_of_product_l1401_140144

theorem cost_of_product (x : ℝ) (a : ℝ) (h : a > 0) :
  (1 + a / 100) * (x / (1 + a / 100)) = x :=
by
  field_simp [ne_of_gt h]
  sorry

end NUMINAMATH_GPT_cost_of_product_l1401_140144


namespace NUMINAMATH_GPT_new_mix_concentration_l1401_140118

theorem new_mix_concentration 
  (capacity1 capacity2 capacity_mix : ℝ)
  (alc_percent1 alc_percent2 : ℝ)
  (amount1 amount2 : capacity1 = 3 ∧ capacity2 = 5 ∧ capacity_mix = 10)
  (percent1: alc_percent1 = 0.25)
  (percent2: alc_percent2 = 0.40)
  (total_volume : ℝ)
  (eight_liters : total_volume = 8) :
  (alc_percent1 * capacity1 + alc_percent2 * capacity2) / total_volume * 100 = 34.375 :=
by
  sorry

end NUMINAMATH_GPT_new_mix_concentration_l1401_140118


namespace NUMINAMATH_GPT_actual_revenue_is_60_percent_of_projected_l1401_140151

variable (R : ℝ)

-- Condition: Projected revenue is 25% more than last year's revenue
def projected_revenue (R : ℝ) : ℝ := 1.25 * R

-- Condition: Actual revenue decreased by 25% compared to last year's revenue
def actual_revenue (R : ℝ) : ℝ := 0.75 * R

-- Theorem: Prove that the actual revenue is 60% of the projected revenue
theorem actual_revenue_is_60_percent_of_projected :
  (actual_revenue R) = 0.6 * (projected_revenue R) :=
  sorry

end NUMINAMATH_GPT_actual_revenue_is_60_percent_of_projected_l1401_140151


namespace NUMINAMATH_GPT_range_of_a_if_p_and_not_q_l1401_140148

open Real

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a_if_p_and_not_q : 
  (∃ a : ℝ, (p a ∧ ¬q a)) → 
  (∀ a : ℝ, (p a ∧ ¬q a) → (-1 ≤ a ∧ a < 0)) :=
sorry

end NUMINAMATH_GPT_range_of_a_if_p_and_not_q_l1401_140148


namespace NUMINAMATH_GPT_map_distance_l1401_140163

/--
On a map, 8 cm represents 40 km. Prove that 20 cm represents 100 km.
-/
theorem map_distance (scale_factor : ℕ) (distance_cm : ℕ) (distance_km : ℕ) 
  (h_scale : scale_factor = 5) (h_distance_cm : distance_cm = 20) : 
  distance_km = 20 * scale_factor := 
by {
  sorry
}

end NUMINAMATH_GPT_map_distance_l1401_140163


namespace NUMINAMATH_GPT_find_symmetric_point_l1401_140170

def slope_angle (l : ℝ → ℝ → Prop) (θ : ℝ) := ∃ m, m = Real.tan θ ∧ ∀ x y, l x y ↔ y = m * (x - 1) + 1
def passes_through (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) := l P.fst P.snd
def symmetric_point (A A' : ℝ × ℝ) (l : ℝ → ℝ → Prop) := 
  (A'.snd - A.snd = A'.fst - A.fst) ∧ 
  ((A'.fst + A.fst) / 2 + (A'.snd + A.snd) / 2 - 2 = 0)

theorem find_symmetric_point :
  ∃ l : ℝ → ℝ → Prop, 
    slope_angle l (135 : ℝ) ∧ 
    passes_through l (1, 1) ∧ 
    (∀ x y, l x y ↔ x + y = 2) ∧ 
    symmetric_point (3, 4) (-2, -1) l :=
by sorry

end NUMINAMATH_GPT_find_symmetric_point_l1401_140170


namespace NUMINAMATH_GPT_RectangleAreaDiagonalk_l1401_140192

theorem RectangleAreaDiagonalk {length width : ℝ} {d : ℝ}
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : 2 * (length + width) = 42)
  (h_diagonal : d = Real.sqrt (length^2 + width^2))
  : (∃ k, k = 10 / 29 ∧ ∀ A, A = k * d^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_RectangleAreaDiagonalk_l1401_140192


namespace NUMINAMATH_GPT_factorial_not_div_by_two_pow_l1401_140116

theorem factorial_not_div_by_two_pow (n : ℕ) : ¬ (2^n ∣ n!) :=
sorry

end NUMINAMATH_GPT_factorial_not_div_by_two_pow_l1401_140116


namespace NUMINAMATH_GPT_joe_total_spending_at_fair_l1401_140101

-- Definitions based on conditions
def entrance_fee (age : ℕ) : ℝ := if age < 18 then 5 else 6
def ride_cost (rides : ℕ) : ℝ := rides * 0.5

-- Given conditions
def joe_age := 19
def twin_age := 6

def total_cost (joe_age : ℕ) (twin_age : ℕ) (rides_per_person : ℕ) :=
  entrance_fee joe_age + 2 * entrance_fee twin_age + 3 * ride_cost rides_per_person

-- The main statement to be proven
theorem joe_total_spending_at_fair : total_cost joe_age twin_age 3 = 20.5 :=
by
  sorry

end NUMINAMATH_GPT_joe_total_spending_at_fair_l1401_140101


namespace NUMINAMATH_GPT_power_equation_l1401_140136

theorem power_equation (x a : ℝ) (h : x^(-a) = 3) : x^(2 * a) = 1 / 9 :=
sorry

end NUMINAMATH_GPT_power_equation_l1401_140136


namespace NUMINAMATH_GPT_range_of_k_real_roots_l1401_140180

variable (k : ℝ)
def quadratic_has_real_roots : Prop :=
  let a := k - 1
  let b := 2
  let c := 1
  let Δ := b^2 - 4 * a * c
  Δ ≥ 0 ∧ a ≠ 0

theorem range_of_k_real_roots :
  quadratic_has_real_roots k ↔ (k ≤ 2 ∧ k ≠ 1) := by
  sorry

end NUMINAMATH_GPT_range_of_k_real_roots_l1401_140180

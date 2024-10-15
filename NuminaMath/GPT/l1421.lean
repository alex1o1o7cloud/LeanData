import Mathlib

namespace NUMINAMATH_GPT_least_number_of_marbles_divisible_l1421_142109

theorem least_number_of_marbles_divisible (n : ℕ) : 
  (∀ k ∈ [2, 3, 4, 5, 6, 7, 8], n % k = 0) -> n >= 840 :=
by sorry

end NUMINAMATH_GPT_least_number_of_marbles_divisible_l1421_142109


namespace NUMINAMATH_GPT_minimum_point_translation_l1421_142142

noncomputable def f (x : ℝ) : ℝ := |x| - 2

theorem minimum_point_translation :
  let minPoint := (0, f 0)
  let newMinPoint := (minPoint.1 + 4, minPoint.2 + 5)
  newMinPoint = (4, 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_point_translation_l1421_142142


namespace NUMINAMATH_GPT_total_time_of_flight_l1421_142158

variables {V_0 g t t_1 H : ℝ}  -- Define variables

-- Define conditions
def initial_condition (V_0 g t_1 H : ℝ) : Prop :=
H = (1/2) * g * t_1^2

def return_condition (V_0 g t : ℝ) : Prop :=
t = 2 * (V_0 / g)

theorem total_time_of_flight
  (V_0 g : ℝ)
  (h1 : initial_condition V_0 g (V_0 / g) (1/2 * g * (V_0 / g)^2))
  : return_condition V_0 g (2 * V_0 / g) :=
by
  sorry

end NUMINAMATH_GPT_total_time_of_flight_l1421_142158


namespace NUMINAMATH_GPT_trapezoid_area_l1421_142162

theorem trapezoid_area (A_outer A_inner : ℝ) (n : ℕ)
  (h_outer : A_outer = 36)
  (h_inner : A_inner = 4)
  (h_n : n = 4) :
  (A_outer - A_inner) / n = 8 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1421_142162


namespace NUMINAMATH_GPT_problem1_problem2_l1421_142199

-- Define the quadratic equation and condition for real roots
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Problem 1
theorem problem1 (m : ℝ) : ((m - 2) * (m - 2) * (m - 2) + 2 * 2 * (2 - m) * 2 * (-1) ≥ 0) → (m ≤ 3 ∧ m ≠ 2) := sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (∀ x, (x = 1 ∨ x = 2) → (m - 2) * x^2 + 2 * x + 1 = 0) → (-1 ≤ m ∧ m < (3 / 4)) := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1421_142199


namespace NUMINAMATH_GPT_relationship_of_y_l1421_142182

theorem relationship_of_y (k y1 y2 y3 : ℝ)
  (hk : k < 0)
  (hy1 : y1 = k / -2)
  (hy2 : y2 = k / 1)
  (hy3 : y3 = k / 2) :
  y2 < y3 ∧ y3 < y1 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_relationship_of_y_l1421_142182


namespace NUMINAMATH_GPT_sqrt_expression_simplification_l1421_142197

theorem sqrt_expression_simplification :
  (Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt (3) - 1| + Real.sqrt 3) = -13 / 4 + 2 * Real.sqrt 3 :=
by
  have h1 : Real.sqrt (1 / 16) = 1 / 4 := sorry
  have h2 : Real.sqrt (25 / 4) = 5 / 2 := sorry
  have h3 : |Real.sqrt 3 - 1| = Real.sqrt 3 - 1 := sorry
  linarith [h1, h2, h3]

end NUMINAMATH_GPT_sqrt_expression_simplification_l1421_142197


namespace NUMINAMATH_GPT_inequality_proof_l1421_142164

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l1421_142164


namespace NUMINAMATH_GPT_book_price_net_change_l1421_142100

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P * 0.70
  let increased_price := decreased_price * 1.20
  let net_change := (increased_price - P) / P * 100
  net_change = -16 := 
by
  sorry

end NUMINAMATH_GPT_book_price_net_change_l1421_142100


namespace NUMINAMATH_GPT_rectangle_not_equal_118_l1421_142166

theorem rectangle_not_equal_118 
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (A : ℕ) (P : ℕ)
  (h₂ : A = a * b) (h₃ : P = 2 * (a + b)) :
  (a + 2) * (b + 2) - 2 ≠ 118 :=
sorry

end NUMINAMATH_GPT_rectangle_not_equal_118_l1421_142166


namespace NUMINAMATH_GPT_red_beads_cost_l1421_142167

theorem red_beads_cost (R : ℝ) (H : 4 * R + 4 * 2 = 10 * 1.72) : R = 2.30 :=
by
  sorry

end NUMINAMATH_GPT_red_beads_cost_l1421_142167


namespace NUMINAMATH_GPT_max_integer_value_of_k_l1421_142115

theorem max_integer_value_of_k :
  ∀ x y k : ℤ,
    x - 4 * y = k - 1 →
    2 * x + y = k →
    x - y ≤ 0 →
    k ≤ 0 :=
by
  intros x y k h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_integer_value_of_k_l1421_142115


namespace NUMINAMATH_GPT_range_of_p_l1421_142121

noncomputable def proof_problem (p : ℝ) : Prop :=
  (∀ x : ℝ, (4 * x + p < 0) → (x < -1 ∨ x > 2)) → (p ≥ 4)

theorem range_of_p (p : ℝ) : proof_problem p :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_p_l1421_142121


namespace NUMINAMATH_GPT_range_of_reciprocal_sum_l1421_142139

theorem range_of_reciprocal_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := 
sorry

end NUMINAMATH_GPT_range_of_reciprocal_sum_l1421_142139


namespace NUMINAMATH_GPT_regular_polygon_sides_l1421_142176

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1421_142176


namespace NUMINAMATH_GPT_base_of_isosceles_triangle_l1421_142198

theorem base_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : 3 * a = 45) 
  (h₂ : 2 * b + c = 40) 
  (h₃ : b = a ∨ b = a) : c = 10 := 
sorry

end NUMINAMATH_GPT_base_of_isosceles_triangle_l1421_142198


namespace NUMINAMATH_GPT_solve_for_q_l1421_142108

variable (R t m q : ℝ)

def given_condition : Prop :=
  R = t / ((2 + m) ^ q)

theorem solve_for_q (h : given_condition R t m q) : 
  q = (Real.log (t / R)) / (Real.log (2 + m)) := 
sorry

end NUMINAMATH_GPT_solve_for_q_l1421_142108


namespace NUMINAMATH_GPT_combined_mpg_l1421_142172

theorem combined_mpg
  (R_eff : ℝ) (T_eff : ℝ)
  (R_dist : ℝ) (T_dist : ℝ)
  (H_R_eff : R_eff = 35)
  (H_T_eff : T_eff = 15)
  (H_R_dist : R_dist = 420)
  (H_T_dist : T_dist = 300)
  : (R_dist + T_dist) / (R_dist / R_eff + T_dist / T_eff) = 22.5 := 
by
  rw [H_R_eff, H_T_eff, H_R_dist, H_T_dist]
  -- Proof steps would go here, but we'll use sorry to skip it.
  sorry

end NUMINAMATH_GPT_combined_mpg_l1421_142172


namespace NUMINAMATH_GPT_distinct_convex_quadrilaterals_l1421_142114

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem distinct_convex_quadrilaterals (n : ℕ) (h : n > 4) 
  (no_three_collinear : ℕ → Prop) :
  ∃ k, k ≥ combinations n 5 / (n - 4) :=
by
  sorry

end NUMINAMATH_GPT_distinct_convex_quadrilaterals_l1421_142114


namespace NUMINAMATH_GPT_largest_sum_is_three_fourths_l1421_142168

-- Definitions of sums
def sum1 := (1 / 4) + (1 / 2)
def sum2 := (1 / 4) + (1 / 9)
def sum3 := (1 / 4) + (1 / 3)
def sum4 := (1 / 4) + (1 / 10)
def sum5 := (1 / 4) + (1 / 6)

-- The theorem stating that sum1 is the maximum of the sums
theorem largest_sum_is_three_fourths : max (max (max (max sum1 sum2) sum3) sum4) sum5 = 3 / 4 := 
sorry

end NUMINAMATH_GPT_largest_sum_is_three_fourths_l1421_142168


namespace NUMINAMATH_GPT_number_of_sturgeons_l1421_142148

def number_of_fishes := 145
def number_of_pikes := 30
def number_of_herrings := 75

theorem number_of_sturgeons : (number_of_fishes - (number_of_pikes + number_of_herrings) = 40) :=
  by
  sorry

end NUMINAMATH_GPT_number_of_sturgeons_l1421_142148


namespace NUMINAMATH_GPT_expression_bounds_l1421_142189

noncomputable def expression (p q r s : ℝ) : ℝ :=
  Real.sqrt (p^2 + (2 - q)^2) + Real.sqrt (q^2 + (2 - r)^2) +
  Real.sqrt (r^2 + (2 - s)^2) + Real.sqrt (s^2 + (2 - p)^2)

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2)
  (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) : 
  4 * Real.sqrt 2 ≤ expression p q r s ∧ expression p q r s ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_expression_bounds_l1421_142189


namespace NUMINAMATH_GPT_x4_y4_value_l1421_142144

theorem x4_y4_value (x y : ℝ) (h1 : x^4 + x^2 = 3) (h2 : y^4 - y^2 = 3) : x^4 + y^4 = 7 := by
  sorry

end NUMINAMATH_GPT_x4_y4_value_l1421_142144


namespace NUMINAMATH_GPT_pieces_on_third_day_impossibility_of_2014_pieces_l1421_142103

-- Define the process of dividing and eating chocolate pieces.
def chocolate_pieces (n : ℕ) : ℕ :=
  9 + 8 * n

-- The number of pieces after the third day.
theorem pieces_on_third_day : chocolate_pieces 3 = 25 :=
sorry

-- It's impossible for Maria to have exactly 2014 pieces on any given day.
theorem impossibility_of_2014_pieces : ∀ n : ℕ, chocolate_pieces n ≠ 2014 :=
sorry

end NUMINAMATH_GPT_pieces_on_third_day_impossibility_of_2014_pieces_l1421_142103


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l1421_142186

noncomputable def cost_price (SP : ℝ) : ℝ := 0.81 * SP

noncomputable def profit (SP CP : ℝ) : ℝ := SP - CP

noncomputable def profit_percentage (profit CP : ℝ) : ℝ := (profit / CP) * 100

theorem profit_percentage_is_correct (SP : ℝ) (h : SP = 100) :
  profit_percentage (profit SP (cost_price SP)) (cost_price SP) = 23.46 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l1421_142186


namespace NUMINAMATH_GPT_total_number_of_animals_l1421_142138

-- Definitions based on conditions
def number_of_females : ℕ := 35
def males_outnumber_females_by : ℕ := 7
def number_of_males : ℕ := number_of_females + males_outnumber_females_by

-- Theorem to prove the total number of animals
theorem total_number_of_animals :
  number_of_females + number_of_males = 77 := by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l1421_142138


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1421_142145

theorem ratio_of_larger_to_smaller
  (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h3 : x + y = 6 * (x - y)) :
  x / y = 7 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1421_142145


namespace NUMINAMATH_GPT_smallest_value_l1421_142118

theorem smallest_value 
  (x1 x2 x3 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2) 
  (hx3 : 0 < x3)
  (h : 2 * x1 + 3 * x2 + 4 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 = 10000 / 29 := by
  sorry

end NUMINAMATH_GPT_smallest_value_l1421_142118


namespace NUMINAMATH_GPT_correct_calculation_result_l1421_142116

theorem correct_calculation_result (x : ℝ) (h : x / 12 = 8) : 12 * x = 1152 :=
sorry

end NUMINAMATH_GPT_correct_calculation_result_l1421_142116


namespace NUMINAMATH_GPT_solution_l1421_142155

def is_prime (n : ℕ) : Prop := ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

noncomputable def find_pairs : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_prime (a * b^2 / (a + b)) ∧ ((a = 6 ∧ b = 2) ∨ (a = 2 ∧ b = 6))

theorem solution :
  find_pairs := sorry

end NUMINAMATH_GPT_solution_l1421_142155


namespace NUMINAMATH_GPT_farmer_plow_l1421_142170

theorem farmer_plow (P : ℕ) (M : ℕ) (H1 : M = 12) (H2 : 8 * P + M * (8 - (55 / P)) = 30) (H3 : 55 % P = 0) : P = 10 :=
by
  sorry

end NUMINAMATH_GPT_farmer_plow_l1421_142170


namespace NUMINAMATH_GPT_brick_height_l1421_142149

theorem brick_height (length width : ℕ) (num_bricks : ℕ) (wall_length wall_width wall_height : ℕ) (h : ℕ) :
  length = 20 ∧ width = 10 ∧ num_bricks = 25000 ∧ wall_length = 2500 ∧ wall_width = 200 ∧ wall_height = 75 ∧
  ( 20 * 10 * h = (wall_length * wall_width * wall_height) / 25000 ) -> 
  h = 75 :=
by
  sorry

end NUMINAMATH_GPT_brick_height_l1421_142149


namespace NUMINAMATH_GPT_car_actual_speed_is_40_l1421_142104

variable (v : ℝ) -- actual speed (we will prove it is 40 km/h)

-- Conditions
variable (hyp_speed : ℝ := v + 20) -- hypothetical speed
variable (distance : ℝ := 60) -- distance traveled
variable (time_difference : ℝ := 0.5) -- time difference in hours

-- Define the equation derived from the given conditions:
def speed_equation : Prop :=
  (distance / v) - (distance / hyp_speed) = time_difference

-- The theorem to prove:
theorem car_actual_speed_is_40 : speed_equation v → v = 40 :=
by
  sorry

end NUMINAMATH_GPT_car_actual_speed_is_40_l1421_142104


namespace NUMINAMATH_GPT_max_value_quadratic_max_value_quadratic_attained_l1421_142123

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic : ∀ (x : ℝ), quadratic (-8) 32 (-1) x ≤ 31 :=
by
  sorry

theorem max_value_quadratic_attained : 
  quadratic (-8) 32 (-1) 2 = 31 :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_max_value_quadratic_attained_l1421_142123


namespace NUMINAMATH_GPT_boxes_contain_same_number_of_apples_l1421_142192

theorem boxes_contain_same_number_of_apples (total_apples boxes : ℕ) (h1 : total_apples = 49) (h2 : boxes = 7) : 
  total_apples / boxes = 7 :=
by
  sorry

end NUMINAMATH_GPT_boxes_contain_same_number_of_apples_l1421_142192


namespace NUMINAMATH_GPT_part1_part2_part3_part4_l1421_142126

section QuadraticFunction

variable {x : ℝ} {y : ℝ} 

-- 1. Prove that if a quadratic function y = x^2 + bx - 3 intersects the x-axis at (3, 0), 
-- then b = -2 and the other intersection point is (-1, 0).
theorem part1 (b : ℝ) : 
  ((3:ℝ) ^ 2 + b * (3:ℝ) - 3 = 0) → 
  b = -2 ∧ ∃ x : ℝ, (x = -1 ∧ x^2 + b * x - 3 = 0) := 
  sorry

-- 2. For the function y = x^2 + bx - 3 where b = -2, 
-- prove that when 0 < y < 5, x is in -2 < x < -1 or 3 < x < 4.
theorem part2 (b : ℝ) :
  b = -2 → 
  (0 < y ∧ y < 5 → ∃ x : ℝ, (x^2 + b * x - 3 = y) → (-2 < x ∧ x < -1) ∨ (3 < x ∧ x < 4)) :=
  sorry

-- 3. Prove that the value t such that y = x^2 + bx - 3 and y > t always holds for all x
-- is t < -((b ^ 2 + 12) / 4).
theorem part3 (b t : ℝ) :
  (∀ x : ℝ, (x ^ 2 + b * x - 3 > t)) → t < -(b ^ 2 + 12) / 4 :=
  sorry

-- 4. Given y = x^2 - 3x - 3 and 1 < x < 2, 
-- prove that m < y < n with n = -5, b = -3, and m ≤ -21 / 4.
theorem part4 (m n : ℝ) :
  (1 < x ∧ x < 2 → m < x^2 - 3 * x - 3 ∧ x^2 - 3 * x - 3 < n) →
  n = -5 ∧ -21 / 4 ≤ m :=
  sorry

end QuadraticFunction

end NUMINAMATH_GPT_part1_part2_part3_part4_l1421_142126


namespace NUMINAMATH_GPT_largest_integer_is_190_l1421_142111

theorem largest_integer_is_190 (A B C D : ℤ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D) 
  (h4 : (A + B + C + D) / 4 = 76) 
  (h5 : A = 37) 
  (h6 : B = 38) 
  (h7 : C = 39) : 
  D = 190 := 
sorry

end NUMINAMATH_GPT_largest_integer_is_190_l1421_142111


namespace NUMINAMATH_GPT_max_handshakes_l1421_142185

-- Definitions based on the given conditions
def num_people := 30
def handshake_formula (n : ℕ) := n * (n - 1) / 2

-- Formal statement of the problem
theorem max_handshakes : handshake_formula num_people = 435 :=
by
  -- Calculation here would be carried out in the proof, but not included in the statement itself.
  sorry

end NUMINAMATH_GPT_max_handshakes_l1421_142185


namespace NUMINAMATH_GPT_cauchy_schwarz_equivalent_iag_l1421_142196

theorem cauchy_schwarz_equivalent_iag (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (Real.sqrt x * Real.sqrt y) ≤ (x + y) / 2) ↔
  ((a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) := by
  sorry

end NUMINAMATH_GPT_cauchy_schwarz_equivalent_iag_l1421_142196


namespace NUMINAMATH_GPT_values_of_x_l1421_142154

theorem values_of_x (x : ℝ) : (x+2)*(x-9) < 0 ↔ -2 < x ∧ x < 9 := 
by
  sorry

end NUMINAMATH_GPT_values_of_x_l1421_142154


namespace NUMINAMATH_GPT_algae_difference_l1421_142161

-- Define the original number of algae plants.
def original_algae := 809

-- Define the current number of algae plants.
def current_algae := 3263

-- Statement to prove: The difference between the current number of algae plants and the original number of algae plants is 2454.
theorem algae_difference : current_algae - original_algae = 2454 := by
  sorry

end NUMINAMATH_GPT_algae_difference_l1421_142161


namespace NUMINAMATH_GPT_complex_division_example_l1421_142175

theorem complex_division_example (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + i) = (1/2 : ℂ) - (3/2 : ℂ) * i :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_complex_division_example_l1421_142175


namespace NUMINAMATH_GPT_toms_dog_age_is_twelve_l1421_142136

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end NUMINAMATH_GPT_toms_dog_age_is_twelve_l1421_142136


namespace NUMINAMATH_GPT_correct_operation_C_l1421_142113

theorem correct_operation_C (m : ℕ) : m^7 / m^3 = m^4 := by
  sorry

end NUMINAMATH_GPT_correct_operation_C_l1421_142113


namespace NUMINAMATH_GPT_standard_spherical_coordinates_l1421_142107

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 4 → φ = 9 * Real.pi / 5 →
  (ρ > 0) →
  (0 ≤ θ ∧ θ < 2 * Real.pi) →
  (0 ≤ φ ∧ φ ≤ Real.pi) →
  (ρ, θ, φ) = (5, 7 * Real.pi / 4, Real.pi / 5) :=
by sorry

end NUMINAMATH_GPT_standard_spherical_coordinates_l1421_142107


namespace NUMINAMATH_GPT_factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l1421_142180

-- Problem 1
theorem factorize_2x2_minus_4x (x : ℝ) : 
  2 * x^2 - 4 * x = 2 * x * (x - 2) := 
by 
  sorry

-- Problem 2
theorem factorize_xy2_minus_2xy_plus_x (x y : ℝ) :
  x * y^2 - 2 * x * y + x = x * (y - 1)^2 :=
by 
  sorry

end NUMINAMATH_GPT_factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l1421_142180


namespace NUMINAMATH_GPT_quadratic_has_minimum_l1421_142193

theorem quadratic_has_minimum (a b : ℝ) (h : a > b^2) :
  ∃ (c : ℝ), c = (4 * b^2 / a) - 3 ∧ (∃ x : ℝ, a * x ^ 2 + 2 * b * x + c < 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_has_minimum_l1421_142193


namespace NUMINAMATH_GPT_algebraic_expression_value_l1421_142105

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9 * a * b = 27 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1421_142105


namespace NUMINAMATH_GPT_jordan_book_pages_l1421_142151

theorem jordan_book_pages (avg_first_4_days : ℕ)
                           (avg_next_2_days : ℕ)
                           (pages_last_day : ℕ)
                           (total_pages : ℕ) :
  avg_first_4_days = 42 → 
  avg_next_2_days = 38 → 
  pages_last_day = 20 → 
  total_pages = 4 * avg_first_4_days + 2 * avg_next_2_days + pages_last_day →
  total_pages = 264 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_jordan_book_pages_l1421_142151


namespace NUMINAMATH_GPT_richard_more_pins_than_patrick_l1421_142156

theorem richard_more_pins_than_patrick :
  ∀ (R P R2 P2 : ℕ), 
    P = 70 → 
    R > P →
    P2 = 2 * R →
    R2 = P2 - 3 → 
    (R + R2) = (P + P2) + 12 → 
    R = 70 + 15 := 
by 
  intros R P R2 P2 hP hRp hP2 hR2 hTotal
  sorry

end NUMINAMATH_GPT_richard_more_pins_than_patrick_l1421_142156


namespace NUMINAMATH_GPT_sin2alpha_cos2beta_l1421_142110

variable (α β : ℝ)

-- Conditions
def tan_add_eq : Prop := Real.tan (α + β) = -3
def tan_sub_eq : Prop := Real.tan (α - β) = 2

-- Question
theorem sin2alpha_cos2beta (h1 : tan_add_eq α β) (h2 : tan_sub_eq α β) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = -1 / 7 := 
  sorry

end NUMINAMATH_GPT_sin2alpha_cos2beta_l1421_142110


namespace NUMINAMATH_GPT_y_intercept_of_line_l1421_142143

theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 7) (h₃ : y₀ = 0) :
  ∃ (b : ℝ), (0, b) = (0, 21) :=
by
  -- Our goal is to prove the y-intercept is (0, 21)
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1421_142143


namespace NUMINAMATH_GPT_find_g_l1421_142187

-- Given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x - 10
def parameterization (g : ℝ → ℝ) (t : ℝ) : Prop := 20 * t - 8 = 2 * g t - 10

-- Statement to prove
theorem find_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ x y, line_equation x y → parameterization g t) →
  g t = 10 * t + 1 :=
sorry

end NUMINAMATH_GPT_find_g_l1421_142187


namespace NUMINAMATH_GPT_bags_on_monday_l1421_142153

/-- Define the problem conditions -/
def t : Nat := 8  -- total number of bags
def f : Nat := 4  -- number of bags found the next day

-- Define the statement to be proven
theorem bags_on_monday : t - f = 4 := by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_bags_on_monday_l1421_142153


namespace NUMINAMATH_GPT_ball_hits_ground_at_correct_time_l1421_142132

def initial_velocity : ℝ := 7
def initial_height : ℝ := 10

-- The height function as given by the condition
def height_function (t : ℝ) : ℝ := -4.9 * t^2 + initial_velocity * t + initial_height

-- Statement
theorem ball_hits_ground_at_correct_time :
  ∃ t : ℝ, height_function t = 0 ∧ t = 2313 / 1000 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_correct_time_l1421_142132


namespace NUMINAMATH_GPT_problem_probability_ao_drawn_second_l1421_142119

def is_ao_drawn_second (pair : ℕ × ℕ) : Bool :=
  pair.snd = 3

def random_pairs : List (ℕ × ℕ) := [
  (1, 3), (2, 4), (1, 2), (3, 2), (4, 3), (1, 4), (2, 4), (3, 2), (3, 1), (2, 1), 
  (2, 3), (1, 3), (3, 2), (2, 1), (2, 4), (4, 2), (1, 3), (3, 2), (2, 1), (3, 4)
]

def count_ao_drawn_second : ℕ :=
  (random_pairs.filter is_ao_drawn_second).length

def probability_ao_drawn_second : ℚ :=
  count_ao_drawn_second / random_pairs.length

theorem problem_probability_ao_drawn_second :
  probability_ao_drawn_second = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_probability_ao_drawn_second_l1421_142119


namespace NUMINAMATH_GPT_square_area_l1421_142160

theorem square_area (side_length : ℕ) (h : side_length = 17) : side_length * side_length = 289 :=
by sorry

end NUMINAMATH_GPT_square_area_l1421_142160


namespace NUMINAMATH_GPT_desks_in_classroom_l1421_142157

theorem desks_in_classroom (d c : ℕ) (h1 : c = 4 * d) (h2 : 4 * c + 6 * d = 728) : d = 33 :=
by
  -- The proof is omitted, this placeholder is to indicate that it is required to complete the proof.
  sorry

end NUMINAMATH_GPT_desks_in_classroom_l1421_142157


namespace NUMINAMATH_GPT_anna_candy_division_l1421_142140

theorem anna_candy_division : 
  ∀ (total_candies friends : ℕ), 
  total_candies = 30 → 
  friends = 4 → 
  ∃ (candies_to_remove : ℕ), 
  candies_to_remove = 2 ∧ 
  (total_candies - candies_to_remove) % friends = 0 := 
by
  sorry

end NUMINAMATH_GPT_anna_candy_division_l1421_142140


namespace NUMINAMATH_GPT_find_equation_of_tangent_line_l1421_142152

def is_tangent_at_point (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := 
  ∃ x y, (x - 1)^2 + (y + 2)^2 = 1 ∧ l x₀ y₀ ∧ l x y

def equation_of_line (l : ℝ → ℝ → Prop) := 
  ∀ x y, l x y ↔ (x = 2 ∨ 12 * x - 5 * y - 9 = 0)

theorem find_equation_of_tangent_line : 
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ (x - 1)^2 + (y + 2)^2 ≠ 1 ∧ (x, y) = (2,3))
  → is_tangent_at_point l 2 3
  → equation_of_line l := 
sorry

end NUMINAMATH_GPT_find_equation_of_tangent_line_l1421_142152


namespace NUMINAMATH_GPT_min_a_b_sum_l1421_142191

theorem min_a_b_sum (a b : ℕ) (x : ℕ → ℕ)
  (h0 : x 1 = a)
  (h1 : x 2 = b)
  (h2 : ∀ n, x (n+2) = x n + x (n+1))
  (h3 : ∃ n, x n = 1000) : a + b = 10 :=
sorry

end NUMINAMATH_GPT_min_a_b_sum_l1421_142191


namespace NUMINAMATH_GPT_sum_of_a_equals_five_l1421_142188

theorem sum_of_a_equals_five
  (f : ℕ → ℕ → ℕ)  -- Represents the function f defined by Table 1
  (a : ℕ → ℕ)  -- Represents the occurrences a₀, a₁, ..., a₄
  (h1 : a 0 + a 1 + a 2 + a 3 + a 4 = 5)  -- Condition 1
  (h2 : 0 * a 0 + 1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 = 5)  -- Condition 2
  : a 0 + a 1 + a 2 + a 3 = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_a_equals_five_l1421_142188


namespace NUMINAMATH_GPT_percentage_of_stock_l1421_142102

noncomputable def investment_amount : ℝ := 6000
noncomputable def income_derived : ℝ := 756
noncomputable def brokerage_percentage : ℝ := 0.25
noncomputable def brokerage_fee : ℝ := investment_amount * (brokerage_percentage / 100)
noncomputable def net_investment_amount : ℝ := investment_amount - brokerage_fee
noncomputable def dividend_yield : ℝ := (income_derived / net_investment_amount) * 100

theorem percentage_of_stock :
  ∃ (percentage_of_stock : ℝ), percentage_of_stock = dividend_yield := by
  sorry

end NUMINAMATH_GPT_percentage_of_stock_l1421_142102


namespace NUMINAMATH_GPT_kyle_and_miles_total_marble_count_l1421_142183

noncomputable def kyle_marble_count (F : ℕ) (K : ℕ) : Prop :=
  F = 4 * K

noncomputable def miles_marble_count (F : ℕ) (M : ℕ) : Prop :=
  F = 9 * M

theorem kyle_and_miles_total_marble_count :
  ∀ (F K M : ℕ), F = 36 → kyle_marble_count F K → miles_marble_count F M → K + M = 13 :=
by
  intros F K M hF hK hM
  sorry

end NUMINAMATH_GPT_kyle_and_miles_total_marble_count_l1421_142183


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1421_142178

noncomputable def side1 := 13
noncomputable def side2 := 13
noncomputable def side3 := 10
noncomputable def s := (side1 + side2 + side3) / 2
noncomputable def area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
noncomputable def inradius := area / s

theorem inscribed_circle_radius :
  inradius = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1421_142178


namespace NUMINAMATH_GPT_number_multiplied_by_approx_l1421_142129

variable (X : ℝ)

theorem number_multiplied_by_approx (h : (0.0048 * X) / (0.05 * 0.1 * 0.004) = 840) : X = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_number_multiplied_by_approx_l1421_142129


namespace NUMINAMATH_GPT_number_of_b_values_l1421_142130

theorem number_of_b_values (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), ∀ (x : ℤ), x^2 + b * x + 6 ≤ 0 ↔ x = x1 ∨ x = x2 ∨ x = x3) ↔ (b = -6 ∨ b = -5 ∨ b = 5 ∨ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_b_values_l1421_142130


namespace NUMINAMATH_GPT_all_edges_same_color_l1421_142171

-- Define the vertices in the two pentagons and the set of all vertices
inductive vertex
| A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5
open vertex

-- Predicate to identify edges between vertices
def edge (v1 v2 : vertex) : Prop :=
  match (v1, v2) with
  | (A1, A2) | (A2, A3) | (A3, A4) | (A4, A5) | (A5, A1) => true
  | (B1, B2) | (B2, B3) | (B3, B4) | (B4, B5) | (B5, B1) => true
  | (A1, B1) | (A1, B2) | (A1, B3) | (A1, B4) | (A1, B5) => true
  | (A2, B1) | (A2, B2) | (A2, B3) | (A2, B4) | (A2, B5) => true
  | (A3, B1) | (A3, B2) | (A3, B3) | (A3, B4) | (A3, B5) => true
  | (A4, B1) | (A4, B2) | (A4, B3) | (A4, B4) | (A4, B5) => true
  | (A5, B1) | (A5, B2) | (A5, B3) | (A5, B4) | (A5, B5) => true
  | _ => false

-- Edge coloring predicate 'black' or 'white'
inductive color
| black | white
open color

def edge_color (v1 v2 : vertex) : color → Prop :=
  sorry -- Coloring function needs to be defined accordingly

-- Predicate to check for monochrome triangles
def no_monochrome_triangle : Prop :=
  ∀ v1 v2 v3 : vertex,
    (edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v1) →
    ¬ (∃ c : color, edge_color v1 v2 c ∧ edge_color v2 v3 c ∧ edge_color v3 v1 c)

-- Main theorem statement
theorem all_edges_same_color (no_mt : no_monochrome_triangle) :
  ∃ c : color, ∀ v1 v2 : vertex,
    (edge v1 v2 ∧ (v1 = A1 ∨ v1 = A2 ∨ v1 = A3 ∨ v1 = A4 ∨ v1 = A5) ∧
                 (v2 = A1 ∨ v2 = A2 ∨ v2 = A3 ∨ v2 = A4 ∨ v2 = A5) ) →
    edge_color v1 v2 c ∧
    (edge v1 v2 ∧ (v1 = B1 ∨ v1 = B2 ∨ v1 = B3 ∨ v1 = B4 ∨ v1 = B5) ∧
                 (v2 = B1 ∨ v2 = B2 ∨ v2 = B3 ∨ v2 = B4 ∨ v2 = B5) ) →
    edge_color v1 v2 c := sorry

end NUMINAMATH_GPT_all_edges_same_color_l1421_142171


namespace NUMINAMATH_GPT_new_class_mean_l1421_142147

theorem new_class_mean 
  (n1 n2 : ℕ) 
  (mean1 mean2 : ℝ)
  (students_total : ℕ)
  (total_score1 total_score2 : ℝ)
  (h1 : n1 = 45)
  (h2 : n2 = 5)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : students_total = 50)
  (h6 : total_score1 = n1 * mean1)
  (h7 : total_score2 = n2 * mean2) :
  (total_score1 + total_score2) / students_total = 81 :=
by
  sorry

end NUMINAMATH_GPT_new_class_mean_l1421_142147


namespace NUMINAMATH_GPT_inequality_selection_l1421_142122

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := 
by sorry

end NUMINAMATH_GPT_inequality_selection_l1421_142122


namespace NUMINAMATH_GPT_angle_in_first_quadrant_l1421_142169

-- Define the condition and equivalence proof problem in Lean 4
theorem angle_in_first_quadrant (deg : ℤ) (h1 : deg = 721) : (deg % 360) > 0 := 
by 
  have : deg % 360 = 1 := sorry
  exact sorry

end NUMINAMATH_GPT_angle_in_first_quadrant_l1421_142169


namespace NUMINAMATH_GPT_selection_methods_l1421_142137

-- Conditions
def volunteers : ℕ := 5
def friday_slots : ℕ := 1
def saturday_slots : ℕ := 2
def sunday_slots : ℕ := 1

-- Function to calculate combinatorial n choose k
def choose (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Function to calculate permutations of n P k
def perm (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

-- The target proposition
theorem selection_methods : choose volunteers saturday_slots * perm (volunteers - saturday_slots) (friday_slots + sunday_slots) = 60 :=
by
  -- assumption here leads to the property required, usually this would be more detailed computation.
  sorry

end NUMINAMATH_GPT_selection_methods_l1421_142137


namespace NUMINAMATH_GPT_branches_number_l1421_142150

-- Conditions (converted into Lean definitions)
def total_leaves : ℕ := 12690
def twigs_per_branch : ℕ := 90
def leaves_per_twig_percentage_4 : ℝ := 0.3
def leaves_per_twig_percentage_5 : ℝ := 0.7
def leaves_per_twig_4 : ℕ := 4
def leaves_per_twig_5 : ℕ := 5

-- The goal
theorem branches_number (B : ℕ) 
  (h1 : twigs_per_branch = 90) 
  (h2 : leaves_per_twig_percentage_4 = 0.3) 
  (h3 : leaves_per_twig_percentage_5 = 0.7) 
  (h4 : leaves_per_twig_4 = 4) 
  (h5 : leaves_per_twig_5 = 5) 
  (h6 : total_leaves = 12690) :
  B = 30 := 
sorry

end NUMINAMATH_GPT_branches_number_l1421_142150


namespace NUMINAMATH_GPT_num_ways_write_100_as_distinct_squares_l1421_142101

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end NUMINAMATH_GPT_num_ways_write_100_as_distinct_squares_l1421_142101


namespace NUMINAMATH_GPT_f_value_at_5pi_over_6_l1421_142163

noncomputable def f (x ω : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 3))

theorem f_value_at_5pi_over_6
  (ω : ℝ) (ω_pos : ω > 0)
  (α β : ℝ)
  (h1 : f α ω = 2)
  (h2 : f β ω = 0)
  (h3 : Real.sqrt ((α - β)^2 + 4) = Real.sqrt (4 + (Real.pi^2 / 4))) :
  f (5 * Real.pi / 6) ω = -1 := 
sorry

end NUMINAMATH_GPT_f_value_at_5pi_over_6_l1421_142163


namespace NUMINAMATH_GPT_range_function_1_l1421_142146

theorem range_function_1 (y : ℝ) : 
  (∃ x : ℝ, x ≥ -1 ∧ y = (1/3) ^ x) ↔ (0 < y ∧ y ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_function_1_l1421_142146


namespace NUMINAMATH_GPT_find_b_when_a_is_negative12_l1421_142190

theorem find_b_when_a_is_negative12 (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∃ k, a * b = k) : b = -56.25 :=
sorry

end NUMINAMATH_GPT_find_b_when_a_is_negative12_l1421_142190


namespace NUMINAMATH_GPT_chemistry_marks_l1421_142194

-- Definitions based on given conditions
def total_marks (P C M : ℕ) : Prop := P + C + M = 210
def avg_physics_math (P M : ℕ) : Prop := (P + M) / 2 = 90
def physics_marks (P : ℕ) : Prop := P = 110
def avg_physics_other_subject (P C : ℕ) : Prop := (P + C) / 2 = 70

-- The proof problem statement
theorem chemistry_marks {P C M : ℕ} (h1 : total_marks P C M) (h2 : avg_physics_math P M) (h3 : physics_marks P) : C = 30 ∧ avg_physics_other_subject P C :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_chemistry_marks_l1421_142194


namespace NUMINAMATH_GPT_acrobat_count_range_l1421_142128

def animal_legs (elephants monkeys acrobats : ℕ) : ℕ :=
  4 * elephants + 2 * monkeys + 2 * acrobats

def animal_heads (elephants monkeys acrobats : ℕ) : ℕ :=
  elephants + monkeys + acrobats

theorem acrobat_count_range (e m a : ℕ) (h1 : animal_heads e m a = 18)
  (h2 : animal_legs e m a = 50) : 0 ≤ a ∧ a ≤ 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_acrobat_count_range_l1421_142128


namespace NUMINAMATH_GPT_polygon_sides_twice_diagonals_l1421_142177

theorem polygon_sides_twice_diagonals (n : ℕ) (h1 : n ≥ 3) (h2 : n * (n - 3) / 2 = 2 * n) : n = 7 :=
sorry

end NUMINAMATH_GPT_polygon_sides_twice_diagonals_l1421_142177


namespace NUMINAMATH_GPT_polar_to_rect_l1421_142159

theorem polar_to_rect (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (2.5, 5 * Real.sqrt 3 / 2) :=
by
  rw [hr, hθ]
  sorry

end NUMINAMATH_GPT_polar_to_rect_l1421_142159


namespace NUMINAMATH_GPT_solve_for_a_l1421_142181

theorem solve_for_a (a : ℝ) (h : 2 * a + (1 - 4 * a) = 0) : a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1421_142181


namespace NUMINAMATH_GPT_corn_plants_multiple_of_nine_l1421_142125

theorem corn_plants_multiple_of_nine 
  (num_sunflowers : ℕ) (num_tomatoes : ℕ) (num_corn : ℕ) (max_plants_per_row : ℕ)
  (h1 : num_sunflowers = 45) (h2 : num_tomatoes = 63) (h3 : max_plants_per_row = 9)
  : ∃ k : ℕ, num_corn = 9 * k :=
by
  sorry

end NUMINAMATH_GPT_corn_plants_multiple_of_nine_l1421_142125


namespace NUMINAMATH_GPT_age_difference_l1421_142127

/-- 
The overall age of x and y is some years greater than the overall age of y and z. Z is 12 years younger than X.
Prove: The overall age of x and y is 12 years greater than the overall age of y and z.
-/
theorem age_difference {X Y Z : ℕ} (h1: X + Y > Y + Z) (h2: Z = X - 12) : 
  (X + Y) - (Y + Z) = 12 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_age_difference_l1421_142127


namespace NUMINAMATH_GPT_arccos_cos_11_eq_l1421_142174

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end NUMINAMATH_GPT_arccos_cos_11_eq_l1421_142174


namespace NUMINAMATH_GPT_tan_difference_l1421_142124

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4 / 3) :
  Real.tan (α - β) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_difference_l1421_142124


namespace NUMINAMATH_GPT_spring_summer_work_hours_l1421_142141

def john_works_spring_summer : Prop :=
  ∀ (work_hours_winter_week : ℕ) (weeks_winter : ℕ) (earnings_winter : ℕ)
    (weeks_spring_summer : ℕ) (earnings_spring_summer : ℕ) (hourly_rate : ℕ),
    work_hours_winter_week = 40 →
    weeks_winter = 8 →
    earnings_winter = 3200 →
    weeks_spring_summer = 24 →
    earnings_spring_summer = 4800 →
    hourly_rate = earnings_winter / (work_hours_winter_week * weeks_winter) →
    (earnings_spring_summer / hourly_rate) / weeks_spring_summer = 20

theorem spring_summer_work_hours : john_works_spring_summer :=
  sorry

end NUMINAMATH_GPT_spring_summer_work_hours_l1421_142141


namespace NUMINAMATH_GPT_ratio_37m48s_2h13m15s_l1421_142195

-- Define the total seconds for 37 minutes and 48 seconds
def t1 := 37 * 60 + 48

-- Define the total seconds for 2 hours, 13 minutes, and 15 seconds
def t2 := 2 * 3600 + 13 * 60 + 15

-- Prove the ratio t1 / t2 = 2268 / 7995
theorem ratio_37m48s_2h13m15s : t1 / t2 = 2268 / 7995 := 
by sorry

end NUMINAMATH_GPT_ratio_37m48s_2h13m15s_l1421_142195


namespace NUMINAMATH_GPT_center_cell_value_l1421_142184

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end NUMINAMATH_GPT_center_cell_value_l1421_142184


namespace NUMINAMATH_GPT_vertex_parabola_shape_l1421_142117

theorem vertex_parabola_shape
  (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (P : ℝ → ℝ → Prop), 
  (∀ t : ℝ, ∃ (x y : ℝ), P x y ∧ (x = (-t / (2 * a))) ∧ (y = -a * (x^2) + d)) ∧
  (∀ x y : ℝ, P x y ↔ (y = -a * (x^2) + d)) :=
by
  sorry

end NUMINAMATH_GPT_vertex_parabola_shape_l1421_142117


namespace NUMINAMATH_GPT_problem8x_eq_5_200timesreciprocal_l1421_142131

theorem problem8x_eq_5_200timesreciprocal (x : ℚ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := 
by 
  sorry

end NUMINAMATH_GPT_problem8x_eq_5_200timesreciprocal_l1421_142131


namespace NUMINAMATH_GPT_max_movies_watched_l1421_142179

-- Conditions given in the problem
def movie_duration : Nat := 90
def tuesday_minutes : Nat := 4 * 60 + 30
def tuesday_movies : Nat := tuesday_minutes / movie_duration
def wednesday_movies : Nat := 2 * tuesday_movies

-- Problem statement: Total movies watched in two days
theorem max_movies_watched : 
  tuesday_movies + wednesday_movies = 9 := 
by
  -- We add the placeholder for the proof here
  sorry

end NUMINAMATH_GPT_max_movies_watched_l1421_142179


namespace NUMINAMATH_GPT_zhang_san_not_losing_probability_l1421_142165

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end NUMINAMATH_GPT_zhang_san_not_losing_probability_l1421_142165


namespace NUMINAMATH_GPT_cos_neg_13pi_div_4_l1421_142106

theorem cos_neg_13pi_div_4 : (Real.cos (-13 * Real.pi / 4)) = -Real.sqrt 2 / 2 := 
by sorry

end NUMINAMATH_GPT_cos_neg_13pi_div_4_l1421_142106


namespace NUMINAMATH_GPT_bruce_money_left_l1421_142133

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end NUMINAMATH_GPT_bruce_money_left_l1421_142133


namespace NUMINAMATH_GPT_distance_between_trees_l1421_142134

theorem distance_between_trees (l : ℕ) (n : ℕ) (d : ℕ) (h_length : l = 225) (h_trees : n = 26) (h_segments : n - 1 = 25) : d = 9 :=
sorry

end NUMINAMATH_GPT_distance_between_trees_l1421_142134


namespace NUMINAMATH_GPT_find_larger_number_l1421_142173

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l1421_142173


namespace NUMINAMATH_GPT_actual_time_when_car_clock_shows_10PM_l1421_142135

def car_clock_aligned (aligned_time wristwatch_time : ℕ) : Prop :=
  aligned_time = wristwatch_time

def car_clock_time (rate: ℚ) (hours_elapsed_real_time hours_elapsed_car_time : ℚ) : Prop :=
  rate = hours_elapsed_car_time / hours_elapsed_real_time

def actual_time (current_car_time car_rate : ℚ) : ℚ :=
  current_car_time / car_rate

theorem actual_time_when_car_clock_shows_10PM :
  let accurate_start_time := 9 -- 9:00 AM
  let car_start_time := 9 -- Synchronized at 9:00 AM
  let wristwatch_time_wristwatch := 13 -- 1:00 PM in hours
  let car_time_car := 13 + 48 / 60 -- 1:48 PM in hours
  let rate := car_time_car / wristwatch_time_wristwatch
  let current_car_time := 22 -- 10:00 PM in hours
  let real_time := actual_time current_car_time rate
  real_time = 19.8333 := -- which converts to 7:50 PM (Option B)
sorry

end NUMINAMATH_GPT_actual_time_when_car_clock_shows_10PM_l1421_142135


namespace NUMINAMATH_GPT_margo_walks_total_distance_l1421_142120

theorem margo_walks_total_distance :
  let time_to_house := 15
  let time_to_return := 25
  let total_time_minutes := time_to_house + time_to_return
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let avg_rate := 3  -- units: miles per hour
  (avg_rate * total_time_hours = 2) := 
sorry

end NUMINAMATH_GPT_margo_walks_total_distance_l1421_142120


namespace NUMINAMATH_GPT_michael_total_fish_l1421_142112

-- Definitions based on conditions
def michael_original_fish : ℕ := 31
def ben_fish_given : ℕ := 18

-- Theorem to prove the total number of fish Michael has now
theorem michael_total_fish : (michael_original_fish + ben_fish_given) = 49 :=
by sorry

end NUMINAMATH_GPT_michael_total_fish_l1421_142112

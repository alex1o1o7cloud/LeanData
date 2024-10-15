import Mathlib

namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1260_126082

/-- The equation of the common chord of two circles -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 - 4*x - 3 = 0) ∧ (x^2 + y^2 - 4*y - 3 = 0) → (x - y = 0) := by
  sorry

#check common_chord_of_circles

end NUMINAMATH_CALUDE_common_chord_of_circles_l1260_126082


namespace NUMINAMATH_CALUDE_cardinality_difference_constant_l1260_126034

/-- Given a finite set of positive integers, S_n is the set of all sums of exactly n elements from the set -/
def S_n (A : Finset Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem stating the existence of N and k -/
theorem cardinality_difference_constant (A : Finset Nat) :
  ∃ (N k : Nat), ∀ n ≥ N, (S_n A (n + 1)).card = (S_n A n).card + k :=
sorry

end NUMINAMATH_CALUDE_cardinality_difference_constant_l1260_126034


namespace NUMINAMATH_CALUDE_pins_purchased_proof_l1260_126040

/-- Calculates the number of pins purchased given the original price, discount percentage, and total amount spent. -/
def calculate_pins_purchased (original_price : ℚ) (discount_percent : ℚ) (total_spent : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  total_spent / discounted_price

/-- Proves that purchasing pins at a 15% discount from $20 each, spending $170 results in 10 pins. -/
theorem pins_purchased_proof :
  calculate_pins_purchased 20 15 170 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pins_purchased_proof_l1260_126040


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1260_126048

theorem quadratic_form_ratio (x : ℝ) : ∃ b c : ℝ, 
  x^2 + 500*x + 1000 = (x + b)^2 + c ∧ c / b = -246 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1260_126048


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1260_126041

theorem cost_price_percentage (cost_price selling_price : ℝ) 
  (h : selling_price = 4 * cost_price) : 
  cost_price / selling_price = 1 / 4 := by
  sorry

#check cost_price_percentage

end NUMINAMATH_CALUDE_cost_price_percentage_l1260_126041


namespace NUMINAMATH_CALUDE_gcd_values_count_l1260_126016

theorem gcd_values_count (a b : ℕ+) (h : Nat.gcd a.val b.val * Nat.lcm a.val b.val = 180) :
  ∃ S : Finset ℕ+, (∀ x ∈ S, x = Nat.gcd a.val b.val) ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_values_count_l1260_126016


namespace NUMINAMATH_CALUDE_least_k_value_l1260_126033

theorem least_k_value (a b c d : ℝ) : 
  ∃ k : ℝ, k = 4 ∧ 
  (∀ a b c d : ℝ, 
    Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
    Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
    Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
    Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) ≥ 
    2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ a b c d : ℝ, 
      Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
      Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
      Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
      Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) < 
      2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k') :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l1260_126033


namespace NUMINAMATH_CALUDE_problem_statement_l1260_126035

theorem problem_statement :
  (¬ (∃ x : ℝ, x^2 - x + 1 < 0)) ∧
  (¬ (∀ x : ℝ, x^2 - 4 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1260_126035


namespace NUMINAMATH_CALUDE_expression_simplification_l1260_126054

theorem expression_simplification (x y : ℝ) :
  x * (4 * x^3 - 3 * x^2 + 2 * y) - 6 * (x^3 - 3 * x^2 + 2 * x + 8) =
  4 * x^4 - 9 * x^3 + 18 * x^2 + 2 * x * y - 12 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1260_126054


namespace NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l1260_126090

/-- Given two similar right triangles, where the first triangle has a side of 15 and a hypotenuse of 17,
    and the second triangle has a hypotenuse of 102, the shortest side of the second triangle is 48. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a ^ 2 + 15 ^ 2 = 17 ^ 2 → -- First triangle is right-angled with side 15 and hypotenuse 17
  a ≤ 15 → -- a is the shortest side of the first triangle
  ∃ (k : ℝ), k > 0 ∧ k * 17 = 102 ∧ k * a = 48 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l1260_126090


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l1260_126089

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l1260_126089


namespace NUMINAMATH_CALUDE_series_convergence_l1260_126057

/-- The series ∑(n=1 to ∞) [x^(2n-1) / ((n^2 + 1) * 3^n)] converges absolutely if and only if -√3 ≤ x ≤ √3 -/
theorem series_convergence (x : ℝ) : 
  (∑' n, (x^(2*n-1) / ((n^2 + 1) * 3^n))) ≠ 0 ↔ -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l1260_126057


namespace NUMINAMATH_CALUDE_field_length_difference_l1260_126045

/-- 
Given a rectangular field with length 24 meters and width 13.5 meters,
prove that the difference between twice the width and the length is 3 meters.
-/
theorem field_length_difference (length width : ℝ) 
  (h1 : length = 24)
  (h2 : width = 13.5) :
  2 * width - length = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_length_difference_l1260_126045


namespace NUMINAMATH_CALUDE_sequence_problem_l1260_126032

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) 
    (h_geo : geometric_sequence a)
    (h_arith : arithmetic_sequence b)
    (h_a : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
    (h_b : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1260_126032


namespace NUMINAMATH_CALUDE_angle_value_l1260_126058

def A (θ : ℝ) : Set ℝ := {1, Real.cos θ}
def B : Set ℝ := {0, 1/2, 1}

theorem angle_value (θ : ℝ) (h1 : A θ ⊆ B) (h2 : 0 < θ ∧ θ < π / 2) : θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l1260_126058


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l1260_126022

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : IncreasingFunction f) (h_sum_positive : a + b > 0) :
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l1260_126022


namespace NUMINAMATH_CALUDE_distance_traveled_l1260_126099

/-- Given a speed of 75 km/hr and a time of 4 hours, prove that the distance traveled is 300 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 75) (h2 : time = 4) :
  speed * time = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1260_126099


namespace NUMINAMATH_CALUDE_circle_area_with_complex_conditions_l1260_126062

theorem circle_area_with_complex_conditions (z₁ z₂ : ℂ) 
  (h1 : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0)
  (h2 : Complex.abs z₂ = 2) :
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_complex_conditions_l1260_126062


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1260_126002

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := 2 * x / (x + 1) ≥ 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x | 1/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | x < -1 ∨ x ≥ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -1 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1260_126002


namespace NUMINAMATH_CALUDE_houses_with_neither_amenity_l1260_126019

/-- Given a development with houses, some of which have a two-car garage and/or an in-the-ground swimming pool, 
    this theorem proves the number of houses with neither amenity. -/
theorem houses_with_neither_amenity 
  (total : ℕ) 
  (garage : ℕ) 
  (pool : ℕ) 
  (both : ℕ) 
  (h1 : total = 90) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 35 := by
  sorry


end NUMINAMATH_CALUDE_houses_with_neither_amenity_l1260_126019


namespace NUMINAMATH_CALUDE_unoccupied_volume_correct_l1260_126087

/-- Represents the dimensions of a cube in inches -/
structure CubeDimensions where
  side : ℝ

/-- Calculates the volume of a cube given its dimensions -/
def cubeVolume (d : CubeDimensions) : ℝ := d.side ^ 3

/-- Represents the container and its contents -/
structure Container where
  dimensions : CubeDimensions
  waterFillRatio : ℝ
  iceCubes : ℕ
  iceCubeDimensions : CubeDimensions

/-- Calculates the unoccupied volume in the container -/
def unoccupiedVolume (c : Container) : ℝ :=
  let containerVolume := cubeVolume c.dimensions
  let waterVolume := c.waterFillRatio * containerVolume
  let iceCubeVolume := cubeVolume c.iceCubeDimensions
  let totalIceVolume := c.iceCubes * iceCubeVolume
  containerVolume - waterVolume - totalIceVolume

/-- The main theorem to prove -/
theorem unoccupied_volume_correct (c : Container) : 
  c.dimensions.side = 12 ∧ 
  c.waterFillRatio = 3/4 ∧ 
  c.iceCubes = 6 ∧ 
  c.iceCubeDimensions.side = 1.5 → 
  unoccupiedVolume c = 411.75 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_volume_correct_l1260_126087


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l1260_126072

theorem p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → |x| ≤ 3) ∧
  (∃ x : ℝ, |x| ≤ 3 ∧ x^2 + 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l1260_126072


namespace NUMINAMATH_CALUDE_large_envelopes_count_l1260_126088

theorem large_envelopes_count (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ) : 
  total_letters = 80 →
  small_envelope_letters = 20 →
  letters_per_large_envelope = 2 →
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 := by
sorry

end NUMINAMATH_CALUDE_large_envelopes_count_l1260_126088


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l1260_126003

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both_traits ≥ 3 →
  N ≤ 27 :=
by
  sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l1260_126003


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_a_squared_greater_b_squared_l1260_126029

theorem sufficiency_not_necessity_a_squared_greater_b_squared (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_a_squared_greater_b_squared_l1260_126029


namespace NUMINAMATH_CALUDE_gasoline_reduction_l1260_126050

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.20 * P
  let new_total_cost := 1.08 * (P * Q)
  let new_quantity := new_total_cost / new_price
  (Q - new_quantity) / Q = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_reduction_l1260_126050


namespace NUMINAMATH_CALUDE_stream_speed_l1260_126086

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 10 →
  distance = 90 →
  time = 5 →
  ∃ (stream_speed : ℝ), 
    distance = (rowing_speed + stream_speed) * time ∧
    stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1260_126086


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l1260_126039

/-- Calculates the remaining money after grocery shopping --/
def remaining_money (initial_amount : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (sauce_quantity : ℝ) : ℝ :=
  initial_amount - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 after shopping --/
theorem jerry_remaining_money :
  remaining_money 50 13 2 4 3 5 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l1260_126039


namespace NUMINAMATH_CALUDE_cookies_per_box_l1260_126037

/-- Proof of the number of cookies per box in Brenda's banana pudding problem -/
theorem cookies_per_box 
  (num_trays : ℕ) 
  (cookies_per_tray : ℕ) 
  (cost_per_box : ℚ) 
  (total_cost : ℚ) 
  (h1 : num_trays = 3)
  (h2 : cookies_per_tray = 80)
  (h3 : cost_per_box = 7/2)
  (h4 : total_cost = 14) :
  (num_trays * cookies_per_tray) / (total_cost / cost_per_box) = 60 := by
  sorry

#eval (3 * 80) / (14 / (7/2)) -- Should evaluate to 60

end NUMINAMATH_CALUDE_cookies_per_box_l1260_126037


namespace NUMINAMATH_CALUDE_tangent_line_to_two_curves_l1260_126056

/-- A line y = kx + t is tangent to both curves y = exp x + 2 and y = exp (x + 1) -/
theorem tangent_line_to_two_curves (k t : ℝ) : 
  (∃ x₁ : ℝ, k * x₁ + t = Real.exp x₁ + 2 ∧ k = Real.exp x₁) →
  (∃ x₂ : ℝ, k * x₂ + t = Real.exp (x₂ + 1) ∧ k = Real.exp (x₂ + 1)) →
  t = 4 - 2 * Real.log 2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_to_two_curves_l1260_126056


namespace NUMINAMATH_CALUDE_probability_at_least_one_grade_12_l1260_126028

def total_sample_size : ℕ := 6
def grade_10_size : ℕ := 54
def grade_11_size : ℕ := 18
def grade_12_size : ℕ := 36

def grade_10_selected : ℕ := 3
def grade_11_selected : ℕ := 1
def grade_12_selected : ℕ := 2

def selected_size : ℕ := 3

theorem probability_at_least_one_grade_12 :
  let total_combinations := Nat.choose total_sample_size selected_size
  let favorable_combinations := total_combinations - Nat.choose (total_sample_size - grade_12_selected) selected_size
  (favorable_combinations : ℚ) / total_combinations = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_grade_12_l1260_126028


namespace NUMINAMATH_CALUDE_simplify_expression_l1260_126008

-- Define the expression
def expression (x y : ℝ) : ℝ := (15*x + 45*y) + (7*x + 18*y) - (6*x + 35*y)

-- State the theorem
theorem simplify_expression :
  ∀ x y : ℝ, expression x y = 16*x + 28*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1260_126008


namespace NUMINAMATH_CALUDE_factorization_proof_l1260_126077

theorem factorization_proof (z : ℝ) : 
  88 * z^19 + 176 * z^38 + 264 * z^57 = 88 * z^19 * (1 + 2 * z^19 + 3 * z^38) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l1260_126077


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1260_126051

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 5 * x + c = 0 ↔ x = (-5 + Real.sqrt 21) / 4 ∨ x = (-5 - Real.sqrt 21) / 4) →
  c = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1260_126051


namespace NUMINAMATH_CALUDE_perpendicular_condition_acute_angle_condition_l1260_126014

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acute_angle (u v : Fin 2 → ℝ) : Prop := dot_product u v > 0

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

theorem perpendicular_condition (x : ℝ) : 
  perpendicular (λ i => a i + 2 * b x i) (λ i => 2 * a i - b x i) ↔ x = -2 ∨ x = 7/2 := by
  sorry

theorem acute_angle_condition (x : ℝ) :
  acute_angle a (b x) ∧ ¬ parallel a (b x) ↔ x > -2 ∧ x ≠ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_acute_angle_condition_l1260_126014


namespace NUMINAMATH_CALUDE_inequality_theorem_l1260_126049

theorem inequality_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z ≥ 9*(x*y + y*z + z*x) ∧
  ((x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z = 9*(x*y + y*z + z*x) ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1260_126049


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1260_126064

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3) / 2 : ℚ) = 2 * n ∧ 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1260_126064


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1260_126046

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), (∀ n ∈ S, (Real.sqrt (n + 1) ≤ Real.sqrt (3 * n + 2) ∧ 
    Real.sqrt (3 * n + 2) < Real.sqrt (2 * n + 7))) ∧ 
    S.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1260_126046


namespace NUMINAMATH_CALUDE_kayla_driving_years_l1260_126070

/-- The minimum driving age in Kayla's state -/
def minimum_driving_age : ℕ := 18

/-- Kimiko's age -/
def kimiko_age : ℕ := 26

/-- Kayla's current age -/
def kayla_age : ℕ := kimiko_age / 2

/-- The number of years before Kayla can reach the minimum driving age -/
def years_until_driving : ℕ := minimum_driving_age - kayla_age

theorem kayla_driving_years :
  years_until_driving = 5 :=
by sorry

end NUMINAMATH_CALUDE_kayla_driving_years_l1260_126070


namespace NUMINAMATH_CALUDE_min_area_quadrilateral_l1260_126065

/-- Given a rectangle ABCD with points A₁, B₁, C₁, D₁ on the rays AB, BC, CD, DA respectively,
    such that AA₁/AB = BB₁/BC = CC₁/CD = DD₁/DA = k > 0,
    prove that the area of quadrilateral A₁B₁C₁D₁ is minimized when k = 1/2 -/
theorem min_area_quadrilateral (a b : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  let area := a * b * (1 - k + k^2)
  (∀ k' > 0, area ≤ a * b * (1 - k' + k'^2)) ↔ k = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_min_area_quadrilateral_l1260_126065


namespace NUMINAMATH_CALUDE_fraction_addition_l1260_126018

theorem fraction_addition (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) :
  (a + b) / b = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1260_126018


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1260_126007

/-- Calculates the percentage of valid votes a candidate received in an election. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_vote_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_vote_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 380800) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_vote_percentage) * total_votes) = 80 / 100 := by
sorry


end NUMINAMATH_CALUDE_candidate_vote_percentage_l1260_126007


namespace NUMINAMATH_CALUDE_equation_solution_l1260_126085

theorem equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (1 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 10 + 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1260_126085


namespace NUMINAMATH_CALUDE_brian_shirts_l1260_126074

theorem brian_shirts (steven_shirts andrew_shirts brian_shirts : ℕ) 
  (h1 : steven_shirts = 4 * andrew_shirts)
  (h2 : andrew_shirts = 6 * brian_shirts)
  (h3 : steven_shirts = 72) : 
  brian_shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_brian_shirts_l1260_126074


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1260_126013

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 969 := by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1260_126013


namespace NUMINAMATH_CALUDE_frog_eggs_eaten_percentage_l1260_126079

theorem frog_eggs_eaten_percentage
  (total_eggs : ℕ)
  (dry_up_percentage : ℚ)
  (hatch_fraction : ℚ)
  (hatched_frogs : ℕ)
  (h_total_eggs : total_eggs = 800)
  (h_dry_up_percentage : dry_up_percentage = 1 / 10)
  (h_hatch_fraction : hatch_fraction = 1 / 4)
  (h_hatched_frogs : hatched_frogs = 40) :
  (total_eggs : ℚ) * (1 - dry_up_percentage - hatch_fraction * (1 - dry_up_percentage)) = 70 / 100 * total_eggs :=
sorry

end NUMINAMATH_CALUDE_frog_eggs_eaten_percentage_l1260_126079


namespace NUMINAMATH_CALUDE_gcd_1887_2091_l1260_126075

theorem gcd_1887_2091 : Nat.gcd 1887 2091 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1887_2091_l1260_126075


namespace NUMINAMATH_CALUDE_specific_student_not_front_l1260_126047

/-- The number of ways to arrange n students in a line. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with a specific student at the front. -/
def arrangementsWithSpecificFront (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of students. -/
def numStudents : ℕ := 5

theorem specific_student_not_front :
  arrangements numStudents - arrangementsWithSpecificFront numStudents = 96 :=
sorry

end NUMINAMATH_CALUDE_specific_student_not_front_l1260_126047


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1260_126063

theorem complex_fraction_simplification :
  1007 * ((7/4 / (3/4) + 3 / (9/4) + 1/3) / ((1+2+3+4+5) * 5 - 22)) / 19 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1260_126063


namespace NUMINAMATH_CALUDE_garment_pricing_problem_l1260_126001

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 400

-- Define the profit function without donation
def profit_function (x : ℝ) : ℝ := (x - 60) * (sales_function x)

-- Define the profit function with donation
def profit_function_with_donation (x : ℝ) : ℝ := (x - 70) * (sales_function x)

theorem garment_pricing_problem :
  -- The linear function fits the given data points
  (sales_function 80 = 240) ∧
  (sales_function 90 = 220) ∧
  (sales_function 100 = 200) ∧
  (sales_function 110 = 180) ∧
  -- The smaller solution to the profit equation is 100
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    profit_function x₁ = 8000 ∧ 
    profit_function x₂ = 8000 ∧ 
    x₁ = 100) ∧
  -- The profit function with donation has a maximum at 135
  (∃ max_profit : ℝ, 
    profit_function_with_donation 135 = max_profit ∧
    ∀ x : ℝ, profit_function_with_donation x ≤ max_profit) :=
by sorry

end NUMINAMATH_CALUDE_garment_pricing_problem_l1260_126001


namespace NUMINAMATH_CALUDE_grandmother_age_problem_l1260_126080

theorem grandmother_age_problem (yuna_initial_age grandmother_initial_age : ℕ) 
  (h1 : yuna_initial_age = 12)
  (h2 : grandmother_initial_age = 72) :
  ∃ (years_passed : ℕ), 
    grandmother_initial_age + years_passed = 5 * (yuna_initial_age + years_passed) ∧
    grandmother_initial_age + years_passed = 75 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_problem_l1260_126080


namespace NUMINAMATH_CALUDE_mothers_salary_l1260_126023

theorem mothers_salary (mother_salary : ℝ) : 
  let father_salary := 1.3 * mother_salary
  let combined_salary := mother_salary + father_salary
  let method1_savings := (combined_salary / 10) * 6
  let method2_savings := (combined_salary / 2) * (1 + 0.03 * 10)
  method1_savings = method2_savings - 2875 →
  mother_salary = 25000 := by
sorry

end NUMINAMATH_CALUDE_mothers_salary_l1260_126023


namespace NUMINAMATH_CALUDE_tangent_line_proofs_l1260_126083

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_proofs :
  let e := Real.exp 1
  -- Tangent line at (e, e^e)
  ∃ (m : ℝ), ∀ x y : ℝ,
    (y = f x) → (x = e ∧ y = f e) →
    (m * (x - e) + f e = y ∧ m * x - y - m * e + f e = 0) →
    (Real.exp e * x - y - Real.exp (e + 1) = 0) ∧
  -- Tangent line from origin
  ∃ (k : ℝ), ∀ x y : ℝ,
    (y = f x) → (y = k * x) →
    (k = f x ∧ k = (f x) / x) →
    (e * x - y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_proofs_l1260_126083


namespace NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l1260_126027

theorem odd_prime_fifth_power_difference (p : ℕ) (h_prime : Prime p) (h_odd : Odd p)
  (h_fifth_power_diff : ∃ (a b : ℕ), p = a^5 - b^5) :
  ∃ (n : ℕ), Odd n ∧ (((4 * p + 1) : ℚ) / 5).sqrt = ((n^2 + 1) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l1260_126027


namespace NUMINAMATH_CALUDE_original_bananas_count_l1260_126004

/-- The number of bananas originally in the jar. -/
def original_bananas : ℕ := sorry

/-- The number of bananas removed from the jar. -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal. -/
def remaining_bananas : ℕ := 41

/-- Theorem: The original number of bananas is equal to 46. -/
theorem original_bananas_count : original_bananas = 46 := by
  sorry

end NUMINAMATH_CALUDE_original_bananas_count_l1260_126004


namespace NUMINAMATH_CALUDE_expected_value_is_1866_l1260_126038

/-- Represents the available keys on the calculator -/
inductive Key
| One
| Two
| Three
| Plus
| Minus

/-- A sequence of 5 keystrokes -/
def Sequence := Vector Key 5

/-- Evaluates a sequence of keystrokes according to the problem rules -/
def evaluate : Sequence → ℤ := sorry

/-- The probability of pressing any specific key -/
def keyProbability : ℚ := 1 / 5

/-- The expected value of the result after evaluating a random sequence -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value is 1866 -/
theorem expected_value_is_1866 : expectedValue = 1866 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_1866_l1260_126038


namespace NUMINAMATH_CALUDE_square_side_length_for_unit_area_l1260_126026

theorem square_side_length_for_unit_area (s : ℝ) :
  s > 0 → s * s = 1 → s = 1 := by sorry

end NUMINAMATH_CALUDE_square_side_length_for_unit_area_l1260_126026


namespace NUMINAMATH_CALUDE_candy_problem_l1260_126068

theorem candy_problem :
  ∀ (S : ℕ) (N : ℕ),
    (∀ (i : ℕ), i < N → S / N = (S - S / N - 11)) →
    (S / N > 1) →
    (N > 1) →
    S = 33 :=
by sorry

end NUMINAMATH_CALUDE_candy_problem_l1260_126068


namespace NUMINAMATH_CALUDE_probability_genuine_after_defective_l1260_126017

theorem probability_genuine_after_defective :
  ∀ (total genuine defective : ℕ),
    total = genuine + defective →
    total = 7 →
    genuine = 4 →
    defective = 3 →
    (genuine : ℚ) / (total - 1 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_genuine_after_defective_l1260_126017


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1260_126096

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1260_126096


namespace NUMINAMATH_CALUDE_subtraction_problem_l1260_126020

theorem subtraction_problem (x : ℤ) : 
  (x - 48 = 22) → (x - 32 = 38) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1260_126020


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1260_126021

/-- Given an arithmetic sequence where:
    - n is a positive integer
    - The sum of the first n terms is 48
    - The sum of the first 2n terms is 60
    This theorem states that the sum of the first 3n terms is 36 -/
theorem arithmetic_sequence_sum (n : ℕ+) 
  (sum_n : ℕ) (sum_2n : ℕ) (h1 : sum_n = 48) (h2 : sum_2n = 60) :
  ∃ (sum_3n : ℕ), sum_3n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1260_126021


namespace NUMINAMATH_CALUDE_x_equals_y_l1260_126055

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l1260_126055


namespace NUMINAMATH_CALUDE_two_cakes_left_l1260_126043

/-- The number of cakes left at a restaurant -/
def cakes_left (baked_today baked_yesterday sold : ℕ) : ℕ :=
  baked_today + baked_yesterday - sold

/-- Theorem: Given the conditions, prove that 2 cakes are left -/
theorem two_cakes_left : cakes_left 5 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cakes_left_l1260_126043


namespace NUMINAMATH_CALUDE_units_digit_of_A_is_one_l1260_126097

-- Define the function for the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression for A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_one : unitsDigit A = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_A_is_one_l1260_126097


namespace NUMINAMATH_CALUDE_inequality_chain_l1260_126084

theorem inequality_chain (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x + y = 1) :
  x < 2*x*y ∧ 2*x*y < (x + y)/2 ∧ (x + y)/2 < y := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1260_126084


namespace NUMINAMATH_CALUDE_horner_method_operations_l1260_126031

/-- The number of arithmetic operations required to evaluate a polynomial using Horner's method -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- Theorem: For a polynomial of degree n, Horner's method requires 2n arithmetic operations -/
theorem horner_method_operations (n : ℕ) :
  horner_operations n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l1260_126031


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1260_126012

/-- Given a hyperbola with equation y²/12 - x²/4 = 1, prove that the equation of the ellipse
    that has the foci of the hyperbola as its vertices and the vertices of the hyperbola as its foci
    is y²/16 + x²/4 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (y^2 / 12 - x^2 / 4 = 1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
    (y^2 / a^2 + x^2 / b^2 = 1) ∧
    a = 4 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1260_126012


namespace NUMINAMATH_CALUDE_quarters_percentage_l1260_126024

theorem quarters_percentage (num_dimes : ℕ) (num_quarters : ℕ) : num_dimes = 70 → num_quarters = 30 → 
  (num_quarters * 25 : ℚ) / ((num_dimes * 10 + num_quarters * 25) : ℚ) * 100 = 51724 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_quarters_percentage_l1260_126024


namespace NUMINAMATH_CALUDE_card_probability_ratio_l1260_126052

/-- Given a box of 60 cards numbered 1 to 12, with 5 cards for each number,
    prove that the ratio of probabilities q/p is 275, where:
    p = probability of drawing 5 cards with the same number
    q = probability of drawing 4 cards with one number and 1 card with a different number -/
theorem card_probability_ratio :
  let total_cards : ℕ := 60
  let num_values : ℕ := 12
  let cards_per_value : ℕ := 5
  let draw_size : ℕ := 5
  let p := (num_values * Nat.choose cards_per_value draw_size) / Nat.choose total_cards draw_size
  let q := (num_values * (num_values - 1) * Nat.choose cards_per_value 4 * Nat.choose cards_per_value 1) / Nat.choose total_cards draw_size
  q / p = 275 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_ratio_l1260_126052


namespace NUMINAMATH_CALUDE_students_liking_both_l1260_126042

theorem students_liking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (neither : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  neither = 6 →
  ∃ (both : ℕ), both = 12 ∧ total = fries + burgers - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_l1260_126042


namespace NUMINAMATH_CALUDE_mary_potatoes_l1260_126066

/-- The number of potatoes Mary initially had -/
def initial_potatoes : ℕ := 8

/-- The number of potatoes eaten by rabbits -/
def eaten_potatoes : ℕ := 3

/-- The number of potatoes Mary has now -/
def remaining_potatoes : ℕ := initial_potatoes - eaten_potatoes

theorem mary_potatoes : remaining_potatoes = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l1260_126066


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1260_126005

theorem rectangle_diagonal (perimeter : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  perimeter = 72 →
  ratio_length = 3 →
  ratio_width = 2 →
  let length := (perimeter / 2) * (ratio_length / (ratio_length + ratio_width))
  let width := (perimeter / 2) * (ratio_width / (ratio_length + ratio_width))
  (length ^ 2 + width ^ 2) = 673.92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1260_126005


namespace NUMINAMATH_CALUDE_tuesday_bags_count_l1260_126081

/-- The number of bags of leaves raked on Tuesday -/
def bags_on_tuesday (price_per_bag : ℕ) (bags_monday : ℕ) (bags_other_day : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - price_per_bag * (bags_monday + bags_other_day)) / price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Tuesday is 3 -/
theorem tuesday_bags_count :
  bags_on_tuesday 4 5 9 68 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_bags_count_l1260_126081


namespace NUMINAMATH_CALUDE_elizabeth_steak_knife_cost_l1260_126073

def steak_knife_cost (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℚ) : ℚ :=
  (sets * cost_per_set) / (sets * knives_per_set)

theorem elizabeth_steak_knife_cost :
  steak_knife_cost 2 4 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_steak_knife_cost_l1260_126073


namespace NUMINAMATH_CALUDE_right_triangle_equations_l1260_126061

/-- A right-angled triangle ABC with specified coordinates -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : B.1 = 1 ∧ B.2 = Real.sqrt 3
  A_on_x_axis : A = (-2, 0)
  C_on_x_axis : C.2 = 0

/-- The equation of line BC in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The equation of line OB (median to hypotenuse) in the form y = kx -/
def median_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

theorem right_triangle_equations (t : RightTriangle) :
  (∃ (a b c : ℝ), a = Real.sqrt 3 ∧ b = 1 ∧ c = -2 * Real.sqrt 3 ∧
    ∀ (x y : ℝ), line_equation a b c x y ↔ (x, y) ∈ ({t.B, t.C} : Set (ℝ × ℝ))) ∧
  (∃ (k : ℝ), k = Real.sqrt 3 ∧
    ∀ (x y : ℝ), median_equation k x y ↔ (x, y) ∈ ({(0, 0), t.B} : Set (ℝ × ℝ))) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_equations_l1260_126061


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l1260_126030

theorem largest_triangle_perimeter : ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) ∧ 
  (8 : ℝ) + (x : ℝ) > 11 ∧ 
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l1260_126030


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1260_126094

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1260_126094


namespace NUMINAMATH_CALUDE_tshirt_pricing_theorem_l1260_126009

/-- Represents the cost and pricing information for two batches of T-shirts --/
structure TShirtBatches where
  first_batch_cost : ℕ
  second_batch_cost : ℕ
  quantity_ratio : ℚ
  price_difference : ℕ
  first_batch_selling_price : ℕ
  min_total_profit : ℕ

/-- Calculates the cost price of each T-shirt in the first batch --/
def cost_price_first_batch (b : TShirtBatches) : ℚ :=
  sorry

/-- Calculates the minimum selling price for the second batch --/
def min_selling_price_second_batch (b : TShirtBatches) : ℕ :=
  sorry

/-- Theorem stating the correct cost price and minimum selling price --/
theorem tshirt_pricing_theorem (b : TShirtBatches) 
  (h1 : b.first_batch_cost = 4000)
  (h2 : b.second_batch_cost = 5400)
  (h3 : b.quantity_ratio = 3/2)
  (h4 : b.price_difference = 5)
  (h5 : b.first_batch_selling_price = 70)
  (h6 : b.min_total_profit = 4060) :
  cost_price_first_batch b = 50 ∧ 
  min_selling_price_second_batch b = 66 :=
  sorry

end NUMINAMATH_CALUDE_tshirt_pricing_theorem_l1260_126009


namespace NUMINAMATH_CALUDE_students_left_l1260_126076

theorem students_left (total : ℕ) (checked_out : ℕ) (h1 : total = 124) (h2 : checked_out = 93) :
  total - checked_out = 31 := by
  sorry

end NUMINAMATH_CALUDE_students_left_l1260_126076


namespace NUMINAMATH_CALUDE_bombardment_percentage_l1260_126060

/-- Proves that the percentage of people who died by bombardment is 5% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4675)
  (h2 : final_population = 3553) :
  ∃ (x : ℝ), x = 5 ∧ 
  (initial_population : ℝ) * ((100 - x) / 100) * 0.8 = final_population := by
  sorry

end NUMINAMATH_CALUDE_bombardment_percentage_l1260_126060


namespace NUMINAMATH_CALUDE_inequality_range_l1260_126006

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1260_126006


namespace NUMINAMATH_CALUDE_f_min_value_solution_set_characterization_l1260_126067

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem 1: The minimum value of f(x) is -3
theorem f_min_value : ∀ x : ℝ, f x ≥ -3 := by sorry

-- Theorem 2: Characterization of the solution set for the inequality
theorem solution_set_characterization :
  ∀ x : ℝ, x^2 - 8*x + 15 + f x < 0 ↔ (5 - Real.sqrt 3 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) := by sorry

end NUMINAMATH_CALUDE_f_min_value_solution_set_characterization_l1260_126067


namespace NUMINAMATH_CALUDE_inverse_log_property_l1260_126011

noncomputable section

variable (a : ℝ)
variable (a_pos : a > 0)
variable (a_ne_one : a ≠ 1)

def f (x : ℝ) := Real.log x / Real.log a

def f_inverse (x : ℝ) := a ^ x

theorem inverse_log_property (h : f_inverse a 2 = 9) : f a 9 + f a 6 = 2 := by
  sorry

#check inverse_log_property

end NUMINAMATH_CALUDE_inverse_log_property_l1260_126011


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1260_126092

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 4^s - s
  r = 16377 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1260_126092


namespace NUMINAMATH_CALUDE_college_ratio_theorem_l1260_126025

/-- Represents the ratio of boys to girls in a college -/
structure CollegeRatio where
  boys : ℕ
  girls : ℕ

/-- Given the total number of students and the number of girls, calculate the ratio of boys to girls -/
def calculateRatio (totalStudents : ℕ) (numGirls : ℕ) : CollegeRatio :=
  { boys := totalStudents - numGirls,
    girls := numGirls }

/-- Theorem stating that for a college with 240 total students and 140 girls, the ratio of boys to girls is 5:7 -/
theorem college_ratio_theorem :
  let ratio := calculateRatio 240 140
  ratio.boys = 5 ∧ ratio.girls = 7 := by
  sorry


end NUMINAMATH_CALUDE_college_ratio_theorem_l1260_126025


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1260_126095

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1260_126095


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1260_126036

def M : ℕ := 18 * 18 * 125 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 14 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1260_126036


namespace NUMINAMATH_CALUDE_new_conveyor_belt_time_l1260_126093

theorem new_conveyor_belt_time (old_time new_time combined_time : ℝ) 
  (h1 : old_time = 21)
  (h2 : combined_time = 8.75)
  (h3 : 1 / old_time + 1 / new_time = 1 / combined_time) : 
  new_time = 15 := by
sorry

end NUMINAMATH_CALUDE_new_conveyor_belt_time_l1260_126093


namespace NUMINAMATH_CALUDE_wyatt_envelopes_l1260_126000

/-- The number of blue envelopes Wyatt has -/
def blue_envelopes : ℕ := 10

/-- The difference between blue and yellow envelopes -/
def envelope_difference : ℕ := 4

/-- The total number of envelopes Wyatt has -/
def total_envelopes : ℕ := blue_envelopes + (blue_envelopes - envelope_difference)

/-- Theorem stating the total number of envelopes Wyatt has -/
theorem wyatt_envelopes : total_envelopes = 16 := by sorry

end NUMINAMATH_CALUDE_wyatt_envelopes_l1260_126000


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1260_126078

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x + 2 ≠ 0 ∧ x - 2 ≠ 0) →
  (1 / (x + 2) + 4 * x / (x^2 - 4) = 1 / (x - 2)) ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1260_126078


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1260_126053

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1260_126053


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l1260_126091

theorem raccoon_lock_problem (first_lock_time second_lock_time both_locks_time : ℕ) :
  second_lock_time = 3 * first_lock_time - 3 →
  both_locks_time = 5 * second_lock_time →
  second_lock_time = 60 →
  first_lock_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l1260_126091


namespace NUMINAMATH_CALUDE_towel_packs_l1260_126044

theorem towel_packs (towels_per_pack : ℕ) (total_towels : ℕ) (num_packs : ℕ) :
  towels_per_pack = 3 →
  total_towels = 27 →
  num_packs * towels_per_pack = total_towels →
  num_packs = 9 := by
  sorry

end NUMINAMATH_CALUDE_towel_packs_l1260_126044


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l1260_126010

theorem no_solution_implies_a_leq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬(x ≥ 3 ∧ x < a)) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l1260_126010


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1260_126015

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1260_126015


namespace NUMINAMATH_CALUDE_pythagorean_triple_in_range_l1260_126071

theorem pythagorean_triple_in_range : 
  ∀ a b c : ℕ, 
    a^2 + b^2 = c^2 → 
    Nat.gcd a (Nat.gcd b c) = 1 → 
    2000 ≤ a ∧ a ≤ 3000 → 
    2000 ≤ b ∧ b ≤ 3000 → 
    2000 ≤ c ∧ c ≤ 3000 → 
    (a, b, c) = (2100, 2059, 2941) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_in_range_l1260_126071


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1260_126069

theorem geometric_sequence_sum_inequality (n : ℕ) : 2^n - 1 < 2^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1260_126069


namespace NUMINAMATH_CALUDE_green_blue_difference_l1260_126059

/-- Represents the number of disks of each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ

/-- The ratio of disks for each color -/
def diskRatio : DiskCounts := {
  blue := 3,
  yellow := 7,
  green := 8,
  red := 4,
  purple := 5
}

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 360

/-- Calculates the total ratio parts -/
def totalRatioParts (ratio : DiskCounts) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red + ratio.purple

/-- Calculates the number of disks for each ratio part -/
def disksPerPart (total : ℕ) (ratioParts : ℕ) : ℕ :=
  total / ratioParts

/-- Calculates the actual disk counts based on the ratio and total disks -/
def actualDiskCounts (ratio : DiskCounts) (total : ℕ) : DiskCounts :=
  let parts := totalRatioParts ratio
  let perPart := disksPerPart total parts
  {
    blue := ratio.blue * perPart,
    yellow := ratio.yellow * perPart,
    green := ratio.green * perPart,
    red := ratio.red * perPart,
    purple := ratio.purple * perPart
  }

theorem green_blue_difference :
  let counts := actualDiskCounts diskRatio totalDisks
  counts.green - counts.blue = 65 := by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l1260_126059


namespace NUMINAMATH_CALUDE_max_value_expression_l1260_126098

theorem max_value_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 10) :
  Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) ≤ 4 * Real.sqrt 79 ∧
  (x = 10 → Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) = 4 * Real.sqrt 79) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1260_126098

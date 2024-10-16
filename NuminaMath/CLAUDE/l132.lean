import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l132_13275

theorem expression_simplification (x : ℝ) (h : x = 2 * Real.sin (60 * π / 180) - Real.tan (45 * π / 180)) :
  (x / (x - 1) - 1 / (x^2 - x)) / ((x + 1)^2 / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l132_13275


namespace NUMINAMATH_CALUDE_cube_face_sum_l132_13238

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1287 →
  (a + d) + (b + e) + (c + f) = 33 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l132_13238


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l132_13239

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l132_13239


namespace NUMINAMATH_CALUDE_unique_intersection_point_l132_13252

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

theorem unique_intersection_point :
  ∃! a : ℝ, f a = a ∧ a = -1 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l132_13252


namespace NUMINAMATH_CALUDE_tan_theta_value_l132_13249

theorem tan_theta_value (θ : Real) 
  (h1 : Real.cos (θ / 2) = 4 / 5) 
  (h2 : Real.sin θ < 0) : 
  Real.tan θ = -24 / 7 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l132_13249


namespace NUMINAMATH_CALUDE_min_faces_for_conditions_l132_13268

/-- Represents a pair of dice --/
structure DicePair :=
  (die1 : ℕ)
  (die2 : ℕ)

/-- Calculates the number of ways to roll a sum on a pair of dice --/
def waysToRollSum (d : DicePair) (sum : ℕ) : ℕ := sorry

/-- Checks if a pair of dice satisfies the given conditions --/
def satisfiesConditions (d : DicePair) : Prop :=
  d.die1 ≥ 6 ∧ d.die2 ≥ 6 ∧
  waysToRollSum d 8 * 12 = waysToRollSum d 11 * 5 ∧
  waysToRollSum d 14 * d.die1 * d.die2 = d.die1 * d.die2 / 15

/-- Theorem stating that the minimum number of faces on two dice satisfying the conditions is 27 --/
theorem min_faces_for_conditions :
  ∀ d : DicePair, satisfiesConditions d → d.die1 + d.die2 ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_min_faces_for_conditions_l132_13268


namespace NUMINAMATH_CALUDE_b_value_when_square_zero_l132_13227

theorem b_value_when_square_zero (b : ℝ) : (b + 5)^2 = 0 → b = -5 := by
  sorry

end NUMINAMATH_CALUDE_b_value_when_square_zero_l132_13227


namespace NUMINAMATH_CALUDE_expression_evaluation_l132_13250

theorem expression_evaluation : 
  -|-(3 + 3/5)| - (-(2 + 2/5)) + 4/5 = -2/5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l132_13250


namespace NUMINAMATH_CALUDE_olympiad_team_formation_l132_13216

theorem olympiad_team_formation (n : ℕ) (k : ℕ) (roles : ℕ) 
  (h1 : n = 20) 
  (h2 : k = 3) 
  (h3 : roles = 3) :
  (n.factorial / ((n - k).factorial * k.factorial)) * (k.factorial / (roles.factorial * (k - roles).factorial)) = 6840 :=
sorry

end NUMINAMATH_CALUDE_olympiad_team_formation_l132_13216


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_450_cube_l132_13273

/-- Given a positive integer n, returns true if n is a perfect cube, false otherwise -/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The smallest positive integer that, when multiplied by 450, results in a perfect cube -/
def smallestMultiplier : ℕ := 60

theorem smallest_multiplier_for_450_cube :
  (isPerfectCube (450 * smallestMultiplier)) ∧
  (∀ n : ℕ, 0 < n → n < smallestMultiplier → ¬(isPerfectCube (450 * n))) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_450_cube_l132_13273


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_min_value_achieved_l132_13270

/-- The function f(x) = |2x-1| - m -/
def f (x m : ℝ) : ℝ := |2*x - 1| - m

/-- The theorem stating the minimum value of a + b -/
theorem min_value_a_plus_b (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h2 : 1/a + 1/(2*b) = m) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achieved -/
theorem min_value_achieved (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = m ∧ a + b = 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_min_value_achieved_l132_13270


namespace NUMINAMATH_CALUDE_tangent_line_equation_l132_13241

def S (x : ℝ) : ℝ := 3*x - x^3

theorem tangent_line_equation (x₀ y₀ : ℝ) (h : y₀ = S x₀) (h₀ : x₀ = 2) (h₁ : y₀ = -2) :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b → (x = x₀ ∧ y = y₀) ∨ (y - y₀ = m*(x - x₀))) ∧
  ((m = -9 ∧ b = 16) ∨ (m = 0 ∧ b = -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l132_13241


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_ge_one_l132_13217

/-- A function f(x) = ln x - ax is monotonically decreasing on (1, +∞) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → Real.log y - a * y < Real.log x - a * x

/-- If f(x) = ln x - ax is monotonically decreasing on (1, +∞), then a ≥ 1 -/
theorem monotone_decreasing_implies_a_ge_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_ge_one_l132_13217


namespace NUMINAMATH_CALUDE_gcd_51457_37958_is_1_l132_13295

theorem gcd_51457_37958_is_1 : Nat.gcd 51457 37958 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_51457_37958_is_1_l132_13295


namespace NUMINAMATH_CALUDE_line_circle_intersection_condition_l132_13248

/-- The line y = x + b intersects the circle x^2 + y^2 = 1 -/
def line_intersects_circle (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = x + b ∧ x^2 + y^2 = 1

/-- The condition 0 < b < 1 is necessary but not sufficient for the intersection -/
theorem line_circle_intersection_condition (b : ℝ) :
  line_intersects_circle b → 0 < b ∧ b < 1 ∧
  ¬(∀ b : ℝ, 0 < b ∧ b < 1 → line_intersects_circle b) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_condition_l132_13248


namespace NUMINAMATH_CALUDE_equation_solution_l132_13243

theorem equation_solution (x : ℝ) : 
  (4 * x - 3 > 0) → 
  (Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8) ↔ 
  (x = 7/4 ∨ x = 39/4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l132_13243


namespace NUMINAMATH_CALUDE_four_digit_number_proof_l132_13236

theorem four_digit_number_proof :
  ∀ (a b : ℕ),
    (2^a * 9^b ≥ 1000) ∧ 
    (2^a * 9^b < 10000) ∧
    (2^a * 9^b = 2000 + 100*a + 90 + b) →
    a = 5 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_proof_l132_13236


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_value_l132_13229

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair 8 people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair boys with girls (no all-girl pairs) -/
def boy_girl_pairings : ℕ := num_boys.factorial

/-- The probability of at least one pair consisting of two girls -/
def prob_at_least_one_girl_pair : ℚ := 1 - (boy_girl_pairings : ℚ) / total_pairings

theorem prob_at_least_one_girl_pair_value :
  prob_at_least_one_girl_pair = 27 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_value_l132_13229


namespace NUMINAMATH_CALUDE_proposition_form_l132_13201

theorem proposition_form : 
  ∃ (p q : Prop), (12 % 4 = 0 ∧ 12 % 3 = 0) ↔ (p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_form_l132_13201


namespace NUMINAMATH_CALUDE_problem_curve_is_line_segment_l132_13287

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ
  t_min : ℝ
  t_max : ℝ

/-- Definition of a line segment -/
def IsLineSegment (curve : ParametricCurve) : Prop :=
  ∃ (a b : ℝ × ℝ),
    (∀ t, curve.t_min ≤ t ∧ t ≤ curve.t_max →
      (curve.x t, curve.y t) = ((1 - t) • a.1 + t • b.1, (1 - t) • a.2 + t • b.2))

/-- The specific parametric curve from the problem -/
def ProblemCurve : ParametricCurve where
  x := λ t => 2 * t
  y := λ _ => 2
  t_min := -1
  t_max := 1

/-- Theorem stating that the problem curve is a line segment -/
theorem problem_curve_is_line_segment : IsLineSegment ProblemCurve := by
  sorry


end NUMINAMATH_CALUDE_problem_curve_is_line_segment_l132_13287


namespace NUMINAMATH_CALUDE_existence_of_integers_l132_13261

theorem existence_of_integers : ∃ (list : List Int), 
  (list.length = 2016) ∧ 
  (list.prod = 9) ∧ 
  (list.sum = 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l132_13261


namespace NUMINAMATH_CALUDE_puzzle_completion_l132_13265

theorem puzzle_completion (P : ℝ) : 
  (P ≥ 0) →
  (P ≤ 1) →
  ((1 - P) * 0.8 * 0.7 * 1000 = 504) →
  (P = 0.1) := by
sorry

end NUMINAMATH_CALUDE_puzzle_completion_l132_13265


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l132_13282

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l132_13282


namespace NUMINAMATH_CALUDE_sequence_product_l132_13203

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  2 * a 3 - (a 7)^2 + 2 * a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l132_13203


namespace NUMINAMATH_CALUDE_lap_time_improvement_l132_13233

def initial_laps : ℕ := 10
def initial_time : ℕ := 25
def current_laps : ℕ := 12
def current_time : ℕ := 24

theorem lap_time_improvement :
  (initial_time : ℚ) / initial_laps - (current_time : ℚ) / current_laps = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l132_13233


namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_empty_solution_set_condition_l132_13211

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -2 ∨ x > 4/3} :=
sorry

-- Theorem for the range of a where f(x) < a has no solution
theorem empty_solution_set_condition (a : ℝ) :
  ({x : ℝ | f x < a} = ∅) ↔ (a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_empty_solution_set_condition_l132_13211


namespace NUMINAMATH_CALUDE_hike_consumption_ratio_l132_13213

/-- Proves the ratio of food to water consumption given hiking conditions --/
theorem hike_consumption_ratio 
  (initial_water : ℝ) 
  (initial_food : ℝ) 
  (initial_gear : ℝ)
  (water_rate : ℝ) 
  (time : ℝ) 
  (final_weight : ℝ) :
  initial_water = 20 →
  initial_food = 10 →
  initial_gear = 20 →
  water_rate = 2 →
  time = 6 →
  final_weight = 34 →
  ∃ (food_rate : ℝ), 
    final_weight = initial_water - water_rate * time + 
                   initial_food - food_rate * time + 
                   initial_gear ∧
    food_rate / water_rate = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hike_consumption_ratio_l132_13213


namespace NUMINAMATH_CALUDE_semicircle_to_cone_volume_l132_13208

/-- The volume of a cone formed by rolling up a semicircle -/
theorem semicircle_to_cone_volume (R : ℝ) (R_pos : R > 0) :
  let r := R / 2
  let h := R * (Real.sqrt 3) / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * R^3 * Real.sqrt 3) / 24 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_to_cone_volume_l132_13208


namespace NUMINAMATH_CALUDE_unique_postage_arrangements_l132_13284

/-- Represents the quantity of stamps for each denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- Represents the denominations of stamps available -/
def stamp_denominations : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- The target postage amount -/
def target_postage : Nat := 12

/-- A function to calculate the number of unique arrangements -/
noncomputable def count_arrangements (quantities : List Nat) (denominations : List Nat) (target : Nat) : Nat :=
  sorry  -- Implementation details omitted

/-- Theorem stating that there are 82 unique arrangements -/
theorem unique_postage_arrangements :
  count_arrangements stamp_quantities stamp_denominations target_postage = 82 := by
  sorry

#check unique_postage_arrangements

end NUMINAMATH_CALUDE_unique_postage_arrangements_l132_13284


namespace NUMINAMATH_CALUDE_function_form_exists_l132_13296

noncomputable def f (a b c x : ℝ) : ℝ := a * b^x + c

theorem function_form_exists :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, x ≥ 0 → -2 ≤ f a b c x ∧ f a b c x < 3) ∧
    (0 < b ∧ b < 1) ∧
    (∀ x : ℝ, x ≥ 0 → f a b c x = -5 * b^x + 3) :=
by sorry

end NUMINAMATH_CALUDE_function_form_exists_l132_13296


namespace NUMINAMATH_CALUDE_unique_solution_implies_coefficients_l132_13272

theorem unique_solution_implies_coefficients
  (a b : ℚ)
  (h1 : ∀ x y : ℚ, a * x + y = 2 ∧ x + b * y = 2 ↔ x = 2 ∧ y = 1) :
  a = 1/2 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_implies_coefficients_l132_13272


namespace NUMINAMATH_CALUDE_sphere_volume_sum_l132_13223

theorem sphere_volume_sum (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4/3) * π * r^3
  sphere_volume 1 + sphere_volume 4 + sphere_volume 6 + sphere_volume 3 = (1232/3) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_sum_l132_13223


namespace NUMINAMATH_CALUDE_scores_mode_is_9_l132_13244

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_9 : mode scores = 9 := by sorry

end NUMINAMATH_CALUDE_scores_mode_is_9_l132_13244


namespace NUMINAMATH_CALUDE_railing_distance_proof_l132_13288

/-- The distance between two railings with bicycles placed between them -/
def railing_distance (interval_distance : ℕ) (num_bicycles : ℕ) : ℕ :=
  interval_distance * (num_bicycles - 1)

/-- Theorem: The distance between two railings is 95 meters -/
theorem railing_distance_proof :
  railing_distance 5 19 = 95 := by
  sorry

end NUMINAMATH_CALUDE_railing_distance_proof_l132_13288


namespace NUMINAMATH_CALUDE_hyperbola_center_is_two_two_l132_13290

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 8^2 - (5 * x - 10)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 2)

/-- Theorem: The center of the given hyperbola is (2, 2) -/
theorem hyperbola_center_is_two_two :
  ∀ x y : ℝ, hyperbola_equation x y → hyperbola_center = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_two_two_l132_13290


namespace NUMINAMATH_CALUDE_theta_half_quadrants_l132_13260

theorem theta_half_quadrants (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : ℤ), 2 * k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + Real.pi) ∨ 
  (∃ (k : ℤ), 2 * k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + 2 * Real.pi) ∨
  (∃ (k : ℤ), θ / 2 = k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_theta_half_quadrants_l132_13260


namespace NUMINAMATH_CALUDE_complex_sum_problem_l132_13254

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 4 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 3 * I → 
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l132_13254


namespace NUMINAMATH_CALUDE_length_of_AB_is_10_l132_13246

-- Define the triangle structures
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem length_of_AB_is_10 
  (ABC : Triangle) 
  (CBD : Triangle) 
  (isIsoscelesABC : isIsosceles ABC)
  (isIsoscelesCBD : isIsosceles CBD)
  (angle_BAC_twice_ABC : True)  -- We can't directly represent angle relationships, so we use a placeholder
  (perim_CBD : perimeter CBD = 21)
  (perim_ABC : perimeter ABC = 26)
  (length_BD : CBD.c = 9)
  : ABC.a = 10 := by
  sorry


end NUMINAMATH_CALUDE_length_of_AB_is_10_l132_13246


namespace NUMINAMATH_CALUDE_payment_difference_l132_13266

/-- Represents the pizza with its properties and how it was shared -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (pepperoni_cost : ℚ)
  (mushroom_cost : ℚ)
  (bob_slices : ℕ)
  (charlie_slices : ℕ)
  (alice_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.pepperoni_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates how much Bob paid -/
def bob_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.bob_slices

/-- Calculates how much Alice paid -/
def alice_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.alice_slices

/-- The main theorem stating the difference in payment between Bob and Alice -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.pepperoni_cost = 3)
  (h4 : p.mushroom_cost = 2)
  (h5 : p.bob_slices = 6)
  (h6 : p.charlie_slices = 5)
  (h7 : p.alice_slices = 3) :
  bob_payment p - alice_payment p = 4.26 := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_l132_13266


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l132_13263

theorem arithmetic_progression_implies_equal_numbers
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h_arith_prog : (a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l132_13263


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l132_13210

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  m : ℕ+
  n : ℕ+
  h_neq : m ≠ n
  S : ℕ+ → ℚ
  h_m : S m = m / n
  h_n : S n = n / m

/-- The sum of the first (m+n) terms is greater than 4 -/
theorem sum_greater_than_four (seq : ArithmeticSequence) : seq.S (seq.m + seq.n) > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l132_13210


namespace NUMINAMATH_CALUDE_jane_vases_last_day_l132_13214

/-- The number of vases Jane arranges on the last day given her daily rate, total vases, and total days --/
def vases_on_last_day (daily_rate : ℕ) (total_vases : ℕ) (total_days : ℕ) : ℕ :=
  if total_vases ≤ daily_rate * (total_days - 1)
  then 0
  else total_vases - daily_rate * (total_days - 1)

theorem jane_vases_last_day :
  vases_on_last_day 25 378 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jane_vases_last_day_l132_13214


namespace NUMINAMATH_CALUDE_f_properties_l132_13297

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem f_properties (a b : ℝ) (h : a * b ≠ 0) :
  (∀ x, f 1 (-Real.sqrt 3) x = 2 * Real.sin (2 * (x - Real.pi / 6))) ∧
  (a = b → ∀ x, f a b (x + Real.pi / 4) = f a b (Real.pi / 4 - x)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l132_13297


namespace NUMINAMATH_CALUDE_common_tangent_range_a_l132_13245

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

def has_common_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁ = deriv g x₂) ∧ 
    (f x₁ - g x₂ = deriv f x₁ * (x₁ - x₂))

theorem common_tangent_range_a :
  ∀ a : ℝ, (∃ x < 0, has_common_tangent f (g a)) → 
    a ∈ Set.Ioi (Real.log (1/(2*Real.exp 1))) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_range_a_l132_13245


namespace NUMINAMATH_CALUDE_ellipse_focus_d_l132_13219

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The value of d for the given ellipse is 15 -/
theorem ellipse_focus_d (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 15 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_d_l132_13219


namespace NUMINAMATH_CALUDE_sock_pairs_combinations_l132_13285

/-- Given 7 pairs of socks, proves that the number of ways to choose 2 socks 
    from different pairs is 84. -/
theorem sock_pairs_combinations (n : ℕ) (h : n = 7) : 
  (2 * n * (2 * n - 2)) / 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_combinations_l132_13285


namespace NUMINAMATH_CALUDE_cryptarithmetic_problem_l132_13218

theorem cryptarithmetic_problem (A B C : ℕ) : 
  A < 10 → B < 10 → C < 10 →  -- Single-digit integers
  A ≠ B → A ≠ C → B ≠ C →     -- Unique digits
  A * B = 24 →                -- First equation
  A - C = 4 →                 -- Second equation
  C = 0 :=                    -- Conclusion
by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_problem_l132_13218


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l132_13251

/-- Calculates the average speed of a round trip journey given the distance and times for each leg. -/
theorem average_speed_round_trip 
  (uphill_distance : ℝ) 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) : 
  (2 * uphill_distance) / (uphill_time + downhill_time) = 4 :=
by
  sorry

#check average_speed_round_trip 2 (45/60) (15/60)

end NUMINAMATH_CALUDE_average_speed_round_trip_l132_13251


namespace NUMINAMATH_CALUDE_rectangle_y_value_l132_13277

/-- Given a rectangle with vertices (-2, y), (8, y), (-2, 3), and (8, 3),
    if the area is 90 square units and y is positive, then y = 12. -/
theorem rectangle_y_value (y : ℝ) : y > 0 → (8 - (-2)) * (y - 3) = 90 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l132_13277


namespace NUMINAMATH_CALUDE_range_of_a_l132_13242

/-- The range of values for a given the conditions -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x → q x) →  -- p is sufficient for q
  (∃ x, q x ∧ ¬(p x)) →  -- p is not necessary for q
  (∀ x, p x ↔ (x^2 - 2*x - 3 < 0)) →  -- definition of p
  (∀ x, q x ↔ (x > a)) →  -- definition of q
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l132_13242


namespace NUMINAMATH_CALUDE_colored_paper_difference_l132_13209

/-- 
Given that Minyoung and Hoseok each start with 150 pieces of colored paper,
Minyoung buys 32 more pieces, and Hoseok buys 49 more pieces,
prove that Hoseok ends up with 17 more pieces than Minyoung.
-/
theorem colored_paper_difference : 
  let initial_paper : ℕ := 150
  let minyoung_bought : ℕ := 32
  let hoseok_bought : ℕ := 49
  let minyoung_total := initial_paper + minyoung_bought
  let hoseok_total := initial_paper + hoseok_bought
  hoseok_total - minyoung_total = 17 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_difference_l132_13209


namespace NUMINAMATH_CALUDE_gumball_count_l132_13202

/-- Represents a gumball machine with red, green, and blue gumballs. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Creates a gumball machine with the given conditions. -/
def createMachine (redCount : ℕ) : GumballMachine :=
  let blueCount := redCount / 2
  let greenCount := blueCount * 4
  { red := redCount, blue := blueCount, green := greenCount }

/-- Calculates the total number of gumballs in the machine. -/
def totalGumballs (machine : GumballMachine) : ℕ :=
  machine.red + machine.blue + machine.green

/-- Theorem stating that a machine with 16 red gumballs has 56 gumballs in total. -/
theorem gumball_count : totalGumballs (createMachine 16) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gumball_count_l132_13202


namespace NUMINAMATH_CALUDE_club_members_count_l132_13240

theorem club_members_count : ∃ (M : ℕ), 
  M > 0 ∧ 
  (2 : ℚ) / 5 * M = (M : ℚ) - (3 : ℚ) / 5 * M ∧ 
  (1 : ℚ) / 3 * ((3 : ℚ) / 5 * M) = (1 : ℚ) / 5 * M ∧ 
  (2 : ℚ) / 5 * M = 6 ∧ 
  M = 15 :=
by sorry

end NUMINAMATH_CALUDE_club_members_count_l132_13240


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l132_13200

/-- The number of rectangles containing exactly one gray cell in a 2x20 grid --/
def num_rectangles_with_one_gray_cell (total_gray_cells : ℕ) 
  (blue_cells : ℕ) (red_cells : ℕ) : ℕ :=
  blue_cells * 4 + red_cells * 8

/-- Theorem stating the number of rectangles with one gray cell in the given grid --/
theorem rectangles_with_one_gray_cell :
  num_rectangles_with_one_gray_cell 40 36 4 = 176 := by
  sorry

#eval num_rectangles_with_one_gray_cell 40 36 4

end NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l132_13200


namespace NUMINAMATH_CALUDE_subset_sum_theorem_l132_13206

theorem subset_sum_theorem (a₁ a₂ a₃ a₄ : ℝ) 
  (h : (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₂ + a₃) + (a₂ + a₄) + (a₃ + a₄) + 
       (a₁ + a₂ + a₃) + (a₁ + a₂ + a₄) + (a₁ + a₃ + a₄) + (a₂ + a₃ + a₄) = 28) :
  a₁ + a₂ + a₃ + a₄ = 4 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_theorem_l132_13206


namespace NUMINAMATH_CALUDE_smallest_with_six_odd_twelve_even_divisors_l132_13222

/-- Count the number of positive odd integer divisors of a natural number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count the number of positive even integer divisors of a natural number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly six positive odd integer divisors and twelve positive even integer divisors -/
def hasSixOddTwelveEvenDivisors (n : ℕ) : Prop :=
  countOddDivisors n = 6 ∧ countEvenDivisors n = 12

theorem smallest_with_six_odd_twelve_even_divisors :
  ∃ (n : ℕ), n > 0 ∧ hasSixOddTwelveEvenDivisors n ∧
  ∀ (m : ℕ), m > 0 → hasSixOddTwelveEvenDivisors m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_smallest_with_six_odd_twelve_even_divisors_l132_13222


namespace NUMINAMATH_CALUDE_sin_C_value_area_when_b_is_6_l132_13293

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 + t.c^2 - Real.sqrt 3 * t.a * t.c = t.b^2 ∧
  3 * t.a = 2 * t.b

-- Theorem for part (I)
theorem sin_C_value (t : Triangle) (h : satisfies_conditions t) :
  Real.sin t.C = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

-- Theorem for part (II)
theorem area_when_b_is_6 (t : Triangle) (h : satisfies_conditions t) (h_b : t.b = 6) :
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_area_when_b_is_6_l132_13293


namespace NUMINAMATH_CALUDE_searchlight_dark_time_l132_13225

/-- The number of revolutions per minute for the searchlight -/
def revolutions_per_minute : ℝ := 4

/-- The probability of staying in the dark for at least a certain number of seconds -/
def probability : ℝ := 0.6666666666666667

/-- The time in seconds for which the probability applies -/
def dark_time : ℝ := 10

theorem searchlight_dark_time :
  revolutions_per_minute = 4 ∧ probability = 0.6666666666666667 →
  dark_time = 10 := by sorry

end NUMINAMATH_CALUDE_searchlight_dark_time_l132_13225


namespace NUMINAMATH_CALUDE_heathers_oranges_l132_13224

/-- The total number of oranges Heather has after receiving oranges from Russell -/
def total_oranges (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem stating that Heather's total oranges is 96.3 given the initial and received amounts -/
theorem heathers_oranges :
  total_oranges 60.5 35.8 = 96.3 := by
  sorry

end NUMINAMATH_CALUDE_heathers_oranges_l132_13224


namespace NUMINAMATH_CALUDE_trebled_result_is_72_l132_13258

theorem trebled_result_is_72 (x : ℕ) (h : x = 9) : 3 * (2 * x + 6) = 72 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_is_72_l132_13258


namespace NUMINAMATH_CALUDE_quadratic_solutions_l132_13289

theorem quadratic_solutions : 
  (∃ (x : ℝ), x^2 - 8*x + 12 = 0) ∧ 
  (∃ (x : ℝ), x^2 - 2*x - 8 = 0) ∧ 
  ({x : ℝ | x^2 - 8*x + 12 = 0} = {2, 6}) ∧
  ({x : ℝ | x^2 - 2*x - 8 = 0} = {-2, 4}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_l132_13289


namespace NUMINAMATH_CALUDE_vector_count_is_two_l132_13298

-- Define the properties of a quantity
structure Quantity where
  has_magnitude : Bool
  has_direction : Bool

-- Define what makes a quantity a vector
def is_vector (q : Quantity) : Bool :=
  q.has_magnitude ∧ q.has_direction

-- Define the given quantities
def density : Quantity := { has_magnitude := true, has_direction := false }
def buoyancy : Quantity := { has_magnitude := true, has_direction := true }
def wind_speed : Quantity := { has_magnitude := true, has_direction := true }
def temperature : Quantity := { has_magnitude := true, has_direction := false }

-- List of quantities
def quantities : List Quantity := [density, buoyancy, wind_speed, temperature]

-- Theorem to prove
theorem vector_count_is_two :
  (quantities.filter is_vector).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_count_is_two_l132_13298


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l132_13235

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (x1^2 - 4*x1 = 5 ∧ x2^2 - 4*x2 = 5) ∧ (x1 = 5 ∧ x2 = -1)) ∧
  (∃ y1 y2 : ℝ, (y1^2 + 7*y1 - 18 = 0 ∧ y2^2 + 7*y2 - 18 = 0) ∧ (y1 = -9 ∧ y2 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l132_13235


namespace NUMINAMATH_CALUDE_system_of_equations_l132_13228

theorem system_of_equations (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l132_13228


namespace NUMINAMATH_CALUDE_count_propositions_and_true_propositions_l132_13259

-- Define the type for statements
inductive Statement
| RhetoricalQuestion
| Question
| Proposition (isTrue : Bool)
| ExclamatoryStatement
| ConstructionLanguage

-- Define the list of statements
def statements : List Statement := [
  Statement.RhetoricalQuestion,
  Statement.Question,
  Statement.Proposition false,
  Statement.ExclamatoryStatement,
  Statement.Proposition false,
  Statement.ConstructionLanguage
]

-- Theorem to prove
theorem count_propositions_and_true_propositions :
  (statements.filter (fun s => match s with
    | Statement.Proposition _ => true
    | _ => false
  )).length = 2 ∧
  (statements.filter (fun s => match s with
    | Statement.Proposition true => true
    | _ => false
  )).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_and_true_propositions_l132_13259


namespace NUMINAMATH_CALUDE_friday_lunch_customers_l132_13220

theorem friday_lunch_customers (breakfast : ℕ) (dinner : ℕ) (saturday_prediction : ℕ) :
  breakfast = 73 →
  dinner = 87 →
  saturday_prediction = 574 →
  ∃ (lunch : ℕ), lunch = saturday_prediction / 2 - breakfast - dinner ∧ lunch = 127 :=
by sorry

end NUMINAMATH_CALUDE_friday_lunch_customers_l132_13220


namespace NUMINAMATH_CALUDE_range_of_a_l132_13234

open Set

theorem range_of_a (A B : Set ℝ) (a : ℝ) :
  A = {0, 1, a} →
  B = {x : ℝ | 0 < x ∧ x < 2} →
  A ∩ B = {1, a} →
  a ∈ Ioo 0 1 ∪ Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l132_13234


namespace NUMINAMATH_CALUDE_expression_with_eight_factors_l132_13294

theorem expression_with_eight_factors
  (x y : ℕ)
  (hx_prime : Nat.Prime x)
  (hy_prime : Nat.Prime y)
  (hx_odd : Odd x)
  (hy_odd : Odd y)
  (hxy_lt : x < y) :
  (Finset.filter (fun d => (x^3 * y) % d = 0) (Finset.range (x^3 * y + 1))).card = 8 :=
sorry

end NUMINAMATH_CALUDE_expression_with_eight_factors_l132_13294


namespace NUMINAMATH_CALUDE_all_bulbs_can_be_turned_off_l132_13281

-- Define a type for the state of light bulbs
def BulbState := Bool

-- Define a type for buttons
structure Button where
  toggle : List BulbState → List BulbState

-- Define the property that for any subset of bulbs, there's a button connected to an odd number of them
def oddConnectionProperty (buttons : List Button) (n : Nat) : Prop :=
  ∀ (subset : List Nat), subset.length ≤ n →
    ∃ (b : Button), b ∈ buttons ∧ (subset.length % 2 = 1)

-- Define the theorem
theorem all_bulbs_can_be_turned_off (n : Nat) (buttons : List Button) :
  oddConnectionProperty buttons n →
  ∃ (sequence : List Button),
    (sequence.foldl (λ acc b => b.toggle acc) (List.replicate n true)).all (· = false) :=
sorry

end NUMINAMATH_CALUDE_all_bulbs_can_be_turned_off_l132_13281


namespace NUMINAMATH_CALUDE_cookie_problem_l132_13264

theorem cookie_problem :
  ∃! N : ℕ, 0 < N ∧ N < 150 ∧ N % 13 = 5 ∧ N % 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l132_13264


namespace NUMINAMATH_CALUDE_largest_angle_in_hexagon_l132_13283

/-- Theorem: In a hexagon ABCDEF with given angle conditions, the largest angle measures 304°. -/
theorem largest_angle_in_hexagon (A B C D E F : ℝ) : 
  A = 100 →
  B = 120 →
  C = D →
  F = 3 * C + 10 →
  A + B + C + D + E + F = 720 →
  max A (max B (max C (max D (max E F)))) = 304 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_hexagon_l132_13283


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l132_13212

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l132_13212


namespace NUMINAMATH_CALUDE_power_function_sum_l132_13291

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (α β : ℝ), ∀ x, f x = α * x ^ β

-- State the theorem
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l132_13291


namespace NUMINAMATH_CALUDE_custom_op_difference_l132_13276

-- Define the custom operator @
def customOp (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem custom_op_difference : (customOp 7 2) - (customOp 2 7) = -20 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l132_13276


namespace NUMINAMATH_CALUDE_hash_example_l132_13204

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d^2

theorem hash_example : hash 2 3 1 4 = 17 := by sorry

end NUMINAMATH_CALUDE_hash_example_l132_13204


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l132_13271

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 > 0}
  let B : Set ℝ := {x | (x - m) * (x - 2*m) ≤ 0}
  A ∩ B = ∅ → m ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l132_13271


namespace NUMINAMATH_CALUDE_investment_interest_proof_l132_13299

/-- Calculates the total interest earned on an investment with compound interest. -/
def total_interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on $1,500 invested at 5% annual interest
    rate compounded annually for 5 years is approximately $414.42. -/
theorem investment_interest_proof :
  ∃ ε > 0, |total_interest_earned 1500 0.05 5 - 414.42| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l132_13299


namespace NUMINAMATH_CALUDE_outfit_combinations_l132_13256

/-- Represents the number of items of each type (shirts, pants, hats) -/
def num_items : ℕ := 7

/-- Represents the number of colors available for each item type -/
def num_colors : ℕ := 7

/-- Calculates the number of valid outfit combinations where no two items are the same color -/
def valid_outfits : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Proves that the number of valid outfit combinations is 210 -/
theorem outfit_combinations :
  valid_outfits = 210 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l132_13256


namespace NUMINAMATH_CALUDE_equilateral_triangle_centroid_sum_l132_13231

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The centroid of a triangle -/
class Centroid (T : Type) where
  point : T → ℝ × ℝ

/-- The length of a segment from a vertex to the centroid -/
def vertex_to_centroid_length (t : EquilateralTriangle) : ℝ := sorry

theorem equilateral_triangle_centroid_sum 
  (t : EquilateralTriangle) 
  [Centroid EquilateralTriangle] : 
  3 * vertex_to_centroid_length t = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_centroid_sum_l132_13231


namespace NUMINAMATH_CALUDE_unique_arithmetic_grid_solution_l132_13274

/-- Represents a 5x5 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 5) Int

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → Int) : Prop :=
  ∃ d : Int, ∀ i : Fin 5, i.val < 4 → seq (i + 1) - seq i = d

/-- The initial grid with given values -/
def initialGrid : Grid :=
  fun i j => if i = 0 ∧ j = 0 then 2
             else if i = 0 ∧ j = 4 then 14
             else if i = 1 ∧ j = 1 then 8
             else if i = 2 ∧ j = 1 then 11
             else if i = 2 ∧ j = 2 then 16
             else if i = 4 ∧ j = 0 then 10
             else 0  -- placeholder for unknown values

/-- Theorem stating the existence and uniqueness of the solution -/
theorem unique_arithmetic_grid_solution :
  ∃! g : Grid,
    (∀ i j, initialGrid i j ≠ 0 → g i j = initialGrid i j) ∧
    (∀ i, isArithmeticSequence (fun j => g i j)) ∧
    (∀ j, isArithmeticSequence (fun i => g i j)) := by
  sorry

end NUMINAMATH_CALUDE_unique_arithmetic_grid_solution_l132_13274


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l132_13269

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (5 * x₁^2 - 2 * x₁ + c = 0) ∧ 
    (5 * x₂^2 - 2 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/5)) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l132_13269


namespace NUMINAMATH_CALUDE_equation_roots_l132_13247

theorem equation_roots : 
  {x : ℝ | (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0} = {-3, 2} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l132_13247


namespace NUMINAMATH_CALUDE_AC_greater_than_CK_l132_13232

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = BC ∧ AC = 2 * Real.sqrt 7 ∧ AB = 8

-- Define point D as the foot of the height from B
def HeightFoot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - D.1) * (A.1 - C.1) + (B.2 - D.2) * (A.2 - C.2) = 0 ∧
  D.1 = (A.1 + C.1) / 2 ∧ D.2 = (A.2 + C.2) / 2

-- Define point K on BD
def PointK (B D K : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 2/5 ∧ K = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)

-- Main theorem
theorem AC_greater_than_CK (A B C D K : ℝ × ℝ) :
  Triangle A B C → HeightFoot A B C D → PointK B D K →
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) > Real.sqrt ((C.1 - K.1)^2 + (C.2 - K.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_AC_greater_than_CK_l132_13232


namespace NUMINAMATH_CALUDE_jakes_weight_l132_13221

/-- Represents the weights of Jake, his sister, and Mark -/
structure SiblingWeights where
  jake : ℝ
  sister : ℝ
  mark : ℝ

/-- The conditions of the problem -/
def weightConditions (w : SiblingWeights) : Prop :=
  w.jake - 12 = 2 * (w.sister + 4) ∧
  w.mark = w.jake + w.sister + 50 ∧
  w.jake + w.sister + w.mark = 385

/-- The theorem stating Jake's current weight -/
theorem jakes_weight (w : SiblingWeights) :
  weightConditions w → w.jake = 118 := by
  sorry

#check jakes_weight

end NUMINAMATH_CALUDE_jakes_weight_l132_13221


namespace NUMINAMATH_CALUDE_impossibleToGet2015Stacks_l132_13215

/-- Represents a collection of token stacks -/
structure TokenStacks where
  stacks : List Nat
  inv : stacks.sum = 2014

/-- Represents the allowed operations on token stacks -/
inductive Operation
  | Split : Nat → Nat → Operation  -- Split a stack into two
  | Merge : Nat → Nat → Operation  -- Merge two stacks

/-- Applies an operation to the token stacks -/
def applyOperation (ts : TokenStacks) (op : Operation) : TokenStacks :=
  match op with
  | Operation.Split i j => { stacks := i :: j :: ts.stacks.tail, inv := sorry }
  | Operation.Merge i j => { stacks := (i + j) :: ts.stacks.tail.tail, inv := sorry }

/-- The main theorem to prove -/
theorem impossibleToGet2015Stacks (ts : TokenStacks) :
  ¬∃ (ops : List Operation), (ops.foldl applyOperation ts).stacks = List.replicate 2015 1 :=
sorry

end NUMINAMATH_CALUDE_impossibleToGet2015Stacks_l132_13215


namespace NUMINAMATH_CALUDE_min_value_theorem_l132_13205

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ (m : ℝ), m = 15 ∧ ∀ (a b : ℝ), a > 1 → b > 1 → a * b - 2 * a - b + 1 = 0 → (3/2) * a^2 + b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l132_13205


namespace NUMINAMATH_CALUDE_smallest_n_for_gcd_lcm_condition_l132_13230

theorem smallest_n_for_gcd_lcm_condition : ∃ (n : ℕ), 
  (∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = n.factorial) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = m.factorial) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_gcd_lcm_condition_l132_13230


namespace NUMINAMATH_CALUDE_total_winter_clothing_l132_13262

def number_of_boxes : ℕ := 8
def scarves_per_box : ℕ := 4
def mittens_per_box : ℕ := 6

theorem total_winter_clothing : 
  number_of_boxes * (scarves_per_box + mittens_per_box) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l132_13262


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_4_intersection_A_B_equals_A_l132_13255

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Statement 1
theorem union_A_B_when_a_is_4 : 
  A 4 ∪ B = {x | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Statement 2
theorem intersection_A_B_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_4_intersection_A_B_equals_A_l132_13255


namespace NUMINAMATH_CALUDE_inequalities_system_k_range_l132_13278

theorem inequalities_system_k_range :
  ∀ k : ℚ,
  (∀ x : ℤ, x^2 - x - 2 > 0 ∧ 2*x^2 + (5 + 2*k)*x + 5 < 0 ↔ x = -2) →
  3/4 < k ∧ k ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequalities_system_k_range_l132_13278


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l132_13280

theorem perfect_square_trinomial 
  (a b c : ℝ) 
  (h : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y^2) :
  ∃ (d e : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (d * x + e)^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l132_13280


namespace NUMINAMATH_CALUDE_one_by_one_tile_position_l132_13267

/-- Represents a tile with width and height -/
structure Tile where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Square where
  side_length : ℕ

/-- Represents the position of a tile in the square -/
structure TilePosition where
  row : ℕ
  col : ℕ

/-- Checks if a position is in the center or adjacent to the boundary of the square -/
def is_center_or_adjacent_boundary (pos : TilePosition) (square : Square) : Prop :=
  (pos.row = square.side_length / 2 + 1 ∧ pos.col = square.side_length / 2 + 1) ∨
  (pos.row = 1 ∨ pos.row = square.side_length ∨ pos.col = 1 ∨ pos.col = square.side_length)

/-- Theorem: In a 7x7 square formed by sixteen 1x3 tiles and one 1x1 tile,
    the 1x1 tile must be either in the center or adjacent to the boundary -/
theorem one_by_one_tile_position
  (square : Square)
  (large_tiles : Finset Tile)
  (small_tile : Tile)
  (tile_arrangement : Square → Finset Tile → Tile → TilePosition) :
  square.side_length = 7 →
  large_tiles.card = 16 →
  (∀ t ∈ large_tiles, t.width = 1 ∧ t.height = 3) →
  small_tile.width = 1 ∧ small_tile.height = 1 →
  is_center_or_adjacent_boundary (tile_arrangement square large_tiles small_tile) square :=
by sorry

end NUMINAMATH_CALUDE_one_by_one_tile_position_l132_13267


namespace NUMINAMATH_CALUDE_division_count_is_eight_l132_13207

/-- Represents an L-shaped piece consisting of three cells -/
structure LPiece where
  -- Define properties of an L-shaped piece if needed

/-- Represents a 3 × 6 rectangle -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a division of the rectangle into L-shaped pieces -/
structure Division where
  pieces : List LPiece

/-- Function to count valid divisions of the rectangle into L-shaped pieces -/
def countValidDivisions (rect : Rectangle) : Nat :=
  sorry

/-- Theorem stating that the number of ways to divide a 3 × 6 rectangle 
    into L-shaped pieces of three cells is 8 -/
theorem division_count_is_eight :
  let rect : Rectangle := { width := 6, height := 3 }
  countValidDivisions rect = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_count_is_eight_l132_13207


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l132_13253

theorem shopping_tax_calculation (total_amount : ℝ) (h_positive : total_amount > 0) :
  let clothing_percent : ℝ := 0.5
  let food_percent : ℝ := 0.25
  let other_percent : ℝ := 0.25
  let clothing_tax_rate : ℝ := 0.1
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.2
  
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let other_amount := other_percent * total_amount
  
  let clothing_tax := clothing_amount * clothing_tax_rate
  let food_tax := food_amount * food_tax_rate
  let other_tax := other_amount * other_tax_rate
  
  let total_tax := clothing_tax + food_tax + other_tax
  
  (total_tax / total_amount) = 0.1 := by sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l132_13253


namespace NUMINAMATH_CALUDE_train_passing_time_l132_13292

/-- Prove that a train with given speed and platform crossing time will take 16 seconds to pass a stationary point -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 280 →
  platform_crossing_time = 30 →
  (train_speed_kmph * 1000 / 3600) * ((platform_length + train_speed_kmph * 1000 / 3600 * platform_crossing_time) / (train_speed_kmph * 1000 / 3600)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l132_13292


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l132_13286

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference : lunch_cost - breakfast_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l132_13286


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l132_13226

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Prime p → Prime q → (p * q ∣ 3^p + 3^q) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ 
   (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l132_13226


namespace NUMINAMATH_CALUDE_set_operations_l132_13257

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (C_R_B ∪ A = {x : ℝ | x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l132_13257


namespace NUMINAMATH_CALUDE_triangle_area_l132_13279

-- Define the lines that bound the triangle
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Theorem statement
theorem triangle_area : 
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := |A.1 - B.1|
  let height := |line3 - O.2|
  (1/2 : ℝ) * base * height = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l132_13279


namespace NUMINAMATH_CALUDE_robin_camera_pictures_l132_13237

/-- The number of pictures Robin uploaded from her camera -/
def camera_pictures (phone_pictures total_albums pictures_per_album : ℕ) : ℕ :=
  total_albums * pictures_per_album - phone_pictures

/-- Proof that Robin uploaded 5 pictures from her camera -/
theorem robin_camera_pictures :
  camera_pictures 35 5 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_camera_pictures_l132_13237

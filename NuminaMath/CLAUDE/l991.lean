import Mathlib

namespace NUMINAMATH_CALUDE_population_average_age_l991_99166

/-- Given a population with females and males, calculate the average age -/
theorem population_average_age
  (female_ratio male_ratio : ℕ)
  (female_avg_age male_avg_age : ℝ)
  (h_ratio : female_ratio = 11 ∧ male_ratio = 10)
  (h_female_age : female_avg_age = 34)
  (h_male_age : male_avg_age = 32) :
  let total_people := female_ratio + male_ratio
  let total_age_sum := female_ratio * female_avg_age + male_ratio * male_avg_age
  total_age_sum / total_people = 33 + 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l991_99166


namespace NUMINAMATH_CALUDE_g_value_at_800_l991_99177

def g_property (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y

theorem g_value_at_800 (g : ℝ → ℝ) (h : g_property g) (h1000 : g 1000 = 4) :
  g 800 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_800_l991_99177


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l991_99187

/-- The cost price of a bicycle for seller A, given the following conditions:
  - A sells the bicycle to B at a profit of 20%
  - B sells it to C at a profit of 25%
  - C pays Rs. 225 for the bicycle
-/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l991_99187


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l991_99100

theorem equation_root_implies_m_value (m x : ℝ) : 
  (m / (x - 3) - 1 / (3 - x) = 2) → 
  (∃ x > 0, m / (x - 3) - 1 / (3 - x) = 2) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l991_99100


namespace NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l991_99143

theorem goose_egg_hatch_fraction (E : ℕ) (F : ℚ) : 
  E > 0 → 
  F > 0 → 
  F ≤ 1 → 
  E * F * (4/5) * (2/5) = 120 → 
  F = 1 := by
sorry

end NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l991_99143


namespace NUMINAMATH_CALUDE_binomial_coefficient_12_5_l991_99171

theorem binomial_coefficient_12_5 : Nat.choose 12 5 = 792 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_12_5_l991_99171


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l991_99151

theorem trig_expression_equals_one : 
  let tan_30 : ℝ := 1 / Real.sqrt 3
  let sin_30 : ℝ := 1 / 2
  (tan_30^2 - sin_30^2) / (tan_30^2 * sin_30^2) = 1 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l991_99151


namespace NUMINAMATH_CALUDE_solution_pairs_l991_99130

theorem solution_pairs (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0 ∧
   x^2 + 81 * x^2 * y^4 = 2 * y^2) ↔
  ((x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3) ∨
   (x = 1/3 ∧ y = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l991_99130


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l991_99147

-- Define the conditions
def p (a : ℝ) : Prop := a ≤ 2
def q (a : ℝ) : Prop := a * (a - 2) ≤ 0

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ a : ℝ, ¬(p a) → ¬(q a)) ∧
  ¬(∀ a : ℝ, ¬(q a) → ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l991_99147


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l991_99185

/-- Given a hyperbola with center at the origin, foci on the y-axis,
    and an asymptote passing through (-2, 4), its eccentricity is √5/2 -/
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a/b * x → (x = -2 ∧ y = 4)) →  -- asymptote passes through (-2, 4)
  a^2 = c^2 - b^2 →                              -- hyperbola equation
  c^2 / a^2 = (5:ℝ)/4 :=                         -- eccentricity squared
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l991_99185


namespace NUMINAMATH_CALUDE_rain_hours_calculation_l991_99167

/-- Given a 9-hour period where it rained for 4 hours, prove that it did not rain for 5 hours. -/
theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_rain_hours_calculation_l991_99167


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l991_99199

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h_sum : a + b + c + d + e + f = 10) :
  2/a + 3/b + 9/c + 16/d + 25/e + 36/f ≥ (329 + 38 * Real.sqrt 6) / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l991_99199


namespace NUMINAMATH_CALUDE_quadratic_function_range_l991_99169

theorem quadratic_function_range (k m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + m > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4 * x₁ + k = 0 ∧ x₂^2 - 4 * x₂ + k = 0) →
  (∀ k' : ℤ, k' > k → 
    (∃ x : ℝ, 2 * x^2 - 2 * k' * x + m ≤ 0) ∨
    (∀ x₁ x₂ : ℝ, x₁ = x₂ ∨ x₁^2 - 4 * x₁ + k' ≠ 0 ∨ x₂^2 - 4 * x₂ + k' ≠ 0)) →
  m > 9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l991_99169


namespace NUMINAMATH_CALUDE_total_problems_solved_l991_99114

theorem total_problems_solved (initial_problems : Nat) (additional_problems : Nat) : 
  initial_problems = 45 → additional_problems = 18 → initial_problems + additional_problems = 63 :=
by sorry

end NUMINAMATH_CALUDE_total_problems_solved_l991_99114


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l991_99142

noncomputable def f (x : ℝ) : ℝ := 3 * x^(1/4) - x^(1/2)

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = (1/4) * x + 7/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l991_99142


namespace NUMINAMATH_CALUDE_units_digit_problem_l991_99195

theorem units_digit_problem : ∃ n : ℕ, (8 * 13 * 1989 - 8^3) % 10 = 4 ∧ n * 10 ≤ 8 * 13 * 1989 - 8^3 ∧ 8 * 13 * 1989 - 8^3 < (n + 1) * 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l991_99195


namespace NUMINAMATH_CALUDE_average_score_five_subjects_l991_99191

theorem average_score_five_subjects 
  (avg_three : ℝ) 
  (score_four : ℝ) 
  (score_five : ℝ) 
  (h1 : avg_three = 92) 
  (h2 : score_four = 90) 
  (h3 : score_five = 95) : 
  (3 * avg_three + score_four + score_five) / 5 = 92.2 := by
sorry

end NUMINAMATH_CALUDE_average_score_five_subjects_l991_99191


namespace NUMINAMATH_CALUDE_four_dogs_food_consumption_l991_99196

/-- The total daily food consumption of four dogs -/
def total_dog_food_consumption (dog1 dog2 dog3 dog4 : ℚ) : ℚ :=
  dog1 + dog2 + dog3 + dog4

/-- Theorem stating the total daily food consumption of four specific dogs -/
theorem four_dogs_food_consumption :
  total_dog_food_consumption (1/8) (1/4) (3/8) (1/2) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_four_dogs_food_consumption_l991_99196


namespace NUMINAMATH_CALUDE_fraction_addition_l991_99174

theorem fraction_addition : (2 : ℚ) / 5 + (1 : ℚ) / 3 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l991_99174


namespace NUMINAMATH_CALUDE_top_is_multiple_of_four_l991_99145

/-- Represents a number pyramid with 4 rows -/
structure NumberPyramid where
  bottom_row : Fin 4 → ℤ
  second_row : Fin 3 → ℤ
  third_row : Fin 2 → ℤ
  top : ℤ

/-- Defines a valid number pyramid where each cell above the bottom row
    is the sum of the two cells below it, and the second row contains equal integers -/
def is_valid_pyramid (p : NumberPyramid) : Prop :=
  (∃ n : ℤ, ∀ i : Fin 3, p.second_row i = n) ∧
  (∀ i : Fin 2, p.third_row i = p.second_row i + p.second_row (i + 1)) ∧
  p.top = p.third_row 0 + p.third_row 1

theorem top_is_multiple_of_four (p : NumberPyramid) (h : is_valid_pyramid p) :
  ∃ k : ℤ, p.top = 4 * k :=
sorry

end NUMINAMATH_CALUDE_top_is_multiple_of_four_l991_99145


namespace NUMINAMATH_CALUDE_barn_painted_area_l991_99134

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintedArea (d : BarnDimensions) : ℝ :=
  2 * (2 * (d.width * d.height + d.length * d.height) + 2 * (d.width * d.length))

/-- Theorem stating that the total area to be painted for the given barn is 1368 square yards -/
theorem barn_painted_area :
  let d : BarnDimensions := { width := 12, length := 15, height := 6 }
  totalPaintedArea d = 1368 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l991_99134


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l991_99135

theorem rectangle_area_perimeter_relation :
  ∀ a b : ℕ,
  a > 10 →
  a * b = 5 * (2 * a + 2 * b) →
  2 * a + 2 * b = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l991_99135


namespace NUMINAMATH_CALUDE_max_rectangle_area_l991_99170

theorem max_rectangle_area (l w : ℝ) (h_perimeter : l + w = 10) :
  l * w ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l991_99170


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_l991_99126

theorem units_digit_factorial_sum : 
  (1 + 2 + 6 + (24 % 10)) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_l991_99126


namespace NUMINAMATH_CALUDE_clothes_spending_fraction_l991_99124

theorem clothes_spending_fraction (initial_amount remaining_amount : ℝ) : 
  initial_amount = 1249.9999999999998 →
  remaining_amount = 500 →
  ∃ (F : ℝ), 
    F > 0 ∧ F < 1 ∧
    remaining_amount = (1 - 1/4) * (1 - 1/5) * (1 - F) * initial_amount ∧
    F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clothes_spending_fraction_l991_99124


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l991_99198

theorem sum_of_quadratic_solutions :
  let f (x : ℝ) := x^2 - 6*x - 8 - (2*x + 18)
  let solutions := {x : ℝ | f x = 0}
  (∃ x₁ x₂ : ℝ, solutions = {x₁, x₂}) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s ∧ s = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l991_99198


namespace NUMINAMATH_CALUDE_max_points_32_points_32_achievable_l991_99184

/-- Represents a basketball game where a player only attempts three-point and two-point shots -/
structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  threePointSuccessRate : ℚ
  twoPointSuccessRate : ℚ

/-- Calculates the total points scored in a basketball game -/
def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccessRate * game.threePointAttempts +
  2 * game.twoPointSuccessRate * game.twoPointAttempts

/-- Theorem stating that under the given conditions, the maximum points scored is 32 -/
theorem max_points_32 (game : BasketballGame) 
    (h1 : game.threePointAttempts + game.twoPointAttempts = 40)
    (h2 : game.threePointSuccessRate = 1/4)
    (h3 : game.twoPointSuccessRate = 2/5) :
  totalPoints game ≤ 32 := by
  sorry

/-- Theorem stating that 32 points can be achieved -/
theorem points_32_achievable : 
  ∃ (game : BasketballGame), 
    game.threePointAttempts + game.twoPointAttempts = 40 ∧
    game.threePointSuccessRate = 1/4 ∧
    game.twoPointSuccessRate = 2/5 ∧
    totalPoints game = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_points_32_points_32_achievable_l991_99184


namespace NUMINAMATH_CALUDE_custom_operation_result_l991_99144

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b - c)^2

/-- Main theorem -/
theorem custom_operation_result (x y z : ℝ) :
  dollar ((x - z)^2) ((y - x)^2) ((y - z)^2) = (-2*x*z + z^2 + 2*y*x - 2*y*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l991_99144


namespace NUMINAMATH_CALUDE_first_two_satisfying_numbers_l991_99190

def satisfiesConditions (n : ℕ) : Prop :=
  n % 7 = 3 ∧ n % 9 = 4

theorem first_two_satisfying_numbers :
  ∃ (a b : ℕ), a < b ∧
  satisfiesConditions a ∧
  satisfiesConditions b ∧
  (∀ (x : ℕ), x < a → ¬satisfiesConditions x) ∧
  (∀ (x : ℕ), a < x → x < b → ¬satisfiesConditions x) ∧
  a = 31 ∧ b = 94 := by
  sorry

end NUMINAMATH_CALUDE_first_two_satisfying_numbers_l991_99190


namespace NUMINAMATH_CALUDE_interior_angle_sum_l991_99181

/-- 
Given a convex polygon where the sum of interior angles is 1800°,
prove that the sum of interior angles of a polygon with 3 fewer sides is 1260°.
-/
theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n - 3) - 2) = 1260) := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l991_99181


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l991_99180

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * (1 + Real.cos A * Real.cos B * Real.cos C) ∧
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l991_99180


namespace NUMINAMATH_CALUDE_function_inequality_l991_99120

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), deriv f x * sin x < f x * cos x) →
  Real.sqrt 3 * f (π / 4) > Real.sqrt 2 * f (π / 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l991_99120


namespace NUMINAMATH_CALUDE_cyclic_inequality_l991_99197

def cyclic_sum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

def cyclic_prod (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c * f b c a * f c a b

theorem cyclic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  cyclic_sum (fun x y z => x / (y + z)) a b c ≥ 2 - 4 * cyclic_prod (fun x y z => x / (y + z)) a b c := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l991_99197


namespace NUMINAMATH_CALUDE_circle_op_proof_l991_99112

def circle_op (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

theorem circle_op_proof (M N : Set ℕ) 
  (hM : M = {0, 2, 4, 6, 8, 10}) 
  (hN : N = {0, 3, 6, 9, 12, 15}) : 
  (circle_op (circle_op M N) M) = N := by
  sorry

#check circle_op_proof

end NUMINAMATH_CALUDE_circle_op_proof_l991_99112


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l991_99136

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - p - 1 = 0) → 
  (q^3 - q - 1 = 0) → 
  (r^3 - r - 1 = 0) → 
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 11 / 7) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l991_99136


namespace NUMINAMATH_CALUDE_football_players_count_l991_99183

/-- Represents the number of players for each sport type -/
structure PlayerCounts where
  cricket : Nat
  hockey : Nat
  softball : Nat
  total : Nat

/-- Calculates the number of football players given the counts of other players -/
def footballPlayers (counts : PlayerCounts) : Nat :=
  counts.total - (counts.cricket + counts.hockey + counts.softball)

/-- Theorem stating that the number of football players is 11 given the specific counts -/
theorem football_players_count (counts : PlayerCounts)
  (h1 : counts.cricket = 12)
  (h2 : counts.hockey = 17)
  (h3 : counts.softball = 10)
  (h4 : counts.total = 50) :
  footballPlayers counts = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l991_99183


namespace NUMINAMATH_CALUDE_cube_root_nested_l991_99157

theorem cube_root_nested (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/3))^(1/3) = N^(13/27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_nested_l991_99157


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_l991_99156

theorem exists_m_for_all_n : ∀ (n : ℤ), ∃ (m : ℤ), n * m = m := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_l991_99156


namespace NUMINAMATH_CALUDE_correct_number_of_children_l991_99188

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 5

/-- The total number of crayons -/
def total_crayons : ℕ := 50

/-- The number of children -/
def number_of_children : ℕ := total_crayons / crayons_per_child

theorem correct_number_of_children : number_of_children = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_children_l991_99188


namespace NUMINAMATH_CALUDE_sector_arc_length_l991_99161

theorem sector_arc_length (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 3 → θ = 2 * π / 3 → l = r * θ → l = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l991_99161


namespace NUMINAMATH_CALUDE_sum_and_product_problem_l991_99103

theorem sum_and_product_problem (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 ∧ x^2 + y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_problem_l991_99103


namespace NUMINAMATH_CALUDE_expression_simplification_l991_99193

theorem expression_simplification (x : ℝ) (h : x = Real.pi ^ 0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l991_99193


namespace NUMINAMATH_CALUDE_function_properties_l991_99110

-- Define the function f(x) and its derivative
def f (x : ℝ) (c : ℝ) : ℝ := x^3 - 3*x^2 + c
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem function_properties :
  -- f'(x) passes through (0,0) and (2,0)
  f_derivative 0 = 0 ∧ f_derivative 2 = 0 ∧
  -- f(x) attains its minimum at x = 2
  (∀ x : ℝ, f x (-1) ≥ f 2 (-1)) ∧
  -- The minimum value is -5
  f 2 (-1) = -5 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l991_99110


namespace NUMINAMATH_CALUDE_fraction_value_l991_99175

theorem fraction_value (x y : ℝ) (h1 : -1 < (y - x) / (x + y)) (h2 : (y - x) / (x + y) < 2) 
  (h3 : ∃ n : ℤ, y / x = n) : y / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l991_99175


namespace NUMINAMATH_CALUDE_proportion_equality_l991_99123

theorem proportion_equality (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 4 * y) : x / 4 = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l991_99123


namespace NUMINAMATH_CALUDE_speed_conversion_l991_99172

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

/-- Theorem: A speed of 70.0056 meters per second is equivalent to 252.02016 kilometers per hour -/
theorem speed_conversion : mps_to_kmph 70.0056 = 252.02016 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l991_99172


namespace NUMINAMATH_CALUDE_juhyes_money_l991_99129

theorem juhyes_money (initial_money : ℝ) : 
  (1/3 : ℝ) * (3/4 : ℝ) * initial_money = 2500 → initial_money = 10000 := by
  sorry

end NUMINAMATH_CALUDE_juhyes_money_l991_99129


namespace NUMINAMATH_CALUDE_decimal_difference_equals_fraction_l991_99164

/-- The repeating decimal 0.2̅3̅ -/
def repeating_decimal : ℚ := 23 / 99

/-- The terminating decimal 0.23 -/
def terminating_decimal : ℚ := 23 / 100

/-- The difference between the repeating decimal 0.2̅3̅ and the terminating decimal 0.23 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_equals_fraction : decimal_difference = 23 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_equals_fraction_l991_99164


namespace NUMINAMATH_CALUDE_reading_completion_time_l991_99113

/-- Represents a reader with their reading speed and number of books to read -/
structure Reader where
  speed : ℕ  -- hours per book
  books : ℕ

/-- Represents the reading schedule constraints -/
structure ReadingConstraints where
  hours_per_day : ℕ

/-- Calculate the total reading time for a reader -/
def total_reading_time (reader : Reader) : ℕ :=
  reader.speed * reader.books

/-- Calculate the number of days needed to finish reading -/
def days_to_finish (reader : Reader) (constraints : ReadingConstraints) : ℕ :=
  (total_reading_time reader + constraints.hours_per_day - 1) / constraints.hours_per_day

theorem reading_completion_time 
  (peter kristin : Reader) 
  (constraints : ReadingConstraints) 
  (h1 : peter.speed = 12)
  (h2 : kristin.speed = 3 * peter.speed)
  (h3 : peter.books = 20)
  (h4 : kristin.books = 20)
  (h5 : constraints.hours_per_day = 16) :
  kristin.speed = 36 ∧ 
  days_to_finish peter constraints = days_to_finish kristin constraints ∧
  days_to_finish kristin constraints = 45 := by
  sorry

end NUMINAMATH_CALUDE_reading_completion_time_l991_99113


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l991_99104

/-- Proves that given a journey of 540 miles, where the last 120 miles are traveled at 40 mph,
    and the average speed for the entire journey is 54 mph, the speed for the first 420 miles
    must be 60 mph. -/
theorem journey_speed_calculation (v : ℝ) : 
  v > 0 →                           -- Assume positive speed
  540 / (420 / v + 120 / 40) = 54 → -- Average speed equation
  v = 60 :=                         -- Conclusion: speed for first part is 60 mph
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l991_99104


namespace NUMINAMATH_CALUDE_road_repair_hours_l991_99152

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 42)
  (h2 : people2 = 30)
  (h3 : days1 = 12)
  (h4 : days2 = 14)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l991_99152


namespace NUMINAMATH_CALUDE_beautiful_points_of_A_beautiful_points_coincide_original_point_C_l991_99153

-- Define the type for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the beautiful points of a given point
def beautifulPoints (p : Point2D) : (Point2D × Point2D) :=
  let a := -p.x
  let b := p.x - p.y
  ({x := a, y := b}, {x := b, y := a})

-- Theorem 1: Beautiful points of A(4,1)
theorem beautiful_points_of_A :
  let A : Point2D := {x := 4, y := 1}
  let (M, N) := beautifulPoints A
  M = {x := -4, y := 3} ∧ N = {x := 3, y := -4} := by sorry

-- Theorem 2: When beautiful points of B(2,y) coincide
theorem beautiful_points_coincide :
  ∀ y : ℝ, let B : Point2D := {x := 2, y := y}
  let (M, N) := beautifulPoints B
  M = N → y = 4 := by sorry

-- Theorem 3: Original point C given a beautiful point (-2,7)
theorem original_point_C :
  ∀ C : Point2D, let (M, N) := beautifulPoints C
  (M = {x := -2, y := 7} ∨ N = {x := -2, y := 7}) →
  (C = {x := 2, y := -5} ∨ C = {x := -7, y := -5}) := by sorry

end NUMINAMATH_CALUDE_beautiful_points_of_A_beautiful_points_coincide_original_point_C_l991_99153


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l991_99109

theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (grasshopper_extra : ℕ) 
  (h1 : frog_jump = 11) 
  (h2 : grasshopper_extra = 2) : 
  frog_jump + grasshopper_extra = 13 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l991_99109


namespace NUMINAMATH_CALUDE_range_of_a_l991_99159

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

-- State the theorem
theorem range_of_a : 
  (∃ a : ℝ, C a ∪ (Set.univ \ B) = Set.univ) ↔ 
  (∃ a : ℝ, a ≤ -3 ∧ C a ∪ (Set.univ \ B) = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l991_99159


namespace NUMINAMATH_CALUDE_exists_h_not_divisible_l991_99192

theorem exists_h_not_divisible : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end NUMINAMATH_CALUDE_exists_h_not_divisible_l991_99192


namespace NUMINAMATH_CALUDE_max_true_statements_l991_99141

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i]) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l991_99141


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l991_99131

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l991_99131


namespace NUMINAMATH_CALUDE_eighth_term_is_23_l991_99115

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighth_term_is_23 :
  arithmetic_sequence 2 3 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_23_l991_99115


namespace NUMINAMATH_CALUDE_product_decimal_places_l991_99178

/-- A function that returns the number of decimal places in a decimal number -/
def decimal_places (x : ℚ) : ℕ :=
  sorry

/-- The product of two decimal numbers with one and two decimal places respectively has three decimal places -/
theorem product_decimal_places (a b : ℚ) :
  decimal_places a = 1 → decimal_places b = 2 → decimal_places (a * b) = 3 :=
sorry

end NUMINAMATH_CALUDE_product_decimal_places_l991_99178


namespace NUMINAMATH_CALUDE_kayla_bought_fifteen_items_l991_99128

/-- Represents the number of chocolate bars bought by Theresa -/
def theresa_chocolate_bars : ℕ := 12

/-- Represents the number of soda cans bought by Theresa -/
def theresa_soda_cans : ℕ := 18

/-- Represents the ratio of items bought by Theresa compared to Kayla -/
def theresa_to_kayla_ratio : ℕ := 2

/-- Calculates the total number of items bought by Kayla -/
def kayla_total_items : ℕ := 
  (theresa_chocolate_bars / theresa_to_kayla_ratio) + 
  (theresa_soda_cans / theresa_to_kayla_ratio)

/-- Theorem stating that Kayla bought 15 items in total -/
theorem kayla_bought_fifteen_items : kayla_total_items = 15 := by
  sorry

end NUMINAMATH_CALUDE_kayla_bought_fifteen_items_l991_99128


namespace NUMINAMATH_CALUDE_proportional_relation_l991_99125

theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ m : ℝ, x = m * y^3) →  -- x is directly proportional to y^3
  (∃ n : ℝ, y * z = n) →    -- y is inversely proportional to z
  (x = 5 ∧ z = 16) →        -- x = 5 when z = 16
  (z = 64 → x = 5/64) :=    -- x = 5/64 when z = 64
by sorry

end NUMINAMATH_CALUDE_proportional_relation_l991_99125


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l991_99121

-- Define the vector type
def Vec2D := ℝ × ℝ

-- Define the angle between vectors a and b
def angle_between (a b : Vec2D) : ℝ := sorry

-- Define the magnitude of a vector
def magnitude (v : Vec2D) : ℝ := sorry

-- Define the dot product of two vectors
def dot_product (a b : Vec2D) : ℝ := sorry

-- Define the vector subtraction
def vec_sub (a b : Vec2D) : Vec2D := sorry

-- Define the vector scalar multiplication
def vec_scalar_mul (r : ℝ) (v : Vec2D) : Vec2D := sorry

theorem magnitude_of_vector_combination (a b : Vec2D) :
  angle_between a b = 2 * Real.pi / 3 →
  a = (3/5, -4/5) →
  magnitude b = 2 →
  magnitude (vec_sub (vec_scalar_mul 2 a) b) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l991_99121


namespace NUMINAMATH_CALUDE_seashells_found_l991_99160

/-- The number of seashells found by Joan and Jessica -/
theorem seashells_found (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashells_found_l991_99160


namespace NUMINAMATH_CALUDE_range_of_a_l991_99155

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → 
  a > -1 ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l991_99155


namespace NUMINAMATH_CALUDE_first_part_length_l991_99127

/-- Proves that given a 60 km trip with two parts, where the second part is traveled at half the speed
    of the first part, and the average speed of the entire trip is 32 km/h, the length of the first
    part of the trip is 30 km. -/
theorem first_part_length
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_second_part = speed_first_part / 2)
  (h3 : average_speed = 32)
  : ∃ (first_part_length : ℝ),
    first_part_length = 30 ∧
    first_part_length / speed_first_part +
    (total_distance - first_part_length) / speed_second_part =
    total_distance / average_speed :=
by sorry

end NUMINAMATH_CALUDE_first_part_length_l991_99127


namespace NUMINAMATH_CALUDE_inverse_composition_l991_99176

noncomputable section

-- Define the functions f and h
variable (f h : ℂ → ℂ)

-- Define the inverse functions
variable (f_inv h_inv : ℂ → ℂ)

-- Assume f and h are bijective
variable (hf : Function.Bijective f)
variable (hh : Function.Bijective h)

-- Define the given condition
axiom condition : ∀ x, f_inv (h x) = 2 * x^2 + 4

-- State the theorem
theorem inverse_composition :
  h_inv (f 3) = Complex.I / Real.sqrt 2 ∨ h_inv (f 3) = -Complex.I / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_l991_99176


namespace NUMINAMATH_CALUDE_prob_not_adjacent_l991_99139

/-- The number of desks in the classroom -/
def num_desks : ℕ := 9

/-- The number of students choosing desks -/
def num_students : ℕ := 2

/-- The number of ways two adjacent desks can be chosen -/
def adjacent_choices : ℕ := num_desks - 1

/-- The probability that two students do not sit next to each other when randomly choosing from a row of desks -/
theorem prob_not_adjacent (n : ℕ) (k : ℕ) (h : n ≥ 2 ∧ k = 2) : 
  (1 : ℚ) - (adjacent_choices : ℚ) / (n.choose k) = 7/9 :=
sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_l991_99139


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l991_99158

theorem circle_line_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + Real.sqrt 3 * y + m = 0 ∧ 
   (x + Real.sqrt 3 * y + m + 1)^2 + y^2 = 4 * ((x + Real.sqrt 3 * y + m - 1)^2 + y^2)) → 
  -13/3 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l991_99158


namespace NUMINAMATH_CALUDE_londolozi_lion_population_l991_99154

/-- Calculates the lion population after a given number of months -/
def lionPopulation (initialPopulation birthRate deathRate months : ℕ) : ℕ :=
  initialPopulation + birthRate * months - deathRate * months

/-- Theorem: The lion population in Londolozi after 12 months -/
theorem londolozi_lion_population :
  lionPopulation 100 5 1 12 = 148 := by
  sorry

#eval lionPopulation 100 5 1 12

end NUMINAMATH_CALUDE_londolozi_lion_population_l991_99154


namespace NUMINAMATH_CALUDE_inequality_solution_l991_99137

theorem inequality_solution (x : ℝ) :
  x ≠ 4 →
  (x * (x + 1) / (x - 4)^2 ≥ 15 ↔ x ∈ Set.Iic 3 ∪ Set.Ioo (40/7) 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l991_99137


namespace NUMINAMATH_CALUDE_prove_M_value_l991_99163

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : Int
  diff : Int

/-- The row sequence -/
def rowSeq : ArithmeticSequence := { first := 12, diff := -7 }

/-- The first column sequence -/
def col1Seq : ArithmeticSequence := { first := -11, diff := 9 }

/-- The second column sequence -/
def col2Seq : ArithmeticSequence := { first := -35, diff := 5 }

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : Nat) : Int :=
  seq.first + seq.diff * (n - 1)

theorem prove_M_value : 
  nthTerm rowSeq 1 = 12 ∧ 
  nthTerm col1Seq 4 = 7 ∧ 
  nthTerm col1Seq 5 = 16 ∧
  nthTerm col2Seq 5 = -10 ∧
  col2Seq.first = -35 := by sorry

end NUMINAMATH_CALUDE_prove_M_value_l991_99163


namespace NUMINAMATH_CALUDE_range_of_a_l991_99146

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3 - a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ 0) → a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l991_99146


namespace NUMINAMATH_CALUDE_johny_total_distance_l991_99179

def johny_journey (south_distance : ℕ) : ℕ :=
  let east_distance := south_distance + 20
  let north_distance := 2 * east_distance
  south_distance + east_distance + north_distance

theorem johny_total_distance :
  johny_journey 40 = 220 := by
  sorry

end NUMINAMATH_CALUDE_johny_total_distance_l991_99179


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l991_99150

/-- Given a loan with simple interest where the loan period equals the interest rate,
    prove that the interest rate is 3% when the principal is $1200 and the total interest is $108. -/
theorem interest_rate_calculation (principal : ℝ) (total_interest : ℝ) :
  principal = 1200 →
  total_interest = 108 →
  ∃ (rate : ℝ),
    total_interest = principal * rate * rate / 100 ∧
    rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l991_99150


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l991_99173

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (3, 4, 5). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = -3 →
  ρ * Real.sin φ * Real.sin θ = -4 →
  ρ * Real.cos φ = 5 →
  ρ * Real.sin (-φ) * Real.cos (θ + π) = 3 ∧
  ρ * Real.sin (-φ) * Real.sin (θ + π) = 4 ∧
  ρ * Real.cos (-φ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l991_99173


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l991_99182

theorem complex_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l991_99182


namespace NUMINAMATH_CALUDE_max_mice_two_kittens_max_mice_two_males_l991_99111

/-- Represents the production possibility frontier (PPF) for a kitten --/
structure KittenPPF where
  maxMice : ℕ  -- Maximum number of mice caught when K = 0
  slope : ℚ    -- Rate of decrease in mice caught per hour of therapy

/-- Calculates the number of mice caught given hours of therapy --/
def micesCaught (ppf : KittenPPF) (therapyHours : ℚ) : ℚ :=
  ppf.maxMice - ppf.slope * therapyHours

/-- Male kitten PPF --/
def malePPF : KittenPPF := { maxMice := 80, slope := 4 }

/-- Female kitten PPF --/
def femalePPF : KittenPPF := { maxMice := 16, slope := 1/4 }

/-- Theorem: The maximum number of mice caught by 2 kittens is 160 --/
theorem max_mice_two_kittens :
  ∀ (k1 k2 : KittenPPF), ∀ (h1 h2 : ℚ),
    micesCaught k1 h1 + micesCaught k2 h2 ≤ 160 :=
by sorry

/-- Corollary: The maximum is achieved with two male kittens and zero therapy hours --/
theorem max_mice_two_males :
  micesCaught malePPF 0 + micesCaught malePPF 0 = 160 :=
by sorry

end NUMINAMATH_CALUDE_max_mice_two_kittens_max_mice_two_males_l991_99111


namespace NUMINAMATH_CALUDE_sum_a_b_equals_five_l991_99107

theorem sum_a_b_equals_five (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 3*a + 4*b = 18) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_five_l991_99107


namespace NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l991_99108

theorem tenth_power_sum_of_roots (r s : ℂ) : 
  (r^2 - 2*r + 4 = 0) → (s^2 - 2*s + 4 = 0) → r^10 + s^10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l991_99108


namespace NUMINAMATH_CALUDE_f_shifted_up_is_g_l991_99148

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the shifted function g
def g : ℝ → ℝ := sorry

-- Theorem stating that g is f shifted up by 1
theorem f_shifted_up_is_g : ∀ x : ℝ, g x = f x + 1 := by sorry

end NUMINAMATH_CALUDE_f_shifted_up_is_g_l991_99148


namespace NUMINAMATH_CALUDE_mean_median_difference_l991_99140

-- Define the frequency distribution of sick days
def sick_days_freq : List (Nat × Nat) := [(0, 4), (1, 2), (2, 5), (3, 2), (4, 1), (5, 1)]

-- Total number of students
def total_students : Nat := 15

-- Function to calculate the median
def median (freq : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

-- Function to calculate the mean
def mean (freq : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

-- Theorem statement
theorem mean_median_difference :
  mean sick_days_freq total_students = (median sick_days_freq total_students : Rat) - 1/5 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l991_99140


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l991_99105

theorem simplify_fraction_sum (n d : Nat) : 
  n = 75 → d = 100 → ∃ (a b : Nat), (a.gcd b = 1) ∧ (n * b = d * a) ∧ (a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l991_99105


namespace NUMINAMATH_CALUDE_fraction_calculation_l991_99116

theorem fraction_calculation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) * (2 : ℚ) / 3 = 98 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l991_99116


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l991_99101

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ 105 ∧ (∃ (P Q R : ℕ+), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P * Q * R = 3003 ∧ P + Q + R = 105) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l991_99101


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l991_99132

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y = x + 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l991_99132


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l991_99168

/-- The amount of money Chris had before his birthday. -/
def money_before_birthday (grandmother_gift aunt_uncle_gift parents_gift total_now : ℕ) : ℕ :=
  total_now - (grandmother_gift + aunt_uncle_gift + parents_gift)

/-- Theorem stating that Chris had $239 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday 25 20 75 359 = 239 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l991_99168


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l991_99118

theorem quadratic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = -2 ∨ x = 1) → 
  a = 1 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l991_99118


namespace NUMINAMATH_CALUDE_tournament_probability_l991_99149

/-- The number of teams in the tournament -/
def num_teams : ℕ := 50

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The probability of a team winning any single game -/
def win_probability : ℚ := 1 / 2

/-- The number of possible outcomes in the tournament -/
def total_outcomes : ℕ := 2^total_games

/-- The number of favorable outcomes (where no two teams win the same number of games) -/
def favorable_outcomes : ℕ := num_teams.factorial

/-- The probability that no two teams win the same number of games -/
def probability : ℚ := favorable_outcomes / total_outcomes

/-- The denominator of the probability fraction in its lowest terms -/
def q : ℕ := 2^1178

theorem tournament_probability : 
  probability = favorable_outcomes / total_outcomes ∧ 
  (probability.den : ℕ) = q :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l991_99149


namespace NUMINAMATH_CALUDE_parabola_area_l991_99165

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the roots of the parabola
def root1 : ℝ := 1
def root2 : ℝ := 3

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in root1..root2, -f x) = 4/3 := by sorry

end NUMINAMATH_CALUDE_parabola_area_l991_99165


namespace NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l991_99119

/-- A structure representing a scalene triangle with its properties -/
structure ScaleneTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The area of the triangle
  S : ℝ
  -- The longest angle bisector
  l₁ : ℝ
  -- The shortest angle bisector
  l₂ : ℝ
  -- Conditions for a scalene triangle
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Positive sides and area
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < S
  -- Triangle inequality
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  -- l₁ is the longest bisector
  l₁_longest : l₁ ≥ l₂
  -- l₂ is the shortest bisector
  l₂_shortest : l₂ ≤ l₁

/-- The main theorem stating the inequality for scalene triangles -/
theorem scalene_triangle_bisector_inequality (t : ScaleneTriangle) :
  t.l₁^2 > Real.sqrt 3 * t.S ∧ Real.sqrt 3 * t.S > t.l₂^2 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_bisector_inequality_l991_99119


namespace NUMINAMATH_CALUDE_martha_apples_problem_l991_99189

/-- Given that Martha has 20 apples initially, gives 5 to Jane and 2 more than that to James,
    prove that she needs to give away 4 more apples to be left with exactly 4 apples. -/
theorem martha_apples_problem (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : james_extra_apples = 2) :
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - 4 = 4 := by
  sorry

#check martha_apples_problem

end NUMINAMATH_CALUDE_martha_apples_problem_l991_99189


namespace NUMINAMATH_CALUDE_acorn_theorem_l991_99117

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
theorem acorn_theorem (shawna sheila danny : ℕ) : 
  shawna = 7 →
  sheila = 5 * shawna →
  danny = sheila + 3 →
  shawna + sheila + danny = 80 := by
  sorry

end NUMINAMATH_CALUDE_acorn_theorem_l991_99117


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l991_99122

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def angle (a b : E) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem vector_angle_theorem (a b : E) (k : ℝ) (hk : k ≠ 0) 
  (h : norm (a + k • b) = norm (a - b)) : 
  (k = -1 → angle a b = Real.pi / 2) ∧ 
  (k ≠ -1 → angle a b = Real.arccos (-1 / (k + 1))) :=
sorry

end

end NUMINAMATH_CALUDE_vector_angle_theorem_l991_99122


namespace NUMINAMATH_CALUDE_triangle_side_length_l991_99186

theorem triangle_side_length (b c : ℝ) (C : ℝ) (h1 : b = 6 * Real.sqrt 3) (h2 : c = 6) (h3 : C = 30 * π / 180) :
  ∃ (a : ℝ), (a = 6 ∨ a = 12) ∧ c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l991_99186


namespace NUMINAMATH_CALUDE_triangle_existence_l991_99102

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
  sorry

/-- Checks if an angle is obtuse -/
def isObtuse (θ : ℝ) : Prop :=
  θ > Real.pi / 2

/-- Constructs a triangle given A₀, A₁, and A₂ -/
noncomputable def constructTriangle (A₀ A₁ A₂ : Point) : Option Triangle :=
  sorry

theorem triangle_existence (A₀ A₁ A₂ : Point) :
  ¬collinear A₀ A₁ A₂ →
  isObtuse (angle A₀ A₁ A₂) →
  ∃! (t : Triangle),
    (constructTriangle A₀ A₁ A₂ = some t) ∧
    (A₀.x = (t.B.x + t.C.x) / 2 ∧ A₀.y = (t.B.y + t.C.y) / 2) ∧
    collinear A₁ t.B t.C ∧
    (let midAlt := Point.mk ((t.A.x + A₀.x) / 2) ((t.A.y + A₀.y) / 2);
     A₂ = midAlt) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l991_99102


namespace NUMINAMATH_CALUDE_mens_haircut_time_is_correct_l991_99106

/-- The time it takes to cut a man's hair -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a woman's hair -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a kid's hair -/
def kids_haircut_time : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def num_womens_haircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def num_mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def num_kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair -/
def total_time : ℕ := 255

theorem mens_haircut_time_is_correct :
  num_womens_haircuts * womens_haircut_time +
  num_mens_haircuts * mens_haircut_time +
  num_kids_haircuts * kids_haircut_time = total_time := by
sorry

end NUMINAMATH_CALUDE_mens_haircut_time_is_correct_l991_99106


namespace NUMINAMATH_CALUDE_sqrt_calculation_l991_99194

theorem sqrt_calculation : 
  Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2)^2 = 8 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l991_99194


namespace NUMINAMATH_CALUDE_chain_store_max_profit_l991_99133

/-- Annual profit function for a chain store -/
def L (x a : ℝ) : ℝ := (x - 4 - a) * (10 - x)^2

/-- Maximum annual profit for the chain store -/
theorem chain_store_max_profit (a : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) :
  ∃ (L_max : ℝ),
    (∀ x, 7 ≤ x → x ≤ 9 → L x a ≤ L_max) ∧
    ((1 ≤ a ∧ a ≤ 3/2 → L_max = 27 - 9*a) ∧
     (3/2 < a ∧ a ≤ 3 → L_max = 4*(2 - a/3)^3)) :=
sorry

end NUMINAMATH_CALUDE_chain_store_max_profit_l991_99133


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l991_99138

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
               Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l991_99138


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l991_99162

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ (n : ℕ), n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l991_99162

import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_function_periodic_l4131_413182

/-- A function f: ℝ → ℝ satisfying certain symmetry properties -/
def symmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (2 - x)) ∧ (∀ x, f (x + 7) = f (7 - x))

/-- Theorem stating that a symmetric function is periodic with period 10 -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (h : symmetricFunction f) :
  ∀ x, f (x + 10) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_periodic_l4131_413182


namespace NUMINAMATH_CALUDE_modulus_of_complex_l4131_413103

theorem modulus_of_complex (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l4131_413103


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l4131_413140

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l4131_413140


namespace NUMINAMATH_CALUDE_arithmetic_computation_l4131_413160

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l4131_413160


namespace NUMINAMATH_CALUDE_base_approximation_l4131_413142

/-- The base value we're looking for -/
def base : ℝ := 21.5

/-- The function representing the left side of the inequality -/
def f (b : ℝ) : ℝ := 2.134 * b^3

theorem base_approximation :
  ∀ b : ℝ, f b < 21000 → b ≤ base :=
sorry

end NUMINAMATH_CALUDE_base_approximation_l4131_413142


namespace NUMINAMATH_CALUDE_min_groups_for_students_l4131_413163

theorem min_groups_for_students (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ), 
    num_groups * (total_students / num_groups) = total_students ∧
    (total_students / num_groups) ≤ max_group_size ∧
    ∀ (k : ℕ), k * (total_students / k) = total_students ∧ (total_students / k) ≤ max_group_size → k ≥ num_groups :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_students_l4131_413163


namespace NUMINAMATH_CALUDE_factorization_proof_l4131_413130

theorem factorization_proof (x y : ℝ) : 91 * x^7 - 273 * x^14 * y^3 = 91 * x^7 * (1 - 3 * x^7 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4131_413130


namespace NUMINAMATH_CALUDE_emilys_initial_lives_l4131_413135

theorem emilys_initial_lives :
  ∀ (initial_lives : ℕ),
  initial_lives - 25 + 24 = 41 →
  initial_lives = 42 :=
by sorry

end NUMINAMATH_CALUDE_emilys_initial_lives_l4131_413135


namespace NUMINAMATH_CALUDE_range_of_a_l4131_413146

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 4, x^2 ≥ a) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a = 1 ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4131_413146


namespace NUMINAMATH_CALUDE_apples_basket_value_l4131_413153

/-- Given a total number of apples, number of baskets, and price per apple,
    calculates the value of apples in one basket. -/
def value_of_basket (total_apples : ℕ) (num_baskets : ℕ) (price_per_apple : ℕ) : ℕ :=
  (total_apples / num_baskets) * price_per_apple

/-- Theorem stating that the value of apples in one basket is 6000 won
    given the specific conditions of the problem. -/
theorem apples_basket_value :
  value_of_basket 180 6 200 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_apples_basket_value_l4131_413153


namespace NUMINAMATH_CALUDE_total_cost_calculation_l4131_413177

def cabin_cost : ℕ := 6000
def land_cost_multiplier : ℕ := 4

theorem total_cost_calculation :
  cabin_cost + land_cost_multiplier * cabin_cost = 30000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l4131_413177


namespace NUMINAMATH_CALUDE_three_dozens_equals_42_l4131_413137

/-- Calculates the total number of flowers a customer receives when buying dozens of flowers with a free flower promotion. -/
def totalFlowers (dozens : ℕ) : ℕ :=
  let boughtFlowers := dozens * 12
  let freeFlowers := dozens * 2
  boughtFlowers + freeFlowers

/-- Theorem stating that buying 3 dozens of flowers results in 42 total flowers. -/
theorem three_dozens_equals_42 :
  totalFlowers 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_three_dozens_equals_42_l4131_413137


namespace NUMINAMATH_CALUDE_lightest_pumpkin_weight_l4131_413144

/-- Given three pumpkins with weights A, B, and C, prove that A = 5 -/
theorem lightest_pumpkin_weight (A B C : ℕ) 
  (h1 : A ≤ B) (h2 : B ≤ C)
  (h3 : A + B = 12) (h4 : A + C = 13) (h5 : B + C = 15) : 
  A = 5 := by
  sorry

end NUMINAMATH_CALUDE_lightest_pumpkin_weight_l4131_413144


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l4131_413150

noncomputable def f (x : ℝ) : ℝ := 1 + (Real.sin x) / (2 + Real.cos x)

theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l4131_413150


namespace NUMINAMATH_CALUDE_f_composition_of_three_l4131_413139

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_composition_of_three : f (f (f (f 3))) = 187 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l4131_413139


namespace NUMINAMATH_CALUDE_power_inequality_l4131_413169

theorem power_inequality (a b n : ℕ) (ha : a > b) (hb : b > 1) (hodd : Odd b) 
  (hn : n > 0) (hdiv : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a : ℝ)^b > (3 : ℝ)^n / n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l4131_413169


namespace NUMINAMATH_CALUDE_cube_edge_length_is_twelve_l4131_413178

/-- Represents a cube with integer edge length -/
structure Cube where
  edge_length : ℕ

/-- Calculates the number of small cubes with three painted faces -/
def three_painted_faces (c : Cube) : ℕ := 8

/-- Calculates the number of small cubes with two painted faces -/
def two_painted_faces (c : Cube) : ℕ := 12 * (c.edge_length - 2)

/-- Theorem stating that when the number of small cubes with two painted faces
    is 15 times the number of small cubes with three painted faces,
    the edge length of the cube must be 12 -/
theorem cube_edge_length_is_twelve (c : Cube) :
  two_painted_faces c = 15 * three_painted_faces c → c.edge_length = 12 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_is_twelve_l4131_413178


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l4131_413148

theorem lcm_hcf_problem (n : ℕ) : 
  Nat.lcm 8 n = 24 → Nat.gcd 8 n = 4 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l4131_413148


namespace NUMINAMATH_CALUDE_carrot_eating_problem_l4131_413110

theorem carrot_eating_problem :
  ∃ (x y z : ℕ), x + y + z = 15 ∧ z % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_carrot_eating_problem_l4131_413110


namespace NUMINAMATH_CALUDE_find_number_l4131_413196

theorem find_number : ∃ N : ℕ,
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  N / sum = quotient ∧ N % sum = 20 ∧ N = 220020 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4131_413196


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l4131_413166

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 144 = 0 ↔ x = -13 ∨ x = 11 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 4*x - 32 = 0 ↔ x = 8 ∨ x = -4 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3 := by sorry

-- Equation 4
theorem equation_four_solution (x : ℝ) :
  (x + 3)^2 = 2*x + 5 ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l4131_413166


namespace NUMINAMATH_CALUDE_simple_interest_principal_l4131_413154

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℚ)
  (time : ℚ)
  (interest : ℚ)
  (h_rate : rate = 4 / 100)
  (h_time : time = 1)
  (h_interest : interest = 400) :
  interest = (10000 : ℚ) * rate * time :=
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l4131_413154


namespace NUMINAMATH_CALUDE_prob_at_least_one_8_l4131_413109

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting at least one 8 when rolling two fair 8-sided dice -/
def probAtLeastOne8 : ℚ := 15 / 64

/-- Theorem: The probability of getting at least one 8 when rolling two fair 8-sided dice is 15/64 -/
theorem prob_at_least_one_8 : 
  probAtLeastOne8 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_8_l4131_413109


namespace NUMINAMATH_CALUDE_curve_intersection_l4131_413125

theorem curve_intersection :
  ∃ (θ t : ℝ),
    0 ≤ θ ∧ θ ≤ π ∧
    Real.sqrt 5 * Real.cos θ = 5/6 ∧
    Real.sin θ = 2/3 ∧
    (5/4) * t = 5/6 ∧
    t = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_curve_intersection_l4131_413125


namespace NUMINAMATH_CALUDE_floor_sqrt_200_l4131_413111

theorem floor_sqrt_200 : ⌊Real.sqrt 200⌋ = 14 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_200_l4131_413111


namespace NUMINAMATH_CALUDE_ellipse_implies_a_greater_than_one_l4131_413119

/-- Represents the condition that the curve is an ellipse with foci on the x-axis -/
def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (3 - t) + y^2 / (t + 1) = 1 → -1 < t ∧ t < 1

/-- Represents the inequality condition -/
def satisfies_inequality (t a : ℝ) : Prop :=
  t^2 - (a - 1) * t - a < 0

/-- The main theorem statement -/
theorem ellipse_implies_a_greater_than_one :
  (∀ t : ℝ, is_ellipse t → (∃ a : ℝ, satisfies_inequality t a)) ∧
  (∃ t a : ℝ, satisfies_inequality t a ∧ ¬is_ellipse t) →
  ∀ a : ℝ, (∃ t : ℝ, is_ellipse t → satisfies_inequality t a) → a > 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_implies_a_greater_than_one_l4131_413119


namespace NUMINAMATH_CALUDE_watch_cost_price_l4131_413116

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 5 →
  additional_amount = 180 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l4131_413116


namespace NUMINAMATH_CALUDE_product_of_fractions_l4131_413192

theorem product_of_fractions : (1 : ℚ) / 3 * 2 / 5 * 3 / 7 * 4 / 8 = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l4131_413192


namespace NUMINAMATH_CALUDE_mika_stickers_l4131_413108

/-- The number of stickers Mika has after all events -/
def final_stickers (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 28 stickers -/
theorem mika_stickers : final_stickers 45 53 35 19 86 = 28 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l4131_413108


namespace NUMINAMATH_CALUDE_find_divisor_l4131_413104

theorem find_divisor : ∃ (d : ℕ), d > 0 ∧ 
  (13603 - 31) % d = 0 ∧
  (∀ (n : ℕ), n < 31 → (13603 - n) % d ≠ 0) ∧
  d = 13572 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l4131_413104


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l4131_413132

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of spaces where zeros can be placed -/
def total_spaces : ℕ := num_ones + 1

/-- The probability that two zeros are not adjacent when randomly arranged with four ones -/
theorem zeros_not_adjacent_probability : 
  (Nat.choose total_spaces num_zeros : ℚ) / 
  (Nat.choose total_spaces 1 + Nat.choose total_spaces num_zeros : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l4131_413132


namespace NUMINAMATH_CALUDE_monkey_family_size_l4131_413145

/-- The number of monkeys in a family that collected bananas -/
def number_of_monkeys : ℕ := by sorry

theorem monkey_family_size :
  let total_piles : ℕ := 10
  let piles_type1 : ℕ := 6
  let hands_per_pile_type1 : ℕ := 9
  let bananas_per_hand_type1 : ℕ := 14
  let piles_type2 : ℕ := total_piles - piles_type1
  let hands_per_pile_type2 : ℕ := 12
  let bananas_per_hand_type2 : ℕ := 9
  let bananas_per_monkey : ℕ := 99

  let total_bananas : ℕ := 
    piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2

  number_of_monkeys = total_bananas / bananas_per_monkey := by sorry

end NUMINAMATH_CALUDE_monkey_family_size_l4131_413145


namespace NUMINAMATH_CALUDE_x_eighth_is_zero_l4131_413123

theorem x_eighth_is_zero (x : ℝ) (h : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_is_zero_l4131_413123


namespace NUMINAMATH_CALUDE_equal_function_values_l4131_413131

/-- Given a function f(x) = ax^2 - 2ax + 1 where a > 1, prove that f(x₁) = f(x₂) when x₁ < x₂ and x₁ + x₂ = 1 + a -/
theorem equal_function_values
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 1)
  (hx : x₁ < x₂)
  (hsum : x₁ + x₂ = 1 + a)
  : a * x₁^2 - 2*a*x₁ + 1 = a * x₂^2 - 2*a*x₂ + 1 :=
by sorry

end NUMINAMATH_CALUDE_equal_function_values_l4131_413131


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l4131_413173

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) ≤ (1 : ℝ) / 4 ∧
  (xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) = (1 : ℝ) / 4 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l4131_413173


namespace NUMINAMATH_CALUDE_solve_equation_l4131_413183

theorem solve_equation (y : ℝ) : 3/4 + 1/y = 7/8 ↔ y = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l4131_413183


namespace NUMINAMATH_CALUDE_set_existence_condition_l4131_413122

theorem set_existence_condition (r : ℝ) (hr : 0 < r ∧ r < 1) :
  (∃ S : Set ℝ, 
    (∀ t : ℝ, (t ∈ S ∨ (t + r) ∈ S ∨ (t + 1) ∈ S) ∧
              (t ∉ S ∨ (t + r) ∉ S) ∧ ((t + r) ∉ S ∨ (t + 1) ∉ S) ∧ (t ∉ S ∨ (t + 1) ∉ S)) ∧
    (∀ t : ℝ, (t ∈ S ∨ (t - r) ∈ S ∨ (t - 1) ∈ S) ∧
              (t ∉ S ∨ (t - r) ∉ S) ∧ ((t - r) ∉ S ∨ (t - 1) ∉ S) ∧ (t ∉ S ∨ (t - 1) ∉ S))) ↔
  (¬ ∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ)) ∨
  (∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ) ∧ 3 ∣ (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_set_existence_condition_l4131_413122


namespace NUMINAMATH_CALUDE_distinct_triangles_on_circle_l4131_413176

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of distinct triangles that can be drawn -/
def num_triangles : ℕ := Nat.choose n k

theorem distinct_triangles_on_circle :
  num_triangles = 220 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_on_circle_l4131_413176


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l4131_413199

/-- Proves that given the initial conditions, the amount of water evaporated per day is correct -/
theorem water_evaporation_per_day 
  (initial_water : ℝ) 
  (evaporation_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_water = 10) 
  (h2 : evaporation_percentage = 7.000000000000001) 
  (h3 : days = 50) : 
  (initial_water * evaporation_percentage / 100) / days = 0.014000000000000002 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l4131_413199


namespace NUMINAMATH_CALUDE_circle_ratio_new_circumference_to_area_increase_l4131_413101

/-- The ratio of new circumference to increase in area when a circle's radius is increased -/
theorem circle_ratio_new_circumference_to_area_increase 
  (r k : ℝ) (h : k > 0) : 
  (2 * Real.pi * (r + k)) / (Real.pi * ((r + k)^2 - r^2)) = 2 * (r + k) / (2 * r * k + k^2) :=
sorry

end NUMINAMATH_CALUDE_circle_ratio_new_circumference_to_area_increase_l4131_413101


namespace NUMINAMATH_CALUDE_not_5x_representation_l4131_413115

-- Define the expressions
def expr_A (x : ℝ) : ℝ := 5 * x
def expr_B (x : ℝ) : ℝ := x^5
def expr_C (x : ℝ) : ℝ := x + x + x + x + x

-- Theorem stating that B is not equal to 5x, while A and C are
theorem not_5x_representation (x : ℝ) : 
  expr_A x = 5 * x ∧ expr_C x = 5 * x ∧ expr_B x ≠ 5 * x :=
sorry

end NUMINAMATH_CALUDE_not_5x_representation_l4131_413115


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l4131_413152

/-- Theorem: Ratio of boys to girls in a class --/
theorem boys_to_girls_ratio 
  (boys_avg : ℝ) 
  (girls_avg : ℝ) 
  (class_avg : ℝ) 
  (missing_scores : ℕ) 
  (missing_avg : ℝ) 
  (h1 : boys_avg = 90) 
  (h2 : girls_avg = 96) 
  (h3 : class_avg = 94) 
  (h4 : missing_scores = 3) 
  (h5 : missing_avg = 92) :
  ∃ (boys girls : ℕ), 
    boys > 0 ∧ 
    girls > 0 ∧ 
    (boys : ℝ) / girls = 1 / 5 ∧
    class_avg * (boys + girls + missing_scores : ℝ) = 
      boys_avg * boys + girls_avg * girls + missing_avg * missing_scores :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l4131_413152


namespace NUMINAMATH_CALUDE_marquita_garden_width_marquita_garden_width_proof_l4131_413171

/-- The width of Marquita's gardens given the conditions of the problem -/
theorem marquita_garden_width : ℝ :=
  let mancino_garden_count : ℕ := 3
  let mancino_garden_length : ℝ := 16
  let mancino_garden_width : ℝ := 5
  let marquita_garden_count : ℕ := 2
  let marquita_garden_length : ℝ := 8
  let total_area : ℝ := 304

  let mancino_total_area := mancino_garden_count * mancino_garden_length * mancino_garden_width
  let marquita_total_area := total_area - mancino_total_area
  let marquita_garden_area := marquita_total_area / marquita_garden_count
  let marquita_garden_width := marquita_garden_area / marquita_garden_length

  4

theorem marquita_garden_width_proof : marquita_garden_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_width_marquita_garden_width_proof_l4131_413171


namespace NUMINAMATH_CALUDE_sum_of_first_50_digits_l4131_413159

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => match n % 5 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | _ => 0  -- This case is technically unreachable

/-- Sum of the first n digits in the decimal expansion -/
def sum_of_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- The theorem stating the sum of the first 50 digits after the decimal point in 1/10101 -/
theorem sum_of_first_50_digits :
  sum_of_digits 50 = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_50_digits_l4131_413159


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l4131_413124

theorem movie_theater_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : adult_tickets = 500)
  : adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = 5100 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_l4131_413124


namespace NUMINAMATH_CALUDE_bank_profit_maximization_l4131_413189

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) (h_k_pos : k > 0) :
  let deposit_amount (x : ℝ) := k * x
  let profit (x : ℝ) := 0.048 * deposit_amount x - x * deposit_amount x
  ∃ (x_max : ℝ), x_max ∈ Set.Ioo 0 0.048 ∧
    ∀ (x : ℝ), x ∈ Set.Ioo 0 0.048 → profit x ≤ profit x_max ∧
    x_max = 0.024 :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_maximization_l4131_413189


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4131_413120

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1) ∧
    P = -6 ∧ Q = 8 ∧ R = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4131_413120


namespace NUMINAMATH_CALUDE_unique_x_intercept_l4131_413113

/-- The parabola equation: x = -3y^2 + 2y + 4 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

theorem unique_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l4131_413113


namespace NUMINAMATH_CALUDE_special_triangle_unique_values_l4131_413105

/-- An isosceles triangle with a specific internal point -/
structure SpecialTriangle where
  -- The side length of the two equal sides
  s : ℝ
  -- The base length
  t : ℝ
  -- Coordinates of the internal point P
  px : ℝ
  py : ℝ
  -- Assertion that the triangle is isosceles
  h_isosceles : s > 0
  -- Assertion that P is inside the triangle
  h_inside : 0 < px ∧ px < t ∧ 0 < py ∧ py < s
  -- Distance from A to P is 2
  h_ap : px^2 + py^2 = 4
  -- Distance from B to P is 2√2
  h_bp : (t - px)^2 + py^2 = 8
  -- Distance from C to P is 3
  h_cp : px^2 + (s - py)^2 = 9

/-- The theorem stating the unique values of s and t -/
theorem special_triangle_unique_values (tri : SpecialTriangle) : 
  tri.s = 2 * Real.sqrt 3 ∧ tri.t = 6 := by sorry

end NUMINAMATH_CALUDE_special_triangle_unique_values_l4131_413105


namespace NUMINAMATH_CALUDE_sin_product_equality_l4131_413112

theorem sin_product_equality : 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) * Real.sin (60 * π / 180) * Real.sin (85 * π / 180) =
  Real.sin (20 * π / 180) * Real.sin (40 * π / 180) * Real.sin (75 * π / 180) * Real.sin (80 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_sin_product_equality_l4131_413112


namespace NUMINAMATH_CALUDE_percentage_of_seniors_with_cars_l4131_413168

theorem percentage_of_seniors_with_cars :
  ∀ (total_students : ℕ) 
    (seniors : ℕ) 
    (lower_grades : ℕ) 
    (lower_grades_car_percentage : ℚ) 
    (total_car_percentage : ℚ),
  total_students = 1200 →
  seniors = 300 →
  lower_grades = 900 →
  lower_grades_car_percentage = 1/10 →
  total_car_percentage = 1/5 →
  (↑seniors * (seniors_car_percentage : ℚ) + ↑lower_grades * lower_grades_car_percentage) / ↑total_students = total_car_percentage →
  seniors_car_percentage = 1/2 :=
by
  sorry

#check percentage_of_seniors_with_cars

end NUMINAMATH_CALUDE_percentage_of_seniors_with_cars_l4131_413168


namespace NUMINAMATH_CALUDE_a_2016_div_2017_l4131_413162

/-- The sequence a defined by the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) + 41 * a n

/-- The theorem stating that the 2016th term of the sequence is divisible by 2017 -/
theorem a_2016_div_2017 : 2017 ∣ a 2016 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_div_2017_l4131_413162


namespace NUMINAMATH_CALUDE_intersection_M_N_l4131_413106

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4131_413106


namespace NUMINAMATH_CALUDE_product_evaluation_l4131_413165

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l4131_413165


namespace NUMINAMATH_CALUDE_tablet_screen_area_difference_l4131_413136

theorem tablet_screen_area_difference : 
  let diagonal_8 : ℝ := 8
  let diagonal_7 : ℝ := 7
  let area_8 : ℝ := (diagonal_8^2) / 2
  let area_7 : ℝ := (diagonal_7^2) / 2
  area_8 - area_7 = 7.5 := by sorry

end NUMINAMATH_CALUDE_tablet_screen_area_difference_l4131_413136


namespace NUMINAMATH_CALUDE_jasons_journey_l4131_413188

/-- The distance to Jason's home --/
def distance_to_home (total_time : ℝ) (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * (total_time - time1)

/-- Theorem statement for Jason's journey --/
theorem jasons_journey :
  let total_time : ℝ := 1.5
  let speed1 : ℝ := 60
  let time1 : ℝ := 0.5
  let speed2 : ℝ := 90
  distance_to_home total_time speed1 time1 speed2 = 120 := by
sorry

end NUMINAMATH_CALUDE_jasons_journey_l4131_413188


namespace NUMINAMATH_CALUDE_optimal_game_result_exact_distinct_rows_l4131_413141

/-- Represents the game board -/
def GameBoard := Fin (2^100) → Fin 100 → Bool

/-- Player A's strategy -/
def StrategyA := GameBoard → Fin 100 → Fin 100

/-- Player B's strategy -/
def StrategyB := GameBoard → Fin 100 → Fin 100

/-- Counts the number of distinct rows in a game board -/
def countDistinctRows (board : GameBoard) : ℕ := sorry

/-- Simulates the game with given strategies -/
def playGame (stratA : StrategyA) (stratB : StrategyB) : GameBoard := sorry

/-- The main theorem stating the result of the game -/
theorem optimal_game_result :
  ∀ (stratA : StrategyA) (stratB : StrategyB),
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA stratB) ≥ 2^50 ∧
  countDistinctRows (playGame stratA optimalB) ≤ 2^50 := by sorry

/-- The final theorem stating the exact number of distinct rows -/
theorem exact_distinct_rows :
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA optimalB) = 2^50 := by sorry

end NUMINAMATH_CALUDE_optimal_game_result_exact_distinct_rows_l4131_413141


namespace NUMINAMATH_CALUDE_certain_number_value_l4131_413175

theorem certain_number_value (x : ℝ) (certain_number : ℝ) : 
  (100 + 200 + 300 + x) / 4 = 250 ∧ 
  (300 + 150 + 100 + x + certain_number) / 5 = 200 → 
  certain_number = 50 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l4131_413175


namespace NUMINAMATH_CALUDE_power_product_inequality_l4131_413149

/-- Given positive real numbers a, b, and c, 
    a^a * b^b * c^c ≥ (a * b * c)^((a+b+c)/3) -/
theorem power_product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := by
  sorry

end NUMINAMATH_CALUDE_power_product_inequality_l4131_413149


namespace NUMINAMATH_CALUDE_linear_equation_solution_range_l4131_413156

theorem linear_equation_solution_range (x k : ℝ) : 
  (2 * x - 5 * k = x + 4) → (x > 0) → (k > -4/5) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_range_l4131_413156


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l4131_413121

theorem power_multiplication_equality : (512 : ℝ)^(2/3) * 8 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l4131_413121


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l4131_413172

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line to be parallel to y = 4x - 1
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the point P₀
structure Point_P₀ where
  x : ℝ
  y : ℝ
  on_curve : f x = y
  tangent_parallel : tangent_parallel x

-- State the theorem
theorem tangent_point_coordinates :
  ∀ p : Point_P₀, (p.x = 1 ∧ p.y = 0) ∨ (p.x = -1 ∧ p.y = -4) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l4131_413172


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l4131_413190

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c m n : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ m < x ∧ x < n

-- State the theorem
theorem quadratic_inequality_properties
  (a b c m n : ℝ)
  (h_sol : solution_set a b c m n)
  (h_m_pos : m > 0)
  (h_n_gt_m : n > m) :
  a < 0 ∧
  b > 0 ∧
  (∀ x, f c b a x > 0 ↔ 1/n < x ∧ x < 1/m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l4131_413190


namespace NUMINAMATH_CALUDE_fair_coin_same_side_four_times_l4131_413133

theorem fair_coin_same_side_four_times (p : ℝ) :
  (p = 1 / 2) →                        -- The coin is fair (equal probability for each side)
  (p ^ 4 : ℝ) = 1 / 16 := by            -- Probability of same side 4 times is 1/16
sorry


end NUMINAMATH_CALUDE_fair_coin_same_side_four_times_l4131_413133


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l4131_413197

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49,16,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l4131_413197


namespace NUMINAMATH_CALUDE_grocery_store_soda_count_l4131_413157

/-- Given a grocery store inventory, prove the number of regular soda bottles -/
theorem grocery_store_soda_count 
  (diet_soda : ℕ) 
  (regular_soda_diff : ℕ) 
  (h1 : diet_soda = 53)
  (h2 : regular_soda_diff = 26) :
  diet_soda + regular_soda_diff = 79 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_count_l4131_413157


namespace NUMINAMATH_CALUDE_tangent_circles_m_range_l4131_413187

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 25 - m^2 = 0

-- Define the property of being externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_range :
  ∀ m : ℝ, externally_tangent m ↔ m ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 0 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_range_l4131_413187


namespace NUMINAMATH_CALUDE_curve_and_function_relation_l4131_413128

-- Define a curve C as a set of points in ℝ²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem curve_and_function_relation :
  (∀ p : ℝ × ℝ, p ∈ C → F p.1 p.2 = 0) ∧
  (∀ p : ℝ × ℝ, F p.1 p.2 ≠ 0 → p ∉ C) :=
sorry

end NUMINAMATH_CALUDE_curve_and_function_relation_l4131_413128


namespace NUMINAMATH_CALUDE_problem_solution_l4131_413127

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - t*x + 1

-- Define the predicate p
def p (t : ℝ) : Prop := ∃ x, f t x = 0

-- Define the predicate q
def q (t : ℝ) : Prop := ∀ x, |x - 1| ≥ 2 - t^2

theorem problem_solution (t : ℝ) :
  (q t → t ∈ Set.Ici (Real.sqrt 2) ∪ Set.Iic (-Real.sqrt 2)) ∧
  (¬p t ∧ ¬q t → t ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l4131_413127


namespace NUMINAMATH_CALUDE_square_root_sum_l4131_413102

theorem square_root_sum (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l4131_413102


namespace NUMINAMATH_CALUDE_rectangle_formations_l4131_413181

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem rectangle_formations :
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l4131_413181


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_minus_circles_l4131_413107

/-- The shaded area in a rectangle after subtracting two circles -/
theorem shaded_area_rectangle_minus_circles 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (h1 : rectangle_length = 16)
  (h2 : rectangle_width = 8)
  (h3 : circle1_radius = 4)
  (h4 : circle2_radius = 2) :
  rectangle_length * rectangle_width - π * (circle1_radius^2 + circle2_radius^2) = 128 - 20 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_minus_circles_l4131_413107


namespace NUMINAMATH_CALUDE_decreasing_power_function_l4131_413100

/-- A power function y = ax^b is decreasing on (0, +∞) if and only if b = -3 -/
theorem decreasing_power_function (a : ℝ) (b : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → a * x₁^b > a * x₂^b) ↔ b = -3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l4131_413100


namespace NUMINAMATH_CALUDE_log_simplification_l4131_413174

theorem log_simplification : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l4131_413174


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l4131_413179

/-- Regular triangular pyramid with given properties -/
structure RegularPyramid where
  base_side : ℝ
  lateral_edge : ℝ
  base_side_eq_one : base_side = 1
  lateral_edge_eq_two : lateral_edge = 2

/-- Section that divides the pyramid volume equally -/
structure EqualVolumeSection (p : RegularPyramid) where
  passes_through_AB : Bool
  divides_equally : Bool

/-- Dihedral angle between the section and the base -/
def dihedralAngle (p : RegularPyramid) (s : EqualVolumeSection p) : ℝ := sorry

theorem dihedral_angle_cosine 
  (p : RegularPyramid) 
  (s : EqualVolumeSection p) : 
  Real.cos (dihedralAngle p s) = 2 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l4131_413179


namespace NUMINAMATH_CALUDE_john_money_l4131_413185

theorem john_money (total nada ali john : ℕ) : 
  total = 67 →
  ali = nada - 5 →
  john = 4 * nada →
  total = nada + ali + john →
  john = 48 := by
sorry

end NUMINAMATH_CALUDE_john_money_l4131_413185


namespace NUMINAMATH_CALUDE_expression_value_l4131_413161

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 3) :
  (x + 2*y)^2 - (x + y)*(2*x - y) = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4131_413161


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l4131_413164

/-- A polynomial g satisfying g(x + 1) - g(x) = 6x + 6 for all x has leading coefficient 3 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x + 6) :
  ∃ a b c : ℝ, (∀ x, g x = 3 * x^2 + a * x + b) ∧ c = 3 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l4131_413164


namespace NUMINAMATH_CALUDE_fat_thin_eating_time_l4131_413151

/-- The time it takes for two people to eat a certain amount of fruit together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 46.875 minutes to eat 5 pounds of fruit together -/
theorem fat_thin_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 5         -- Amount of fruit in pounds
  eating_time fat_rate thin_rate amount = 46875 / 1000 := by
sorry

end NUMINAMATH_CALUDE_fat_thin_eating_time_l4131_413151


namespace NUMINAMATH_CALUDE_fred_gave_233_marbles_l4131_413155

/-- The number of black marbles Fred gave to Sara -/
def marbles_from_fred (initial_marbles final_marbles : ℕ) : ℕ :=
  final_marbles - initial_marbles

/-- Theorem stating that Fred gave Sara 233 black marbles -/
theorem fred_gave_233_marbles :
  let initial_marbles : ℕ := 792
  let final_marbles : ℕ := 1025
  marbles_from_fred initial_marbles final_marbles = 233 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_233_marbles_l4131_413155


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l4131_413191

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 20) → 
  (a / (1 - r^2) = 8) → 
  (r ≠ 1) →
  r = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l4131_413191


namespace NUMINAMATH_CALUDE_sqrt_four_cubes_sum_l4131_413138

theorem sqrt_four_cubes_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_cubes_sum_l4131_413138


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l4131_413170

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (4 : ℝ)^x + (2 : ℝ)^x - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l4131_413170


namespace NUMINAMATH_CALUDE_no_positive_integer_triples_l4131_413184

theorem no_positive_integer_triples : 
  ¬∃ (a b c : ℕ+), (Nat.factorial a.val + b.val^3 = 18 + c.val^3) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_triples_l4131_413184


namespace NUMINAMATH_CALUDE_boat_speed_upstream_l4131_413134

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still_water : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still_water - speed_current

/-- Theorem: The speed of a boat upstream is 30 kmph when its speed in still water is 50 kmph and the current speed is 20 kmph. -/
theorem boat_speed_upstream :
  speed_upstream 50 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_upstream_l4131_413134


namespace NUMINAMATH_CALUDE_big_crash_frequency_is_20_l4131_413198

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency (total_accidents : ℕ) (total_time : ℕ) (collision_frequency : ℕ) : ℕ :=
  let regular_collisions := total_time / collision_frequency
  let big_crashes := total_accidents - regular_collisions
  total_time / big_crashes

/-- Theorem stating the frequency of big crashes given the problem conditions -/
theorem big_crash_frequency_is_20 :
  big_crash_frequency 36 (4 * 60) 10 = 20 := by
  sorry

#eval big_crash_frequency 36 (4 * 60) 10

end NUMINAMATH_CALUDE_big_crash_frequency_is_20_l4131_413198


namespace NUMINAMATH_CALUDE_division_chain_l4131_413195

theorem division_chain : (180 / 6) / 3 / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_division_chain_l4131_413195


namespace NUMINAMATH_CALUDE_b_has_property_P_l4131_413118

-- Define property P for a sequence
def has_property_P (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, (a (n + 1) + a (n + 2)) = q * (a n + a (n + 1))

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n + (-1)^n

-- Theorem statement
theorem b_has_property_P : has_property_P b := by
  sorry

end NUMINAMATH_CALUDE_b_has_property_P_l4131_413118


namespace NUMINAMATH_CALUDE_g_difference_l4131_413143

/-- The function g(x) = 3x^3 + 4x^2 - 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 - 3 * x + 2

/-- Theorem stating that g(x + h) - g(x) = h(9x^2 + 8x + 9xh + 4h + 3h^2 - 3) for all x and h -/
theorem g_difference (x h : ℝ) : 
  g (x + h) - g x = h * (9 * x^2 + 8 * x + 9 * x * h + 4 * h + 3 * h^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l4131_413143


namespace NUMINAMATH_CALUDE_largest_increase_2007_2008_l4131_413180

/-- Represents the number of students taking AMC 10 for each year from 2002 to 2008 -/
def students : Fin 7 → ℕ
  | 0 => 50  -- 2002
  | 1 => 55  -- 2003
  | 2 => 60  -- 2004
  | 3 => 65  -- 2005
  | 4 => 72  -- 2006
  | 5 => 80  -- 2007
  | 6 => 90  -- 2008

/-- Calculates the percentage increase between two consecutive years -/
def percentageIncrease (year : Fin 6) : ℚ :=
  (students (year.succ) - students year) / students year * 100

/-- Theorem stating that the percentage increase between 2007 and 2008 is the largest -/
theorem largest_increase_2007_2008 :
  ∀ year : Fin 6, percentageIncrease 5 ≥ percentageIncrease year :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2007_2008_l4131_413180


namespace NUMINAMATH_CALUDE_double_scientific_notation_doubling_2_4_times_10_to_8_l4131_413147

theorem double_scientific_notation (x : Real) (n : Nat) :
  2 * (x * (10 ^ n)) = (2 * x) * (10 ^ n) := by sorry

theorem doubling_2_4_times_10_to_8 :
  2 * (2.4 * (10 ^ 8)) = 4.8 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_double_scientific_notation_doubling_2_4_times_10_to_8_l4131_413147


namespace NUMINAMATH_CALUDE_vote_count_proof_l4131_413129

theorem vote_count_proof (T : ℕ) (F : ℕ) (A : ℕ) : 
  F = A + 68 →  -- 68 more votes in favor than against
  A = (40 * T) / 100 →  -- 40% of total votes were against
  T = F + A →  -- total votes is sum of for and against
  T = 340 :=
by sorry

end NUMINAMATH_CALUDE_vote_count_proof_l4131_413129


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4131_413158

theorem complex_equation_sum (a b : ℝ) : 
  (2 : ℂ) / (1 - Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4131_413158


namespace NUMINAMATH_CALUDE_tangent_lines_with_slope_one_l4131_413117

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_lines_with_slope_one :
  ∃! (points : Finset ℝ), 
    (Finset.card points = 2) ∧ 
    (∀ x ∈ points, f' x = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_with_slope_one_l4131_413117


namespace NUMINAMATH_CALUDE_ed_lost_marbles_ed_lost_eleven_marbles_l4131_413114

theorem ed_lost_marbles (doug : ℕ) : ℕ :=
  let ed_initial := doug + 19
  let ed_final := doug + 8
  ed_initial - ed_final

theorem ed_lost_eleven_marbles (doug : ℕ) :
  ed_lost_marbles doug = 11 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_marbles_ed_lost_eleven_marbles_l4131_413114


namespace NUMINAMATH_CALUDE_smallest_n_value_l4131_413126

/-- The number of ordered triplets (a, b, c) satisfying the conditions -/
def num_triplets : ℕ := 27000

/-- The greatest common divisor of a, b, and c -/
def gcd_value : ℕ := 91

/-- A function that counts the number of valid triplets for a given n -/
noncomputable def count_triplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value :
  ∃ (n : ℕ), n = 17836000 ∧
  count_triplets n = num_triplets ∧
  (∀ m : ℕ, m < n → count_triplets m ≠ num_triplets) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l4131_413126


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4131_413194

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_10 : a + b + c + d = 10) :
  (1/a + 9/b + 25/c + 49/d) ≥ 25.6 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 0 < d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 10 ∧
    1/a₀ + 9/b₀ + 25/c₀ + 49/d₀ = 25.6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4131_413194


namespace NUMINAMATH_CALUDE_expenditure_difference_l4131_413186

theorem expenditure_difference
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percent : ℝ)
  (purchased_quantity_percent : ℝ)
  (h1 : price_increase_percent = 25)
  (h2 : purchased_quantity_percent = 72)
  : (1 + price_increase_percent / 100) * (purchased_quantity_percent / 100) - 1 = -0.1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_difference_l4131_413186


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l4131_413193

/-- Given real numbers a, b, c, if f(x) = ax^2 + bx + c, g(x) = ax + b, 
    and the maximum value of g(x) is 2 when -1 ≤ x ≤ 1, then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination (a b c : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x, -1 ≤ x → x ≤ 1 → g x ≤ 2)
  (h_reaches_max : ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l4131_413193


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4131_413167

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -995 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4131_413167

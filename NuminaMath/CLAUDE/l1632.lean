import Mathlib

namespace sum_of_roots_l1632_163243

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 10*a*x - 11*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 10*c*x - 11*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1210 := by
sorry

end sum_of_roots_l1632_163243


namespace polynomial_factorization_l1632_163280

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end polynomial_factorization_l1632_163280


namespace product_in_fourth_quadrant_l1632_163290

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem product_in_fourth_quadrant :
  let z₁ : ℂ := Complex.mk 3 1
  let z₂ : ℂ := Complex.mk 1 (-1)
  let z : ℂ := complex_multiply z₁.re z₁.im z₂.re z₂.im
  fourth_quadrant z := by sorry

end product_in_fourth_quadrant_l1632_163290


namespace inverse_tangent_sum_l1632_163278

theorem inverse_tangent_sum : Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = π/4 := by
  sorry

end inverse_tangent_sum_l1632_163278


namespace sin_arccos_twelve_thirteenths_l1632_163254

theorem sin_arccos_twelve_thirteenths : 
  Real.sin (Real.arccos (12 / 13)) = 5 / 13 := by
  sorry

end sin_arccos_twelve_thirteenths_l1632_163254


namespace quadratic_inequality_solution_is_all_reals_l1632_163261

theorem quadratic_inequality_solution_is_all_reals :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 5 * x + 6
  ∀ x : ℝ, f x < 0 ∨ f x > 0 :=
by sorry

end quadratic_inequality_solution_is_all_reals_l1632_163261


namespace base_5_to_octal_conversion_l1632_163229

def base_5_to_decimal (n : ℕ) : ℕ := n

def decimal_to_octal (n : ℕ) : ℕ := n

theorem base_5_to_octal_conversion :
  decimal_to_octal (base_5_to_decimal 1234) = 302 := by
  sorry

end base_5_to_octal_conversion_l1632_163229


namespace units_digit_of_product_units_digit_27_times_46_l1632_163241

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by
  sorry

/-- The main theorem: the units digit of 27 * 46 is equal to the units digit of 7 * 6 -/
theorem units_digit_27_times_46 :
  unitsDigit (27 * 46) = unitsDigit (7 * 6) := by
  sorry

end units_digit_of_product_units_digit_27_times_46_l1632_163241


namespace unique_solution_condition_l1632_163253

/-- 
Given a system of linear equations:
  a * x + b * y - b * z = c
  a * y + b * x - b * z = c
  a * z + b * y - b * x = c
This theorem states that the system has a unique solution if and only if 
a ≠ 0, a - b ≠ 0, and a + b ≠ 0.
-/
theorem unique_solution_condition (a b c : ℝ) :
  (∃! x y z : ℝ, (a * x + b * y - b * z = c) ∧ 
                 (a * y + b * x - b * z = c) ∧ 
                 (a * z + b * y - b * x = c)) ↔ 
  (a ≠ 0 ∧ a - b ≠ 0 ∧ a + b ≠ 0) :=
by sorry

end unique_solution_condition_l1632_163253


namespace last_two_nonzero_digits_80_factorial_l1632_163200

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The last two nonzero digits of 80! are 08 -/
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 8 := by sorry

end last_two_nonzero_digits_80_factorial_l1632_163200


namespace region_area_correct_l1632_163228

/-- Given a circle with radius 36, two chords of length 66 intersecting at a point 12 units from
    the center at a 45° angle, this function calculates the area of one region formed by the
    intersection of the chords. -/
def calculate_region_area (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the calculated area is correct for the given conditions. -/
theorem region_area_correct (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) :
  radius = 36 ∧ 
  chord_length = 66 ∧ 
  intersection_distance = 12 ∧ 
  intersection_angle = 45 * π / 180 →
  calculate_region_area radius chord_length intersection_distance intersection_angle > 0 :=
by sorry

end region_area_correct_l1632_163228


namespace math_textbooks_in_one_box_l1632_163286

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def num_boxes : ℕ := 3
def box_capacities : Fin 3 → ℕ := ![4, 5, 6]

def probability_all_math_in_one_box : ℚ := 1 / 143

theorem math_textbooks_in_one_box :
  let total_arrangements := (total_textbooks.choose (box_capacities 0)) * 
                            ((total_textbooks - box_capacities 0).choose (box_capacities 1)) * 
                            ((total_textbooks - box_capacities 0 - box_capacities 1).choose (box_capacities 2))
  let favorable_outcomes := (box_capacities 0).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 0 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 0).choose (box_capacities 1)) +
                            (box_capacities 1).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 1 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 1).choose (box_capacities 0)) +
                            (box_capacities 2).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 2 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 2).choose (box_capacities 0))
  (favorable_outcomes : ℚ) / (total_arrangements : ℚ) = probability_all_math_in_one_box := by
  sorry

end math_textbooks_in_one_box_l1632_163286


namespace square_sum_theorem_l1632_163270

theorem square_sum_theorem (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    n % 2 = 0 ∧ 
    n > 2 ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 :=
by sorry

end square_sum_theorem_l1632_163270


namespace composition_ratio_l1632_163268

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 2)) / g (f (g 2)) = 115 / 73 := by
  sorry

end composition_ratio_l1632_163268


namespace solution_of_linear_equation_l1632_163201

theorem solution_of_linear_equation :
  ∃ x : ℝ, 2 * x + 6 = 0 ∧ x = -3 :=
by sorry

end solution_of_linear_equation_l1632_163201


namespace cycle_sale_result_l1632_163266

/-- Calculates the final selling price and overall profit percentage for a cycle sale --/
def cycle_sale_analysis (initial_cost upgrade_cost : ℚ) (profit_margin sales_tax : ℚ) :
  ℚ × ℚ :=
  let total_cost := initial_cost + upgrade_cost
  let selling_price_before_tax := total_cost * (1 + profit_margin)
  let final_selling_price := selling_price_before_tax * (1 + sales_tax)
  let overall_profit := final_selling_price - total_cost
  let overall_profit_percentage := (overall_profit / total_cost) * 100
  (final_selling_price, overall_profit_percentage)

/-- Theorem stating the correct final selling price and overall profit percentage --/
theorem cycle_sale_result :
  let (final_price, profit_percentage) := cycle_sale_analysis 1400 600 (10/100) (5/100)
  final_price = 2310 ∧ profit_percentage = 15.5 := by
  sorry

end cycle_sale_result_l1632_163266


namespace max_xy_value_l1632_163225

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + 3*y = 6) :
  ∃ (max_val : ℝ), max_val = 3/2 ∧ ∀ (z : ℝ), x*y ≤ z → z ≤ max_val :=
sorry

end max_xy_value_l1632_163225


namespace trig_function_problem_l1632_163204

theorem trig_function_problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 := by
  sorry

end trig_function_problem_l1632_163204


namespace sin_alpha_value_l1632_163238

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α - π / 3) = 2 / 3) : 
  Real.sin α = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := by
  sorry

end sin_alpha_value_l1632_163238


namespace arithmetic_mean_of_fractions_l1632_163259

theorem arithmetic_mean_of_fractions :
  (1 / 3 : ℚ) * (3 / 7 + 5 / 9 + 7 / 11) = 1123 / 2079 := by
  sorry

end arithmetic_mean_of_fractions_l1632_163259


namespace shoeing_time_proof_l1632_163273

/-- Calculates the minimum time required for blacksmiths to shoe horses -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_hooves := num_horses * 4
  let total_time := total_hooves * time_per_shoe
  total_time / num_blacksmiths

/-- Proves that the minimum time for 48 blacksmiths to shoe 60 horses is 25 minutes -/
theorem shoeing_time_proof :
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end shoeing_time_proof_l1632_163273


namespace smith_initial_markers_l1632_163249

/-- The number of new boxes of markers Mr. Smith buys -/
def new_boxes : ℕ := 6

/-- The number of markers in each new box -/
def markers_per_box : ℕ := 9

/-- The total number of markers Mr. Smith has after buying new boxes -/
def total_markers : ℕ := 86

/-- The number of markers Mr. Smith had initially -/
def initial_markers : ℕ := total_markers - (new_boxes * markers_per_box)

theorem smith_initial_markers :
  initial_markers = 32 := by sorry

end smith_initial_markers_l1632_163249


namespace cubic_equation_solution_l1632_163232

theorem cubic_equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 6 / x + 9 / x^2 - 2 / x^3 = 0 →
  (3 / x = 3 / 2) ∨ (3 / x = 3 / (2 + Real.sqrt 3)) ∨ (3 / x = 3 / (2 - Real.sqrt 3)) :=
by sorry

end cubic_equation_solution_l1632_163232


namespace group_formation_count_l1632_163212

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of officers --/
def totalOfficers : ℕ := 10

/-- The total number of jawans --/
def totalJawans : ℕ := 15

/-- The number of officers in each group --/
def officersPerGroup : ℕ := 3

/-- The number of jawans in each group --/
def jawansPerGroup : ℕ := 5

/-- The number of ways to form groups --/
def numberOfGroups : ℕ := 
  totalOfficers * 
  (choose (totalOfficers - 1) (officersPerGroup - 1)) * 
  (choose totalJawans jawansPerGroup)

theorem group_formation_count : numberOfGroups = 1081080 := by
  sorry

end group_formation_count_l1632_163212


namespace line_intercept_sum_l1632_163224

theorem line_intercept_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
  sorry

end line_intercept_sum_l1632_163224


namespace original_equals_scientific_l1632_163214

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 12060000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  coefficient := 1.206
  exponent := 7
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent := by
  sorry

end original_equals_scientific_l1632_163214


namespace decagon_diagonals_l1632_163281

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l1632_163281


namespace umbrella_arrangement_count_l1632_163207

/-- The number of ways to arrange 7 people in an umbrella shape -/
def umbrella_arrangements : ℕ := sorry

/-- The binomial coefficient (n choose k) -/
def choose (n k : ℕ) : ℕ := sorry

theorem umbrella_arrangement_count : umbrella_arrangements = 20 := by
  sorry

end umbrella_arrangement_count_l1632_163207


namespace quadratic_two_zeros_l1632_163279

/-- A quadratic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 1) * x + k + 4

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * (k + 4)

/-- The function has two distinct zeros iff the discriminant is positive -/
def has_two_distinct_zeros (k : ℝ) : Prop := discriminant k > 0

/-- The range of k for which the function has two distinct zeros -/
def k_range (k : ℝ) : Prop := k < -3 ∨ k > 5

theorem quadratic_two_zeros (k : ℝ) : 
  has_two_distinct_zeros k ↔ k_range k := by sorry

end quadratic_two_zeros_l1632_163279


namespace ellipse_from_hyperbola_l1632_163294

/-- The hyperbola from which we derive the ellipse parameters -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = -1

/-- The vertex of the hyperbola becomes the focus of the ellipse -/
def hyperbola_vertex_to_ellipse_focus (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 2 ∨ y = -2))

/-- The focus of the hyperbola becomes the vertex of the ellipse -/
def hyperbola_focus_to_ellipse_vertex (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 4 ∨ y = -4))

/-- The resulting ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 16 = 1

/-- Theorem stating that the ellipse with the given properties has the specified equation -/
theorem ellipse_from_hyperbola :
  (∀ x y, hyperbola_vertex_to_ellipse_focus x y) →
  (∀ x y, hyperbola_focus_to_ellipse_vertex x y) →
  (∀ x y, ellipse x y) :=
sorry

end ellipse_from_hyperbola_l1632_163294


namespace triangle_problem_l1632_163295

/-- Triangle sum for the nth row -/
def triangle_sum (n : ℕ) (a d : ℕ) : ℕ := 2^n * a + (2^n - 2) * d

/-- The problem statement -/
theorem triangle_problem (a d : ℕ) (ha : a > 0) (hd : d > 0) :
  (∃ n : ℕ, triangle_sum n a d = 1988) →
  (∃ n : ℕ, n = 6 ∧ a = 2 ∧ d = 30 ∧ 
    (∀ m : ℕ, triangle_sum m a d = 1988 → m ≤ n)) :=
by sorry

end triangle_problem_l1632_163295


namespace asymptotes_of_hyperbola_l1632_163258

/-- Hyperbola C with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point on the right branch of hyperbola C -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  h_right_branch : 0 < x

/-- Equilateral triangle with vertices on hyperbola -/
structure EquilateralTriangleOnHyperbola (h : Hyperbola) where
  A : PointOnHyperbola h
  B : PointOnHyperbola h
  c : ℝ
  h_equilateral : c^2 = A.x^2 + A.y^2 ∧ c^2 = B.x^2 + B.y^2
  h_side_length : c^2 = h.a^2 + h.b^2

/-- Theorem: Asymptotes of hyperbola C are y = ±x -/
theorem asymptotes_of_hyperbola (h : Hyperbola) 
  (t : EquilateralTriangleOnHyperbola h) :
  ∃ (k : ℝ), k = 1 ∧ 
  (∀ (x y : ℝ), (y = k*x ∨ y = -k*x) ↔ 
    (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
      x'^2/h.a^2 - y'^2/h.b^2 = 1 → 
      x' > δ → |y'/x' - k| < ε ∨ |y'/x' + k| < ε)) :=
sorry

end asymptotes_of_hyperbola_l1632_163258


namespace minuend_value_l1632_163262

theorem minuend_value (minuend subtrahend : ℝ) 
  (h : minuend + subtrahend + (minuend - subtrahend) = 25) : 
  minuend = 12.5 := by
sorry

end minuend_value_l1632_163262


namespace polynomial_factorization_l1632_163246

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 81

-- Define the factored form
def f (x : ℝ) : ℝ := (x-3)*(x+3)*(x^2+9)

-- Theorem stating the equality of the polynomial and its factored form
theorem polynomial_factorization :
  ∀ x : ℝ, p x = f x :=
by
  sorry

end polynomial_factorization_l1632_163246


namespace alexis_has_60_mangoes_l1632_163284

/-- Represents the number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℕ
  dilan : ℕ
  ashley : ℕ

/-- Defines the conditions of the mango distribution problem -/
def validDistribution (d : MangoDistribution) : Prop :=
  (d.alexis = 4 * (d.dilan + d.ashley)) ∧
  (d.alexis + d.dilan + d.ashley = 75)

/-- Theorem stating that Alexis has 60 mangoes in a valid distribution -/
theorem alexis_has_60_mangoes (d : MangoDistribution) 
  (h : validDistribution d) : d.alexis = 60 := by
  sorry

end alexis_has_60_mangoes_l1632_163284


namespace area_of_similar_rectangle_l1632_163250

/-- Given a rectangle R1 with one side of 3 inches and an area of 18 square inches,
    and a similar rectangle R2 with a diagonal of 18 inches,
    prove that the area of R2 is 14.4 square inches. -/
theorem area_of_similar_rectangle (r1_side : ℝ) (r1_area : ℝ) (r2_diagonal : ℝ) :
  r1_side = 3 →
  r1_area = 18 →
  r2_diagonal = 18 →
  ∃ (r2_side1 r2_side2 : ℝ),
    r2_side1 * r2_side2 = 14.4 ∧
    r2_side1^2 + r2_side2^2 = r2_diagonal^2 ∧
    r2_side2 / r2_side1 = r1_area / r1_side^2 :=
by sorry

end area_of_similar_rectangle_l1632_163250


namespace emily_big_garden_seeds_l1632_163223

/-- The number of seeds Emily started with -/
def total_seeds : ℕ := 41

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The number of seeds Emily planted in the big garden -/
def seeds_in_big_garden : ℕ := total_seeds - (num_small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds : seeds_in_big_garden = 29 := by
  sorry

end emily_big_garden_seeds_l1632_163223


namespace custom_op_difference_l1632_163244

-- Define the custom operation
def customOp (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem custom_op_difference : (customOp 7 4) - (customOp 4 7) = -9 := by
  sorry

end custom_op_difference_l1632_163244


namespace fraction_change_l1632_163285

theorem fraction_change (a b k : ℕ+) :
  (a : ℚ) / b < 1 → (a + k : ℚ) / (b + k) > a / b ∧
  (a : ℚ) / b > 1 → (a + k : ℚ) / (b + k) < a / b :=
by sorry

end fraction_change_l1632_163285


namespace complement_A_intersect_B_l1632_163267

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3} := by sorry

end complement_A_intersect_B_l1632_163267


namespace temp_difference_changtai_beijing_l1632_163213

/-- The difference in temperature between two locations -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp1 - temp2

/-- The lowest temperature in Beijing (in °C) -/
def beijing_temp : Int := -6

/-- The lowest temperature in Changtai County (in °C) -/
def changtai_temp : Int := 15

/-- Theorem: The temperature difference between Changtai County and Beijing is 21°C -/
theorem temp_difference_changtai_beijing :
  temperature_difference changtai_temp beijing_temp = 21 := by
  sorry

end temp_difference_changtai_beijing_l1632_163213


namespace problem_proof_l1632_163260

theorem problem_proof : 289 + 2 * 17 * 8 + 64 = 625 := by
  sorry

end problem_proof_l1632_163260


namespace gem_selection_count_is_22_l1632_163202

/-- The number of ways to choose gems under given constraints -/
def gem_selection_count : ℕ :=
  let red_gems : ℕ := 9
  let blue_gems : ℕ := 5
  let green_gems : ℕ := 6
  let total_to_choose : ℕ := 10
  let min_red : ℕ := 2
  let min_blue : ℕ := 2
  let max_green : ℕ := 3
  
  (Finset.range (max_green + 1)).sum (λ g =>
    (Finset.range (red_gems + 1)).sum (λ r =>
      if r ≥ min_red ∧ 
         total_to_choose - r - g ≥ min_blue ∧ 
         total_to_choose - r - g ≤ blue_gems
      then 1
      else 0
    )
  )

/-- Theorem stating that the number of ways to choose gems is 22 -/
theorem gem_selection_count_is_22 : gem_selection_count = 22 := by
  sorry

end gem_selection_count_is_22_l1632_163202


namespace angle_expression_equality_l1632_163291

theorem angle_expression_equality (θ : Real) 
  (h1 : ∃ (x y : Real), x < 0 ∧ y < 0 ∧ Real.cos θ = x ∧ Real.sin θ = y) 
  (h2 : Real.tan θ ^ 2 = -2 * Real.sqrt 2) : 
  Real.sin θ ^ 2 - Real.sin (3 * Real.pi + θ) * Real.cos (Real.pi + θ) - Real.sqrt 2 * Real.cos θ ^ 2 = (2 - 2 * Real.sqrt 2) / 3 := by
  sorry


end angle_expression_equality_l1632_163291


namespace factor_implies_p_value_l1632_163255

theorem factor_implies_p_value (m p : ℤ) : 
  (m - 8) ∣ (m^2 - p*m - 24) → p = 5 := by
sorry

end factor_implies_p_value_l1632_163255


namespace parallelogram_sum_l1632_163233

/-- A parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1 -/
structure Parallelogram (x y : ℝ) :=
  (side1 : ℝ := 5)
  (side2 : ℝ := 11)
  (side3 : ℝ := 3 * y + 2)
  (side4 : ℝ := 4 * x - 1)

/-- Theorem: In a parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1, x + y = 4 -/
theorem parallelogram_sum (x y : ℝ) (p : Parallelogram x y) : x + y = 4 := by
  sorry

end parallelogram_sum_l1632_163233


namespace rhombus_diagonal_l1632_163239

theorem rhombus_diagonal (A : ℝ) (d : ℝ) : 
  d > 0 →  -- shorter diagonal is positive
  3 * d > 0 →  -- longer diagonal is positive
  A = (1/2) * d * (3*d) →  -- area formula
  40 = 4 * (((d/2)^2 + ((3*d)/2)^2)^(1/2)) →  -- perimeter formula
  d = (1/3) * (10 * A)^(1/2) := by sorry

end rhombus_diagonal_l1632_163239


namespace passes_through_origin_symmetric_about_y_axis_l1632_163217

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Theorem for passing through the origin
theorem passes_through_origin (m : ℝ) : 
  (f m 0 = 0) ↔ (m = 1 ∨ m = -2) := by sorry

-- Theorem for symmetry about y-axis
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 0 := by sorry

end passes_through_origin_symmetric_about_y_axis_l1632_163217


namespace unique_integer_solution_fourth_power_equation_l1632_163205

theorem unique_integer_solution_fourth_power_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 :=
by sorry

end unique_integer_solution_fourth_power_equation_l1632_163205


namespace inequality_solution_l1632_163206

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end inequality_solution_l1632_163206


namespace max_profit_for_container_l1632_163274

/-- Represents the container and goods properties --/
structure Container :=
  (volume : ℝ)
  (weight_capacity : ℝ)
  (chemical_volume_per_ton : ℝ)
  (paper_volume_per_ton : ℝ)
  (chemical_profit_per_ton : ℝ)
  (paper_profit_per_ton : ℝ)

/-- Calculates the profit for a given allocation of goods --/
def profit (c : Container) (chemical_tons : ℝ) (paper_tons : ℝ) : ℝ :=
  c.chemical_profit_per_ton * chemical_tons + c.paper_profit_per_ton * paper_tons

/-- Checks if the allocation satisfies the container constraints --/
def is_valid_allocation (c : Container) (chemical_tons : ℝ) (paper_tons : ℝ) : Prop :=
  chemical_tons ≥ 0 ∧ paper_tons ≥ 0 ∧
  chemical_tons + paper_tons ≤ c.weight_capacity ∧
  c.chemical_volume_per_ton * chemical_tons + c.paper_volume_per_ton * paper_tons ≤ c.volume

/-- Theorem stating the maximum profit for the given container --/
theorem max_profit_for_container :
  ∃ (c : Container) (chemical_tons paper_tons : ℝ),
    c.volume = 12 ∧
    c.weight_capacity = 5 ∧
    c.chemical_volume_per_ton = 1 ∧
    c.paper_volume_per_ton = 3 ∧
    c.chemical_profit_per_ton = 100000 ∧
    c.paper_profit_per_ton = 200000 ∧
    is_valid_allocation c chemical_tons paper_tons ∧
    profit c chemical_tons paper_tons = 850000 ∧
    ∀ (other_chemical other_paper : ℝ),
      is_valid_allocation c other_chemical other_paper →
      profit c other_chemical other_paper ≤ 850000 :=
sorry

end max_profit_for_container_l1632_163274


namespace fraction_simplification_l1632_163293

theorem fraction_simplification (x : ℝ) : 
  (x^2 + 2*x + 3) / 4 + (3*x - 5) / 6 = (3*x^2 + 12*x - 1) / 12 := by sorry

end fraction_simplification_l1632_163293


namespace log_equation_l1632_163245

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by sorry

end log_equation_l1632_163245


namespace expand_product_l1632_163242

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l1632_163242


namespace base_number_inequality_l1632_163210

theorem base_number_inequality (x : ℝ) : 64^8 > x^22 ↔ x = 2^(24/11) := by
  sorry

end base_number_inequality_l1632_163210


namespace optimal_bicycle_point_l1632_163257

/-- Represents the problem of finding the optimal point to leave a bicycle --/
theorem optimal_bicycle_point 
  (total_distance : ℝ) 
  (cycling_speed walking_speed : ℝ) 
  (h1 : total_distance = 30) 
  (h2 : cycling_speed = 20) 
  (h3 : walking_speed = 5) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance → 
      max ((total_distance / 2 - y) / cycling_speed + y / walking_speed)
          (y / walking_speed + (total_distance / 2 - y) / cycling_speed)
      ≥ 
      max ((total_distance / 2 - x) / cycling_speed + x / walking_speed)
          (x / walking_speed + (total_distance / 2 - x) / cycling_speed)) :=
by sorry

end optimal_bicycle_point_l1632_163257


namespace complex_imaginary_part_l1632_163230

theorem complex_imaginary_part (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end complex_imaginary_part_l1632_163230


namespace jamie_balls_l1632_163247

theorem jamie_balls (R : ℕ) : 
  (R - 6) + 2 * R + 32 = 74 → R = 16 := by
sorry

end jamie_balls_l1632_163247


namespace graph_horizontal_shift_l1632_163296

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 2

-- Theorem stating that y = f(x + 2) is equivalent to shifting the graph of y = f(x) 2 units left
theorem graph_horizontal_shift :
  y = f (x + h) ↔ y = f ((x + h) - h) :=
sorry

end graph_horizontal_shift_l1632_163296


namespace divisibility_by_30_l1632_163218

theorem divisibility_by_30 (p : ℕ) (h1 : p.Prime) (h2 : p ≥ 7) : 30 ∣ p^2 - 1 := by
  sorry

end divisibility_by_30_l1632_163218


namespace car_comparison_l1632_163221

-- Define the speeds and times for both cars
def speed_M : ℝ := 1  -- Arbitrary unit speed for Car M
def speed_N : ℝ := 3 * speed_M
def start_time_M : ℝ := 0
def start_time_N : ℝ := 2
def total_time : ℝ := 3  -- From the solution, but not directly given in the problem

-- Define the distance traveled by each car
def distance_M (t : ℝ) : ℝ := speed_M * t
def distance_N (t : ℝ) : ℝ := speed_N * (t - start_time_N)

-- Theorem statement
theorem car_comparison :
  ∃ (t : ℝ), t > start_time_N ∧
  distance_M t = distance_N t ∧
  speed_N = 3 * speed_M ∧
  start_time_N - start_time_M = 2 := by
  sorry

end car_comparison_l1632_163221


namespace two_quadratic_solving_algorithms_l1632_163277

/-- A quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An algorithm to solve a quadratic equation -/
structure QuadraticSolver where
  solve : QuadraticEquation → Set ℝ

/-- The specific quadratic equation x^2 - 5x + 6 = 0 -/
def specificEquation : QuadraticEquation :=
  { a := 1, b := -5, c := 6 }

theorem two_quadratic_solving_algorithms :
  ∃ (algo1 algo2 : QuadraticSolver), algo1 ≠ algo2 ∧
    algo1.solve specificEquation = algo2.solve specificEquation :=
sorry

end two_quadratic_solving_algorithms_l1632_163277


namespace exists_divisibility_property_l1632_163220

theorem exists_divisibility_property (n : ℕ+) : ∃ (a b : ℕ+), n ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end exists_divisibility_property_l1632_163220


namespace hyperbola_asymptotes_l1632_163275

/-- Given a hyperbola and related geometric conditions, prove its asymptotes. -/
theorem hyperbola_asymptotes (a b c : ℝ) (E F₁ F₂ D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧
  (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) E.1 E.2 ∧  -- E is on the hyperbola
  (λ (x y : ℝ) => 4*x^2 + 4*y^2 = b^2) D.1 D.2 ∧       -- D is on the circle
  (E.1 - F₁.1) * D.1 + (E.2 - F₁.2) * D.2 = 0 ∧       -- EF₁ is tangent to circle at D
  2 * D.1 = E.1 + F₁.1 ∧ 2 * D.2 = E.2 + F₁.2 →       -- D is midpoint of EF₁
  (λ (x y : ℝ) => x + 2*y = 0 ∨ x - 2*y = 0) E.1 E.2  -- Asymptotes equations
  := by sorry

end hyperbola_asymptotes_l1632_163275


namespace marble_problem_l1632_163265

theorem marble_problem (x : ℕ) : 
  (((x / 2) * (1 / 3)) * (85 / 100) : ℚ) = 432 → x = 3052 :=
by sorry

end marble_problem_l1632_163265


namespace class_size_l1632_163222

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end class_size_l1632_163222


namespace smallest_mu_inequality_l1632_163287

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + 2*a*d ≥ μ*(a*b + b*c + c*d)) → μ ≥ 2) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + 2*a*d ≥ 2*(a*b + b*c + c*d)) :=
by sorry

end smallest_mu_inequality_l1632_163287


namespace root_sum_reciprocal_l1632_163235

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 36*x^2 + 215*x - 470

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) (D E F : ℝ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  p a = 0 ∧ p b = 0 ∧ p c = 0 →
  (∀ t : ℝ, t ≠ a ∧ t ≠ b ∧ t ≠ c →
    1 / (t^3 - 36*t^2 + 215*t - 470) = D / (t - a) + E / (t - b) + F / (t - c)) →
  1 / D + 1 / E + 1 / F = 105 := by
sorry

end root_sum_reciprocal_l1632_163235


namespace tommy_bike_ride_l1632_163216

theorem tommy_bike_ride (E : ℕ) : 
  (4 * (E + 2) : ℕ) * 4 = 80 → E = 3 := by sorry

end tommy_bike_ride_l1632_163216


namespace smallest_m_with_integer_price_l1632_163219

theorem smallest_m_with_integer_price : ∃ m : ℕ+, 
  (∀ k < m, ¬∃ x : ℕ, (107 : ℚ) * x = 100 * k) ∧
  (∃ x : ℕ, (107 : ℚ) * x = 100 * m) ∧
  m = 107 := by
sorry

end smallest_m_with_integer_price_l1632_163219


namespace arithmetic_sequence_property_l1632_163256

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 := by
  sorry

end arithmetic_sequence_property_l1632_163256


namespace inequalities_solution_l1632_163209

theorem inequalities_solution (x : ℝ) : 
  (((2*x - 4)*(x - 5) < 0) ↔ (x > 2 ∧ x < 5)) ∧
  ((3*x^2 + 5*x + 1 > 0) ↔ (x < (-5 - Real.sqrt 13) / 6 ∨ x > (-5 + Real.sqrt 13) / 6)) ∧
  (∀ x, -x^2 + x < 2) ∧
  (¬∃ x, 7*x^2 + 5*x + 1 ≤ 0) ∧
  ((4*x ≥ 4*x^2 + 1) ↔ (x = 1/2)) :=
by sorry

end inequalities_solution_l1632_163209


namespace binomial_max_fifth_term_l1632_163299

/-- 
If in the binomial expansion of (√x + 2/x)^n, only the fifth term has the maximum 
binomial coefficient, then n = 8.
-/
theorem binomial_max_fifth_term (n : ℕ) : 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧ 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k < Nat.choose n 4) → 
  n = 8 := by sorry

end binomial_max_fifth_term_l1632_163299


namespace z_in_third_quadrant_l1632_163298

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem z_in_third_quadrant :
  let z : ℂ := -1 - 2*I
  is_in_third_quadrant z :=
by sorry

end z_in_third_quadrant_l1632_163298


namespace smallest_odd_between_2_and_7_l1632_163271

theorem smallest_odd_between_2_and_7 : 
  ∀ n : ℕ, (2 < n ∧ n < 7 ∧ Odd n) → 3 ≤ n :=
by sorry

end smallest_odd_between_2_and_7_l1632_163271


namespace polynomial_factorization_l1632_163248

theorem polynomial_factorization (x : ℤ) :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2 =
  (3 * x^2 + 59 * x + 231) * (x + 7) * (3 * x + 11) := by
  sorry

end polynomial_factorization_l1632_163248


namespace lines_skew_iff_b_neq_l1632_163227

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of skew lines -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, l1.point + t • l1.direction ≠ l2.point + u • l2.direction

/-- The problem statement -/
theorem lines_skew_iff_b_neq (b : ℝ) :
  let l1 : Line3D := ⟨(1, 2, b), (2, 3, 4)⟩
  let l2 : Line3D := ⟨(3, 0, -1), (5, 3, 1)⟩
  are_skew l1 l2 ↔ b ≠ 11/3 := by
  sorry

end lines_skew_iff_b_neq_l1632_163227


namespace mary_garbage_bill_calculation_l1632_163292

/-- Calculates Mary's garbage bill given the specified conditions -/
def maryGarbageBill (weeksInMonth : ℕ) (trashBinCost recyclingBinCost greenWasteBinCost : ℚ)
  (trashBinCount recyclingBinCount greenWasteBinCount : ℕ) (serviceFee : ℚ)
  (discountRate : ℚ) (inappropriateItemsFine lateFee : ℚ) : ℚ :=
  let weeklyBinCost := trashBinCost * trashBinCount + recyclingBinCost * recyclingBinCount +
                       greenWasteBinCost * greenWasteBinCount
  let monthlyBinCost := weeklyBinCost * weeksInMonth
  let totalBeforeDiscount := monthlyBinCost + serviceFee
  let discountAmount := totalBeforeDiscount * discountRate
  let totalAfterDiscount := totalBeforeDiscount - discountAmount
  totalAfterDiscount + inappropriateItemsFine + lateFee

/-- Theorem stating that Mary's garbage bill is $134.14 under the given conditions -/
theorem mary_garbage_bill_calculation :
  maryGarbageBill 4 10 5 3 2 1 1 15 (18/100) 20 10 = 134.14 := by
  sorry

end mary_garbage_bill_calculation_l1632_163292


namespace find_integers_with_sum_and_lcm_l1632_163251

theorem find_integers_with_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 3972 ∧ 
    Nat.lcm a b = 985928 ∧ 
    a = 1964 ∧ 
    b = 2008 := by
  sorry

end find_integers_with_sum_and_lcm_l1632_163251


namespace cost_increase_percentage_l1632_163282

/-- Proves that under given conditions, the cost increase percentage is 25% -/
theorem cost_increase_percentage (C : ℝ) (P : ℝ) : 
  C > 0 → -- Ensure cost is positive
  let S := 4.2 * C -- Original selling price
  let new_profit := 0.7023809523809523 * S -- New profit after cost increase
  3.2 * C - (P / 100) * C = new_profit → -- Equation relating new profit to cost increase
  P = 25 := by
  sorry

end cost_increase_percentage_l1632_163282


namespace math_team_selection_count_l1632_163288

-- Define the number of boys and girls in the math club
def num_boys : ℕ := 10
def num_girls : ℕ := 10

-- Define the required number of boys and girls in the team
def required_boys : ℕ := 4
def required_girls : ℕ := 3

-- Define the total team size
def team_size : ℕ := required_boys + required_girls

-- Theorem statement
theorem math_team_selection_count :
  (Nat.choose num_boys required_boys) * (Nat.choose num_girls required_girls) = 25200 := by
  sorry

end math_team_selection_count_l1632_163288


namespace patrol_officer_results_l1632_163236

/-- Represents the travel record of the patrol officer -/
def travel_record : List Int := [10, -8, 6, -13, 7, -12, 3, -3]

/-- Position of the gas station relative to the guard post -/
def gas_station_position : Int := 6

/-- Fuel consumption rate of the motorcycle in liters per kilometer -/
def fuel_consumption_rate : ℚ := 0.05

/-- Calculates the final position of the patrol officer relative to the guard post -/
def final_position (record : List Int) : Int :=
  record.sum

/-- Counts the number of times the patrol officer passes the gas station -/
def gas_station_passes (record : List Int) (gas_station_pos : Int) : Nat :=
  sorry

/-- Calculates the total distance traveled by the patrol officer -/
def total_distance (record : List Int) : Int :=
  record.map (Int.natAbs) |>.sum

/-- Calculates the total fuel consumed during the patrol -/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  rate * distance.toNat

theorem patrol_officer_results :
  (final_position travel_record = -10) ∧
  (gas_station_passes travel_record gas_station_position = 4) ∧
  (total_fuel_consumed (total_distance travel_record) fuel_consumption_rate = 3.1) :=
sorry

end patrol_officer_results_l1632_163236


namespace sally_grew_113_turnips_l1632_163231

/-- The number of turnips Sally grew -/
def sallys_turnips : ℕ := 113

/-- The number of pumpkins Sally grew -/
def sallys_pumpkins : ℕ := 118

/-- The number of turnips Mary grew -/
def marys_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- Theorem stating that Sally grew 113 turnips -/
theorem sally_grew_113_turnips :
  sallys_turnips = total_turnips - marys_turnips :=
by sorry

end sally_grew_113_turnips_l1632_163231


namespace sum_quadratic_residues_divisible_l1632_163283

theorem sum_quadratic_residues_divisible (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ s : ℕ, (s > 0) ∧ (s < p) ∧ (∀ x : ℕ, x < p → (∃ y : ℕ, y < p ∧ y^2 ≡ x [ZMOD p]) → s ≡ s + x [ZMOD p]) :=
sorry

end sum_quadratic_residues_divisible_l1632_163283


namespace parabola_through_points_intersects_interval_l1632_163203

/-- Represents a parabola of the form y = ax² + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a_neg : a < 0

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.c

theorem parabola_through_points_intersects_interval
  (p : Parabola)
  (h_through_A : p.y_at 0 = 9)
  (h_through_P : p.y_at 2 = 8.1) :
  -9/49 < p.a ∧ p.a < -1/4 ∧
  ∃ x, 6 < x ∧ x < 7 ∧ p.y_at x = 0 :=
sorry

end parabola_through_points_intersects_interval_l1632_163203


namespace quadratic_root_problem_l1632_163237

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) →
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -3) :=
by sorry

end quadratic_root_problem_l1632_163237


namespace N_is_positive_l1632_163289

theorem N_is_positive (a b : ℝ) : 
  let N := 4*a^2 - 12*a*b + 13*b^2 - 6*a + 4*b + 13
  0 < N := by sorry

end N_is_positive_l1632_163289


namespace projection_incircle_inequality_l1632_163276

/-- Represents a right triangle with legs a and b, hypotenuse c, projections p and q, and incircle radii ρ_a and ρ_b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  ρ_a : ℝ
  ρ_b : ℝ
  h_right : a^2 + b^2 = c^2
  h_a_lt_b : a < b
  h_p_proj : p * c = a^2
  h_q_proj : q * c = b^2
  h_ρ_a_def : ρ_a * (a + c - b) = a * b
  h_ρ_b_def : ρ_b * (b + c - a) = a * b

/-- Theorem stating the inequalities for projections and incircle radii in a right triangle -/
theorem projection_incircle_inequality (t : RightTriangle) : t.p < t.ρ_a ∧ t.q > t.ρ_b := by
  sorry

end projection_incircle_inequality_l1632_163276


namespace samuel_bought_two_dozen_l1632_163215

/-- The number of dozens of doughnuts Samuel bought -/
def samuel_dozens : ℕ := sorry

/-- The number of dozens of doughnuts Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The total number of people sharing the doughnuts -/
def total_people : ℕ := 10

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- Theorem stating that Samuel bought 2 dozen doughnuts -/
theorem samuel_bought_two_dozen : samuel_dozens = 2 := by
  sorry

end samuel_bought_two_dozen_l1632_163215


namespace train_length_calculation_l1632_163252

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 27) 
  (h2 : man_speed = 6) 
  (h3 : passing_time = 11.999040076793857) : 
  ∃ (length : ℝ), abs (length - 110) < 0.1 := by
  sorry


end train_length_calculation_l1632_163252


namespace trailing_zeroes_500_factorial_l1632_163226

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end trailing_zeroes_500_factorial_l1632_163226


namespace problem_statement_l1632_163272

theorem problem_statement (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 2 + 1)) 
  (hy : y = 1 / (Real.sqrt 2 - 1)) : 
  x^2 - 3*x*y + y^2 = 3 := by
sorry

end problem_statement_l1632_163272


namespace two_x_minus_y_value_l1632_163297

theorem two_x_minus_y_value (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 3) 
  (hxy : x / y < 0) : 
  2 * x - y = 7 ∨ 2 * x - y = -7 := by
sorry

end two_x_minus_y_value_l1632_163297


namespace e_value_l1632_163263

-- Define variables
variable (p j t b a e : ℝ)

-- Define conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.8 * t
def condition3 : Prop := t = p * (1 - e / 100)
def condition4 : Prop := b = 1.4 * j
def condition5 : Prop := a = 0.85 * b
def condition6 : Prop := e = 2 * ((p - a) / p) * 100

-- Theorem statement
theorem e_value (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t e)
                (h4 : condition4 j b) (h5 : condition5 b a) (h6 : condition6 p a e) :
  e = 21.5 := by
  sorry

end e_value_l1632_163263


namespace cyclist_heartbeats_l1632_163208

/-- Calculates the total number of heartbeats during a cycling race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats is 24000 for the given conditions -/
theorem cyclist_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let race_distance : ℕ := 50  -- miles
  let pace : ℕ := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 24000 := by
sorry


end cyclist_heartbeats_l1632_163208


namespace copper_ion_beakers_l1632_163234

theorem copper_ion_beakers (total_beakers : ℕ) (drops_per_test : ℕ) (total_drops_used : ℕ) (negative_beakers : ℕ) : 
  total_beakers = 22 → 
  drops_per_test = 3 → 
  total_drops_used = 45 → 
  negative_beakers = 7 → 
  total_beakers - negative_beakers = 15 := by
sorry

end copper_ion_beakers_l1632_163234


namespace largest_product_of_three_exists_product_72_largest_product_is_72_l1632_163240

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a * b * c : Int) ≤ 72 :=
sorry

theorem exists_product_72 : 
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72 :=
sorry

theorem largest_product_is_72 : 
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a * b * c : Int) ≤ 72) ∧
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72) :=
sorry

end largest_product_of_three_exists_product_72_largest_product_is_72_l1632_163240


namespace final_price_calculation_l1632_163269

/-- Calculate the final price of a shirt and pants after a series of price changes -/
theorem final_price_calculation (S P : ℝ) :
  let shirt_price_1 := S * 1.20
  let pants_price_1 := P * 0.90
  let combined_price_1 := shirt_price_1 + pants_price_1
  let combined_price_2 := combined_price_1 * 1.15
  let final_price := combined_price_2 * 0.95
  final_price = 1.311 * S + 0.98325 * P := by sorry

end final_price_calculation_l1632_163269


namespace madeline_work_hours_l1632_163264

def monthly_expenses : ℕ := 1200 + 400 + 200 + 60
def emergency_savings : ℕ := 200
def daytime_hourly_rate : ℕ := 15
def bakery_hourly_rate : ℕ := 12
def bakery_weekly_hours : ℕ := 5
def tax_rate : ℚ := 15 / 100

theorem madeline_work_hours :
  ∃ (h : ℕ), h ≥ 146 ∧
  (h * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) ≥ 
  (monthly_expenses + emergency_savings : ℚ) ∧
  ∀ (k : ℕ), k < h →
  (k * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) <
  (monthly_expenses + emergency_savings : ℚ) :=
by sorry

end madeline_work_hours_l1632_163264


namespace always_integer_l1632_163211

theorem always_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) :
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) / (k + 2 : ℤ) * (n.choose k) = m :=
sorry

end always_integer_l1632_163211

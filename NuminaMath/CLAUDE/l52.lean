import Mathlib

namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l52_5296

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 1000 → (∃ k : ℕ, n = k^2) →
  (∀ m : ℕ, m < 1000 → (∃ l : ℕ, m = l^2) → m ≤ n) →
  n = 961 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l52_5296


namespace NUMINAMATH_CALUDE_decimal_division_equivalence_l52_5280

theorem decimal_division_equivalence : 
  (∃ n : ℕ, (0.1 : ℚ) = n * (0.001 : ℚ) ∧ n = 100) ∧ 
  (∃ m : ℕ, (1 : ℚ) = m * (0.01 : ℚ) ∧ m = 100) := by
  sorry

#check decimal_division_equivalence

end NUMINAMATH_CALUDE_decimal_division_equivalence_l52_5280


namespace NUMINAMATH_CALUDE_simplify_expression_l52_5224

theorem simplify_expression (r : ℝ) : 120*r - 38*r + 25*r = 107*r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l52_5224


namespace NUMINAMATH_CALUDE_sum_thirteen_is_156_l52_5293

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  third_sum : sum 3 = 6
  specific_sum : a 9 + a 11 + a 13 = 60

/-- The sum of the first 13 terms of the arithmetic sequence is 156 -/
theorem sum_thirteen_is_156 (seq : ArithmeticSequence) : seq.sum 13 = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_thirteen_is_156_l52_5293


namespace NUMINAMATH_CALUDE_jack_morning_emails_l52_5250

/-- Given that Jack received 3 emails in the afternoon, 1 email in the evening,
    and a total of 10 emails in the day, prove that he received 6 emails in the morning. -/
theorem jack_morning_emails
  (total : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (h1 : total = 10)
  (h2 : afternoon = 3)
  (h3 : evening = 1) :
  total - (afternoon + evening) = 6 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l52_5250


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l52_5284

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 →
    1 / (x^3 + 2*x^2 - 19*x - 30) = A / (x + 3) + B / (x - 2) + C / ((x - 2)^2)) →
  A = 1/25 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l52_5284


namespace NUMINAMATH_CALUDE_problem_1_l52_5204

theorem problem_1 (x : ℝ) : (x - 1)^2 + x*(3 - x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l52_5204


namespace NUMINAMATH_CALUDE_round_37_396_to_nearest_tenth_l52_5217

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.396396... -/
def x : RepeatingDecimal :=
  { integerPart := 37, repeatingPart := 396 }

theorem round_37_396_to_nearest_tenth :
  roundToNearestTenth x = 37.4 := by
  sorry

end NUMINAMATH_CALUDE_round_37_396_to_nearest_tenth_l52_5217


namespace NUMINAMATH_CALUDE_x_intercept_of_l_equation_of_l_l52_5286

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 4 = 0
def l₃ (x y : ℝ) : Prop := 4*x + 5*y - 12 = 0

-- Define the intersection point of l₁ and l₂
def intersection (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define line l
def l (x y : ℝ) : Prop := ∃ (ix iy : ℝ), intersection ix iy ∧ (y - iy) = ((x - ix) * (3 - iy)) / (3 - ix)

-- Theorem for part 1
theorem x_intercept_of_l : 
  (∃ (x y : ℝ), intersection x y) → 
  l 3 3 → 
  (∃ (x : ℝ), l x 0 ∧ x = -3) :=
sorry

-- Theorem for part 2
theorem equation_of_l :
  (∃ (x y : ℝ), intersection x y) →
  (∀ (x y : ℝ), l x y ↔ l₃ (x + a) (y + b)) →
  (∀ (x y : ℝ), l x y ↔ 4*x + 5*y - 14 = 0) :=
sorry

end NUMINAMATH_CALUDE_x_intercept_of_l_equation_of_l_l52_5286


namespace NUMINAMATH_CALUDE_max_value_of_five_numbers_l52_5222

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five distinct natural numbers
  (a + b + c + d + e) / 5 = 15 →  -- average is 15
  c = 18 →  -- median is 18
  e ≤ 37 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_five_numbers_l52_5222


namespace NUMINAMATH_CALUDE_highway_distance_l52_5277

/-- Proves the distance a car can travel on the highway given its city fuel efficiency and efficiency increase -/
theorem highway_distance (city_efficiency : ℝ) (efficiency_increase : ℝ) (highway_gas : ℝ) :
  city_efficiency = 30 →
  efficiency_increase = 0.2 →
  highway_gas = 7 →
  (city_efficiency * (1 + efficiency_increase)) * highway_gas = 252 := by
  sorry

end NUMINAMATH_CALUDE_highway_distance_l52_5277


namespace NUMINAMATH_CALUDE_correct_operation_l52_5220

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l52_5220


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l52_5231

theorem max_value_sine_cosine_sum :
  let f : ℝ → ℝ := λ x ↦ 6 * Real.sin x + 8 * Real.cos x
  ∃ M : ℝ, M = 10 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l52_5231


namespace NUMINAMATH_CALUDE_equation_solution_l52_5229

theorem equation_solution (x : ℂ) : 
  (x^2 + x + 1) / (x + 1) = x^2 + 2*x + 2 ↔ 
  (x = -1 ∨ x = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ x = (-1 - Complex.I * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l52_5229


namespace NUMINAMATH_CALUDE_A_and_D_independent_l52_5214

def num_balls : ℕ := 5

def prob_A : ℚ := 1 / num_balls
def prob_B : ℚ := 1 / num_balls
def prob_C : ℚ := 3 / (num_balls * num_balls)
def prob_D : ℚ := 1 / num_balls

def prob_AD : ℚ := 1 / (num_balls * num_balls)

theorem A_and_D_independent : prob_AD = prob_A * prob_D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l52_5214


namespace NUMINAMATH_CALUDE_pancakes_remaining_l52_5267

theorem pancakes_remaining (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : 
  total = 21 → bobby_ate = 5 → dog_ate = 7 → total - (bobby_ate + dog_ate) = 9 := by
  sorry

end NUMINAMATH_CALUDE_pancakes_remaining_l52_5267


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l52_5205

theorem triangle_angle_inequalities (α β γ : Real) 
  (h : α + β + γ = π) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l52_5205


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_equality_l52_5281

theorem binomial_coefficient_sum_equality (n : ℕ) : 4^n = 2^10 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_equality_l52_5281


namespace NUMINAMATH_CALUDE_rational_pair_sum_reciprocal_natural_l52_5208

theorem rational_pair_sum_reciprocal_natural (x y : ℚ) :
  (∃ (m n : ℕ), x + 1 / y = m ∧ y + 1 / x = n) →
  ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_rational_pair_sum_reciprocal_natural_l52_5208


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l52_5274

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l52_5274


namespace NUMINAMATH_CALUDE_toothbrush_ratio_l52_5268

theorem toothbrush_ratio (total brushes_jan brushes_feb brushes_mar : ℕ)
  (busiest_slowest_diff : ℕ) :
  total = 330 →
  brushes_jan = 53 →
  brushes_feb = 67 →
  brushes_mar = 46 →
  busiest_slowest_diff = 36 →
  ∃ (brushes_apr brushes_may : ℕ),
    brushes_apr + brushes_may = total - (brushes_jan + brushes_feb + brushes_mar) ∧
    brushes_apr - brushes_may = busiest_slowest_diff ∧
    brushes_apr * 16 = brushes_may * 25 :=
by sorry

end NUMINAMATH_CALUDE_toothbrush_ratio_l52_5268


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l52_5255

/-- Given a blueprint scale and a length on the blueprint, calculates the actual length in meters. -/
def actual_length (scale : ℚ) (blueprint_length : ℚ) : ℚ :=
  blueprint_length * scale / 100

/-- Proves that for a blueprint scale of 1:50 and a line segment of 10 cm on the blueprint,
    the actual length is 5 m. -/
theorem blueprint_to_actual_length :
  let scale : ℚ := 50
  let blueprint_length : ℚ := 10
  actual_length scale blueprint_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l52_5255


namespace NUMINAMATH_CALUDE_complex_division_equality_l52_5213

theorem complex_division_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l52_5213


namespace NUMINAMATH_CALUDE_price_adjustment_l52_5270

theorem price_adjustment (original_price : ℝ) (original_price_positive : 0 < original_price) : 
  let reduced_price := 0.8 * original_price
  let final_price := reduced_price * 1.375
  final_price = 1.1 * original_price :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l52_5270


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l52_5225

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l52_5225


namespace NUMINAMATH_CALUDE_incorrect_stability_statement_l52_5226

/-- Represents the variance of an individual's high jump scores -/
structure JumpVariance where
  value : ℝ
  is_positive : value > 0

/-- Represents the stability of an individual's high jump scores -/
def more_stable (a b : JumpVariance) : Prop :=
  a.value < b.value

theorem incorrect_stability_statement :
  ∃ (a b : JumpVariance),
    a.value = 1.1 ∧
    b.value = 2.5 ∧
    ¬(more_stable a b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_stability_statement_l52_5226


namespace NUMINAMATH_CALUDE_four_point_circle_theorem_l52_5259

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is on or inside a circle -/
def Point.onOrInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Check if a point is on the circumference of a circle -/
def Point.onCircumference (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem four_point_circle_theorem (a b c d : Point) 
  (h : ¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) :
  ∃ (circ : Circle), 
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onOrInside d circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference d circ ∧ Point.onOrInside c circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside b circ) ∨
    (Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside a circ) :=
sorry

end NUMINAMATH_CALUDE_four_point_circle_theorem_l52_5259


namespace NUMINAMATH_CALUDE_solve_equation_l52_5221

/-- Custom remainder operation Θ -/
def theta (m n : ℕ) : ℕ :=
  if m ≥ n then m % n else n % m

/-- Main theorem -/
theorem solve_equation :
  ∃ (A : ℕ), 0 < A ∧ A < 40 ∧ theta 20 (theta A 20) = 7 ∧ A = 33 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l52_5221


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l52_5228

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l52_5228


namespace NUMINAMATH_CALUDE_all_cells_colored_l52_5261

/-- Represents a 6x6 grid where cells can be colored -/
structure Grid :=
  (colored : Fin 6 → Fin 6 → Bool)

/-- Returns the number of colored cells in a 2x2 square starting at (i, j) -/
def count_2x2 (g : Grid) (i j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat +
  (g.colored (i + 1) j).toNat + (g.colored (i + 1) (j + 1)).toNat

/-- Returns the number of colored cells in a 1x3 stripe starting at (i, j) -/
def count_1x3 (g : Grid) (i : Fin 6) (j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat + (g.colored i (j + 2)).toNat

/-- The main theorem -/
theorem all_cells_colored (g : Grid) 
  (h1 : ∀ i j : Fin 4, count_2x2 g i j = count_2x2 g 0 0)
  (h2 : ∀ i : Fin 6, ∀ j : Fin 4, count_1x3 g i j = count_1x3 g 0 0) :
  ∀ i j : Fin 6, g.colored i j = true := by
  sorry

end NUMINAMATH_CALUDE_all_cells_colored_l52_5261


namespace NUMINAMATH_CALUDE_kotelmel_triangle_area_error_l52_5218

/-- The margin of error between Kotelmel's formula and the correct formula for the area of an equilateral triangle --/
theorem kotelmel_triangle_area_error :
  let a : ℝ := 1  -- We can use any positive real number for a
  let kotelmel_area := (1/3 + 1/10) * a^2
  let correct_area := (a^2 / 4) * Real.sqrt 3
  let error_percentage := |correct_area - kotelmel_area| / correct_area * 100
  ∃ ε > 0, error_percentage < 0.075 + ε ∧ error_percentage > 0.075 - ε :=
by sorry


end NUMINAMATH_CALUDE_kotelmel_triangle_area_error_l52_5218


namespace NUMINAMATH_CALUDE_inequality_proof_l52_5235

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l52_5235


namespace NUMINAMATH_CALUDE_cosine_sum_17th_roots_l52_5265

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (10 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_17th_roots_l52_5265


namespace NUMINAMATH_CALUDE_sum_inequality_l52_5295

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l52_5295


namespace NUMINAMATH_CALUDE_inscribed_rhombus_square_area_l52_5207

/-- Represents a rhombus inscribed in a square -/
structure InscribedRhombus where
  -- Square side length
  a : ℝ
  -- Distances from square vertices to rhombus vertices
  pb : ℝ
  bq : ℝ
  pr : ℝ
  qs : ℝ
  -- Conditions
  pb_positive : pb > 0
  bq_positive : bq > 0
  pr_positive : pr > 0
  qs_positive : qs > 0
  pb_plus_bq : pb + bq = a
  pr_plus_qs : pr + qs = a

/-- The area of the square given the inscribed rhombus properties -/
def square_area (r : InscribedRhombus) : ℝ :=
  r.a ^ 2

/-- Theorem: The area of the square with the given inscribed rhombus is 40000/58 -/
theorem inscribed_rhombus_square_area :
  ∀ r : InscribedRhombus,
  r.pb = 10 ∧ r.bq = 25 ∧ r.pr = 20 ∧ r.qs = 40 →
  square_area r = 40000 / 58 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_square_area_l52_5207


namespace NUMINAMATH_CALUDE_complex_subtraction_l52_5244

theorem complex_subtraction : (7 - 3*I) - (2 + 4*I) = 5 - 7*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l52_5244


namespace NUMINAMATH_CALUDE_smallest_n_value_l52_5289

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def num_quadruplets : ℕ := 90000

/-- The greatest common divisor of the quadruplets -/
def quadruplet_gcd : ℕ := 90

/-- 
  The function that counts the number of ordered quadruplets (a, b, c, d) 
  satisfying gcd(a, b, c, d) = quadruplet_gcd and lcm(a, b, c, d) = n
-/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value : 
  (∃ (n : ℕ), n > 0 ∧ count_quadruplets n = num_quadruplets) ∧ 
  (∀ (m : ℕ), m > 0 ∧ count_quadruplets m = num_quadruplets → m ≥ 32400) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l52_5289


namespace NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l52_5291

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : 
  ∃ k : ℤ, 4 * x + t = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l52_5291


namespace NUMINAMATH_CALUDE_vector_parallel_to_a_l52_5271

/-- Given a vector a = (-5, 4), prove that (-5k, 4k) is parallel to a for any scalar k. -/
theorem vector_parallel_to_a (k : ℝ) : 
  ∃ (t : ℝ), ((-5 : ℝ), (4 : ℝ)) = t • ((-5*k : ℝ), (4*k : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_to_a_l52_5271


namespace NUMINAMATH_CALUDE_storage_area_wheels_l52_5288

theorem storage_area_wheels (bicycles tricycles unicycles cars : ℕ)
  (h_bicycles : bicycles = 24)
  (h_tricycles : tricycles = 14)
  (h_unicycles : unicycles = 10)
  (h_cars : cars = 18) :
  let total_wheels := bicycles * 2 + tricycles * 3 + unicycles * 1 + cars * 4
  let unicycle_wheels := unicycles * 1
  let ratio_numerator := unicycle_wheels
  let ratio_denominator := total_wheels
  (total_wheels = 172) ∧ 
  (ratio_numerator = 5 ∧ ratio_denominator = 86) :=
by sorry

end NUMINAMATH_CALUDE_storage_area_wheels_l52_5288


namespace NUMINAMATH_CALUDE_problem_solution_l52_5249

theorem problem_solution (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 2) : 
  (a + b)^2 = 17 ∧ a^2 - 6*a*b + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l52_5249


namespace NUMINAMATH_CALUDE_evaluate_expression_l52_5297

theorem evaluate_expression : 
  (7 ^ (1/4 : ℝ)) / (3 ^ (1/3 : ℝ)) / ((7 ^ (1/2 : ℝ)) / (3 ^ (1/6 : ℝ))) = 
  (1/7 : ℝ) ^ (1/4 : ℝ) * (1/3 : ℝ) ^ (1/6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l52_5297


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l52_5285

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 18*x^2 + 91*x - 170

-- State the theorem
theorem partial_fraction_decomposition 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_roots : p a = 0 ∧ p b = 0 ∧ p c = 0) 
  (D E F : ℝ) 
  (h_decomp : ∀ (s : ℝ), s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 18*s^2 + 91*s - 170) = D / (s - a) + E / (s - b) + F / (s - c)) :
  D + E + F = 0 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l52_5285


namespace NUMINAMATH_CALUDE_tan_equality_solution_l52_5236

theorem tan_equality_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) →
  n = 60 ∨ n = -120 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l52_5236


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l52_5275

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x₀ : ℝ, 2^x₀ + x₀^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l52_5275


namespace NUMINAMATH_CALUDE_luke_stickers_l52_5215

/-- Calculates the number of stickers Luke has left after a series of events. -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Proves that Luke has 39 stickers left after the given events. -/
theorem luke_stickers : stickers_left 20 12 20 5 8 = 39 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l52_5215


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l52_5232

theorem alcohol_mixture_problem (original_volume : ℝ) (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 15 →
  added_water = 2 →
  new_percentage = 17.647058823529413 →
  (new_percentage / 100) * (original_volume + added_water) = (20 / 100) * original_volume :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l52_5232


namespace NUMINAMATH_CALUDE_sum_value_l52_5256

def S : ℚ := 3003 + (1/3) * (3002 + (1/6) * (3001 + (1/9) * (3000 + (1/(3*1000)) * 3)))

theorem sum_value : S = 3002.5 := by sorry

end NUMINAMATH_CALUDE_sum_value_l52_5256


namespace NUMINAMATH_CALUDE_spinster_count_spinster_count_proof_l52_5242

theorem spinster_count : ℕ → ℕ → Prop :=
  fun spinsters cats =>
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 ∧
    cats = spinsters + 55 →
    spinsters = 22

-- The proof is omitted
theorem spinster_count_proof : spinster_count 22 77 := by sorry

end NUMINAMATH_CALUDE_spinster_count_spinster_count_proof_l52_5242


namespace NUMINAMATH_CALUDE_investment_income_percentage_l52_5227

/-- Proves that the total annual income from two investments is 6% of the total invested amount -/
theorem investment_income_percentage : ∀ (investment1 investment2 rate1 rate2 : ℝ),
  investment1 = 2400 →
  investment2 = 2399.9999999999995 →
  rate1 = 0.04 →
  rate2 = 0.08 →
  let total_investment := investment1 + investment2
  let total_income := investment1 * rate1 + investment2 * rate2
  (total_income / total_investment) * 100 = 6 := by sorry

end NUMINAMATH_CALUDE_investment_income_percentage_l52_5227


namespace NUMINAMATH_CALUDE_marble_selection_problem_l52_5211

theorem marble_selection_problem (n : ℕ) (k : ℕ) (s : ℕ) (t : ℕ) :
  n = 15 ∧ k = 5 ∧ s = 4 ∧ t = 2 →
  (Nat.choose s t) * (Nat.choose (n - s) (k - t)) = 990 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l52_5211


namespace NUMINAMATH_CALUDE_tree_shadow_length_l52_5253

/-- Given a tree and a flag pole, proves the length of the tree's shadow. -/
theorem tree_shadow_length 
  (tree_height : ℝ) 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : tree_height = 12)
  (h2 : flagpole_height = 150)
  (h3 : flagpole_shadow = 100) : 
  (tree_height * flagpole_shadow) / flagpole_height = 8 :=
by sorry

end NUMINAMATH_CALUDE_tree_shadow_length_l52_5253


namespace NUMINAMATH_CALUDE_homer_candy_crush_ratio_l52_5276

/-- Proves that the ratio of points scored on the third try to points scored on the second try is 2:1 in Homer's Candy Crush game -/
theorem homer_candy_crush_ratio :
  ∀ (first_try second_try third_try : ℕ),
    first_try = 400 →
    second_try = first_try - 70 →
    ∃ (m : ℕ), third_try = m * second_try →
    first_try + second_try + third_try = 1390 →
    third_try = 2 * second_try :=
by
  sorry

#check homer_candy_crush_ratio

end NUMINAMATH_CALUDE_homer_candy_crush_ratio_l52_5276


namespace NUMINAMATH_CALUDE_percentage_problem_l52_5262

/-- The percentage that, when applied to 12356, results in 6.178 is 0.05% -/
theorem percentage_problem : ∃ p : ℝ, p * 12356 = 6.178 ∧ p = 0.0005 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l52_5262


namespace NUMINAMATH_CALUDE_wendy_bouquets_l52_5299

/-- Represents the number of flowers of each type -/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  daisies : ℕ

/-- Calculates the number of complete bouquets that can be made -/
def max_bouquets (initial : FlowerCount) (wilted : FlowerCount) (bouquet : FlowerCount) : ℕ :=
  let remaining : FlowerCount := ⟨
    initial.roses - wilted.roses,
    initial.lilies - wilted.lilies,
    initial.daisies - wilted.daisies
  ⟩
  min (remaining.roses / bouquet.roses)
      (min (remaining.lilies / bouquet.lilies)
           (remaining.daisies / bouquet.daisies))

/-- The main theorem stating that the maximum number of complete bouquets is 2 -/
theorem wendy_bouquets :
  let initial : FlowerCount := ⟨20, 15, 10⟩
  let wilted : FlowerCount := ⟨12, 8, 5⟩
  let bouquet : FlowerCount := ⟨3, 2, 1⟩
  max_bouquets initial wilted bouquet = 2 := by
  sorry


end NUMINAMATH_CALUDE_wendy_bouquets_l52_5299


namespace NUMINAMATH_CALUDE_angle_measure_l52_5247

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l52_5247


namespace NUMINAMATH_CALUDE_indeterminate_f_five_l52_5251

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem indeterminate_f_five
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_shift : ∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) :
  ¬∃ y, ∀ f, IsOdd f → (∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) → f 5 = y :=
sorry

end NUMINAMATH_CALUDE_indeterminate_f_five_l52_5251


namespace NUMINAMATH_CALUDE_complex_equation_solution_l52_5239

theorem complex_equation_solution (z : ℂ) : (3 - I) * z = 1 - I → z = 2/5 - 1/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l52_5239


namespace NUMINAMATH_CALUDE_student_contribution_l52_5279

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19)
  : (total_contribution - class_funds) / num_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_contribution_l52_5279


namespace NUMINAMATH_CALUDE_wendy_pictures_l52_5278

theorem wendy_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) (first_album : ℕ) : 
  total = 79 →
  num_albums = 5 →
  pics_per_album = 7 →
  first_album + num_albums * pics_per_album = total →
  first_album = 44 := by
sorry

end NUMINAMATH_CALUDE_wendy_pictures_l52_5278


namespace NUMINAMATH_CALUDE_second_category_amount_is_720_l52_5223

/-- Represents a budget with three categories -/
structure Budget where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the amount allocated to the second category in a budget -/
def amount_second_category (b : Budget) : ℕ :=
  b.total * b.ratio2 / (b.ratio1 + b.ratio2 + b.ratio3)

/-- Theorem stating that for a budget with ratio 5:4:1 and total $1800, 
    the amount allocated to the second category is $720 -/
theorem second_category_amount_is_720 :
  ∀ (b : Budget), b.total = 1800 ∧ b.ratio1 = 5 ∧ b.ratio2 = 4 ∧ b.ratio3 = 1 →
  amount_second_category b = 720 := by
  sorry

end NUMINAMATH_CALUDE_second_category_amount_is_720_l52_5223


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l52_5248

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : distance = 12) 
  (h3 : time = 2) : 
  ∃ (speed : ℝ), speed - current_speed = distance / time ∧ speed = 10 :=
sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l52_5248


namespace NUMINAMATH_CALUDE_inconsistent_average_and_sum_l52_5203

theorem inconsistent_average_and_sum :
  let numbers : List ℕ := [54, 55, 57, 58, 62, 62, 63, 65, 65]
  let average : ℕ := 60
  let total_sum : ℕ := average * numbers.length
  let sum_of_numbers : ℕ := numbers.sum
  sum_of_numbers > total_sum := by sorry

end NUMINAMATH_CALUDE_inconsistent_average_and_sum_l52_5203


namespace NUMINAMATH_CALUDE_f_5_eq_neg_f_3_l52_5216

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def quadratic_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_5_eq_neg_f_3 (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_negative f) 
  (h3 : quadratic_on_interval f) : 
  f 5 = -f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_5_eq_neg_f_3_l52_5216


namespace NUMINAMATH_CALUDE_multiply_divide_example_l52_5273

theorem multiply_divide_example : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_example_l52_5273


namespace NUMINAMATH_CALUDE_shaded_grid_percentage_l52_5206

theorem shaded_grid_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 36) (h2 : shaded_squares = 16) : 
  (shaded_squares : ℚ) / total_squares * 100 = 44.4444444444444444 := by
  sorry

end NUMINAMATH_CALUDE_shaded_grid_percentage_l52_5206


namespace NUMINAMATH_CALUDE_composite_function_sum_l52_5254

/-- Given a function f(x) = px + q where p and q are real numbers,
    if f(f(f(x))) = 8x + 21, then p + q = 5 -/
theorem composite_function_sum (p q : ℝ) :
  (∀ x, ∃ f : ℝ → ℝ, f x = p * x + q) →
  (∀ x, ∃ f : ℝ → ℝ, f (f (f x)) = 8 * x + 21) →
  p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_sum_l52_5254


namespace NUMINAMATH_CALUDE_nail_size_fraction_l52_5283

theorem nail_size_fraction (x : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 1) 
  (h2 : x + 0.5 = 0.75) : 
  x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_nail_size_fraction_l52_5283


namespace NUMINAMATH_CALUDE_five_travelers_three_rooms_l52_5238

/-- The number of ways to arrange travelers into guest rooms -/
def arrange_travelers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are at least 1 traveler in each room -/
axiom at_least_one (n k : ℕ) : arrange_travelers n k > 0 → k ≤ n

theorem five_travelers_three_rooms :
  arrange_travelers 5 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_five_travelers_three_rooms_l52_5238


namespace NUMINAMATH_CALUDE_william_washed_two_normal_cars_l52_5241

/-- The time William spends washing a normal car's windows -/
def window_time : ℕ := 4

/-- The time William spends washing a normal car's body -/
def body_time : ℕ := 7

/-- The time William spends cleaning a normal car's tires -/
def tire_time : ℕ := 4

/-- The time William spends waxing a normal car -/
def wax_time : ℕ := 9

/-- The total time William spends on one normal car -/
def normal_car_time : ℕ := window_time + body_time + tire_time + wax_time

/-- The time William spends on one big SUV -/
def suv_time : ℕ := 2 * normal_car_time

/-- The total time William spent washing all vehicles -/
def total_time : ℕ := 96

/-- The number of normal cars William washed -/
def normal_cars : ℕ := (total_time - suv_time) / normal_car_time

theorem william_washed_two_normal_cars : normal_cars = 2 := by
  sorry

end NUMINAMATH_CALUDE_william_washed_two_normal_cars_l52_5241


namespace NUMINAMATH_CALUDE_restriction_surjective_l52_5233

theorem restriction_surjective
  (f : Set.Ioc 0 1 → Set.Ioo 0 1)
  (hf_continuous : Continuous f)
  (hf_surjective : Function.Surjective f) :
  ∀ a ∈ Set.Ioo 0 1,
    Function.Surjective (fun x => f ⟨x, by sorry⟩ : Set.Ioo a 1 → Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_restriction_surjective_l52_5233


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l52_5260

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  sum_constraint : red + blue = total

/-- Represents the probability of drawing two marbles of the same color -/
def drawProbability (box1 box2 : MarbleBox) (color : ℕ → ℕ) : ℚ :=
  (color box1.red / box1.total) * (color box2.red / box2.total)

theorem marble_probability_theorem (box1 box2 : MarbleBox) :
  box1.total + box2.total = 34 →
  drawProbability box1 box2 (fun x => x) = 19/34 →
  drawProbability box1 box2 (fun x => box1.total - x) = 64/289 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l52_5260


namespace NUMINAMATH_CALUDE_xy_plus_one_is_square_l52_5263

theorem xy_plus_one_is_square (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = 1 / (x + 2) + 1 / (y - 2)) : 
  ∃ (n : ℤ), (x * y + 1 : ℤ) = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_xy_plus_one_is_square_l52_5263


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l52_5272

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l52_5272


namespace NUMINAMATH_CALUDE_equal_increment_implies_linear_l52_5230

/-- A function with the property that equal increments in input correspond to equal increments in output -/
def EqualIncrementFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ : ℝ, x₂ - x₁ = x₄ - x₃ → f x₂ - f x₁ = f x₄ - f x₃

/-- The main theorem: if a function has the equal increment property, then it is linear -/
theorem equal_increment_implies_linear (f : ℝ → ℝ) (h : EqualIncrementFunction f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_equal_increment_implies_linear_l52_5230


namespace NUMINAMATH_CALUDE_absolute_difference_m_n_l52_5202

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_difference_m_n (m n : ℝ) 
  (h : (m + 2 * i) / i = n + i) : 
  |m - n| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_m_n_l52_5202


namespace NUMINAMATH_CALUDE_hyperbola_condition_l52_5287

/-- The equation (x^2)/(k-2) + (y^2)/(5-k) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- The general form of the equation -/
def equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k → is_hyperbola k) ∧
  (is_hyperbola k → ∃ x y, equation x y k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l52_5287


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l52_5234

/-- Given digits A and B in base d > 8, if AB̄_d + AĀ_d = 194_d, then A_d - B_d = 5_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 8) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d * d + 9 * d + 4) : 
  A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l52_5234


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l52_5269

theorem solution_satisfies_equations :
  let x : ℚ := 599 / 204
  let y : ℚ := 65 / 136
  (7 * x - 50 * y = -3) ∧ (3 * x - 2 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l52_5269


namespace NUMINAMATH_CALUDE_three_digit_square_mod_1000_l52_5258

theorem three_digit_square_mod_1000 (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000]) ↔ (n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_mod_1000_l52_5258


namespace NUMINAMATH_CALUDE_quadratic_minimum_at_positive_l52_5200

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 6

-- Theorem statement
theorem quadratic_minimum_at_positive (x : ℝ) :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (y : ℝ), f y ≥ f x_min :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_at_positive_l52_5200


namespace NUMINAMATH_CALUDE_meal_cost_l52_5243

theorem meal_cost (total_cost : ℕ) (num_meals : ℕ) (h1 : total_cost = 21) (h2 : num_meals = 3) :
  total_cost / num_meals = 7 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_l52_5243


namespace NUMINAMATH_CALUDE_leila_earnings_proof_l52_5264

-- Define the given conditions
def voltaire_daily_viewers : ℕ := 50
def leila_daily_viewers : ℕ := 2 * voltaire_daily_viewers
def earnings_per_view : ℚ := 1/2

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Leila's weekly earnings
def leila_weekly_earnings : ℚ := leila_daily_viewers * earnings_per_view * days_in_week

-- Theorem statement
theorem leila_earnings_proof : leila_weekly_earnings = 350 := by
  sorry

end NUMINAMATH_CALUDE_leila_earnings_proof_l52_5264


namespace NUMINAMATH_CALUDE_find_number_l52_5257

theorem find_number : ∃! x : ℚ, (172 / 4 - 28) * x + 7 = 172 := by sorry

end NUMINAMATH_CALUDE_find_number_l52_5257


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l52_5201

/-- The probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.4

/-- The probability of rain on Sunday -/
def prob_rain_sunday : ℝ := 0.5

/-- The probabilities are independent -/
axiom independence : True

/-- The probability of rain on at least one day over the weekend -/
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

theorem weekend_rain_probability :
  prob_rain_weekend = 0.7 :=
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l52_5201


namespace NUMINAMATH_CALUDE_average_difference_l52_5290

/-- The average of an arithmetic sequence with first term a and last term b -/
def arithmeticMean (a b : Int) : Rat := (a + b) / 2

/-- The set of even integers from a to b inclusive -/
def evenIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 0}

/-- The set of odd integers from a to b inclusive -/
def oddIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 1}

theorem average_difference :
  (arithmeticMean 20 60 - arithmeticMean 10 140 = -35) ∧
  (arithmeticMean 21 59 - arithmeticMean 11 139 = -35) := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l52_5290


namespace NUMINAMATH_CALUDE_inequality_proof_l52_5298

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l52_5298


namespace NUMINAMATH_CALUDE_crayons_remaining_l52_5245

theorem crayons_remaining (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 440 → given_away = 111 → lost = 106 → initial - given_away - lost = 223 := by
  sorry

end NUMINAMATH_CALUDE_crayons_remaining_l52_5245


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l52_5282

theorem smallest_distance_between_circles (z w : ℂ) : 
  Complex.abs (z - (2 + 2 * Complex.I)) = 2 →
  Complex.abs (w - (5 + 6 * Complex.I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = 11 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 2 * Complex.I)) = 2 →
                   Complex.abs (w' - (5 + 6 * Complex.I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l52_5282


namespace NUMINAMATH_CALUDE_sin_pi_plus_alpha_l52_5292

theorem sin_pi_plus_alpha (α : Real) :
  (∃ (x y : Real), x = Real.sqrt 5 ∧ y = -2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (Real.pi + α) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_plus_alpha_l52_5292


namespace NUMINAMATH_CALUDE_quadratic_second_root_l52_5246

theorem quadratic_second_root 
  (p q r : ℝ) 
  (h : 2*p*(q-r)*2^2 + 3*q*(r-p)*2 + 4*r*(p-q) = 0) :
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    2*p*(q-r)*x^2 + 3*q*(r-p)*x + 4*r*(p-q) = 0 ∧ 
    x = (r*(p-q)) / (p*(q-r)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_second_root_l52_5246


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_is_79_l52_5210

/-- The expression to be simplified -/
def expression (y : ℝ) : ℝ := 5 * (y^2 - 3*y + 3) - 6 * (y^3 - 2*y + 2)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squares_of_coefficients : ℕ := 79

/-- Theorem stating that the sum of squares of coefficients of the simplified expression is 79 -/
theorem sum_of_squares_of_coefficients_is_79 : 
  sum_of_squares_of_coefficients = 79 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_is_79_l52_5210


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l52_5294

/-- The number of ways to arrange 5 people in a row for a photo, with one person fixed in the middle -/
def photo_arrangements : ℕ := 24

/-- The number of people in the photo -/
def total_people : ℕ := 5

/-- The number of people who can be arranged in non-middle positions -/
def non_middle_people : ℕ := total_people - 1

theorem photo_arrangement_count : photo_arrangements = non_middle_people! := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l52_5294


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l52_5240

theorem min_value_quadratic (x y : ℝ) : 
  y = 5 * x^2 - 8 * x + 20 → x ≥ 1 → y ≥ 13 := by
  sorry

theorem min_value_achieved (x : ℝ) : 
  x ≥ 1 → ∃ y : ℝ, y = 5 * x^2 - 8 * x + 20 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l52_5240


namespace NUMINAMATH_CALUDE_g_12_equals_155_l52_5237

/-- The function g defined for all integers n -/
def g (n : ℤ) : ℤ := n^2 - n + 23

/-- Theorem stating that g(12) equals 155 -/
theorem g_12_equals_155 : g 12 = 155 := by
  sorry

end NUMINAMATH_CALUDE_g_12_equals_155_l52_5237


namespace NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l52_5266

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Proof that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l52_5266


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l52_5212

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

-- Define the distance between foci
def distance_between_foci (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance :
  distance_between_foci ellipse_equation = 2 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l52_5212


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l52_5209

theorem sum_of_squares_roots (h : ℝ) : 
  (∃ r s : ℝ, r^2 - 4*h*r - 8 = 0 ∧ s^2 - 4*h*s - 8 = 0 ∧ r^2 + s^2 = 20) → 
  h = 1/2 ∨ h = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l52_5209


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l52_5252

/-- For an equilateral triangle where the square of each side's length
    is equal to the perimeter, the area of the triangle is 9√3/4 square units. -/
theorem equilateral_triangle_area (s : ℝ) (h1 : s > 0) (h2 : s^2 = 3*s) :
  (s^2 * Real.sqrt 3) / 4 = 9 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l52_5252


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l52_5219

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a + b ∧ a^2 + 2*a*b + 4*b^2 = 6) →
    -Real.sqrt 6 ≤ w ∧ w ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l52_5219

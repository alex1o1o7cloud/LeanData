import Mathlib

namespace NUMINAMATH_CALUDE_min_non_isosceles_2008gon_l3678_367852

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  diagonals : ℕ
  diag_bound : diagonals ≤ p.sides - 3

/-- The number of non-isosceles triangles in a triangulation -/
def non_isosceles_count (p : RegularPolygon) (t : Triangulation p) : ℕ := sorry

theorem min_non_isosceles_2008gon :
  ∀ (p : RegularPolygon) (t : Triangulation p),
    p.sides = 2008 →
    t.diagonals = 2005 →
    non_isosceles_count p t ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_non_isosceles_2008gon_l3678_367852


namespace NUMINAMATH_CALUDE_triangle_conditions_l3678_367820

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Define the conditions
def conditionA (t : Triangle) : Prop := t.A = t.B - t.C
def conditionB (t : Triangle) : Prop := t.A = t.B ∧ t.C = 2 * t.A
def conditionC (t : Triangle) : Prop := t.b^2 = t.a^2 - t.c^2
def conditionD (t : Triangle) : Prop := ∃ (k : ℝ), t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k

-- The theorem to prove
theorem triangle_conditions (t : Triangle) :
  (conditionA t → isRightTriangle t) ∧
  (conditionB t → isRightTriangle t) ∧
  (conditionC t → isRightTriangle t) ∧
  (conditionD t → ¬isRightTriangle t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_conditions_l3678_367820


namespace NUMINAMATH_CALUDE_sibling_ages_l3678_367895

theorem sibling_ages (x y z : ℕ) : 
  x - y = 3 →                    -- Age difference between siblings
  z - 1 = 2 * (x + y) →          -- Father's age one year ago
  z + 20 = (x + 20) + (y + 20) → -- Father's age in 20 years
  (x = 11 ∧ y = 8) :=            -- Ages of the siblings
by sorry

end NUMINAMATH_CALUDE_sibling_ages_l3678_367895


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3678_367881

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ c = 1/12 ∧ d = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3678_367881


namespace NUMINAMATH_CALUDE_gregory_age_l3678_367878

/-- Represents the ages of Dmitry and Gregory at different points in time -/
structure Ages where
  gregory_past : ℕ
  dmitry_past : ℕ
  gregory_current : ℕ
  dmitry_current : ℕ
  gregory_future : ℕ
  dmitry_future : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  a.dmitry_current = 3 * a.gregory_past ∧
  a.gregory_current = a.dmitry_past ∧
  a.gregory_future = a.dmitry_current ∧
  a.gregory_future + a.dmitry_future = 49 ∧
  a.dmitry_future - a.gregory_future = a.dmitry_current - a.gregory_current

theorem gregory_age (a : Ages) : age_conditions a → a.gregory_current = 14 := by
  sorry

#check gregory_age

end NUMINAMATH_CALUDE_gregory_age_l3678_367878


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3678_367831

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 4 / y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3678_367831


namespace NUMINAMATH_CALUDE_cost_of_fencing_square_l3678_367815

/-- The cost of fencing a square -/
theorem cost_of_fencing_square (cost_per_side : ℕ) (h : cost_per_side = 79) : 
  4 * cost_per_side = 316 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_fencing_square_l3678_367815


namespace NUMINAMATH_CALUDE_stratified_sampling_under_40_l3678_367846

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 400

/-- Represents the number of teachers under 40 -/
def teachers_under_40 : ℕ := 250

/-- Represents the total sample size -/
def sample_size : ℕ := 80

/-- Calculates the number of teachers under 40 in the sample -/
def sample_under_40 : ℕ := (teachers_under_40 * sample_size) / total_teachers

theorem stratified_sampling_under_40 :
  sample_under_40 = 50 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_under_40_l3678_367846


namespace NUMINAMATH_CALUDE_power_of_two_minus_one_as_power_l3678_367810

theorem power_of_two_minus_one_as_power (n : ℕ) : 
  (∃ (a k : ℕ), k ≥ 2 ∧ 2^n - 1 = a^k) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_minus_one_as_power_l3678_367810


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3678_367835

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 25
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent (R S : ℝ × ℝ) : Prop :=
  C₁ R.1 R.2 ∧ C₂ S.1 S.2 ∧
  ∀ T : ℝ × ℝ, (T ≠ R ∧ T ≠ S) →
    (C₁ T.1 T.2 → (T.1 - R.1)^2 + (T.2 - R.2)^2 < (S.1 - R.1)^2 + (S.2 - R.2)^2) ∧
    (C₂ T.1 T.2 → (T.1 - S.1)^2 + (T.2 - S.2)^2 < (R.1 - S.1)^2 + (R.2 - S.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ R S : ℝ × ℝ, is_tangent R S ∧
    ∀ R' S' : ℝ × ℝ, is_tangent R' S' →
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) ≤ Real.sqrt ((S'.1 - R'.1)^2 + (S'.2 - R'.2)^2) ∧
      Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 5 * Real.sqrt 15 + 10 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3678_367835


namespace NUMINAMATH_CALUDE_product_of_powers_of_ten_l3678_367863

theorem product_of_powers_of_ten : (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_of_ten_l3678_367863


namespace NUMINAMATH_CALUDE_rachel_money_left_l3678_367862

theorem rachel_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (initial_amount / 4 + initial_amount / 5 + initial_amount / 10 + initial_amount / 8) = 65 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_left_l3678_367862


namespace NUMINAMATH_CALUDE_triangles_count_is_120_l3678_367803

/-- Represents the configuration of points on two line segments -/
structure PointConfiguration :=
  (total_points : ℕ)
  (points_on_segment1 : ℕ)
  (points_on_segment2 : ℕ)
  (end_point : ℕ)

/-- Calculates the number of triangles that can be formed with the given point configuration -/
def count_triangles (config : PointConfiguration) : ℕ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the specific configuration results in 120 triangles -/
theorem triangles_count_is_120 :
  let config := PointConfiguration.mk 11 6 4 1
  count_triangles config = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_count_is_120_l3678_367803


namespace NUMINAMATH_CALUDE_croissant_baking_time_l3678_367899

/-- Calculates the baking time for croissants given the process parameters -/
theorem croissant_baking_time 
  (num_folds : ℕ) 
  (fold_time : ℕ) 
  (rest_time : ℕ) 
  (mixing_time : ℕ) 
  (total_time : ℕ) 
  (h1 : num_folds = 4)
  (h2 : fold_time = 5)
  (h3 : rest_time = 75)
  (h4 : mixing_time = 10)
  (h5 : total_time = 360) :
  total_time - (mixing_time + num_folds * fold_time + num_folds * rest_time) = 30 := by
  sorry

#check croissant_baking_time

end NUMINAMATH_CALUDE_croissant_baking_time_l3678_367899


namespace NUMINAMATH_CALUDE_deposit_percentage_l3678_367843

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 55 → remaining = 495 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l3678_367843


namespace NUMINAMATH_CALUDE_gcd_equation_solutions_l3678_367802

theorem gcd_equation_solutions (x y d : ℕ) :
  d = Nat.gcd x y →
  d + x * y / d = x + y →
  (∃ k : ℕ, (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_equation_solutions_l3678_367802


namespace NUMINAMATH_CALUDE_second_question_percentage_l3678_367860

/-- Represents the percentage of boys in a test scenario -/
structure TestPercentages where
  first : ℝ  -- Percentage who answered the first question correctly
  neither : ℝ  -- Percentage who answered neither question correctly
  both : ℝ  -- Percentage who answered both questions correctly

/-- 
Given the percentages of boys who answered the first question correctly, 
neither question correctly, and both questions correctly, 
proves that the percentage who answered the second question correctly is 55%.
-/
theorem second_question_percentage (p : TestPercentages) 
  (h1 : p.first = 75)
  (h2 : p.neither = 20)
  (h3 : p.both = 50) : 
  ∃ second : ℝ, second = 55 := by
  sorry

#check second_question_percentage

end NUMINAMATH_CALUDE_second_question_percentage_l3678_367860


namespace NUMINAMATH_CALUDE_thirty_percent_of_number_l3678_367822

theorem thirty_percent_of_number (x : ℝ) (h : (3 / 7) * x = 0.4 * x + 12) : 0.3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_number_l3678_367822


namespace NUMINAMATH_CALUDE_west_asian_percentage_approx_46_percent_l3678_367832

/-- Represents the Asian population (in millions) for each region of the U.S. -/
structure AsianPopulation where
  ne : ℕ
  mw : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asian population living in the West -/
def westAsianPercentage (pop : AsianPopulation) : ℚ :=
  (pop.west : ℚ) / (pop.ne + pop.mw + pop.south + pop.west)

/-- The given Asian population data for 1990 -/
def population1990 : AsianPopulation :=
  { ne := 2, mw := 3, south := 2, west := 6 }

theorem west_asian_percentage_approx_46_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
  |westAsianPercentage population1990 - (46 : ℚ) / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_west_asian_percentage_approx_46_percent_l3678_367832


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3678_367890

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.2 * L
  let new_surface_area := 6 * new_edge_length^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 44 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3678_367890


namespace NUMINAMATH_CALUDE_max_absolute_value_of_Z_l3678_367841

theorem max_absolute_value_of_Z (Z : ℂ) (h : Complex.abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), Complex.abs (W - (3 + 4*I)) = 1 → Complex.abs W ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_Z_l3678_367841


namespace NUMINAMATH_CALUDE_derivative_at_zero_l3678_367865

/-- Given a function f where f(x) = x^2 + 2x * f'(1), prove that f'(0) = -4 -/
theorem derivative_at_zero (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l3678_367865


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l3678_367864

/-- Given a circle with circumference 80 feet, proves that an arc corresponding
    to a central angle of 45° has a length of 10 feet. -/
theorem arc_length_45_degrees (circle : Real) (arc : Real) : 
  circle = 80 → -- The circumference of the circle is 80 feet
  arc = circle * (45 / 360) → -- The arc length is proportional to its central angle (45°)
  arc = 10 := by -- The arc length is 10 feet
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l3678_367864


namespace NUMINAMATH_CALUDE_laundry_dishes_multiple_l3678_367814

theorem laundry_dishes_multiple : ∃ m : ℝ, 46 = m * 20 + 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_laundry_dishes_multiple_l3678_367814


namespace NUMINAMATH_CALUDE_triangle_side_range_l3678_367808

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the sine law
def sineLaw (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.c / Real.sin t.C = t.a / Real.sin t.A

-- Define the condition for two solutions
def hasTwoSolutions (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ sineLaw t1 ∧ sineLaw t2 ∧
    t1.b = t.b ∧ t1.B = t.B ∧ t2.b = t.b ∧ t2.B = t.B

-- State the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.B = π/4)  -- 45° in radians
  (h3 : hasTwoSolutions t) :
  2 < t.a ∧ t.a < 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3678_367808


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3678_367839

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p ∧ p ∣ (25^3 - 27^2)) ∧ 
  (∀ p, Nat.Prime p → p ∣ (25^3 - 27^2) → p ∈ s) ∧
  (s.sum id = 28) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3678_367839


namespace NUMINAMATH_CALUDE_smallest_q_is_6_l3678_367893

/-- Three consecutive terms of an arithmetic sequence -/
structure ArithmeticTriple where
  p : ℝ
  q : ℝ
  r : ℝ
  positive : 0 < p ∧ 0 < q ∧ 0 < r
  consecutive : ∃ d : ℝ, p + d = q ∧ q + d = r

/-- The product of the three terms equals 216 -/
def productIs216 (t : ArithmeticTriple) : Prop :=
  t.p * t.q * t.r = 216

theorem smallest_q_is_6 (t : ArithmeticTriple) (h : productIs216 t) :
    t.q ≥ 6 ∧ ∃ t' : ArithmeticTriple, productIs216 t' ∧ t'.q = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_q_is_6_l3678_367893


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3678_367809

theorem three_numbers_sum (a b c m : ℕ) : 
  a + b + c = 2015 →
  a + b = m + 1 →
  b + c = m + 2011 →
  c + a = m + 2012 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3678_367809


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3678_367840

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.im (2 * i / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3678_367840


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3678_367825

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line that the tangent should be parallel to -/
def m : ℝ := 4

theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = -1 ∧ p.2 = -4 ∨ p.1 = 1 ∧ p.2 = 0} =
  {p : ℝ × ℝ | p.2 = f p.1 ∧ f' p.1 = m} :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3678_367825


namespace NUMINAMATH_CALUDE_jamie_bathroom_theorem_l3678_367874

/-- The amount of liquid (in ounces) that triggers the need to use the bathroom -/
def bathroom_threshold : ℕ := 32

/-- The amount of liquid (in ounces) in a cup -/
def cup_ounces : ℕ := 8

/-- The amount of liquid (in ounces) in a pint -/
def pint_ounces : ℕ := 16

/-- The amount of liquid Jamie consumed before the test -/
def pre_test_consumption : ℕ := cup_ounces + pint_ounces

/-- The maximum amount Jamie can drink during the test without needing the bathroom -/
def max_test_consumption : ℕ := bathroom_threshold - pre_test_consumption

theorem jamie_bathroom_theorem : max_test_consumption = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_bathroom_theorem_l3678_367874


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3678_367837

/-- Given positive real numbers a, b, and c such that a + b + c = 1,
    the maximum value of √(a+1) + √(b+1) + √(c+1) is achieved
    when applying the Cauchy-Schwarz inequality. -/
theorem max_value_sqrt_sum (a b c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hsum : a + b + c = 1) : 
    ∃ (max : ℝ), ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) ≤ max ∧
    Real.sqrt (a + 1) + Real.sqrt (b + 1) + Real.sqrt (c + 1) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3678_367837


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l3678_367829

/-- The probability of drawing two white balls sequentially without replacement
    from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem two_white_balls_probability
  (white_balls : Nat)
  (black_balls : Nat)
  (h_white : white_balls = 7)
  (h_black : black_balls = 8) :
  (white_balls / (white_balls + black_balls)) *
  ((white_balls - 1) / (white_balls + black_balls - 1)) =
  1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l3678_367829


namespace NUMINAMATH_CALUDE_system_solutions_l3678_367857

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * |y| - 4 * |x| = 6
def equation2 (x y a : ℝ) : Prop := x^2 + y^2 - 14*y + 49 - a^2 = 0

-- Define the number of solutions
def has_n_solutions (n : ℕ) (a : ℝ) : Prop :=
  ∃ (solutions : Finset (ℝ × ℝ)), 
    solutions.card = n ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ equation1 x y ∧ equation2 x y a

-- Theorem statement
theorem system_solutions (a : ℝ) :
  (has_n_solutions 3 a ↔ |a| = 5 ∨ |a| = 9) ∧
  (has_n_solutions 2 a ↔ |a| = 3 ∨ (5 < |a| ∧ |a| < 9)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l3678_367857


namespace NUMINAMATH_CALUDE_range_of_a_l3678_367868

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | a < x ∧ x < a + 1}
def B := {x : ℝ | 3 + 2*x - x^2 > 0}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a ∩ B = A a) ↔ (∀ a : ℝ, -1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3678_367868


namespace NUMINAMATH_CALUDE_marble_jar_problem_l3678_367845

theorem marble_jar_problem (num_marbles : ℕ) : 
  (∀ (x : ℚ), num_marbles / 24 = x → num_marbles / 26 = x - 1) →
  num_marbles = 312 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l3678_367845


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3678_367834

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 3 5
  let z₂ : ℂ := Complex.mk (-2) 3
  z₁ / z₂ = Complex.mk (-21/13) (-19/13) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3678_367834


namespace NUMINAMATH_CALUDE_vessel_mixture_theorem_main_vessel_theorem_l3678_367821

/-- Represents the contents of a vessel -/
structure VesselContents where
  milk : ℚ
  water : ℚ

/-- Represents a vessel with its volume and contents -/
structure Vessel where
  volume : ℚ
  contents : VesselContents

/-- Theorem stating the relationship between vessel contents and their mixture -/
theorem vessel_mixture_theorem 
  (v1 v2 : Vessel) 
  (h1 : v1.volume / v2.volume = 3 / 5)
  (h2 : v2.contents.milk / v2.contents.water = 3 / 2)
  (h3 : (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1) :
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

/-- Main theorem proving the equivalence of the vessel mixture problem -/
theorem main_vessel_theorem :
  ∀ (v1 v2 : Vessel),
  v1.volume / v2.volume = 3 / 5 →
  v2.contents.milk / v2.contents.water = 3 / 2 →
  (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1 ↔
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vessel_mixture_theorem_main_vessel_theorem_l3678_367821


namespace NUMINAMATH_CALUDE_greatest_value_when_x_is_negative_six_l3678_367856

theorem greatest_value_when_x_is_negative_six :
  let x : ℝ := -6
  (2 - x > 2 + x) ∧
  (2 - x > x - 1) ∧
  (2 - x > x) ∧
  (2 - x > x / 2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_when_x_is_negative_six_l3678_367856


namespace NUMINAMATH_CALUDE_arithmetic_sequence_kth_term_l3678_367885

/-- Given an arithmetic sequence where the sum of the first n terms is 3n^2 + 2n,
    this theorem proves that the k-th term is 6k - 1. -/
theorem arithmetic_sequence_kth_term 
  (S : ℕ → ℝ) -- S represents the sum function of the sequence
  (h : ∀ n : ℕ, S n = 3 * n^2 + 2 * n) -- condition that sum of first n terms is 3n^2 + 2n
  (k : ℕ) -- k represents the index of the term we're looking for
  : S k - S (k-1) = 6 * k - 1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_kth_term_l3678_367885


namespace NUMINAMATH_CALUDE_imaginary_complex_magnitude_l3678_367819

theorem imaginary_complex_magnitude (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + 2 * Complex.I) / (1 + Complex.I)) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_complex_magnitude_l3678_367819


namespace NUMINAMATH_CALUDE_product_of_pairs_l3678_367804

theorem product_of_pairs (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (2 - x₁/y₁) * (2 - x₂/y₂) * (2 - x₃/y₃) = 26219/2016 :=
by sorry

end NUMINAMATH_CALUDE_product_of_pairs_l3678_367804


namespace NUMINAMATH_CALUDE_not_complete_residue_sum_l3678_367898

/-- For an even number n, if a and b are complete residue systems modulo n,
    then their pairwise sum is not a complete residue system modulo n. -/
theorem not_complete_residue_sum
  (n : ℕ) (hn : Even n) 
  (a b : Fin n → ℕ)
  (ha : Function.Surjective (λ i => a i % n))
  (hb : Function.Surjective (λ i => b i % n)) :
  ¬ Function.Surjective (λ i => (a i + b i) % n) :=
sorry

end NUMINAMATH_CALUDE_not_complete_residue_sum_l3678_367898


namespace NUMINAMATH_CALUDE_power_product_equality_l3678_367800

theorem power_product_equality : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) = 3 * (3 ^ 4) ^ (1/5) := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l3678_367800


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l3678_367875

/-- Proves that the borrowing interest rate is 4% given the conditions of the problem -/
theorem borrowing_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (lending_rate : ℝ)
  (gain_per_year : ℝ)
  (h1 : principal = 5000)
  (h2 : time = 2)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : (principal * lending_rate - gain_per_year) / principal = 0.04 := by
  sorry

#eval (5000 * 0.06 - 100) / 5000  -- Should output 0.04

end NUMINAMATH_CALUDE_borrowing_interest_rate_l3678_367875


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l3678_367813

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 24 inches
    and the other has a length of 8 inches, prove that the width of the second rectangle
    is 45 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length : ℕ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 15)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 8) :
    area / jordan_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l3678_367813


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_unit_interval_l3678_367851

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sin x}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (-x^2 + 4*x - 3)}

-- State the theorem
theorem A_intersect_B_equals_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_unit_interval_l3678_367851


namespace NUMINAMATH_CALUDE_reeya_average_score_l3678_367896

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score : 
  (List.sum reeya_scores) / (List.length reeya_scores) = 76.6 := by
  sorry

end NUMINAMATH_CALUDE_reeya_average_score_l3678_367896


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3678_367842

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- Shift a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

/-- Shift a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after shifts -/
def resulting_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_theorem :
  resulting_parabola = { a := 1, b := -6, c := 13 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3678_367842


namespace NUMINAMATH_CALUDE_abc_order_l3678_367870

def a : ℚ := (10^1988 + 1) / (10^1989 + 1)
def b : ℚ := (10^1987 + 1) / (10^1988 + 1)
def c : ℚ := (10^1987 + 9) / (10^1988 + 9)

theorem abc_order : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_order_l3678_367870


namespace NUMINAMATH_CALUDE_poultry_farm_loss_l3678_367880

theorem poultry_farm_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss : ℕ)
  (total_birds_after_week : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : total_birds_after_week = 349)
  (h7 : initial_chickens + initial_turkeys + initial_guinea_fowls
      - (7 * daily_turkey_loss + 7 * daily_guinea_fowl_loss)
      - total_birds_after_week = 7 * daily_chicken_loss) :
  daily_chicken_loss = 20 := by sorry

end NUMINAMATH_CALUDE_poultry_farm_loss_l3678_367880


namespace NUMINAMATH_CALUDE_hcf_of_product_and_greater_l3678_367859

theorem hcf_of_product_and_greater (a b : ℕ) (h1 : a * b = 4107) (h2 : a = 111) :
  Nat.gcd a b = 37 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_product_and_greater_l3678_367859


namespace NUMINAMATH_CALUDE_sara_marbles_l3678_367850

/-- Given that Sara has 10 marbles initially and loses 7 marbles, prove that she will have 3 marbles left. -/
theorem sara_marbles (initial_marbles : ℕ) (lost_marbles : ℕ) (h1 : initial_marbles = 10) (h2 : lost_marbles = 7) :
  initial_marbles - lost_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l3678_367850


namespace NUMINAMATH_CALUDE_expression_evaluation_l3678_367833

theorem expression_evaluation (a b c d : ℚ) : 
  a = 3 → 
  b = a + 3 → 
  c = b - 8 → 
  d = a + 5 → 
  a + 2 ≠ 0 → 
  b - 4 ≠ 0 → 
  c + 5 ≠ 0 → 
  d - 3 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 4) * (c + 9) / (c + 5) * (d + 1) / (d - 3) = 1512 / 75 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3678_367833


namespace NUMINAMATH_CALUDE_candy_bar_savings_l3678_367886

/-- Calculates the number of items saved given weekly receipt, consumption rate, and time period. -/
def items_saved (weekly_receipt : ℕ) (consumption_rate : ℕ) (weeks : ℕ) : ℕ :=
  weekly_receipt * weeks - (weeks / consumption_rate)

/-- Proves that under the given conditions, 28 items are saved after 16 weeks. -/
theorem candy_bar_savings : items_saved 2 4 16 = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_savings_l3678_367886


namespace NUMINAMATH_CALUDE_milburg_population_l3678_367877

theorem milburg_population : 
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by sorry

end NUMINAMATH_CALUDE_milburg_population_l3678_367877


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seven_thirds_l3678_367812

theorem smallest_integer_greater_than_neg_seven_thirds :
  Int.ceil (-7/3 : ℚ) = -2 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seven_thirds_l3678_367812


namespace NUMINAMATH_CALUDE_bacteria_growth_relation_l3678_367892

/-- Represents the amount of bacteria at a given time point -/
structure BacteriaAmount where
  amount : ℝ
  time : ℕ

/-- Represents a growth factor between two time points -/
structure GrowthFactor where
  factor : ℝ
  startTime : ℕ
  endTime : ℕ

/-- Theorem stating the relationship between initial, intermediate, and final bacteria amounts
    and their corresponding growth factors -/
theorem bacteria_growth_relation 
  (A₁ : BacteriaAmount) 
  (A₂ : BacteriaAmount) 
  (A₃ : BacteriaAmount) 
  (g : GrowthFactor) 
  (h : GrowthFactor) : 
  A₁.time = 1 →
  A₂.time = 4 →
  A₃.time = 7 →
  g.startTime = 1 →
  g.endTime = 4 →
  h.startTime = 4 →
  h.endTime = 7 →
  A₁.amount = 10 →
  A₃.amount = 12.1 →
  A₂.amount = A₁.amount * g.factor →
  A₃.amount = A₂.amount * h.factor →
  A₃.amount = A₁.amount * g.factor * h.factor :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_relation_l3678_367892


namespace NUMINAMATH_CALUDE_missing_number_proof_l3678_367823

theorem missing_number_proof (x : ℝ) : 
  let numbers := [1, 22, 23, 24, 25, 26, x, 2]
  (List.sum numbers) / (List.length numbers) = 20 → x = 37 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3678_367823


namespace NUMINAMATH_CALUDE_crayons_added_l3678_367873

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by sorry

end NUMINAMATH_CALUDE_crayons_added_l3678_367873


namespace NUMINAMATH_CALUDE_most_balls_l3678_367884

theorem most_balls (soccerballs basketballs : ℕ) 
  (h1 : soccerballs = 50)
  (h2 : basketballs = 26)
  (h3 : ∃ baseballs : ℕ, baseballs = basketballs + 8) :
  soccerballs > basketballs ∧ soccerballs > basketballs + 8 := by
sorry

end NUMINAMATH_CALUDE_most_balls_l3678_367884


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l3678_367848

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Point Q where extended sides BC and DE meet -/
def Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem regular_octagon_extended_sides_angle 
  (octagon : RegularOctagon) : 
  angle_measure (octagon.vertices 3) (Q octagon) (octagon.vertices 4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l3678_367848


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3678_367879

theorem factorization_of_difference_of_squares (x y : ℝ) :
  4 * x^2 - y^4 = (2*x + y^2) * (2*x - y^2) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3678_367879


namespace NUMINAMATH_CALUDE_magic_deck_cost_l3678_367872

/-- Calculates the cost per deck given the initial number of decks, remaining decks, and total earnings -/
def cost_per_deck (initial_decks : ℕ) (remaining_decks : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_decks - remaining_decks)

/-- Proves that the cost per deck is 7 dollars given the problem conditions -/
theorem magic_deck_cost :
  let initial_decks : ℕ := 16
  let remaining_decks : ℕ := 8
  let total_earnings : ℕ := 56
  cost_per_deck initial_decks remaining_decks total_earnings = 7 := by
  sorry


end NUMINAMATH_CALUDE_magic_deck_cost_l3678_367872


namespace NUMINAMATH_CALUDE_favorite_numbers_parity_l3678_367861

/-- Represents a person's favorite number -/
structure FavoriteNumber where
  value : ℤ

/-- Represents whether a number is even or odd -/
inductive Parity
  | Even
  | Odd

/-- Returns the parity of an integer -/
def parity (n : ℤ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The problem setup -/
structure FavoriteNumbers where
  jan : FavoriteNumber
  dan : FavoriteNumber
  anna : FavoriteNumber
  hana : FavoriteNumber
  h1 : parity (dan.value + 3 * jan.value) = Parity.Odd
  h2 : parity ((anna.value - hana.value) * 5) = Parity.Odd
  h3 : parity (dan.value * hana.value + 17) = Parity.Even

/-- The main theorem to prove -/
theorem favorite_numbers_parity (nums : FavoriteNumbers) :
  parity nums.dan.value = Parity.Odd ∧
  parity nums.hana.value = Parity.Odd ∧
  parity nums.anna.value = Parity.Even ∧
  parity nums.jan.value = Parity.Even :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_parity_l3678_367861


namespace NUMINAMATH_CALUDE_jim_ate_15_cookies_l3678_367826

def cookies_problem (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (num_flour_bags : ℕ) (flour_bag_weight : ℕ) (cookies_left : ℕ) : Prop :=
  let total_flour := num_flour_bags * flour_bag_weight
  let num_batches := total_flour / flour_per_batch
  let total_cookies := num_batches * cookies_per_batch
  let cookies_eaten := total_cookies - cookies_left
  cookies_eaten = 15

theorem jim_ate_15_cookies :
  cookies_problem 12 2 4 5 105 := by
  sorry

end NUMINAMATH_CALUDE_jim_ate_15_cookies_l3678_367826


namespace NUMINAMATH_CALUDE_possible_m_values_l3678_367806

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem possible_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l3678_367806


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_18_l3678_367805

theorem smallest_divisible_by_10_11_18 : ∃ n : ℕ+, (∀ m : ℕ+, 10 ∣ m ∧ 11 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 10 ∣ n ∧ 11 ∣ n ∧ 18 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_18_l3678_367805


namespace NUMINAMATH_CALUDE_lcm_1230_924_l3678_367828

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1230_924_l3678_367828


namespace NUMINAMATH_CALUDE_simplify_fraction_l3678_367849

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3678_367849


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3678_367807

theorem arithmetic_geometric_mean_ratio : ∃ (c d : ℝ), 
  c > d ∧ d > 0 ∧ 
  (c + d) / 2 = 3 * Real.sqrt (c * d) ∧
  |(c / d) - 34| < 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3678_367807


namespace NUMINAMATH_CALUDE_age_ratio_l3678_367876

def sachin_age : ℕ := 63
def age_difference : ℕ := 18

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l3678_367876


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_difference_product_relation_l3678_367817

theorem reciprocal_sum_of_difference_product_relation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x - y = 3 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_difference_product_relation_l3678_367817


namespace NUMINAMATH_CALUDE_ratio_equality_l3678_367854

theorem ratio_equality (x y z : ℝ) (h : x / 4 = y / 3 ∧ y / 3 = z / 2) :
  (x - y + 3 * z) / x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3678_367854


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l3678_367883

theorem infinite_solutions_condition (k : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - k) = 3 * (4 * x + 10)) ↔ k = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l3678_367883


namespace NUMINAMATH_CALUDE_painted_cubes_count_l3678_367897

/-- Given a cube with side length 4, composed of unit cubes, where the interior 2x2x2 cube is unpainted,
    the number of unit cubes with at least one face painted is 56. -/
theorem painted_cubes_count (n : ℕ) (h1 : n = 4) : 
  n^3 - (n - 2)^3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l3678_367897


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3678_367827

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 5| = |x + 3| + 2 := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3678_367827


namespace NUMINAMATH_CALUDE_dara_employment_wait_l3678_367858

/-- The minimum age required for employment at the company -/
def min_age : ℕ := 25

/-- Jane's current age -/
def jane_age : ℕ := 28

/-- The number of years in the future when Dara will be half Jane's age -/
def future_years : ℕ := 6

/-- Calculates Dara's current age based on the given conditions -/
def dara_current_age : ℕ :=
  (jane_age + future_years) / 2 - future_years

/-- The time before Dara reaches the minimum age for employment -/
def time_to_min_age : ℕ := min_age - dara_current_age

/-- Theorem stating that the time before Dara reaches the minimum age for employment is 14 years -/
theorem dara_employment_wait : time_to_min_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_dara_employment_wait_l3678_367858


namespace NUMINAMATH_CALUDE_sam_dimes_count_l3678_367844

def final_dimes (initial : ℕ) (dad_gave : ℕ) (mom_took : ℕ) (sister_sets : ℕ) (dimes_per_set : ℕ) : ℕ :=
  initial + dad_gave - mom_took + sister_sets * dimes_per_set

theorem sam_dimes_count :
  final_dimes 9 7 3 4 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sam_dimes_count_l3678_367844


namespace NUMINAMATH_CALUDE_min_value_is_214_l3678_367836

-- Define the type for our permutations
def Permutation := Fin 9 → Fin 9

-- Define the function we want to minimize
def f (p : Permutation) : ℕ :=
  let x₁ := (p 0).val + 1
  let x₂ := (p 1).val + 1
  let x₃ := (p 2).val + 1
  let y₁ := (p 3).val + 1
  let y₂ := (p 4).val + 1
  let y₃ := (p 5).val + 1
  let z₁ := (p 6).val + 1
  let z₂ := (p 7).val + 1
  let z₃ := (p 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃

theorem min_value_is_214 :
  (∃ (p : Permutation), f p = 214) ∧ (∀ (p : Permutation), f p ≥ 214) := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_214_l3678_367836


namespace NUMINAMATH_CALUDE_function_values_l3678_367818

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_values (A : ℝ) :
  f A (π / 3) = 0 →
  A = 1 ∧ f A (π / 12) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l3678_367818


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l3678_367888

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Theorem 1
theorem necessary_not_sufficient_p_for_q (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) →
  m ≥ 7/2 :=
sorry

-- Theorem 2
theorem necessary_not_sufficient_not_p_for_not_q (m : ℝ) :
  (m > 0) →
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) →
  1 ≤ m ∧ m ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l3678_367888


namespace NUMINAMATH_CALUDE_zach_monday_miles_l3678_367891

/-- Calculates the number of miles driven on Monday given the rental conditions and total cost --/
def miles_driven_monday (flat_fee : ℚ) (cost_per_mile : ℚ) (thursday_miles : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - flat_fee - cost_per_mile * thursday_miles) / cost_per_mile

/-- Proves that Zach drove 620 miles on Monday given the rental conditions and total cost --/
theorem zach_monday_miles :
  let flat_fee : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let thursday_miles : ℚ := 744
  let total_cost : ℚ := 832
  miles_driven_monday flat_fee cost_per_mile thursday_miles total_cost = 620 := by
  sorry

end NUMINAMATH_CALUDE_zach_monday_miles_l3678_367891


namespace NUMINAMATH_CALUDE_ghi_equilateral_same_circumcenter_l3678_367811

-- Define the points
variable (A B C D E F G H I G' H' I' : ℝ × ℝ)

-- Define the triangles
def triangle (P Q R : ℝ × ℝ) := Set.insert P (Set.insert Q (Set.singleton R))

-- Define equilateral triangle
def is_equilateral (t : Set (ℝ × ℝ)) : Prop :=
  ∃ P Q R, t = triangle P Q R ∧ 
    dist P Q = dist Q R ∧ dist Q R = dist R P

-- Define reflection
def reflect (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define circumcenter
def circumcenter (t : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Assumptions
variable (h1 : is_equilateral (triangle A B D))
variable (h2 : is_equilateral (triangle A C E))
variable (h3 : is_equilateral (triangle B C F))
variable (h4 : G = circumcenter (triangle A B D))
variable (h5 : H = circumcenter (triangle A C E))
variable (h6 : I = circumcenter (triangle B C F))
variable (h7 : G' = reflect B A G)
variable (h8 : H' = reflect C B H)
variable (h9 : I' = reflect A C I)

-- Theorem statements
theorem ghi_equilateral :
  is_equilateral (triangle G H I) := sorry

theorem same_circumcenter :
  circumcenter (triangle G H I) = circumcenter (triangle G' H' I') := sorry

end NUMINAMATH_CALUDE_ghi_equilateral_same_circumcenter_l3678_367811


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3678_367830

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3678_367830


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3678_367869

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3678_367869


namespace NUMINAMATH_CALUDE_town_employment_theorem_l3678_367889

/-- Represents the employment statistics of town X -/
structure TownEmployment where
  total_population : ℝ
  employed_percentage : ℝ
  employed_male_percentage : ℝ
  employed_female_percentage : ℝ

/-- The employment theorem for town X -/
theorem town_employment_theorem (stats : TownEmployment) 
  (h1 : stats.employed_male_percentage = 24)
  (h2 : stats.employed_female_percentage = 75) :
  stats.employed_percentage = 96 := by
  sorry

end NUMINAMATH_CALUDE_town_employment_theorem_l3678_367889


namespace NUMINAMATH_CALUDE_wood_measurement_correct_l3678_367853

/-- Represents the length of the wood in feet -/
def wood_length : ℝ := sorry

/-- Represents the length of the rope in feet -/
def rope_length : ℝ := sorry

/-- The system of equations for the wood measurement problem -/
def wood_measurement_equations : Prop :=
  (rope_length - wood_length = 4.5) ∧ (wood_length - 1/2 * rope_length = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_correct : wood_measurement_equations :=
sorry

end NUMINAMATH_CALUDE_wood_measurement_correct_l3678_367853


namespace NUMINAMATH_CALUDE_chandler_saves_49_weeks_l3678_367871

/-- The number of weeks it takes Chandler to save for a mountain bike --/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 620
  let birthday_money : ℕ := 70 + 40 + 20
  let weekly_earnings : ℕ := 18
  let weekly_spending : ℕ := 8
  let weekly_savings : ℕ := weekly_earnings - weekly_spending
  ((bike_cost - birthday_money) + weekly_savings - 1) / weekly_savings

theorem chandler_saves_49_weeks :
  weeks_to_save = 49 :=
sorry

end NUMINAMATH_CALUDE_chandler_saves_49_weeks_l3678_367871


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3678_367824

theorem polynomial_difference_divisibility 
  (a b c d : ℤ) (x y : ℤ) (h : x ≠ y) :
  ∃ k : ℤ, (x - y) * k = 
    (a * x^3 + b * x^2 + c * x + d) - (a * y^3 + b * y^2 + c * y + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3678_367824


namespace NUMINAMATH_CALUDE_one_real_root_l3678_367867

-- Define the determinant function
def det (x a b d : ℝ) : ℝ := x * (x^2 + a^2 + b^2 + d^2)

-- State the theorem
theorem one_real_root (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃! x : ℝ, det x a b d = 0 :=
sorry

end NUMINAMATH_CALUDE_one_real_root_l3678_367867


namespace NUMINAMATH_CALUDE_tournament_rankings_l3678_367801

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(sunday_winners_round_robin : List Team)
(sunday_losers_round_robin : List Team)

/-- Represents a ranking of teams -/
def Ranking := List Team

/-- Function to calculate the number of possible rankings -/
def number_of_rankings (t : Tournament) : Nat :=
  6 * 6

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) :
  number_of_rankings t = 36 :=
sorry

end NUMINAMATH_CALUDE_tournament_rankings_l3678_367801


namespace NUMINAMATH_CALUDE_kimmie_earnings_l3678_367847

theorem kimmie_earnings (kimmie_earnings : ℚ) : 
  (kimmie_earnings / 2 + (2 / 3 * kimmie_earnings) / 2 = 375) → 
  kimmie_earnings = 450 := by
  sorry

end NUMINAMATH_CALUDE_kimmie_earnings_l3678_367847


namespace NUMINAMATH_CALUDE_fermat_like_congruence_l3678_367838

theorem fermat_like_congruence (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  let n : ℕ := (2^(2*p) - 1) / 3
  2^n - 2 ≡ 0 [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_congruence_l3678_367838


namespace NUMINAMATH_CALUDE_range_of_a_l3678_367866

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) → 
  a ≥ 1 ∨ a ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3678_367866


namespace NUMINAMATH_CALUDE_additional_cars_during_play_l3678_367887

/-- Calculates the number of additional cars that parked during a play given the initial conditions. -/
theorem additional_cars_during_play
  (front_initial : ℕ)
  (back_initial : ℕ)
  (total_end : ℕ)
  (h1 : front_initial = 100)
  (h2 : back_initial = 2 * front_initial)
  (h3 : total_end = 700) :
  total_end - (front_initial + back_initial) = 300 :=
by sorry

end NUMINAMATH_CALUDE_additional_cars_during_play_l3678_367887


namespace NUMINAMATH_CALUDE_visitors_scientific_notation_l3678_367855

-- Define the number of visitors
def visitors : ℝ := 256000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.56 * (10 ^ 5)

-- Theorem to prove that the number of visitors is equal to its scientific notation representation
theorem visitors_scientific_notation : visitors = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_visitors_scientific_notation_l3678_367855


namespace NUMINAMATH_CALUDE_wire_circle_square_ratio_l3678_367882

theorem wire_circle_square_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (π * (a / (2 * π))^2 = (b / 4)^2) → (a / b = 2 / Real.sqrt π) := by
  sorry

end NUMINAMATH_CALUDE_wire_circle_square_ratio_l3678_367882


namespace NUMINAMATH_CALUDE_kishore_savings_l3678_367816

def total_expenses : ℕ := 16200
def savings_rate : ℚ := 1 / 10

theorem kishore_savings :
  ∀ (salary : ℕ),
  (salary : ℚ) = (total_expenses : ℚ) + savings_rate * (salary : ℚ) →
  savings_rate * (salary : ℚ) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_l3678_367816


namespace NUMINAMATH_CALUDE_power_of_power_l3678_367894

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3678_367894

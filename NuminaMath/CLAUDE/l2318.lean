import Mathlib

namespace NUMINAMATH_CALUDE_tangent_range_l2318_231845

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation of the tangent line passing through (a, f(a)) and (2, t) --/
def tangent_equation (a t : ℝ) : Prop :=
  t - (f a) = (f' a) * (2 - a)

/-- The condition for three distinct tangent lines --/
def three_tangents (t : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    tangent_equation a t ∧ tangent_equation b t ∧ tangent_equation c t

/-- Theorem: If a point (2, t) can be used to draw three tangent lines to y = f(x),
    then t is in the open interval (-6, 2) --/
theorem tangent_range :
  ∀ t : ℝ, three_tangents t → -6 < t ∧ t < 2 := by sorry

end NUMINAMATH_CALUDE_tangent_range_l2318_231845


namespace NUMINAMATH_CALUDE_polar_point_equivalence_l2318_231840

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to standard form where r > 0 and 0 ≤ θ < 2π -/
def toStandardForm (p : PolarPoint) : PolarPoint :=
  sorry

theorem polar_point_equivalence :
  let p := PolarPoint.mk (-4) (5 * Real.pi / 6)
  let standardP := toStandardForm p
  standardP.r = 4 ∧ standardP.θ = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_polar_point_equivalence_l2318_231840


namespace NUMINAMATH_CALUDE_fraction_absolute_value_less_than_one_l2318_231803

theorem fraction_absolute_value_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_absolute_value_less_than_one_l2318_231803


namespace NUMINAMATH_CALUDE_solve_equation_l2318_231834

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2318_231834


namespace NUMINAMATH_CALUDE_problem_statement_l2318_231853

theorem problem_statement (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_θ_range : π/4 < θ ∧ θ < π/2)
  (h_eq1 : Real.cos θ / x = Real.sin θ / y)
  (h_eq2 : Real.sin θ^2 / x^2 + Real.cos θ^2 / y^2 = 10 / (3 * (x^2 + y^2))) :
  (x + y)^2 / (x^2 + y^2) = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2318_231853


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2318_231869

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  1/a + 1/b + 4/c + 9/d + 16/e + 25/f ≥ 25.6 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 1/b' + 4/c' + 9/d' + 16/e' + 25/f' = 25.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2318_231869


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2318_231802

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2318_231802


namespace NUMINAMATH_CALUDE_count_bases_with_final_digit_one_l2318_231822

/-- The number of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def count_bases : ℕ := 7

/-- The set of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def valid_bases : Finset ℕ := {2, 3, 4, 6, 8, 9, 12}

theorem count_bases_with_final_digit_one :
  (Finset.range 11).filter (fun b => 625 % (b + 2) = 1) = valid_bases ∧
  valid_bases.card = count_bases :=
sorry

end NUMINAMATH_CALUDE_count_bases_with_final_digit_one_l2318_231822


namespace NUMINAMATH_CALUDE_semicircle_radius_is_ten_l2318_231854

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The height of the triangle, which is equal to the length of its legs -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base of the triangle is 20 units -/
  base_eq : base = 20
  /-- The semicircle's diameter is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle is 10 units -/
theorem semicircle_radius_is_ten (t : IsoscelesTriangleWithSemicircle) : t.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_is_ten_l2318_231854


namespace NUMINAMATH_CALUDE_count_valid_house_numbers_l2318_231891

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_house_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  is_prime (n / 1000) ∧ 
  is_prime (n % 1000) ∧
  (n / 1000) < 100 ∧
  (n % 1000) < 500 ∧
  ∀ d : ℕ, d < 5 → (n / 10^d % 10) ≠ 0

theorem count_valid_house_numbers :
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_house_number n) ∧ s.card = 1302 :=
sorry

end NUMINAMATH_CALUDE_count_valid_house_numbers_l2318_231891


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2318_231851

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if two points are perpendicular with respect to the origin -/
def perpendicular (a b : Point) : Prop :=
  a.x * b.x + a.y * b.y = 0

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  point.y = line.m * point.x + line.b

/-- The main theorem -/
theorem parabola_line_intersection 
  (C : Parabola) 
  (F : Point)
  (l : Line)
  (A B : Point)
  (h1 : F.x = 1/2 ∧ F.y = 0)
  (h2 : l.m = 2)
  (h3 : onParabola A C ∧ onParabola B C)
  (h4 : onLine A l ∧ onLine B l)
  (h5 : A ≠ ⟨0, 0⟩ ∧ B ≠ ⟨0, 0⟩)
  (h6 : perpendicular A B) :
  C.p = 1 ∧ l.b = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2318_231851


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_over_2_l2318_231892

theorem cos_2alpha_minus_pi_over_2 (α : ℝ) :
  (Real.cos α = -5/13) → (Real.sin α = 12/13) → Real.cos (2*α - π/2) = -120/169 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_over_2_l2318_231892


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2318_231855

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - 5*x + 1
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧
               x₂ = (5 - Real.sqrt 21) / 2 ∧
               f x₁ = 0 ∧ f x₂ = 0 ∧
               ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2318_231855


namespace NUMINAMATH_CALUDE_bushes_needed_for_perfume_l2318_231860

/-- The number of rose petals needed to make an ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

/-- The number of bottles of perfume to be made -/
def num_bottles : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The theorem stating the number of bushes needed to make the required perfume -/
theorem bushes_needed_for_perfume : 
  (petals_per_ounce * num_bottles * ounces_per_bottle) / (petals_per_rose * roses_per_bush) = 800 := by
  sorry

end NUMINAMATH_CALUDE_bushes_needed_for_perfume_l2318_231860


namespace NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l2318_231865

theorem x_squared_gt_4_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l2318_231865


namespace NUMINAMATH_CALUDE_two_a_minus_b_value_l2318_231859

theorem two_a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 5)
  (hab : |a + b| = -(a + b)) :
  2*a - b = 13 ∨ 2*a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_two_a_minus_b_value_l2318_231859


namespace NUMINAMATH_CALUDE_approx_C_squared_minus_D_squared_for_specific_values_l2318_231874

/-- Given nonnegative real numbers x, y, z, we define C and D as follows:
C = √(x + 3) + √(y + 6) + √(z + 12)
D = √(x + 2) + √(y + 4) + √(z + 8)
This theorem states that when x = 1, y = 2, and z = 3, the value of C² - D² 
is approximately 19.483 with arbitrary precision. -/
theorem approx_C_squared_minus_D_squared_for_specific_values :
  ∀ ε > 0, ∃ C D : ℝ,
  C = Real.sqrt (1 + 3) + Real.sqrt (2 + 6) + Real.sqrt (3 + 12) ∧
  D = Real.sqrt (1 + 2) + Real.sqrt (2 + 4) + Real.sqrt (3 + 8) ∧
  |C^2 - D^2 - 19.483| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_approx_C_squared_minus_D_squared_for_specific_values_l2318_231874


namespace NUMINAMATH_CALUDE_perfect_square_problem_l2318_231835

theorem perfect_square_problem (n : ℕ+) :
  ∃ k : ℕ, (n : ℤ)^2 + 19*(n : ℤ) + 48 = k^2 → n = 33 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l2318_231835


namespace NUMINAMATH_CALUDE_shelbys_driving_time_l2318_231862

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_sun speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_sun = 30)
  (h2 : speed_rain = 20)
  (h3 : total_time = 40)
  (h4 : total_distance = 16)
  (h5 : speed_sun > 0 ∧ speed_rain > 0) :
  ∃ (time_rain : ℝ), 
    time_rain = 24 ∧ 
    time_rain > 0 ∧ 
    time_rain < total_time ∧
    (speed_sun * (total_time - time_rain) / 60 + speed_rain * time_rain / 60 = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_shelbys_driving_time_l2318_231862


namespace NUMINAMATH_CALUDE_world_population_scientific_notation_l2318_231897

/-- The number of people in the global population by the end of 2022 -/
def world_population : ℕ := 8000000000

/-- The scientific notation representation of the world population -/
def scientific_notation : ℝ := 8 * (10 : ℝ) ^ 9

/-- Theorem stating that the world population is equal to its scientific notation representation -/
theorem world_population_scientific_notation : 
  (world_population : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_world_population_scientific_notation_l2318_231897


namespace NUMINAMATH_CALUDE_parallel_line_intersection_not_always_parallel_l2318_231861

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and intersection operations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (m n : Line)
variable (h_distinct_planes : α ≠ β)
variable (h_distinct_lines : m ≠ n)

-- State the theorem
theorem parallel_line_intersection_not_always_parallel :
  ¬(∀ (α β : Plane) (m n : Line),
    α ≠ β → m ≠ n →
    parallel m α → intersect α β n → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_intersection_not_always_parallel_l2318_231861


namespace NUMINAMATH_CALUDE_constant_function_theorem_l2318_231894

theorem constant_function_theorem (f : ℝ → ℝ) : 
  (∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f x * f (y * z) + 1) → 
  (∀ x : ℝ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l2318_231894


namespace NUMINAMATH_CALUDE_root_product_equals_27_l2318_231868

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l2318_231868


namespace NUMINAMATH_CALUDE_evaluate_expression_l2318_231830

theorem evaluate_expression : 3 * Real.sqrt 32 + 2 * Real.sqrt 50 = 22 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2318_231830


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2318_231815

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2318_231815


namespace NUMINAMATH_CALUDE_fraction_power_product_l2318_231875

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l2318_231875


namespace NUMINAMATH_CALUDE_y_derivative_l2318_231813

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x) - Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2318_231813


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2318_231824

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence -/
def SpecificSum (a : ℕ → ℝ) : ℝ :=
  a 2 + a 4 + a 9 + a 11

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SpecificSum a = 32 → a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2318_231824


namespace NUMINAMATH_CALUDE_zenobius_had_more_descendants_l2318_231831

/-- Calculates the total number of descendants for King Pafnutius -/
def pafnutius_descendants : ℕ :=
  2 + 60 * 2 + 20 * 1

/-- Calculates the total number of descendants for King Zenobius -/
def zenobius_descendants : ℕ :=
  4 + 35 * 3 + 35 * 1

/-- Proves that King Zenobius had more descendants than King Pafnutius -/
theorem zenobius_had_more_descendants :
  zenobius_descendants > pafnutius_descendants :=
by sorry

end NUMINAMATH_CALUDE_zenobius_had_more_descendants_l2318_231831


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2318_231888

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles at the same height in the first quadrant -/
def TangentLine (c1 c2 : Circle) :=
  ∃ (y : ℝ), y > 0 ∧
    (y = c1.radius ∨ y = c2.radius) ∧
    (c1.center.1 + c1.radius < c2.center.1 - c2.radius)

theorem tangent_line_y_intercept
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1 = ⟨(3, 0), 3⟩)
  (h2 : c2 = ⟨(8, 0), 2⟩)
  (h3 : TangentLine c1 c2) :
  ∃ (line : ℝ → ℝ), line 0 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2318_231888


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l2318_231879

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 72 → x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l2318_231879


namespace NUMINAMATH_CALUDE_andrews_cheese_pops_l2318_231847

theorem andrews_cheese_pops (hotdogs chicken_nuggets total : ℕ) 
  (hotdogs_count : hotdogs = 30)
  (chicken_nuggets_count : chicken_nuggets = 40)
  (total_count : total = 90)
  (sum_equation : hotdogs + chicken_nuggets + (total - hotdogs - chicken_nuggets) = total) :
  total - hotdogs - chicken_nuggets = 20 := by
  sorry

end NUMINAMATH_CALUDE_andrews_cheese_pops_l2318_231847


namespace NUMINAMATH_CALUDE_fixed_point_existence_l2318_231881

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two line segments have equal length -/
def equal_length (a b c d : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (c.x - d.x)^2 + (c.y - d.y)^2

/-- Check if an angle is 90 degrees -/
def is_right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (b.x - c.x) + (b.y - a.y) * (b.y - c.y) = 0

/-- Check if a quadrilateral is convex -/
def is_convex (a b c d : Point) : Prop := sorry

/-- Check if two points are on the same side of a line -/
def same_side (p q : Point) (l : Line) : Prop := sorry

theorem fixed_point_existence (a b : Point) :
  ∃ p : Point,
    ∀ c d : Point,
      is_convex a b c d →
      equal_length a b b c →
      equal_length a d d c →
      is_right_angle a d c →
      same_side c d (Line.mk (b.y - a.y) (a.x - b.x) (a.y * b.x - a.x * b.y)) →
      ∃ l : Line, d.on_line l ∧ c.on_line l ∧ p.on_line l :=
sorry

end NUMINAMATH_CALUDE_fixed_point_existence_l2318_231881


namespace NUMINAMATH_CALUDE_equation_solution_l2318_231864

theorem equation_solution : ∃ x : ℝ, (x - 3)^4 = 16 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2318_231864


namespace NUMINAMATH_CALUDE_stairs_climbed_l2318_231828

theorem stairs_climbed (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 4872 → 
  julia_stairs = Int.floor (2 * Real.sqrt (jonny_stairs / 2) + 15) → 
  jonny_stairs + julia_stairs = 4986 := by
sorry

end NUMINAMATH_CALUDE_stairs_climbed_l2318_231828


namespace NUMINAMATH_CALUDE_f_max_on_interval_f_greater_than_3x_solution_set_l2318_231852

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for the maximum value of f(x) on [-3, 1]
theorem f_max_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ M :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem f_greater_than_3x_solution_set :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_f_max_on_interval_f_greater_than_3x_solution_set_l2318_231852


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l2318_231808

def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l2318_231808


namespace NUMINAMATH_CALUDE_pentagon_right_angles_l2318_231866

/-- The sum of interior angles in a pentagon in degrees -/
def pentagonAngleSum : ℝ := 540

/-- The measure of a right angle in degrees -/
def rightAngle : ℝ := 90

/-- The set of possible numbers of right angles in a pentagon -/
def possibleRightAngles : Set ℕ := {0, 1, 2, 3}

/-- Theorem: The set of possible numbers of right angles in a pentagon is {0, 1, 2, 3} -/
theorem pentagon_right_angles :
  ∀ n : ℕ, n ∈ possibleRightAngles ↔ 
    (n : ℝ) * rightAngle ≤ pentagonAngleSum ∧ 
    (n + 1 : ℝ) * rightAngle > pentagonAngleSum :=
by sorry

end NUMINAMATH_CALUDE_pentagon_right_angles_l2318_231866


namespace NUMINAMATH_CALUDE_computer_accessories_cost_l2318_231886

def original_amount : ℕ := 48
def snack_cost : ℕ := 8

theorem computer_accessories_cost (remaining_amount : ℕ) 
  (h1 : remaining_amount = original_amount / 2 + 4) 
  (h2 : remaining_amount = original_amount - snack_cost - (original_amount - remaining_amount - snack_cost)) :
  original_amount - remaining_amount - snack_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_computer_accessories_cost_l2318_231886


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l2318_231880

/-- Custom multiplication operation -/
def star_op (a b : ℝ) := a^2 - a*b

/-- Theorem stating that the equation (x+1)*3 = -2 has two distinct real roots -/
theorem equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star_op (x₁ + 1) 3 = -2 ∧ star_op (x₂ + 1) 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l2318_231880


namespace NUMINAMATH_CALUDE_largest_decimal_number_l2318_231800

theorem largest_decimal_number : 
  let a := 0.989
  let b := 0.9098
  let c := 0.9899
  let d := 0.9009
  let e := 0.9809
  c > a ∧ c > b ∧ c > d ∧ c > e := by sorry

end NUMINAMATH_CALUDE_largest_decimal_number_l2318_231800


namespace NUMINAMATH_CALUDE_E_parity_l2318_231801

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : (isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023)) := by sorry

end NUMINAMATH_CALUDE_E_parity_l2318_231801


namespace NUMINAMATH_CALUDE_marcel_total_cost_l2318_231836

/-- The cost of Marcel's purchases -/
def total_cost (pen_price briefcase_price : ℝ) : ℝ :=
  pen_price + briefcase_price

/-- Theorem: Marcel's total cost for a pen and briefcase is $24 -/
theorem marcel_total_cost :
  ∃ (pen_price briefcase_price : ℝ),
    pen_price = 4 ∧
    briefcase_price = 5 * pen_price ∧
    total_cost pen_price briefcase_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_marcel_total_cost_l2318_231836


namespace NUMINAMATH_CALUDE_point_in_plane_region_l2318_231895

/-- The range of values for m such that point A(2, 3) lies within or on the boundary
    of the plane region represented by 3x - 2y + m ≥ 0 -/
theorem point_in_plane_region (m : ℝ) : 
  (3 * 2 - 2 * 3 + m ≥ 0) ↔ (m ≥ 0) := by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l2318_231895


namespace NUMINAMATH_CALUDE_pairball_longest_time_l2318_231882

/-- Represents the pairball game setup -/
structure PairballGame where
  totalTime : ℕ
  numChildren : ℕ
  longestPlayRatio : ℕ

/-- Calculates the playing time of the child who played the longest -/
def longestPlayingTime (game : PairballGame) : ℕ :=
  let totalChildMinutes := 2 * game.totalTime
  let adjustedChildren := game.numChildren - 1 + game.longestPlayRatio
  (totalChildMinutes * game.longestPlayRatio) / adjustedChildren

/-- Theorem stating that the longest playing time in the given scenario is 68 minutes -/
theorem pairball_longest_time :
  let game : PairballGame := {
    totalTime := 120,
    numChildren := 6,
    longestPlayRatio := 2
  }
  longestPlayingTime game = 68 := by
  sorry

end NUMINAMATH_CALUDE_pairball_longest_time_l2318_231882


namespace NUMINAMATH_CALUDE_sector_max_area_l2318_231827

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l2318_231827


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2318_231846

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  (∃ (a₁ b₁ : ℝ), a₁ + b₁ + 6 = 6 ∧ a₁^2 + b₁^2 + 6^2 = 18) ∧ 
  (∃ (a₂ b₂ : ℝ), a₂ + b₂ + (-2) = 6 ∧ a₂^2 + b₂^2 + (-2)^2 = 18) ∧
  (∀ (a₃ b₃ c₃ : ℝ), a₃ + b₃ + c₃ = 6 → a₃^2 + b₃^2 + c₃^2 = 18 → c₃ ≤ 6 ∧ c₃ ≥ -2) ∧
  (6 - (-2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2318_231846


namespace NUMINAMATH_CALUDE_ball_motion_time_formula_l2318_231856

/-- Represents the motion of a ball thrown upward -/
structure BallMotion where
  h : ℝ     -- Initial height
  V₀ : ℝ    -- Initial velocity
  g : ℝ     -- Gravitational acceleration
  t : ℝ     -- Time
  V : ℝ     -- Final velocity
  S : ℝ     -- Displacement

/-- The theorem stating the relationship between time, displacement, velocities, and height -/
theorem ball_motion_time_formula (b : BallMotion) 
  (hS : b.S = b.h + (1/2) * b.g * b.t^2 + b.V₀ * b.t)
  (hV : b.V = b.g * b.t + b.V₀) :
  b.t = (2 * (b.S - b.h)) / (b.V + b.V₀) :=
by sorry

end NUMINAMATH_CALUDE_ball_motion_time_formula_l2318_231856


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2318_231878

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 3 ∧
  a 2 + a 5 = 36

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 6 * n - 3

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  GeneralTermFormula a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2318_231878


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_tangent_line_k_value_l2318_231890

/-- A line is tangent to a parabola if and only if the discriminant of their intersection equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
   ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y)) ↔ 
  k = 9 := by
  sorry

/-- The value of k for which the line 4x - 3y + k = 0 is tangent to the parabola y² = 16x is 9 -/
theorem tangent_line_k_value : 
  ∃! k : ℝ, ∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
  ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_tangent_line_k_value_l2318_231890


namespace NUMINAMATH_CALUDE_exam_pass_count_l2318_231857

theorem exam_pass_count (total : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) :
  total = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ pass_count : ℕ,
    pass_count = 100 ∧
    pass_count ≤ total ∧
    (pass_count : ℚ) * avg_pass + (total - pass_count : ℚ) * avg_fail = (total : ℚ) * avg_all :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l2318_231857


namespace NUMINAMATH_CALUDE_triangle_relations_l2318_231814

/-- Given a triangle with area S, inradius r, exradii r_a, r_b, r_c, 
    side lengths a, b, c, circumradius R, and semiperimeter p -/
theorem triangle_relations (S r r_a r_b r_c a b c R : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_semiperimeter : ∃ p, p = (a + b + c) / 2) :
  (1 / r^3 - 1 / r_a^3 - 1 / r_b^3 - 1 / r_c^3 = 12 * R / S^2) ∧
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_relations_l2318_231814


namespace NUMINAMATH_CALUDE_max_value_abc_l2318_231889

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81/262144 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2318_231889


namespace NUMINAMATH_CALUDE_square_area_14m_l2318_231842

/-- The area of a square with side length 14 meters is 196 square meters. -/
theorem square_area_14m (side_length : ℝ) (h : side_length = 14) : 
  side_length * side_length = 196 := by
  sorry

end NUMINAMATH_CALUDE_square_area_14m_l2318_231842


namespace NUMINAMATH_CALUDE_roots_eq1_roots_eq2_l2318_231839

-- Define the quadratic equations
def eq1 (x : ℝ) := x^2 - 2*x - 8
def eq2 (x : ℝ) := 2*x^2 - 4*x + 1

-- Theorem for the roots of the first equation
theorem roots_eq1 : 
  (eq1 4 = 0 ∧ eq1 (-2) = 0) ∧ 
  ∀ x : ℝ, eq1 x = 0 → x = 4 ∨ x = -2 := by sorry

-- Theorem for the roots of the second equation
theorem roots_eq2 : 
  (eq2 ((2 + Real.sqrt 2) / 2) = 0 ∧ eq2 ((2 - Real.sqrt 2) / 2) = 0) ∧ 
  ∀ x : ℝ, eq2 x = 0 → x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_roots_eq1_roots_eq2_l2318_231839


namespace NUMINAMATH_CALUDE_rational_includes_positive_and_negative_l2318_231843

-- Define rational numbers
def RationalNumber : Type := ℚ

-- Define positive and negative rational numbers
def PositiveRational (q : ℚ) : Prop := q > 0
def NegativeRational (q : ℚ) : Prop := q < 0

-- State the theorem
theorem rational_includes_positive_and_negative :
  (∃ q : ℚ, PositiveRational q) ∧ (∃ q : ℚ, NegativeRational q) :=
sorry

end NUMINAMATH_CALUDE_rational_includes_positive_and_negative_l2318_231843


namespace NUMINAMATH_CALUDE_sqrt_three_cubed_l2318_231810

theorem sqrt_three_cubed : Real.sqrt 3 ^ 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_cubed_l2318_231810


namespace NUMINAMATH_CALUDE_circle_area_with_chord_l2318_231863

theorem circle_area_with_chord (chord_length : ℝ) (center_to_chord : ℝ) (area : ℝ) : 
  chord_length = 10 →
  center_to_chord = 5 →
  area = π * (center_to_chord^2 + (chord_length / 2)^2) →
  area = 50 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_with_chord_l2318_231863


namespace NUMINAMATH_CALUDE_tuesday_temperature_l2318_231816

theorem tuesday_temperature
  (temp_tues wed thur fri : ℝ)
  (avg_tues_wed_thur : (temp_tues + wed + thur) / 3 = 52)
  (avg_wed_thur_fri : (wed + thur + fri) / 3 = 54)
  (fri_temp : fri = 53) :
  temp_tues = 47 := by
sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l2318_231816


namespace NUMINAMATH_CALUDE_solve_age_problem_l2318_231837

def age_problem (rona_age : ℕ) : Prop :=
  let rachel_age := 2 * rona_age
  let collete_age := rona_age / 2
  let tommy_age := collete_age + rona_age
  rachel_age + rona_age + collete_age + tommy_age = 40

theorem solve_age_problem : age_problem 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_age_problem_l2318_231837


namespace NUMINAMATH_CALUDE_no_integer_solution_l2318_231804

theorem no_integer_solution : 
  ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2318_231804


namespace NUMINAMATH_CALUDE_factorial_ones_divisibility_l2318_231850

/-- Definition of [n]! -/
def factorial_ones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (factorial_ones k) * (Nat.ofDigits 2 (List.replicate (k + 1) 1))

/-- Theorem stating that [n+m]! is divisible by [n]! · [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry


end NUMINAMATH_CALUDE_factorial_ones_divisibility_l2318_231850


namespace NUMINAMATH_CALUDE_additional_steps_day3_l2318_231812

def day1_steps : ℕ := 200 + 300

def day2_steps : ℕ := 2 * day1_steps

def total_steps : ℕ := 1600

theorem additional_steps_day3 : 
  total_steps - (day1_steps + day2_steps) = 100 := by sorry

end NUMINAMATH_CALUDE_additional_steps_day3_l2318_231812


namespace NUMINAMATH_CALUDE_x_value_l2318_231811

theorem x_value (y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2318_231811


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l2318_231883

theorem multiply_and_add_equality : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l2318_231883


namespace NUMINAMATH_CALUDE_initial_ace_cards_l2318_231841

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseballCards : ℕ
  finalBaseballCards : ℕ
  finalAceCards : ℕ
  aceBaseballDifference : ℕ

/-- Theorem stating the initial number of Ace cards Nell had --/
theorem initial_ace_cards (n : NellCards) 
  (h1 : n.initialBaseballCards = 239)
  (h2 : n.finalBaseballCards = 111)
  (h3 : n.finalAceCards = 376)
  (h4 : n.aceBaseballDifference = 265)
  (h5 : n.finalAceCards - n.finalBaseballCards = n.aceBaseballDifference) :
  n.finalAceCards + (n.initialBaseballCards - n.finalBaseballCards) = 504 := by
  sorry

end NUMINAMATH_CALUDE_initial_ace_cards_l2318_231841


namespace NUMINAMATH_CALUDE_pen_profit_percentage_retailer_profit_is_20_625_percent_l2318_231877

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := num_pens_price
  let selling_price_per_pen := 1 - (discount_percent / 100)
  let total_selling_price := selling_price_per_pen * num_pens_bought
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  20.625

/-- The retailer's profit percentage is 20.625% -/
theorem retailer_profit_is_20_625_percent : 
  pen_profit_percentage 75 60 3.5 = 20.625 := by
  sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_retailer_profit_is_20_625_percent_l2318_231877


namespace NUMINAMATH_CALUDE_division_problem_l2318_231833

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 760 → 
  divisor = 36 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder → 
  quotient = 21 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2318_231833


namespace NUMINAMATH_CALUDE_total_crayons_l2318_231821

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 3

theorem total_crayons : 
  initial_crayons + added_crayons = 10 := by sorry

end NUMINAMATH_CALUDE_total_crayons_l2318_231821


namespace NUMINAMATH_CALUDE_apples_on_ground_l2318_231884

/-- The number of apples that have fallen to the ground -/
def fallen_apples : ℕ := sorry

/-- The number of apples hanging on the tree -/
def hanging_apples : ℕ := 5

/-- The number of apples eaten by the dog -/
def eaten_apples : ℕ := 3

/-- The number of apples left after the dog eats -/
def remaining_apples : ℕ := 10

theorem apples_on_ground :
  fallen_apples = 13 :=
by sorry

end NUMINAMATH_CALUDE_apples_on_ground_l2318_231884


namespace NUMINAMATH_CALUDE_determinant_scaling_l2318_231872

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 10 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 90 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l2318_231872


namespace NUMINAMATH_CALUDE_regular_decagon_diagonal_intersections_eq_choose_l2318_231829

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def regular_decagon_diagonal_intersections : ℕ := 210

/-- A regular decagon has 10 sides -/
def regular_decagon_sides : ℕ := 10

/-- Theorem: The number of distinct interior intersection points of diagonals 
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem regular_decagon_diagonal_intersections_eq_choose :
  regular_decagon_diagonal_intersections = Nat.choose regular_decagon_sides 4 := by
  sorry

#eval regular_decagon_diagonal_intersections
#eval Nat.choose regular_decagon_sides 4

end NUMINAMATH_CALUDE_regular_decagon_diagonal_intersections_eq_choose_l2318_231829


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2318_231818

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 3 * x + 5 ≥ -1 ∧ 3 - x > (1/2) * x}
  S = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2318_231818


namespace NUMINAMATH_CALUDE_cuboid_height_proof_l2318_231805

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 700 m², length 12 m, and width 14 m is 7 m -/
theorem cuboid_height_proof (surfaceArea length width : ℝ) 
  (hsa : surfaceArea = 700)
  (hl : length = 12)
  (hw : width = 14) :
  ∃ height : ℝ, cuboidSurfaceArea length width height = surfaceArea ∧ height = 7 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_proof_l2318_231805


namespace NUMINAMATH_CALUDE_tg_sum_formula_l2318_231873

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- Define the theorem
theorem tg_sum_formula (α β p q : ℝ) 
  (h1 : tg α + tg β = p) 
  (h2 : ctg α + ctg β = q) :
  (p = 0 ∧ q = 0 → tg (α + β) = 0) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q → tg (α + β) = p * q / (q - p)) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p = q → ¬∃x, x = tg (α + β)) ∧
  ((p = 0 ∨ q = 0) ∧ p ≠ q → False) :=
by sorry


end NUMINAMATH_CALUDE_tg_sum_formula_l2318_231873


namespace NUMINAMATH_CALUDE_triangle_inequality_l2318_231825

theorem triangle_inequality (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2318_231825


namespace NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l2318_231806

def C : ℕ := 123456
def D : ℕ := 166666
def M : ℕ := 48

theorem multiplicative_inverse_mod_million :
  (M * (C * D)) % 1000000 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l2318_231806


namespace NUMINAMATH_CALUDE_det_A_eq_32_l2318_231896

def A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]

theorem det_A_eq_32 : Matrix.det A = 32 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_32_l2318_231896


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l2318_231819

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l2318_231819


namespace NUMINAMATH_CALUDE_initial_children_on_bus_prove_initial_children_on_bus_l2318_231849

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    ∀ (added_children total_children : ℕ),
      added_children = 7 →
      total_children = 25 →
      initial_children + added_children = total_children →
      initial_children = 18

-- Proof
theorem prove_initial_children_on_bus :
  ∃ (initial_children : ℕ), initial_children_on_bus initial_children :=
by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_prove_initial_children_on_bus_l2318_231849


namespace NUMINAMATH_CALUDE_primitive_decomposition_existence_l2318_231899

/-- A decomposition of a square into rectangles. -/
structure SquareDecomposition :=
  (n : ℕ)  -- number of rectangles
  (is_finite : n > 0)
  (parallel_sides : Bool)
  (is_primitive : Bool)

/-- Predicate for a valid primitive square decomposition. -/
def valid_primitive_decomposition (d : SquareDecomposition) : Prop :=
  d.parallel_sides ∧ d.is_primitive

/-- Theorem stating for which n a primitive decomposition exists. -/
theorem primitive_decomposition_existence :
  ∀ n : ℕ, (∃ d : SquareDecomposition, d.n = n ∧ valid_primitive_decomposition d) ↔ (n = 5 ∨ n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_primitive_decomposition_existence_l2318_231899


namespace NUMINAMATH_CALUDE_tic_tac_toe_losses_l2318_231809

theorem tic_tac_toe_losses (total_games wins draws : ℕ) (h1 : total_games = 14) (h2 : wins = 2) (h3 : draws = 10) :
  total_games = wins + (total_games - wins - draws) + draws :=
by sorry

#check tic_tac_toe_losses

end NUMINAMATH_CALUDE_tic_tac_toe_losses_l2318_231809


namespace NUMINAMATH_CALUDE_quadrilateral_has_four_sides_and_angles_l2318_231871

/-- Definition of a quadrilateral -/
structure Quadrilateral where
  sides : Fin 4 → Seg
  angles : Fin 4 → Angle

/-- Theorem: A quadrilateral has four sides and four angles -/
theorem quadrilateral_has_four_sides_and_angles (q : Quadrilateral) :
  (∃ (s : Fin 4 → Seg), q.sides = s) ∧ (∃ (a : Fin 4 → Angle), q.angles = a) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_has_four_sides_and_angles_l2318_231871


namespace NUMINAMATH_CALUDE_exponential_inequality_l2318_231898

theorem exponential_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  2^x * x + 2^y * y ≥ 2^y * x + 2^x * y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2318_231898


namespace NUMINAMATH_CALUDE_equation_solution_l2318_231817

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ = -1) ∧ 
              (3 * x₂^2 - 6 * x₂ = -1) ∧ 
              (x₁ = 1 + Real.sqrt 6 / 3) ∧ 
              (x₂ = 1 - Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2318_231817


namespace NUMINAMATH_CALUDE_trigonometric_simplification_special_angle_simplification_l2318_231887

-- Part 1
theorem trigonometric_simplification (α : ℝ) :
  (Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) *
  Real.sin (α - π) * Real.cos (2 * π - α) = -Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem special_angle_simplification :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (200 * π / 180))) /
  (Real.cos (160 * π / 180) - Real.sqrt (1 - Real.cos (20 * π / 180) ^ 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_special_angle_simplification_l2318_231887


namespace NUMINAMATH_CALUDE_currency_denomination_problem_l2318_231885

theorem currency_denomination_problem (total_notes : ℕ) (total_amount : ℕ) (amount_50 : ℕ) (d : ℕ) :
  total_notes = 85 →
  total_amount = 5000 →
  amount_50 = 3500 →
  (amount_50 / 50 + (total_notes - amount_50 / 50)) = total_notes →
  50 * (amount_50 / 50) + d * (total_notes - amount_50 / 50) = total_amount →
  d = 100 := by
sorry

end NUMINAMATH_CALUDE_currency_denomination_problem_l2318_231885


namespace NUMINAMATH_CALUDE_problem_statement_l2318_231893

theorem problem_statement : 7^2 - 2 * 6 + (3^2 - 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2318_231893


namespace NUMINAMATH_CALUDE_max_distinct_count_is_five_l2318_231867

/-- A type representing a circular arrangement of nine natural numbers -/
def CircularArrangement := Fin 9 → ℕ

/-- Checks if a number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The condition that all adjacent triples in the circle form prime sums -/
def AllAdjacentTriplesPrime (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 9, IsPrime (arr i + arr (i + 1) ^ (arr (i + 2)))

/-- The number of distinct elements in the circular arrangement -/
def DistinctCount (arr : CircularArrangement) : ℕ :=
  Finset.card (Finset.image arr Finset.univ)

/-- The main theorem statement -/
theorem max_distinct_count_is_five (arr : CircularArrangement) 
  (h : AllAdjacentTriplesPrime arr) : 
  DistinctCount arr ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_count_is_five_l2318_231867


namespace NUMINAMATH_CALUDE_line_direction_vector_l2318_231870

/-- Given two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (-3, 2) →
  p2 = (4, -3) →
  ∃ (k : ℝ), k • (p2.1 - p1.1, p2.2 - p1.2) = (b, -2) →
  b = 14/5 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2318_231870


namespace NUMINAMATH_CALUDE_find_x_when_y_is_8_l2318_231848

-- Define the relationship between x and y
def varies_directly (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * Real.sqrt x

-- State the theorem
theorem find_x_when_y_is_8 :
  ∀ x₀ y₀ x y : ℝ,
  varies_directly x₀ y₀ →
  varies_directly x y →
  x₀ = 3 →
  y₀ = 2 →
  y = 8 →
  x = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_find_x_when_y_is_8_l2318_231848


namespace NUMINAMATH_CALUDE_coin_division_problem_l2318_231832

theorem coin_division_problem : 
  ∃ (n : ℕ), n > 0 ∧ 
  n % 8 = 6 ∧ 
  n % 7 = 5 ∧ 
  n % 9 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2318_231832


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l2318_231823

/-- The cost in cents for Plan X given m megabytes -/
def cost_x (m : ℕ) : ℕ := 5 * m

/-- The cost in cents for Plan Y given m megabytes -/
def cost_y (m : ℕ) : ℕ := 3000 + 3 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective -/
def min_megabytes : ℕ := 1501

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l2318_231823


namespace NUMINAMATH_CALUDE_two_numbers_sum_l2318_231838

theorem two_numbers_sum (x y : ℝ) 
  (sum_eq : x + y = 5)
  (diff_eq : x - y = 10)
  (square_diff_eq : x^2 - y^2 = 50) : 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_l2318_231838


namespace NUMINAMATH_CALUDE_lilys_remaining_balance_l2318_231876

/-- Calculates the remaining balance in Lily's account after purchases --/
def remaining_balance (initial_balance : ℕ) (shirt_cost : ℕ) : ℕ :=
  initial_balance - shirt_cost - (3 * shirt_cost)

/-- Theorem stating that Lily's remaining balance is 27 dollars --/
theorem lilys_remaining_balance :
  remaining_balance 55 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_lilys_remaining_balance_l2318_231876


namespace NUMINAMATH_CALUDE_complex_calculation_l2318_231858

theorem complex_calculation (A M N : ℂ) (Q : ℝ) :
  A = 5 - 2*I →
  M = -3 + 2*I →
  N = 3*I →
  Q = 3 →
  (A - M + N - Q) * I = 1 + 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2318_231858


namespace NUMINAMATH_CALUDE_expression_proof_l2318_231826

theorem expression_proof (a b E : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : E / (3 * a - 2 * b) = 3) : 
  E = 6 * b := by
sorry

end NUMINAMATH_CALUDE_expression_proof_l2318_231826


namespace NUMINAMATH_CALUDE_second_divisor_problem_l2318_231820

theorem second_divisor_problem (N : ℕ) (D : ℕ) : 
  N % 35 = 25 → N % D = 4 → D = 31 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l2318_231820


namespace NUMINAMATH_CALUDE_parabola_constant_l2318_231807

/-- A parabola with equation x = dy² + ey + f, vertex at (3, -1), and passing through (5, 1) has f = 7/2 -/
theorem parabola_constant (d e f : ℝ) : 
  (∀ y : ℝ, 3 = d * (-1)^2 + e * (-1) + f) →  -- vertex condition
  (5 = d * 1^2 + e * 1 + f) →                 -- point condition
  (∀ y : ℝ, 3 = d * (y + 1)^2 + 3) →          -- vertex form
  f = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_constant_l2318_231807


namespace NUMINAMATH_CALUDE_third_number_proof_l2318_231844

theorem third_number_proof (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 922) : 
  (∃ (k l m : ℕ), a = 64 * k + 22 ∧ b = 64 * l + 22 ∧ c = 64 * m + 22) ∧ 
  (∀ x : ℕ, b < x ∧ x < c → ¬(∃ n : ℕ, x = 64 * n + 22)) := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l2318_231844

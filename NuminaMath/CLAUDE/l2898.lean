import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2898_289820

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (y : ℝ) : 
  -- The ellipse equation
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1) →
  -- F₁ and F₂ are foci of the ellipse
  (∃ F₁ F₂ : ℝ × ℝ, F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0) →
  -- Point P is on the line x = -a
  (∃ P : ℝ × ℝ, P.1 = -a ∧ P.2 = y) →
  -- |PF₁| = |F₁F₂|
  ((a - c)^2 + y^2 = (2*c)^2) →
  -- ∠PF₁F₂ = 120°
  (y / (a - c) = Real.sqrt 3) →
  -- The eccentricity is 1/2
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2898_289820


namespace NUMINAMATH_CALUDE_function_minimum_implies_a_range_l2898_289866

theorem function_minimum_implies_a_range :
  ∀ (a : ℝ),
  (∀ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x ≥ -3) →
  (∃ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x = -3) →
  a ∈ Set.Icc (-3/2) 12 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_range_l2898_289866


namespace NUMINAMATH_CALUDE_problem1_l2898_289890

theorem problem1 (a b : ℝ) (ha : a = -Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a + b) * (a - b) + b * (a + 2 * b) - (a + b)^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l2898_289890


namespace NUMINAMATH_CALUDE_potato_rows_l2898_289867

theorem potato_rows (seeds_per_row : ℕ) (total_potatoes : ℕ) (h1 : seeds_per_row = 9) (h2 : total_potatoes = 54) :
  total_potatoes / seeds_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_potato_rows_l2898_289867


namespace NUMINAMATH_CALUDE_square_diagonal_l2898_289850

/-- The diagonal of a square with perimeter 800 cm is 200√2 cm. -/
theorem square_diagonal (perimeter : ℝ) (side : ℝ) (diagonal : ℝ) : 
  perimeter = 800 →
  side = perimeter / 4 →
  diagonal = side * Real.sqrt 2 →
  diagonal = 200 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_l2898_289850


namespace NUMINAMATH_CALUDE_extrema_of_squared_sum_l2898_289869

theorem extrema_of_squared_sum (a b c : ℝ) 
  (h : |a + b| + |b + c| + |c + a| = 8) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 16/3 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≥ 16/3) ∧
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 32 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≤ 32) :=
by sorry

end NUMINAMATH_CALUDE_extrema_of_squared_sum_l2898_289869


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2898_289833

/-- Given a line 3x + 5y + c = 0, if the sum of its x-intercept and y-intercept is 55/4, then c = 825/32 -/
theorem line_intercept_sum (c : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + c = 0 ∧ x + y = 55 / 4) → c = 825 / 32 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2898_289833


namespace NUMINAMATH_CALUDE_s_not_lowest_avg_l2898_289886

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T

-- Define a type for race results
def RaceResult := List Runner

-- Define the first race result
def firstRace : RaceResult := sorry

-- Define the second race result
def secondRace : RaceResult := [Runner.R, Runner.P, Runner.T, Runner.Q, Runner.S]

-- Function to calculate the position of a runner in a race
def position (runner : Runner) (race : RaceResult) : Nat := sorry

-- Function to calculate the average position of a runner across two races
def avgPosition (runner : Runner) (race1 race2 : RaceResult) : Rat :=
  (position runner race1 + position runner race2) / 2

-- Theorem stating that S cannot have the lowest average position
theorem s_not_lowest_avg :
  ∀ (r : Runner), r ≠ Runner.S →
    avgPosition Runner.S firstRace secondRace ≥ avgPosition r firstRace secondRace :=
  sorry

end NUMINAMATH_CALUDE_s_not_lowest_avg_l2898_289886


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2898_289879

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 30 and a_4 + a_5 = 120, 
    then a_7 + a_8 = 480. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 1 + a 2 = 30 →                          -- a_1 + a_2 = 30
  a 4 + a 5 = 120 →                         -- a_4 + a_5 = 120
  a 7 + a 8 = 480 :=                        -- a_7 + a_8 = 480
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2898_289879


namespace NUMINAMATH_CALUDE_problem_solution_l2898_289828

theorem problem_solution (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2898_289828


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2898_289831

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2898_289831


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2898_289823

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2898_289823


namespace NUMINAMATH_CALUDE_f_min_value_f_max_value_tangent_line_equation_l2898_289817

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the minimum value
theorem f_min_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x₀ ≤ f x ∧ f x₀ = -2 := by sorry

-- Theorem for the maximum value
theorem f_max_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ f x₀ ∧ f x₀ = 2 := by sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  let P : ℝ × ℝ := (2, -6)
  let tangent_line (x y : ℝ) : Prop := 24 * x - y - 54 = 0
  ∀ x y : ℝ, tangent_line x y ↔ (y - f P.1 = (3 * P.1^2 - 3) * (x - P.1)) := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_max_value_tangent_line_equation_l2898_289817


namespace NUMINAMATH_CALUDE_rational_square_roots_existence_l2898_289871

theorem rational_square_roots_existence : ∃ (x : ℚ), 
  3 < x ∧ x < 4 ∧ 
  ∃ (a b : ℚ), a^2 = x - 3 ∧ b^2 = x + 1 ∧
  x = 481 / 144 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_roots_existence_l2898_289871


namespace NUMINAMATH_CALUDE_f_of_f_zero_l2898_289808

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_of_f_zero : f (f 0) = 1 := by sorry

end NUMINAMATH_CALUDE_f_of_f_zero_l2898_289808


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2898_289861

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+,
    (1 : ℚ) / m + (1 : ℚ) / n - (1 : ℚ) / (m * n) = 2 / 5 ↔
    ((m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2898_289861


namespace NUMINAMATH_CALUDE_league_games_l2898_289884

theorem league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l2898_289884


namespace NUMINAMATH_CALUDE_odd_function_property_l2898_289825

-- Define the domain D
def D : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the properties of the function f
def is_odd_function_on_D (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, f (-x) = -f x

-- State the theorem
theorem odd_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function_on_D f)
  (h_pos : ∀ x > 0, f x = x^2 - x) :
  ∀ x < 0, f x = -x^2 - x :=
sorry

end NUMINAMATH_CALUDE_odd_function_property_l2898_289825


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2898_289832

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59525 / 30964 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2898_289832


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2898_289893

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only define the angles, as that's all we need for this problem
  vertex_angle : ℝ
  base_angle : ℝ
  -- The sum of angles in a triangle is 180°
  angle_sum : vertex_angle + 2 * base_angle = 180
  -- In an isosceles triangle, the base angles are equal

-- Define our specific isosceles triangle with one 40° angle
def triangle_with_40_degree_angle (t : IsoscelesTriangle) : Prop :=
  t.vertex_angle = 40 ∨ t.base_angle = 40

-- Theorem to prove
theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : triangle_with_40_degree_angle t) : 
  t.base_angle = 40 ∨ t.base_angle = 70 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2898_289893


namespace NUMINAMATH_CALUDE_table_free_sides_length_l2898_289874

theorem table_free_sides_length (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length = 2 * width → 
  length * width = 128 → 
  length + 2 * width = 32 := by
sorry

end NUMINAMATH_CALUDE_table_free_sides_length_l2898_289874


namespace NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2898_289865

/-- The length of the real axis of a hyperbola given by the equation 2x^2 - y^2 = 8 -/
def real_axis_length : ℝ := 4

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

theorem real_axis_length_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2898_289865


namespace NUMINAMATH_CALUDE_max_value_of_even_quadratic_function_l2898_289877

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem max_value_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), ∀ y ∈ Set.Icc (a - 1) (2 * a), f a b y ≤ f a b x) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), f a b x = 31 / 27) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_even_quadratic_function_l2898_289877


namespace NUMINAMATH_CALUDE_problem_solution_l2898_289896

theorem problem_solution (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2898_289896


namespace NUMINAMATH_CALUDE_complex_simplification_l2898_289873

theorem complex_simplification : 
  3 * (4 - 2 * Complex.I) + 2 * Complex.I * (3 + Complex.I) = (10 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2898_289873


namespace NUMINAMATH_CALUDE_fraction_of_fraction_l2898_289802

theorem fraction_of_fraction (a b c d e f : ℚ) :
  a = 2 → b = 9 → c = 5 → d = 6 → e = 3 → f = 4 →
  (a/b * c/d) / (e/f) = 20/81 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_l2898_289802


namespace NUMINAMATH_CALUDE_two_functions_satisfy_equation_l2898_289848

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

/-- The main theorem stating that there are exactly two functions satisfying the equation -/
theorem two_functions_satisfy_equation :
  ∃! (s : Set (ℝ → ℝ)), 
    (∀ f ∈ s, SatisfiesFunctionalEquation f) ∧ 
    s = {ZeroFunction, SquareFunction} :=
  sorry

end NUMINAMATH_CALUDE_two_functions_satisfy_equation_l2898_289848


namespace NUMINAMATH_CALUDE_article_cost_price_l2898_289807

theorem article_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.75 * 1250)
  (h2 : selling_price_increased = selling_price + 500)
  (h3 : selling_price_increased = 1.15 * 1250) : 1250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2898_289807


namespace NUMINAMATH_CALUDE_tan_alpha_negative_three_l2898_289809

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = 3 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 2 = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_three_l2898_289809


namespace NUMINAMATH_CALUDE_plot_length_is_65_l2898_289837

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 30 meters more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 30

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMeter * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The main theorem stating that under the given conditions, the length of the plot is 65 meters. -/
theorem plot_length_is_65 (plot : RectangularPlot) 
    (h1 : lengthCondition plot) 
    (h2 : fencingCostCondition plot) 
    (h3 : plot.fencingCostPerMeter = 26.5) 
    (h4 : plot.totalFencingCost = 5300) : 
  plot.length = 65 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_65_l2898_289837


namespace NUMINAMATH_CALUDE_pond_ducks_l2898_289843

/-- The number of ducks in the pond -/
def num_ducks : ℕ := 3

/-- The total number of bread pieces thrown in the pond -/
def total_bread : ℕ := 100

/-- The number of bread pieces left in the water -/
def left_bread : ℕ := 30

/-- The number of bread pieces eaten by the second duck -/
def second_duck_bread : ℕ := 13

/-- The number of bread pieces eaten by the third duck -/
def third_duck_bread : ℕ := 7

/-- Theorem stating that the number of ducks in the pond is 3 -/
theorem pond_ducks : 
  (total_bread / 2 + second_duck_bread + third_duck_bread = total_bread - left_bread) → 
  num_ducks = 3 := by
  sorry


end NUMINAMATH_CALUDE_pond_ducks_l2898_289843


namespace NUMINAMATH_CALUDE_indeterminate_roots_l2898_289844

theorem indeterminate_roots (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ 
    ∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x) :
  ¬∃ (root_nature : Prop), 
    (∀ x : ℝ, (a + 1) * x^2 + (b + 2) * x + (c + 1) = 0 ↔ root_nature) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_l2898_289844


namespace NUMINAMATH_CALUDE_M_equals_N_l2898_289841

def M : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 5 * Real.pi / 6 + 2 * k * Real.pi}

def N : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = -7 * Real.pi / 6 + 2 * k * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2898_289841


namespace NUMINAMATH_CALUDE_orthocenter_position_in_isosceles_triangle_l2898_289805

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Vector2D :=
  sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem orthocenter_position_in_isosceles_triangle 
  (t : Triangle) 
  (h_isosceles : isIsosceles t) 
  (h_sides : t.a = 5 ∧ t.b = 5 ∧ t.c = 6) :
  ∃ (m n : ℝ), 
    let H := orthocenter t
    let A := Vector2D.mk 0 0
    let B := Vector2D.mk t.c 0
    let C := Vector2D.mk (t.c / 2) (Real.sqrt (t.a^2 - (t.c / 2)^2))
    H.x = m * B.x + n * C.x ∧
    H.y = m * B.y + n * C.y ∧
    m + n = 21 / 32 :=
  sorry

end NUMINAMATH_CALUDE_orthocenter_position_in_isosceles_triangle_l2898_289805


namespace NUMINAMATH_CALUDE_speed_equivalence_l2898_289875

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.001199999999999

/-- The calculated speed in km/h -/
def calculated_speed_kmph : ℝ := 54.004319999999996

/-- Theorem stating that the calculated speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : calculated_speed_kmph = given_speed_mps * mps_to_kmph := by
  sorry

#check speed_equivalence

end NUMINAMATH_CALUDE_speed_equivalence_l2898_289875


namespace NUMINAMATH_CALUDE_palindrome_square_base_l2898_289892

theorem palindrome_square_base (r : ℕ) (x : ℕ) (n : ℕ) : 
  r > 3 →
  (∃ (p : ℕ), x = p * r^3 + p * r^2 + 2*p * r + 2*p) →
  (∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + c * r^3 + c * r^2 + b * r + a) →
  r = 3 * n^2 ∧ n > 1 :=
by sorry

end NUMINAMATH_CALUDE_palindrome_square_base_l2898_289892


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l2898_289872

/-- Given a point P on the circle x^2 + y^2 = 16, and M being the midpoint of the perpendicular
    line segment from P to the x-axis, the trajectory of M satisfies the equation x^2/4 + y^2/16 = 1. -/
theorem trajectory_of_midpoint (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 16 →  -- P is on the circle
  x₀ = 2*x →  -- M is the midpoint of PD (x-coordinate)
  y₀ = y →  -- M is the midpoint of PD (y-coordinate)
  x^2/4 + y^2/16 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l2898_289872


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2898_289821

theorem quadratic_inequality_solution_range (m : ℝ) : 
  m > 0 ∧ 
  (∃ a b : ℤ, a ≠ b ∧ 
    (∀ x : ℝ, (2*x^2 - 2*m*x + m < 0) ↔ (a < x ∧ x < b)) ∧
    (∀ c : ℤ, (2*c^2 - 2*m*c + m < 0) → (c = a ∨ c = b)))
  → 
  8/3 < m ∧ m ≤ 18/5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2898_289821


namespace NUMINAMATH_CALUDE_hiking_team_participants_l2898_289816

theorem hiking_team_participants (total_gloves : ℕ) (gloves_per_participant : ℕ) : 
  total_gloves = 164 → gloves_per_participant = 2 → total_gloves / gloves_per_participant = 82 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l2898_289816


namespace NUMINAMATH_CALUDE_part_one_part_two_l2898_289818

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |k * x - 1|

-- Part I
theorem part_one (k : ℝ) :
  (∀ x, f k x ≤ 3 ↔ x ∈ Set.Icc (-2) 1) → k = -2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 1 (x + 2) - f 1 (2 * x + 1) ≤ 3 - 2 * m) → m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2898_289818


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2898_289891

theorem solve_quadratic_equation (x : ℝ) : 3 * (x + 1)^2 = 27 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2898_289891


namespace NUMINAMATH_CALUDE_tomato_drying_l2898_289882

/-- Given an initial mass of tomatoes with a certain water content,
    calculate the final mass after water content reduction -/
theorem tomato_drying (initial_mass : ℝ) (initial_water_content : ℝ) (water_reduction : ℝ)
  (h1 : initial_mass = 1000)
  (h2 : initial_water_content = 0.99)
  (h3 : water_reduction = 0.04)
  : ∃ (final_mass : ℝ), final_mass = 200 := by
  sorry


end NUMINAMATH_CALUDE_tomato_drying_l2898_289882


namespace NUMINAMATH_CALUDE_initial_water_amount_l2898_289889

/-- Given a container with alcohol and water, prove the initial amount of water -/
theorem initial_water_amount (initial_alcohol : ℝ) (added_water : ℝ) (ratio_alcohol : ℝ) (ratio_water : ℝ) :
  initial_alcohol = 4 →
  added_water = 2.666666666666667 →
  ratio_alcohol = 3 →
  ratio_water = 5 →
  ratio_alcohol / ratio_water = initial_alcohol / (initial_alcohol + added_water + x) →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_amount_l2898_289889


namespace NUMINAMATH_CALUDE_expression_evaluation_l2898_289895

theorem expression_evaluation :
  let x : ℚ := -1/4
  let y : ℚ := -1/2
  4*x*y - ((x^2 + 5*x*y - y^2) - (x^2 + 3*x*y - 2*y^2)) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2898_289895


namespace NUMINAMATH_CALUDE_fraction_division_equality_l2898_289810

theorem fraction_division_equality : (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l2898_289810


namespace NUMINAMATH_CALUDE_noras_oranges_l2898_289838

/-- The total number of oranges Nora picked from three trees -/
def total_oranges (tree1 tree2 tree3 : ℕ) : ℕ := tree1 + tree2 + tree3

/-- Theorem stating that the total number of oranges Nora picked is 260 -/
theorem noras_oranges : total_oranges 80 60 120 = 260 := by
  sorry

end NUMINAMATH_CALUDE_noras_oranges_l2898_289838


namespace NUMINAMATH_CALUDE_apps_deleted_l2898_289814

/-- Proves that Dave deleted 8 apps given the initial conditions -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : 
  initial_apps = 16 →
  remaining_apps = initial_apps / 2 →
  initial_apps - remaining_apps = 8 := by
sorry

end NUMINAMATH_CALUDE_apps_deleted_l2898_289814


namespace NUMINAMATH_CALUDE_triangle_minimum_area_l2898_289819

theorem triangle_minimum_area :
  ∀ (S : ℝ), 
  (∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b > c ∧ b + c > a ∧ c + a > b →
    (∃ (h : ℝ), h ≤ 1 ∧ S = 1/2 * (a * h)) →
    (∀ (w : ℝ), w < 1 → 
      ¬(∃ (h : ℝ), h ≤ w ∧ S = 1/2 * (a * h)))) →
  S ≥ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_minimum_area_l2898_289819


namespace NUMINAMATH_CALUDE_rearranged_rectangles_perimeter_l2898_289862

/-- The perimeter of a figure formed by rearranging two equal rectangles cut from a square --/
theorem rearranged_rectangles_perimeter (square_side : ℝ) : square_side = 100 → 
  let rectangle_width := square_side / 2
  let rectangle_length := square_side
  let perimeter := 3 * rectangle_length + 4 * rectangle_width
  perimeter = 500 := by
sorry


end NUMINAMATH_CALUDE_rearranged_rectangles_perimeter_l2898_289862


namespace NUMINAMATH_CALUDE_set_operations_l2898_289830

def A : Set ℕ := {6, 8, 10, 12}
def B : Set ℕ := {1, 6, 8}

theorem set_operations :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (𝒫(A ∩ B) = {∅, {6}, {8}, {6, 8}}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2898_289830


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2898_289839

/-- The radius of an inscribed circle in a sector that is one-third of a larger circle -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 6) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let inscribed_radius : ℝ := R * (Real.sqrt 2 - 1)
  inscribed_radius = R * (Real.sqrt 2 - 1) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2898_289839


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l2898_289829

def n : ℕ := 20  -- Total number of knights
def k : ℕ := 4   -- Number of knights chosen

-- Probability that at least two of the four chosen knights were sitting next to each other
def adjacent_probability : ℚ :=
  1 - (Nat.choose (n - k) (k - 1) : ℚ) / (Nat.choose n k : ℚ)

theorem adjacent_knights_probability :
  adjacent_probability = 66 / 75 :=
sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l2898_289829


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2898_289898

theorem min_reciprocal_sum (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hsum : x + y + z = 2) (hx : x = 2 * y) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b →
  1 / x + 1 / y + 1 / z ≤ 1 / a + 1 / b + 1 / c ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b ∧
  1 / x + 1 / y + 1 / z = 1 / a + 1 / b + 1 / c ∧
  1 / x + 1 / y + 1 / z = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2898_289898


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2898_289849

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 2 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem stating that the given center and radius satisfy the circle equation
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l2898_289849


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2898_289826

theorem sum_of_coefficients (x : ℝ) : 
  (fun x => (x - 2)^6 - (x - 1)^7 + (3*x - 2)^8) 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2898_289826


namespace NUMINAMATH_CALUDE_fathers_age_l2898_289860

theorem fathers_age (f d : ℕ) (h1 : f / d = 4) (h2 : f + d + 10 = 50) : f = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l2898_289860


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2898_289878

theorem sin_2theta_value (θ : ℝ) (h : Real.cos (π/4 - θ) = 1/2) : 
  Real.sin (2*θ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2898_289878


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2898_289812

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (a₂ - a₁) = (a₃ - a₂) ∧ (a₃ - a₂) = (a₄ - a₃) ∧ (a₄ - a₃) = (a₅ - a₄)

theorem middle_term_of_arithmetic_sequence (x z : ℝ) :
  arithmetic_sequence 23 x 38 z 53 → 38 = (23 + 53) / 2 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2898_289812


namespace NUMINAMATH_CALUDE_base_c_is_seven_l2898_289864

theorem base_c_is_seven (c : ℕ) (h : c > 1) : 
  (3 * c + 2)^2 = c^3 + 2 * c^2 + 6 * c + 4 → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_c_is_seven_l2898_289864


namespace NUMINAMATH_CALUDE_lcm_e_n_l2898_289888

theorem lcm_e_n (e n : ℕ) (h1 : e > 0) (h2 : n ≥ 100 ∧ n ≤ 999) 
  (h3 : ¬(3 ∣ n)) (h4 : ¬(2 ∣ e)) (h5 : n = 230) : 
  Nat.lcm e n = 230 := by
  sorry

end NUMINAMATH_CALUDE_lcm_e_n_l2898_289888


namespace NUMINAMATH_CALUDE_intersection_difference_l2898_289834

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 2 * x + 6

theorem intersection_difference :
  ∃ (a c : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x → x = a ∨ x = c) ∧
    c ≥ a ∧
    c - a = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_difference_l2898_289834


namespace NUMINAMATH_CALUDE_five_chicks_per_hen_l2898_289857

/-- Represents the poultry farm scenario --/
structure PoultryFarm where
  num_hens : ℕ
  hen_to_rooster_ratio : ℕ
  total_chickens : ℕ

/-- Calculates the number of chicks per hen --/
def chicks_per_hen (farm : PoultryFarm) : ℕ :=
  let num_roosters := farm.num_hens / farm.hen_to_rooster_ratio
  let num_adult_chickens := farm.num_hens + num_roosters
  let num_chicks := farm.total_chickens - num_adult_chickens
  num_chicks / farm.num_hens

/-- Theorem stating that for the given farm conditions, each hen has 5 chicks --/
theorem five_chicks_per_hen (farm : PoultryFarm) 
    (h1 : farm.num_hens = 12)
    (h2 : farm.hen_to_rooster_ratio = 3)
    (h3 : farm.total_chickens = 76) : 
  chicks_per_hen farm = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_chicks_per_hen_l2898_289857


namespace NUMINAMATH_CALUDE_quadratic_single_intersection_l2898_289800

/-- 
A quadratic function f(x) = ax^2 - ax + 3x + 1 intersects the x-axis at only one point 
if and only if a = 1 or a = 9.
-/
theorem quadratic_single_intersection (a : ℝ) : 
  (∃! x, a * x^2 - a * x + 3 * x + 1 = 0) ↔ (a = 1 ∨ a = 9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_intersection_l2898_289800


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2898_289847

theorem max_value_of_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 2 ∧
    (a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c')) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2898_289847


namespace NUMINAMATH_CALUDE_problem_G10_1_l2898_289840

theorem problem_G10_1 (a : ℝ) : 
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 3 * Real.sqrt a + 6 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_G10_1_l2898_289840


namespace NUMINAMATH_CALUDE_mean_of_playground_counts_l2898_289835

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_playground_counts_l2898_289835


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l2898_289887

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid having a triangular front view --/
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True  -- Assuming it can be laid on its side
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  hasTriangularFrontView s →
  s = Solid.TriangularPyramid ∨
  s = Solid.SquarePyramid ∨
  s = Solid.TriangularPrism ∨
  s = Solid.Cone :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l2898_289887


namespace NUMINAMATH_CALUDE_cone_surface_area_l2898_289836

/-- Given a cone with slant height 4 and cross-sectional area π, 
    its total surface area is 12π. -/
theorem cone_surface_area (s : ℝ) (a : ℝ) (h1 : s = 4) (h2 : a = π) :
  let r := Real.sqrt (a / π)
  let lateral_area := π * r * s
  let base_area := a
  lateral_area + base_area = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2898_289836


namespace NUMINAMATH_CALUDE_largest_fraction_sum_l2898_289803

theorem largest_fraction_sum : 
  let a := (3 : ℚ) / 10 + (2 : ℚ) / 20
  let b := (1 : ℚ) / 6 + (1 : ℚ) / 8
  let c := (1 : ℚ) / 5 + (2 : ℚ) / 15
  let d := (1 : ℚ) / 7 + (4 : ℚ) / 21
  let e := (2 : ℚ) / 9 + (3 : ℚ) / 18
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_l2898_289803


namespace NUMINAMATH_CALUDE_cube_decomposition_91_l2898_289822

/-- Decomposition of a cube into consecutive odd numbers -/
def cube_decomposition (n : ℕ+) : List ℕ :=
  sorry

/-- The smallest number in the decomposition of m³ -/
def smallest_in_decomposition (m : ℕ+) : ℕ :=
  sorry

/-- Theorem: If the smallest number in the decomposition of m³ is 91, then m = 10 -/
theorem cube_decomposition_91 (m : ℕ+) :
  smallest_in_decomposition m = 91 → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_91_l2898_289822


namespace NUMINAMATH_CALUDE_price_reduction_achieves_desired_profit_l2898_289846

/-- Represents the profit and sales scenario for black pork zongzi --/
structure ZongziSales where
  initialProfit : ℝ  -- Initial profit per box
  initialQuantity : ℝ  -- Initial quantity sold
  priceElasticity : ℝ  -- Additional boxes sold per dollar of price reduction
  priceReduction : ℝ  -- Amount of price reduction per box
  desiredTotalProfit : ℝ  -- Desired total profit

/-- Calculates the new profit per box after price reduction --/
def newProfitPerBox (s : ZongziSales) : ℝ :=
  s.initialProfit - s.priceReduction

/-- Calculates the new quantity sold after price reduction --/
def newQuantitySold (s : ZongziSales) : ℝ :=
  s.initialQuantity + s.priceElasticity * s.priceReduction

/-- Calculates the total profit after price reduction --/
def totalProfit (s : ZongziSales) : ℝ :=
  newProfitPerBox s * newQuantitySold s

/-- Theorem stating that a price reduction of 15 achieves the desired total profit --/
theorem price_reduction_achieves_desired_profit (s : ZongziSales)
  (h1 : s.initialProfit = 50)
  (h2 : s.initialQuantity = 50)
  (h3 : s.priceElasticity = 2)
  (h4 : s.priceReduction = 15)
  (h5 : s.desiredTotalProfit = 2800) :
  totalProfit s = s.desiredTotalProfit := by
  sorry

#eval totalProfit { initialProfit := 50, initialQuantity := 50, priceElasticity := 2, priceReduction := 15, desiredTotalProfit := 2800 }

end NUMINAMATH_CALUDE_price_reduction_achieves_desired_profit_l2898_289846


namespace NUMINAMATH_CALUDE_sin_fourth_powers_sum_l2898_289827

theorem sin_fourth_powers_sum : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_fourth_powers_sum_l2898_289827


namespace NUMINAMATH_CALUDE_trees_planted_by_two_classes_l2898_289880

theorem trees_planted_by_two_classes 
  (trees_A : ℕ) 
  (trees_B : ℕ) 
  (h1 : trees_A = 8) 
  (h2 : trees_B = 7) : 
  trees_A + trees_B = 15 := by
sorry

end NUMINAMATH_CALUDE_trees_planted_by_two_classes_l2898_289880


namespace NUMINAMATH_CALUDE_parabola_equation_l2898_289858

/-- A parabola with vertex at the origin, axis of symmetry along a coordinate axis, 
    and passing through the point (√3, -2√3) has the equation y² = 4√3x or x² = -√3/2y -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ((y^2 = 2*p*x ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3) ∨ 
                       (x^2 = -2*p*y ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3))) → 
  (y^2 = 4*Real.sqrt 3*x ∨ x^2 = -(Real.sqrt 3/2)*y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2898_289858


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_one_l2898_289813

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem M_intersect_N_eq_one : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_one_l2898_289813


namespace NUMINAMATH_CALUDE_multiply_121_54_l2898_289815

theorem multiply_121_54 : 121 * 54 = 6534 := by sorry

end NUMINAMATH_CALUDE_multiply_121_54_l2898_289815


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2898_289868

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -8 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2898_289868


namespace NUMINAMATH_CALUDE_jacket_pricing_l2898_289852

theorem jacket_pricing (x : ℝ) : 
  let marked_price : ℝ := 300
  let discount_rate : ℝ := 0.7
  let profit : ℝ := 20
  (marked_price * discount_rate - x = profit) ↔ 
  (300 * 0.7 - x = 20) :=
by sorry

end NUMINAMATH_CALUDE_jacket_pricing_l2898_289852


namespace NUMINAMATH_CALUDE_max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l2898_289885

/-- Represents a surveillance network of stations -/
structure SurveillanceNetwork where
  stations : Finset ℕ
  connections : Finset (ℕ × ℕ)

/-- Checks if a station can communicate with all others directly or through one intermediary -/
def canCommunicateWithAll (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  ∀ t ∈ net.stations, s ≠ t →
    (s, t) ∈ net.connections ∨ ∃ u ∈ net.stations, (s, u) ∈ net.connections ∧ (u, t) ∈ net.connections

/-- Checks if a station has at most three direct connections -/
def hasAtMostThreeConnections (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  (net.connections.filter (λ p => p.1 = s ∨ p.2 = s)).card ≤ 3

/-- A valid surveillance network satisfies all conditions -/
def isValidNetwork (net : SurveillanceNetwork) : Prop :=
  ∀ s ∈ net.stations, canCommunicateWithAll net s ∧ hasAtMostThreeConnections net s

/-- The maximum number of stations in a valid surveillance network is 10 -/
theorem max_stations_is_ten :
  ∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10 :=
sorry

/-- There exists a valid surveillance network with 10 stations -/
theorem exists_valid_network_with_ten_stations :
  ∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10 :=
sorry

/-- The maximum number of stations in a valid surveillance network is exactly 10 -/
theorem max_stations_is_exactly_ten :
  (∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10) ∧
  (∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l2898_289885


namespace NUMINAMATH_CALUDE_jane_reading_pages_l2898_289851

/-- Calculates the number of pages Jane reads in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

theorem jane_reading_pages : pages_read_in_week 5 10 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_jane_reading_pages_l2898_289851


namespace NUMINAMATH_CALUDE_no_solutions_for_sqrt_equation_l2898_289806

theorem no_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 9 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 16 - 8 * Real.sqrt (x - 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_sqrt_equation_l2898_289806


namespace NUMINAMATH_CALUDE_hyperbola_center_l2898_289894

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 c : ℝ × ℝ) : 
  f1 = (3, -2) → f2 = (-1, 6) → c = (1, 2) → 
  c.1 = (f1.1 + f2.1) / 2 ∧ c.2 = (f1.2 + f2.2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2898_289894


namespace NUMINAMATH_CALUDE_exists_three_integers_with_cube_product_l2898_289856

/-- A set of 9 distinct integers with prime factors at most 3 -/
def SetWithPrimeFactorsUpTo3 : Type :=
  { S : Finset ℕ // S.card = 9 ∧ ∀ n ∈ S, ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 3 }

/-- The theorem stating that there exist three distinct integers in S whose product is a perfect cube -/
theorem exists_three_integers_with_cube_product (S : SetWithPrimeFactorsUpTo3) :
  ∃ a b c : ℕ, a ∈ S.val ∧ b ∈ S.val ∧ c ∈ S.val ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ∃ k : ℕ, a * b * c = k^3 :=
sorry

end NUMINAMATH_CALUDE_exists_three_integers_with_cube_product_l2898_289856


namespace NUMINAMATH_CALUDE_count_integers_with_7_or_8_eq_386_l2898_289881

/-- The number of digits in base 9 that do not include 7 or 8 -/
def base7_digits : ℕ := 7

/-- The number of digits we consider in base 9 -/
def num_digits : ℕ := 3

/-- The total number of integers we consider -/
def total_integers : ℕ := 729

/-- The function that calculates the number of integers in base 9 
    from 1 to 729 that contain at least one digit 7 or 8 -/
def count_integers_with_7_or_8 : ℕ := total_integers - base7_digits ^ num_digits

theorem count_integers_with_7_or_8_eq_386 : 
  count_integers_with_7_or_8 = 386 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_7_or_8_eq_386_l2898_289881


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2898_289859

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem (right triangle condition)
  (h2 : a = 3)            -- One non-hypotenuse side length
  (h3 : c = 5)            -- Hypotenuse length
  : b = 4 := by           -- Conclusion: other non-hypotenuse side length
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2898_289859


namespace NUMINAMATH_CALUDE_puppy_feeding_last_two_weeks_l2898_289883

/-- Represents the feeding schedule and amount for a puppy over 4 weeks -/
structure PuppyFeeding where
  total_food : ℚ
  first_day_food : ℚ
  first_two_weeks_daily_feeding : ℚ
  first_two_weeks_feeding_frequency : ℕ
  last_two_weeks_feeding_frequency : ℕ
  days_in_week : ℕ
  total_weeks : ℕ

/-- Calculates the amount of food fed to the puppy twice a day for the last two weeks -/
def calculate_last_two_weeks_feeding (pf : PuppyFeeding) : ℚ :=
  let first_two_weeks_food := pf.first_two_weeks_daily_feeding * pf.first_two_weeks_feeding_frequency * (2 * pf.days_in_week)
  let total_food_minus_first_day := pf.total_food - pf.first_day_food
  let last_two_weeks_food := total_food_minus_first_day - first_two_weeks_food
  let last_two_weeks_feedings := 2 * pf.last_two_weeks_feeding_frequency * pf.days_in_week
  last_two_weeks_food / last_two_weeks_feedings

/-- Theorem stating that the amount of food fed to the puppy twice a day for the last two weeks is 1/2 cup -/
theorem puppy_feeding_last_two_weeks
  (pf : PuppyFeeding)
  (h1 : pf.total_food = 25)
  (h2 : pf.first_day_food = 1/2)
  (h3 : pf.first_two_weeks_daily_feeding = 1/4)
  (h4 : pf.first_two_weeks_feeding_frequency = 3)
  (h5 : pf.last_two_weeks_feeding_frequency = 2)
  (h6 : pf.days_in_week = 7)
  (h7 : pf.total_weeks = 4) :
  calculate_last_two_weeks_feeding pf = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_feeding_last_two_weeks_l2898_289883


namespace NUMINAMATH_CALUDE_cos_sin_2theta_l2898_289801

theorem cos_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) :
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_2theta_l2898_289801


namespace NUMINAMATH_CALUDE_blacksmith_iron_calculation_l2898_289854

/-- The amount of iron needed for one horseshoe in kilograms -/
def iron_per_horseshoe : ℕ := 2

/-- The number of horseshoes needed for one horse -/
def horseshoes_per_horse : ℕ := 4

/-- The number of farms -/
def num_farms : ℕ := 2

/-- The number of horses in each farm -/
def horses_per_farm : ℕ := 2

/-- The number of stables -/
def num_stables : ℕ := 2

/-- The number of horses in each stable -/
def horses_per_stable : ℕ := 5

/-- The number of horses at the riding school -/
def riding_school_horses : ℕ := 36

/-- The total amount of iron the blacksmith had initially in kilograms -/
def initial_iron : ℕ := 400

theorem blacksmith_iron_calculation : 
  initial_iron = 
    (num_farms * horses_per_farm + num_stables * horses_per_stable + riding_school_horses) * 
    horseshoes_per_horse * iron_per_horseshoe :=
by sorry

end NUMINAMATH_CALUDE_blacksmith_iron_calculation_l2898_289854


namespace NUMINAMATH_CALUDE_alice_prob_after_three_turns_l2898_289863

/-- Represents the possessor of the ball -/
inductive Possessor : Type
| Alice : Possessor
| Bob : Possessor

/-- The game state after a turn -/
structure GameState :=
  (possessor : Possessor)

/-- The probability of Alice having the ball after one turn, given the current possessor -/
def aliceProbAfterOneTurn (current : Possessor) : ℚ :=
  match current with
  | Possessor.Alice => 2/3
  | Possessor.Bob => 3/5

/-- The probability of Alice having the ball after three turns, given Alice starts -/
def aliceProbAfterThreeTurns : ℚ := 7/45

theorem alice_prob_after_three_turns :
  aliceProbAfterThreeTurns = 7/45 :=
sorry

end NUMINAMATH_CALUDE_alice_prob_after_three_turns_l2898_289863


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2898_289804

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2898_289804


namespace NUMINAMATH_CALUDE_third_to_second_ratio_l2898_289899

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Verifies if the given hourly questions satisfy the problem conditions -/
def satisfiesConditions (q : HourlyQuestions) : Prop :=
  q.third = 132 ∧
  q.third = 3 * q.first ∧
  q.first + q.second + q.third = 242

/-- Theorem stating that if the conditions are satisfied, the ratio of third to second hour questions is 2:1 -/
theorem third_to_second_ratio (q : HourlyQuestions) 
  (h : satisfiesConditions q) : q.third = 2 * q.second :=
by
  sorry

#check third_to_second_ratio

end NUMINAMATH_CALUDE_third_to_second_ratio_l2898_289899


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l2898_289853

/-- A parabola is tangent to a line if and only if their intersection equation has exactly one solution -/
axiom tangent_condition (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) ↔ (∃ x, a * x^2 + 4 = 2 * x + 1 ∧ ∀ y, a * y^2 + 4 = 2 * y + 1 → y = x)

/-- The main theorem: if a parabola y = ax^2 + 4 is tangent to the line y = 2x + 1, then a = 1/3 -/
theorem parabola_tangent_line (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) → a = 1/3 := by
sorry


end NUMINAMATH_CALUDE_parabola_tangent_line_l2898_289853


namespace NUMINAMATH_CALUDE_limit_of_sequence_l2898_289842

/-- The limit of the sequence (√(n+1) - ∛(n³+1)) / (⁴√(n+1) - ⁵√(n⁵+1)) as n approaches infinity is 1 -/
theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → 
    |((n + 1: ℝ)^(1/2) - (n^3 + 1 : ℝ)^(1/3)) / ((n + 1 : ℝ)^(1/4) - (n^5 + 1 : ℝ)^(1/5)) - 1| < ε :=
by
  sorry

#check limit_of_sequence

end NUMINAMATH_CALUDE_limit_of_sequence_l2898_289842


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2898_289811

/-- A line in 3D space defined by the equation (x-2)/2 = (y-2)/(-1) = (z-4)/3 -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 2*t, 2 - t, 4 + 3*t)

/-- A plane in 3D space defined by the equation x + 3y + 5z - 42 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x + 3*y + 5*z - 42 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (4, 1, 7)

theorem intersection_point_is_unique :
  ∃! t : ℝ, line t = intersection_point ∧ plane (line t) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2898_289811


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l2898_289897

/-- Given a function g(x) = x^2 - x, prove that g(x+3) = x^2 + 5x + 6 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g := fun (x : ℝ) => x^2 - x
  g (x + 3) = x^2 + 5*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l2898_289897


namespace NUMINAMATH_CALUDE_max_b_value_l2898_289845

/-- The function f(x) = ax^3 + bx^2 - a^2x -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - a^2 * x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - a^2

theorem max_b_value (a b : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx : x₁ ≠ x₂)
  (hextreme : f' a b x₁ = 0 ∧ f' a b x₂ = 0)
  (hsum : abs x₁ + abs x₂ = 2 * Real.sqrt 2) :
  b ≤ 4 * Real.sqrt 6 ∧ ∃ b₀, b₀ = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l2898_289845


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l2898_289855

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage
  (cp_radio : ℚ) (cp_speaker : ℚ) (cp_headphones : ℚ)
  (sp_radio : ℚ) (sp_speaker : ℚ) (sp_headphones : ℚ)
  (h1 : cp_radio = 1500)
  (h2 : cp_speaker = 2500)
  (h3 : cp_headphones = 800)
  (h4 : sp_radio = 1275)
  (h5 : sp_speaker = 2300)
  (h6 : sp_headphones = 700) :
  let total_cp := cp_radio + cp_speaker + cp_headphones
  let total_sp := sp_radio + sp_speaker + sp_headphones
  let loss := total_cp - total_sp
  let loss_percentage := (loss / total_cp) * 100
  abs (loss_percentage - 10.94) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l2898_289855


namespace NUMINAMATH_CALUDE_total_bricks_used_l2898_289824

/-- The number of brick walls -/
def total_walls : ℕ := 10

/-- The number of walls of the first type -/
def first_type_walls : ℕ := 5

/-- The number of walls of the second type -/
def second_type_walls : ℕ := 5

/-- The number of bricks in a single row for the first type of wall -/
def first_type_bricks_per_row : ℕ := 60

/-- The number of rows in the first type of wall -/
def first_type_rows : ℕ := 100

/-- The number of bricks in a single row for the second type of wall -/
def second_type_bricks_per_row : ℕ := 80

/-- The number of rows in the second type of wall -/
def second_type_rows : ℕ := 120

/-- Theorem: The total number of bricks used for all ten walls is 78000 -/
theorem total_bricks_used : 
  first_type_walls * first_type_bricks_per_row * first_type_rows +
  second_type_walls * second_type_bricks_per_row * second_type_rows = 78000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_bricks_used_l2898_289824


namespace NUMINAMATH_CALUDE_lowest_temp_is_harbin_l2898_289876

def harbin_temp : ℤ := -20
def beijing_temp : ℤ := -10
def hangzhou_temp : ℤ := 0
def jinhua_temp : ℤ := 2

def city_temps : List ℤ := [harbin_temp, beijing_temp, hangzhou_temp, jinhua_temp]

theorem lowest_temp_is_harbin :
  List.minimum city_temps = some harbin_temp := by
  sorry

end NUMINAMATH_CALUDE_lowest_temp_is_harbin_l2898_289876


namespace NUMINAMATH_CALUDE_sum_of_87th_and_95th_odd_integers_l2898_289870

theorem sum_of_87th_and_95th_odd_integers : 
  (2 * 87 - 1) + (2 * 95 - 1) = 362 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_87th_and_95th_odd_integers_l2898_289870

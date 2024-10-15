import Mathlib

namespace NUMINAMATH_CALUDE_xanths_are_yelps_and_wicks_l3060_306006

-- Define the sets
variable (U : Type) -- Universe set
variable (Zorb Yelp Xanth Wick : Set U)

-- State the given conditions
variable (h1 : Zorb ⊆ Yelp)
variable (h2 : Xanth ⊆ Zorb)
variable (h3 : Xanth ⊆ Wick)

-- State the theorem to be proved
theorem xanths_are_yelps_and_wicks : Xanth ⊆ Yelp ∩ Wick := by
  sorry

end NUMINAMATH_CALUDE_xanths_are_yelps_and_wicks_l3060_306006


namespace NUMINAMATH_CALUDE_concert_theorem_l3060_306002

def concert_probability (n : ℕ) (sure : ℕ) (unsure : ℕ) (p : ℚ) : ℚ :=
  sorry

theorem concert_theorem :
  let n : ℕ := 8
  let sure : ℕ := 4
  let unsure : ℕ := 4
  let p : ℚ := 1/3
  concert_probability n sure unsure p = 1 := by sorry

end NUMINAMATH_CALUDE_concert_theorem_l3060_306002


namespace NUMINAMATH_CALUDE_pulley_system_velocity_l3060_306068

/-- A simple pulley system with two loads and a lever -/
structure PulleySystem where
  /-- Velocity of the left load in m/s -/
  v : ℝ
  /-- Velocity of the right load in m/s -/
  u : ℝ

/-- The pulley system satisfies the given conditions -/
def satisfies_conditions (sys : PulleySystem) : Prop :=
  sys.v = 0.5 ∧ 
  -- The strings are inextensible and weightless, and the lever is rigid
  -- (These conditions are implicitly assumed in the relationship between u and v)
  sys.u = 2/7

theorem pulley_system_velocity : 
  ∀ (sys : PulleySystem), satisfies_conditions sys → sys.u = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_pulley_system_velocity_l3060_306068


namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l3060_306014

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  ∀ x, (x - (a + 3)) * (x - (b + 3)) * (x - (c + 3)) = x^3 - 13*x^2 + 57*x - 84 :=
by sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l3060_306014


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3060_306096

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3060_306096


namespace NUMINAMATH_CALUDE_alyssa_cookie_count_l3060_306055

/-- The number of cookies Aiyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference in cookies between Aiyanna and Alyssa -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookie_count_l3060_306055


namespace NUMINAMATH_CALUDE_tangent_product_l3060_306045

theorem tangent_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.cos (α - β) = 1/5) :
  Real.tan α * Real.tan β = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l3060_306045


namespace NUMINAMATH_CALUDE_pentagon_fraction_sum_l3060_306007

theorem pentagon_fraction_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h₅ : a₅ > 0) : 
  let s := a₁ + a₂ + a₃ + a₄ + a₅
  (a₁ / (s - a₁)) + (a₂ / (s - a₂)) + (a₃ / (s - a₃)) + (a₄ / (s - a₄)) + (a₅ / (s - a₅)) < 2 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_fraction_sum_l3060_306007


namespace NUMINAMATH_CALUDE_sin_inequality_l3060_306040

theorem sin_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l3060_306040


namespace NUMINAMATH_CALUDE_double_average_l3060_306092

theorem double_average (n : ℕ) (original_avg : ℚ) (new_avg : ℚ) : 
  n = 11 → original_avg = 36 → new_avg = 2 * original_avg → new_avg = 72 := by
  sorry

end NUMINAMATH_CALUDE_double_average_l3060_306092


namespace NUMINAMATH_CALUDE_increasing_cubic_function_condition_l3060_306088

/-- The function f(x) = 2x^3 - 3mx^2 + 6x is increasing on (1, +∞) if and only if m ≤ 2 -/
theorem increasing_cubic_function_condition (m : ℝ) :
  (∀ x > 1, Monotone (fun x => 2*x^3 - 3*m*x^2 + 6*x)) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_condition_l3060_306088


namespace NUMINAMATH_CALUDE_polynomial_identity_l3060_306032

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : P 0 = 0) 
  (h2 : ∀ x : ℝ, P (x^2 + 1) = (P x)^2 + 1) : 
  ∀ x : ℝ, P x = x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3060_306032


namespace NUMINAMATH_CALUDE_triangle_area_l3060_306008

/-- The area of the triangle bounded by the x-axis and two lines -/
theorem triangle_area (line1 line2 : ℝ × ℝ → ℝ) : 
  (line1 = fun (x, y) ↦ x - 2*y - 4) →
  (line2 = fun (x, y) ↦ 2*x + y - 5) →
  (∃ x₁ x₂ y : ℝ, 
    line1 (x₁, 0) = 0 ∧ 
    line2 (x₂, 0) = 0 ∧ 
    line1 (x₁, y) = 0 ∧ 
    line2 (x₁, y) = 0 ∧ 
    y > 0) →
  (1/2 * (x₁ - x₂) * y = 9/20) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3060_306008


namespace NUMINAMATH_CALUDE_profit_maximizing_price_optimal_selling_price_is_14_l3060_306035

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ :=
  (100 - 10 * x) * ((10 + x) - 8)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

theorem profit_maximizing_price :
  optimal_price_increase = 4 ∧
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price -/
def optimal_selling_price : ℝ :=
  10 + optimal_price_increase

theorem optimal_selling_price_is_14 :
  optimal_selling_price = 14 :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_optimal_selling_price_is_14_l3060_306035


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l3060_306048

-- Define the complex polynomial in the numerator
def numerator (z : ℂ) : ℂ := z^4 - 1

-- Define the complex polynomial in the denominator
def denominator (z : ℂ) : ℂ := z^3 - 3*z + 2

-- Define the equation
def equation (z : ℂ) : Prop := numerator z = 0 ∧ denominator z ≠ 0

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℂ), s.card = 3 ∧ (∀ z ∈ s, equation z) ∧ (∀ z, equation z → z ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l3060_306048


namespace NUMINAMATH_CALUDE_perception_permutations_count_l3060_306070

/-- The number of letters in the word "PERCEPTION" -/
def total_letters : ℕ := 10

/-- The number of repeating letters (E, P, I, N) in "PERCEPTION" -/
def repeating_letters : Finset ℕ := {2, 2, 2, 2}

/-- The number of distinct permutations of the letters in "PERCEPTION" -/
def perception_permutations : ℕ := total_letters.factorial / (repeating_letters.prod (λ x => x.factorial))

theorem perception_permutations_count :
  perception_permutations = 226800 := by sorry

end NUMINAMATH_CALUDE_perception_permutations_count_l3060_306070


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_345_l3060_306060

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The theorem stating that the sum of the digits in the binary representation of 345 is 5 -/
theorem sum_of_binary_digits_345 : sum_of_binary_digits 345 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_345_l3060_306060


namespace NUMINAMATH_CALUDE_two_valid_configurations_l3060_306065

/-- Represents a quadrant in the yard --/
inductive Quadrant
| I
| II
| III
| IV

/-- Represents a configuration of apple trees in the yard --/
def Configuration := Quadrant → Nat

/-- Checks if a configuration is valid (total of 4 trees) --/
def is_valid_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II + c Quadrant.III + c Quadrant.IV = 4

/-- Checks if a configuration has equal trees on both sides of each path --/
def is_balanced_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II = c Quadrant.III + c Quadrant.IV ∧
  c Quadrant.I + c Quadrant.IV = c Quadrant.II + c Quadrant.III ∧
  c Quadrant.I + c Quadrant.III = c Quadrant.II + c Quadrant.IV

/-- Theorem: There exist at least two different valid and balanced configurations --/
theorem two_valid_configurations : ∃ (c1 c2 : Configuration),
  c1 ≠ c2 ∧
  is_valid_configuration c1 ∧
  is_valid_configuration c2 ∧
  is_balanced_configuration c1 ∧
  is_balanced_configuration c2 :=
sorry

end NUMINAMATH_CALUDE_two_valid_configurations_l3060_306065


namespace NUMINAMATH_CALUDE_money_distribution_l3060_306018

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 310) :
  C = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3060_306018


namespace NUMINAMATH_CALUDE_odd_function_complete_expression_l3060_306000

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (x : ℝ) : ℝ :=
  -x^2 + x + 1

theorem odd_function_complete_expression 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x, f x = if x > 0 then f_positive x
             else if x = 0 then 0
             else x^2 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_complete_expression_l3060_306000


namespace NUMINAMATH_CALUDE_problem_statement_l3060_306044

theorem problem_statement : (-0.125)^2003 * (-8)^2004 = -8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3060_306044


namespace NUMINAMATH_CALUDE_janes_total_hours_l3060_306073

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks : ℕ

/-- Calculate total exercise hours -/
def total_hours (routine : ExerciseRoutine) : ℕ :=
  routine.hours_per_day * routine.days_per_week * routine.weeks

/-- Jane's specific routine -/
def janes_routine : ExerciseRoutine :=
  { hours_per_day := 1
    days_per_week := 5
    weeks := 8 }

/-- Theorem: Jane's total exercise hours equal 40 -/
theorem janes_total_hours : total_hours janes_routine = 40 := by
  sorry

end NUMINAMATH_CALUDE_janes_total_hours_l3060_306073


namespace NUMINAMATH_CALUDE_isosceles_triangle_ratio_l3060_306021

/-- Two isosceles triangles with equal perimeters -/
structure IsoscelesTrianglePair :=
  (base₁ : ℝ)
  (leg₁ : ℝ)
  (base₂ : ℝ)
  (leg₂ : ℝ)
  (base₁_pos : 0 < base₁)
  (leg₁_pos : 0 < leg₁)
  (base₂_pos : 0 < base₂)
  (leg₂_pos : 0 < leg₂)
  (equal_perimeters : base₁ + 2 * leg₁ = base₂ + 2 * leg₂)
  (base_relation : base₂ = 1.15 * base₁)
  (leg_relation : leg₂ = 0.95 * leg₁)

/-- The ratio of the base to the leg of the first triangle is 2:3 -/
theorem isosceles_triangle_ratio (t : IsoscelesTrianglePair) : t.base₁ / t.leg₁ = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ratio_l3060_306021


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3060_306043

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering only rotational symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique arrangements of 8 distinct beads on a bracelet, 
    considering only rotational symmetry, is 5040 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3060_306043


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3060_306038

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group2_students : ℕ) 
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 85/100 →
  group2_mean = 80/100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 84/100 := by
sorry

#eval (40 * (85/100) + 10 * (80/100)) / 50

end NUMINAMATH_CALUDE_class_mean_calculation_l3060_306038


namespace NUMINAMATH_CALUDE_john_weekly_production_l3060_306037

/-- Calculates the number of widgets John makes in a week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Proves that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_production_l3060_306037


namespace NUMINAMATH_CALUDE_second_game_points_l3060_306029

/-- The number of points scored in each of the four games -/
structure GamePoints where
  game1 : ℕ
  game2 : ℕ
  game3 : ℕ
  game4 : ℕ

/-- The conditions of the basketball game scenario -/
def basketball_scenario (p : GamePoints) : Prop :=
  p.game1 = 10 ∧
  p.game3 = 6 ∧
  p.game4 = (p.game1 + p.game2 + p.game3) / 3 ∧
  p.game1 + p.game2 + p.game3 + p.game4 = 40

theorem second_game_points :
  ∃ p : GamePoints, basketball_scenario p ∧ p.game2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_game_points_l3060_306029


namespace NUMINAMATH_CALUDE_special_function_unique_special_function_at_3_l3060_306001

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- Theorem stating that any function satisfying the conditions must be f(x) = 2x -/
theorem special_function_unique (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 2 * x :=
sorry

/-- Corollary: f(3) = 6 for any function satisfying the conditions -/
theorem special_function_at_3 (f : ℝ → ℝ) (h : special_function f) : 
  f 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_special_function_unique_special_function_at_3_l3060_306001


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l3060_306030

theorem cos_two_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/3) : 
  Real.cos (2*α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l3060_306030


namespace NUMINAMATH_CALUDE_ground_lines_perpendicular_l3060_306039

-- Define a type for lines
def Line : Type := ℝ × ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a relation for perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a set of lines on the ground
def GroundLines : Set Line := sorry

-- Define the ruler's orientation
def RulerOrientation : Line := sorry

-- Theorem statement
theorem ground_lines_perpendicular 
  (always_parallel : ∀ (r : Line), ∃ (g : Line), g ∈ GroundLines ∧ Parallel r g) :
  ∀ (l1 l2 : Line), l1 ∈ GroundLines → l2 ∈ GroundLines → l1 ≠ l2 → Perpendicular l1 l2 :=
sorry

end NUMINAMATH_CALUDE_ground_lines_perpendicular_l3060_306039


namespace NUMINAMATH_CALUDE_library_initial_books_l3060_306052

/-- The number of books purchased last year -/
def books_last_year : ℕ := 50

/-- The number of books purchased this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The total number of books in the library now -/
def total_books_now : ℕ := 300

/-- The number of books in the library before the new purchases last year -/
def initial_books : ℕ := total_books_now - books_last_year - books_this_year

theorem library_initial_books :
  initial_books = 100 := by sorry

end NUMINAMATH_CALUDE_library_initial_books_l3060_306052


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l3060_306085

/-- Represents the money redistribution problem among three friends. -/
def MoneyRedistribution (a j t : ℚ) : Prop :=
  -- Initial conditions
  (t = 24) ∧
  -- First redistribution (Amy's turn)
  let a₁ := a - 2*j - t
  let j₁ := 3*j
  let t₁ := 2*t
  -- Second redistribution (Jan's turn)
  let a₂ := 2*a₁
  let j₂ := j₁ - (a₁ + t₁)
  let t₂ := 3*t₁
  -- Final redistribution (Toy's turn)
  let a₃ := 3*a₂
  let j₃ := 3*j₂
  let t₃ := t₂ - (a₃ - a₂ + j₃ - j₂)
  -- Final condition
  (t₃ = 24) →
  -- Conclusion
  (a + j + t = 72)

/-- The total amount of money among the three friends is 72 dollars. -/
theorem money_redistribution_theorem (a j t : ℚ) :
  MoneyRedistribution a j t → (a + j + t = 72) :=
by
  sorry


end NUMINAMATH_CALUDE_money_redistribution_theorem_l3060_306085


namespace NUMINAMATH_CALUDE_triangle_count_is_38_l3060_306012

/-- Represents a rectangle with internal divisions as described in the problem -/
structure DividedRectangle where
  -- Add necessary fields to represent the rectangle and its divisions
  -- This is a simplified representation
  height : ℕ
  width : ℕ

/-- Counts the number of triangles in a DividedRectangle -/
def countTriangles (rect : DividedRectangle) : ℕ := sorry

/-- The specific rectangle from the problem -/
def problemRectangle : DividedRectangle := {
  height := 20,
  width := 30
}

/-- Theorem stating that the number of triangles in the problem rectangle is 38 -/
theorem triangle_count_is_38 : countTriangles problemRectangle = 38 := by sorry

end NUMINAMATH_CALUDE_triangle_count_is_38_l3060_306012


namespace NUMINAMATH_CALUDE_equation_roots_count_l3060_306099

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 1

-- Define the iterative composition of f
def f_n (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- Theorem statement
theorem equation_roots_count :
  ∃! (roots : Finset ℝ), (∀ x ∈ roots, f_n 10 x = -1/2) ∧ (Finset.card roots = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_count_l3060_306099


namespace NUMINAMATH_CALUDE_solve_system_l3060_306053

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 10) (eq2 : 3 * p + 5 * q = 20) : p = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3060_306053


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3060_306041

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 4) :
  (2/x + 3/y) ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 4 ∧ 2/x₀ + 3/y₀ = 25/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3060_306041


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l3060_306062

theorem discount_percentage_calculation (cost_price marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 95 →
  marked_price = 125 →
  profit_percentage = 25 →
  ∃ (discount_percentage : ℝ),
    discount_percentage = 5 ∧
    marked_price * (1 - discount_percentage / 100) = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l3060_306062


namespace NUMINAMATH_CALUDE_number_system_generalization_l3060_306034

-- Define the number systems
inductive NumberSystem
| Natural
| Integer
| Rational
| Real
| Complex

-- Define the basic operations
inductive Operation
| Addition
| Subtraction
| Multiplication
| Division
| SquareRoot

-- Define a function to check if an operation is executable in a given number system
def is_executable (op : Operation) (ns : NumberSystem) : Prop :=
  match op, ns with
  | Operation.Subtraction, NumberSystem.Natural => false
  | Operation.Division, NumberSystem.Integer => false
  | Operation.SquareRoot, NumberSystem.Rational => false
  | Operation.SquareRoot, NumberSystem.Real => false
  | _, _ => true

-- Define the theorem
theorem number_system_generalization (op : Operation) :
  ∃ ns : NumberSystem, is_executable op ns :=
sorry

end NUMINAMATH_CALUDE_number_system_generalization_l3060_306034


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l3060_306086

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 13 * X →
  A = 322 →
  X = 14 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l3060_306086


namespace NUMINAMATH_CALUDE_parabola_properties_l3060_306097

/-- Represents a parabola of the form y = a(x-3)^2 + 2 -/
structure Parabola where
  a : ℝ

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating properties of a specific parabola -/
theorem parabola_properties (p : Parabola) (A B : Point) :
  (p.a * (1 - 3)^2 + 2 = -2) →  -- parabola passes through (1, -2)
  (A.y = p.a * (A.x - 3)^2 + 2) →  -- point A is on the parabola
  (B.y = p.a * (B.x - 3)^2 + 2) →  -- point B is on the parabola
  (A.x < B.x) →  -- m < n
  (B.x < 3) →  -- n < 3
  (p.a = -1 ∧ A.y < B.y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3060_306097


namespace NUMINAMATH_CALUDE_simple_interest_rate_is_five_percent_l3060_306071

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate is 5% given the problem conditions -/
theorem simple_interest_rate_is_five_percent :
  simple_interest_rate 750 900 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_is_five_percent_l3060_306071


namespace NUMINAMATH_CALUDE_centers_connection_line_l3060_306084

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (2, -3)
def center2 : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem centers_connection_line : 
  line_equation (center1.1) (center1.2) ∧ 
  line_equation (center2.1) (center2.2) ∧
  ∀ (x y : ℝ), line_equation x y ↔ ∃ (t : ℝ), 
    x = center1.1 + t * (center2.1 - center1.1) ∧
    y = center1.2 + t * (center2.2 - center1.2) :=
sorry

end NUMINAMATH_CALUDE_centers_connection_line_l3060_306084


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3060_306009

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 + 4 * p.2^2 = 1 ∧ p.2 = m * p.1 + 1) →
  m^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3060_306009


namespace NUMINAMATH_CALUDE_lecture_average_minutes_heard_l3060_306024

/-- Calculates the average number of minutes heard in a lecture --/
theorem lecture_average_minutes_heard 
  (total_duration : ℝ) 
  (total_attendees : ℕ) 
  (full_lecture_percent : ℝ) 
  (missed_lecture_percent : ℝ) 
  (half_lecture_percent : ℝ) 
  (h1 : total_duration = 90)
  (h2 : total_attendees = 200)
  (h3 : full_lecture_percent = 0.3)
  (h4 : missed_lecture_percent = 0.2)
  (h5 : half_lecture_percent = 0.4 * (1 - full_lecture_percent - missed_lecture_percent))
  (h6 : full_lecture_percent + missed_lecture_percent + half_lecture_percent + 
        (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) = 1) :
  (full_lecture_percent * total_duration * total_attendees + 
   0 * missed_lecture_percent * total_attendees + 
   (total_duration / 2) * half_lecture_percent * total_attendees + 
   (3 * total_duration / 4) * (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) * total_attendees) / 
   total_attendees = 56.25 := by
sorry

end NUMINAMATH_CALUDE_lecture_average_minutes_heard_l3060_306024


namespace NUMINAMATH_CALUDE_largest_square_with_three_lattice_points_l3060_306036

/-- A lattice point in a 2D plane. -/
def LatticePoint (p : ℝ × ℝ) : Prop := Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square in a 2D plane. -/
structure Square where
  center : ℝ × ℝ
  sideLength : ℝ
  rotation : ℝ  -- Angle of rotation in radians

/-- Predicate to check if a point is in the interior of a square. -/
def IsInteriorPoint (s : Square) (p : ℝ × ℝ) : Prop := sorry

/-- The number of lattice points in the interior of a square. -/
def InteriorLatticePointCount (s : Square) : ℕ := sorry

/-- Theorem stating that the area of the largest square containing exactly three lattice points in its interior is 5. -/
theorem largest_square_with_three_lattice_points :
  ∃ (s : Square), InteriorLatticePointCount s = 3 ∧
    ∀ (s' : Square), InteriorLatticePointCount s' = 3 → s'.sideLength^2 ≤ s.sideLength^2 ∧
    s.sideLength^2 = 5 := by sorry

end NUMINAMATH_CALUDE_largest_square_with_three_lattice_points_l3060_306036


namespace NUMINAMATH_CALUDE_division_of_power_sixteen_l3060_306047

theorem division_of_power_sixteen (m : ℕ) : m = 16^2024 → m/8 = 8 * 16^2020 := by
  sorry

end NUMINAMATH_CALUDE_division_of_power_sixteen_l3060_306047


namespace NUMINAMATH_CALUDE_power_inequality_l3060_306082

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + a*b^3*c + a*b*c^3 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3060_306082


namespace NUMINAMATH_CALUDE_p_and_not_q_l3060_306023

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 1 → a^2 > a

-- Define proposition q
def q : Prop := ∀ a : ℝ, a > 0 → a > 1/a

-- Theorem to prove
theorem p_and_not_q : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_l3060_306023


namespace NUMINAMATH_CALUDE_prime_square_difference_l3060_306046

theorem prime_square_difference (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_form : ∃ k, p = 4*k + 3) (hq_form : ∃ k, q = 4*k + 3)
  (h_exists : ∃ (x y : ℤ), x^2 - p*q*y^2 = 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (p*a^2 - q*b^2 = 1 ∨ q*b^2 - p*a^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_difference_l3060_306046


namespace NUMINAMATH_CALUDE_perpendicular_line_m_value_l3060_306033

/-- Given a line passing through points (m, 3) and (1, m) that is perpendicular
    to a line with slope -1, prove that m = 2. -/
theorem perpendicular_line_m_value (m : ℝ) : 
  (((m - 3) / (1 - m) = 1) ∧ (1 * (-1) = -1)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_m_value_l3060_306033


namespace NUMINAMATH_CALUDE_probability_one_girl_no_growth_pie_l3060_306080

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ :=
  1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given)

theorem probability_one_girl_no_growth_pie :
  probability_no_growth_pie = 7/10 :=
sorry

end NUMINAMATH_CALUDE_probability_one_girl_no_growth_pie_l3060_306080


namespace NUMINAMATH_CALUDE_binomial_n_value_l3060_306026

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_n_value (X : BinomialRV) 
  (h_exp : expectation X = 2)
  (h_var : variance X = 3/2) :
  X.n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_n_value_l3060_306026


namespace NUMINAMATH_CALUDE_median_of_dataset2_with_X_l3060_306005

def dataset1 : List ℕ := [15, 9, 11, 7]
def dataset2 : List ℕ := [10, 11, 14, 8]

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem median_of_dataset2_with_X (X : ℕ) : 
  mode (X :: dataset1) = 11 → median (X :: dataset2) = 11 := by sorry

end NUMINAMATH_CALUDE_median_of_dataset2_with_X_l3060_306005


namespace NUMINAMATH_CALUDE_square_greater_than_linear_for_less_than_negative_one_l3060_306025

theorem square_greater_than_linear_for_less_than_negative_one (x : ℝ) :
  x < -1 → x^2 > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_linear_for_less_than_negative_one_l3060_306025


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3060_306027

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
  (h1 : compound_interest P r 6 = 6000)
  (h2 : compound_interest P r 7 = 7500) :
  r = 0.25 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3060_306027


namespace NUMINAMATH_CALUDE_person_B_age_l3060_306010

theorem person_B_age (a b c d e f g : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  c = d / 2 →
  d = e - 3 →
  f = a * d →
  g = b + e →
  a + b + c + d + e + f + g = 292 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_person_B_age_l3060_306010


namespace NUMINAMATH_CALUDE_valid_lineup_count_l3060_306004

def total_players : ℕ := 16
def num_starters : ℕ := 7
def num_triplets : ℕ := 3
def num_twins : ℕ := 2

def valid_lineups : ℕ := 8778

theorem valid_lineup_count :
  (Nat.choose total_players num_starters) -
  ((Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) +
   (Nat.choose (total_players - num_twins) (num_starters - num_twins)) -
   (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)))
  = valid_lineups := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l3060_306004


namespace NUMINAMATH_CALUDE_ticket_sales_result_l3060_306066

/-- Represents a section in the stadium -/
structure Section where
  name : String
  seats : Nat
  price : Nat

/-- Represents the stadium configuration -/
def Stadium : List Section := [
  ⟨"A", 40, 10⟩,
  ⟨"B", 30, 15⟩,
  ⟨"C", 25, 20⟩
]

/-- Theorem stating the result of the ticket sales -/
theorem ticket_sales_result 
  (children : Nat) (adults : Nat) (seniors : Nat)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : seniors = 15)
  (h4 : children + adults + seniors = Stadium.foldr (fun s acc => s.seats + acc) 0 + 1) :
  (∀ s : Section, s ∈ Stadium → 
    (if s.name = "A" then adults + seniors else children) ≥ s.seats) ∧
  (Stadium.foldr (fun s acc => s.seats * s.price + acc) 0 = 1350) := by
  sorry

#check ticket_sales_result

end NUMINAMATH_CALUDE_ticket_sales_result_l3060_306066


namespace NUMINAMATH_CALUDE_dual_expression_problem_l3060_306031

theorem dual_expression_problem (x : ℝ) :
  (Real.sqrt (20 - x) + Real.sqrt (4 - x) = 8) →
  (Real.sqrt (20 - x) - Real.sqrt (4 - x) = 2) ∧ (x = -5) := by
  sorry


end NUMINAMATH_CALUDE_dual_expression_problem_l3060_306031


namespace NUMINAMATH_CALUDE_base_b_is_7_l3060_306022

/-- Given a base b, this function represents the number 15 in that base -/
def number_15 (b : ℕ) : ℕ := b + 5

/-- Given a base b, this function represents the number 433 in that base -/
def number_433 (b : ℕ) : ℕ := 4*b^2 + 3*b + 3

/-- The theorem states that if the square of the number represented by 15 in base b
    equals the number represented by 433 in base b, then b must be 7 in base 10 -/
theorem base_b_is_7 : ∃ (b : ℕ), (number_15 b)^2 = number_433 b ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_7_l3060_306022


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3060_306028

/-- Represents the scenario of a power boat and raft on a river --/
structure RiverScenario where
  r : ℝ  -- Speed of the river current (and raft)
  p : ℝ  -- Speed of the power boat relative to the river
  t : ℝ  -- Time taken by power boat from A to B

/-- The conditions of the problem --/
def scenario_conditions (s : RiverScenario) : Prop :=
  s.r > 0 ∧ s.p > 0 ∧ s.t > 0 ∧
  (s.p + s.r) * s.t + (s.p - s.r) * (9 - s.t) = 9 * s.r

/-- The theorem to be proved --/
theorem power_boat_travel_time (s : RiverScenario) :
  scenario_conditions s → s.t = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_power_boat_travel_time_l3060_306028


namespace NUMINAMATH_CALUDE_angle_terminal_side_problem_tan_equation_problem_l3060_306015

-- Problem 1
theorem angle_terminal_side_problem (α : Real) 
  (h : Real.tan α = -3/4 ∧ Real.sin α = 3/5) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (2019*π/2 - α) * Real.tan (9*π/2 + α)) = 9/20 := by sorry

-- Problem 2
theorem tan_equation_problem (x : Real) (h : Real.tan (π/4 + x) = 2018) :
  1 / Real.cos (2*x) + Real.tan (2*x) = 2018 := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_problem_tan_equation_problem_l3060_306015


namespace NUMINAMATH_CALUDE_gcd_factorial_8_factorial_6_squared_l3060_306003

theorem gcd_factorial_8_factorial_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_factorial_6_squared_l3060_306003


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_l3060_306042

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Checks if a quadrilateral has a vertical line of symmetry -/
def hasVerticalSymmetry (q : Quadrilateral) : Prop := sorry

theorem y_coordinate_of_C (q : Quadrilateral) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 1⟩ →
  q.D = ⟨3, 1⟩ →
  q.C.x = q.B.x →
  hasVerticalSymmetry q →
  area q = 18 →
  q.C.y = 11 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_l3060_306042


namespace NUMINAMATH_CALUDE_llama_to_goat_ratio_l3060_306074

def goat_cost : ℕ := 400
def num_goats : ℕ := 3
def total_spent : ℕ := 4800

def llama_cost : ℕ := goat_cost + goat_cost / 2

def num_llamas : ℕ := (total_spent - num_goats * goat_cost) / llama_cost

theorem llama_to_goat_ratio :
  num_llamas * 1 = num_goats * 2 :=
by sorry

end NUMINAMATH_CALUDE_llama_to_goat_ratio_l3060_306074


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3060_306079

theorem inscribed_sphere_volume (r h l : ℝ) (V : ℝ) :
  r = 2 →
  2 * π * r * l = 8 * π →
  h^2 + r^2 = l^2 →
  (h - V^(1/3) * ((3 * r) / (4 * π))^(1/3)) / l = V^(1/3) * ((3 * r) / (4 * π))^(1/3) / r →
  V = (32 * Real.sqrt 3) / 27 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3060_306079


namespace NUMINAMATH_CALUDE_min_team_size_for_handshake_probability_l3060_306091

theorem min_team_size_for_handshake_probability (n : ℕ) : n ≥ 20 ↔ 
  (2 : ℚ) / (n + 1 : ℚ) < (1 : ℚ) / 10 ∧ 
  ∀ m : ℕ, m < n → (2 : ℚ) / (m + 1 : ℚ) ≥ (1 : ℚ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_team_size_for_handshake_probability_l3060_306091


namespace NUMINAMATH_CALUDE_bodies_of_water_is_six_l3060_306013

/-- Represents the aquatic reserve scenario -/
structure AquaticReserve where
  total_fish : ℕ
  fish_per_body : ℕ
  h_total : total_fish = 1050
  h_per_body : fish_per_body = 175

/-- The number of bodies of water in the aquatic reserve -/
def bodies_of_water (reserve : AquaticReserve) : ℕ :=
  reserve.total_fish / reserve.fish_per_body

/-- Theorem stating that the number of bodies of water is 6 -/
theorem bodies_of_water_is_six (reserve : AquaticReserve) : 
  bodies_of_water reserve = 6 := by
  sorry


end NUMINAMATH_CALUDE_bodies_of_water_is_six_l3060_306013


namespace NUMINAMATH_CALUDE_specific_pairings_probability_l3060_306095

/-- The probability of two specific pairings occurring simultaneously in a class of 32 students -/
theorem specific_pairings_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 3) = 1 / 899 :=
sorry

end NUMINAMATH_CALUDE_specific_pairings_probability_l3060_306095


namespace NUMINAMATH_CALUDE_some_number_approximation_l3060_306069

/-- Given that (3.241 * 14) / x = 0.045374000000000005, prove that x ≈ 1000 -/
theorem some_number_approximation (x : ℝ) 
  (h : (3.241 * 14) / x = 0.045374000000000005) : 
  ∃ ε > 0, |x - 1000| < ε :=
sorry

end NUMINAMATH_CALUDE_some_number_approximation_l3060_306069


namespace NUMINAMATH_CALUDE_journey_speed_proof_l3060_306011

/-- Proves that the speed of the first half of a journey is 21 km/hr, given the total journey conditions -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := first_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l3060_306011


namespace NUMINAMATH_CALUDE_cup_arrangement_theorem_l3060_306056

/-- Represents the number of ways to arrange cups in a circular pattern -/
def circularArrangements (yellow blue red : ℕ) : ℕ := sorry

/-- Represents the number of ways to arrange cups in a circular pattern with adjacent red cups -/
def circularArrangementsAdjacentRed (yellow blue red : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid arrangements -/
theorem cup_arrangement_theorem :
  circularArrangements 4 3 2 - circularArrangementsAdjacentRed 4 3 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cup_arrangement_theorem_l3060_306056


namespace NUMINAMATH_CALUDE_work_completion_time_l3060_306094

/-- Given two workers a and b, where a does half as much work as b in 3/4 of the time,
    and b takes 30 days to complete the work alone, prove that they take 18 days
    to complete the work together. -/
theorem work_completion_time (a b : ℝ) : 
  (a * (3/4 * 30) = (1/2) * b * 30) →  -- a does half as much work as b in 3/4 of the time
  (b * 30 = 1) →  -- b completes the work in 30 days
  (a + b) * 18 = 1  -- they complete the work together in 18 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3060_306094


namespace NUMINAMATH_CALUDE_perpendicular_to_vertical_line_l3060_306058

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A vertical line represented by its x-coordinate -/
structure VerticalLine where
  x : ℝ

/-- Two lines are perpendicular if one is vertical and the other is horizontal -/
def isPerpendicular (l : Line) (v : VerticalLine) : Prop :=
  l.slope = 0

theorem perpendicular_to_vertical_line (k : ℝ) :
  isPerpendicular (Line.mk k 1) (VerticalLine.mk 1) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_vertical_line_l3060_306058


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l3060_306050

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2 + 8 * x - 5

/-- The focus of a parabola y = a(x - h)^2 + k is at (h, k + 1/(4a)) -/
def parabola_focus (a h k x y : ℝ) : Prop :=
  x = h ∧ y = k + 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 + 8x - 5 is at (-1, -8.9375) -/
theorem parabola_focus_theorem :
  ∃ (x y : ℝ), parabola_equation x y ∧ parabola_focus 4 (-1) (-9) x y ∧ x = -1 ∧ y = -8.9375 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l3060_306050


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l3060_306054

/-- 
Given a 3-digit number represented by its digits x, y, and z,
if four times the number equals 1464 and the sum of its digits is 15,
then the number is 366.
-/
theorem three_digit_number_problem (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 →
  4 * (100 * x + 10 * y + z) = 1464 →
  x + y + z = 15 →
  100 * x + 10 * y + z = 366 := by
  sorry

#check three_digit_number_problem

end NUMINAMATH_CALUDE_three_digit_number_problem_l3060_306054


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l3060_306078

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 = 100 * k + 12 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l3060_306078


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l3060_306077

theorem sum_of_reciprocals_bound {α β k : ℝ} (hα : α > 0) (hβ : β > 0) (hk : k > 0)
  (hαβ : α ≠ β) (hfα : |Real.log α| = k) (hfβ : |Real.log β| = k) :
  1 / α + 1 / β > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l3060_306077


namespace NUMINAMATH_CALUDE_jenny_total_wins_l3060_306090

/-- The number of games Jenny played against Mark -/
def games_with_mark : ℕ := 10

/-- The number of games Mark won against Jenny -/
def marks_wins : ℕ := 1

/-- The number of games Jenny played against Jill -/
def games_with_jill : ℕ := 2 * games_with_mark

/-- The percentage of games Jill won against Jenny -/
def jills_win_percentage : ℚ := 75 / 100

theorem jenny_total_wins : 
  (games_with_mark - marks_wins) + 
  (games_with_jill - (jills_win_percentage * games_with_jill).num) = 14 := by
sorry

end NUMINAMATH_CALUDE_jenny_total_wins_l3060_306090


namespace NUMINAMATH_CALUDE_hexagon_intersection_area_l3060_306016

-- Define the hexagon
structure Hexagon where
  area : ℝ
  is_regular : Prop

-- Define a function to calculate the expected value
def expected_intersection_area (H : Hexagon) : ℝ :=
  -- The actual calculation of the expected value
  12

-- The theorem to be proved
theorem hexagon_intersection_area (H : Hexagon) 
  (h1 : H.area = 360) 
  (h2 : H.is_regular) : 
  expected_intersection_area H = 12 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_intersection_area_l3060_306016


namespace NUMINAMATH_CALUDE_max_volume_at_one_cm_l3060_306019

/-- The side length of the original square sheet -/
def sheet_side : ℝ := 6

/-- The side length of the small square cut from each corner -/
def cut_side : ℝ := 1

/-- The volume of the box as a function of the cut side length -/
def box_volume (x : ℝ) : ℝ := x * (sheet_side - 2 * x)^2

theorem max_volume_at_one_cm :
  ∀ x, 0 < x → x < sheet_side / 2 → box_volume cut_side ≥ box_volume x :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_one_cm_l3060_306019


namespace NUMINAMATH_CALUDE_problem_solution_l3060_306051

theorem problem_solution (a b : ℕ+) (q r : ℕ) :
  a^2 + b^2 = q * (a + b) + r ∧ q^2 + r = 1977 →
  ((a = 50 ∧ b = 7) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 37 ∧ b = 50)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3060_306051


namespace NUMINAMATH_CALUDE_cosine_equality_l3060_306061

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (820 * π / 180) → 
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_cosine_equality_l3060_306061


namespace NUMINAMATH_CALUDE_magic_shop_change_theorem_final_change_theorem_l3060_306093

/-- Represents the currency system in the magic shop -/
structure MagicShopCurrency where
  silver_to_gold_rate : ℚ
  cloak_price_gold : ℚ

/-- Calculate the change in silver coins when buying a cloak with gold coins -/
def change_in_silver (c : MagicShopCurrency) (gold_paid : ℚ) : ℚ :=
  (gold_paid - c.cloak_price_gold) * (1 / c.silver_to_gold_rate)

/-- Theorem: Buying a cloak with 14 gold coins results in 10 silver coins as change -/
theorem magic_shop_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = 10 := by
  sorry

/-- The correct change is 10 silver coins -/
def correct_change : ℚ := 10

/-- The final theorem stating the correct change -/
theorem final_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = correct_change := by
  sorry

end NUMINAMATH_CALUDE_magic_shop_change_theorem_final_change_theorem_l3060_306093


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3060_306059

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population
  (initial_marked : ℕ)
  (second_catch : ℕ)
  (marked_in_second : ℕ)
  (initial_marked_pos : 0 < initial_marked)
  (second_catch_pos : 0 < second_catch)
  (marked_in_second_pos : 0 < marked_in_second)
  (marked_in_second_le_second_catch : marked_in_second ≤ second_catch)
  (marked_in_second_le_initial_marked : marked_in_second ≤ initial_marked) :
  (initial_marked * second_catch) / marked_in_second = 1500 :=
sorry

#eval (40 * 300) / 8

end NUMINAMATH_CALUDE_estimate_fish_population_l3060_306059


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l3060_306057

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra fruit the cafeteria ended up with -/
def extra_fruit : ℕ := 40

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

theorem cafeteria_green_apples :
  red_apples + green_apples - students_wanting_fruit = extra_fruit :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l3060_306057


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3060_306081

/-- An arithmetic sequence with sum Sn for the first n terms -/
structure ArithmeticSequence where
  Sn : ℕ → ℚ
  a : ℕ → ℚ
  d : ℚ
  sum_formula : ∀ n, Sn n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 1/5 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
    (h1 : seq.Sn 5 = 6)
    (h2 : seq.a 2 = 1) : 
  seq.d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3060_306081


namespace NUMINAMATH_CALUDE_max_value_T_l3060_306064

theorem max_value_T (a b c : ℝ) (ha : 1 ≤ a) (ha' : a ≤ 2) 
                     (hb : 1 ≤ b) (hb' : b ≤ 2)
                     (hc : 1 ≤ c) (hc' : c ≤ 2) : 
  (∃ (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2), 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 = 2) ∧ 
  (∀ (x y z : ℝ), 1 ≤ x ∧ x ≤ 2 → 1 ≤ y ∧ y ≤ 2 → 1 ≤ z ∧ z ≤ 2 → 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_T_l3060_306064


namespace NUMINAMATH_CALUDE_expression_evaluation_l3060_306017

theorem expression_evaluation (x : ℝ) (h : 2*x - 9 ≥ 0) :
  (3 + Real.sqrt (2*x - 9))^2 - 3*x = -x + 6*Real.sqrt (2*x - 9) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3060_306017


namespace NUMINAMATH_CALUDE_yunas_marbles_l3060_306067

/-- Yuna's marble problem -/
theorem yunas_marbles (M : ℕ) : 
  (((M - 12 + 5) / 2 : ℚ) + 3 : ℚ) = 17 → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_yunas_marbles_l3060_306067


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l3060_306072

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists a z in S such that z^n = 1 -/
def property (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ S ∧ z^n = 1

/-- Theorem stating that 13 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_13 : 
  (property 13 ∧ ∀ m : ℕ, 0 < m → m < 13 → ¬property m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l3060_306072


namespace NUMINAMATH_CALUDE_time_to_fill_leaking_pool_l3060_306063

/-- Time to fill a leaking pool -/
theorem time_to_fill_leaking_pool 
  (pool_capacity : ℝ) 
  (filling_rate : ℝ) 
  (leaking_rate : ℝ) 
  (h1 : pool_capacity = 60) 
  (h2 : filling_rate = 1.6) 
  (h3 : leaking_rate = 0.1) : 
  pool_capacity / (filling_rate - leaking_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_time_to_fill_leaking_pool_l3060_306063


namespace NUMINAMATH_CALUDE_function_property_l3060_306049

open Set Function Real

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x, x ≠ a → f x ≠ f a) :
  (∀ x, x ≠ a → f x ≠ f a) ∧ 
  ¬(∀ x, f x ≠ f a → x ≠ a) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3060_306049


namespace NUMINAMATH_CALUDE_trade_value_trade_value_correct_l3060_306098

theorem trade_value (matt_cards : ℕ) (matt_card_value : ℕ) 
  (traded_cards : ℕ) (received_cheap_cards : ℕ) (cheap_card_value : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_traded_value := traded_cards * matt_card_value
  let received_cheap_value := received_cheap_cards * cheap_card_value
  let total_received_value := total_traded_value + profit
  total_received_value - received_cheap_value

#check trade_value 8 6 2 3 2 3 = 9

theorem trade_value_correct : trade_value 8 6 2 3 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_trade_value_trade_value_correct_l3060_306098


namespace NUMINAMATH_CALUDE_sqrt_sum_to_fraction_l3060_306020

theorem sqrt_sum_to_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_to_fraction_l3060_306020


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3060_306076

theorem triangle_side_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let A : ℝ := 2 * Real.pi / 3
  a^2 = 2*b*c + 3*c^2 →
  c / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3060_306076


namespace NUMINAMATH_CALUDE_largest_angle_obtuse_l3060_306083

-- Define a triangle with altitudes
structure TriangleWithAltitudes where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  h₃_pos : h₃ > 0

-- Define the property of having a specific set of altitudes
def hasAltitudes (t : TriangleWithAltitudes) : Prop :=
  t.h₁ = 8 ∧ t.h₂ = 10 ∧ t.h₃ = 25

-- Define an obtuse angle
def isObtuse (θ : ℝ) : Prop :=
  θ > Real.pi / 2 ∧ θ < Real.pi

-- Theorem statement
theorem largest_angle_obtuse (t : TriangleWithAltitudes) (h : hasAltitudes t) :
  ∃ θ, isObtuse θ ∧ (∀ φ, φ ≤ θ) :=
sorry

end NUMINAMATH_CALUDE_largest_angle_obtuse_l3060_306083


namespace NUMINAMATH_CALUDE_negation_and_converse_of_divisibility_proposition_l3060_306075

def last_digit (n : ℤ) : ℤ := n % 10

theorem negation_and_converse_of_divisibility_proposition :
  (¬ (∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → n % 5 = 0) ↔ 
   (∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ n % 5 ≠ 0)) ∧
  ((∀ n : ℤ, n % 5 = 0 → (last_digit n = 0 ∨ last_digit n = 5)) ↔
   (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → n % 5 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_and_converse_of_divisibility_proposition_l3060_306075


namespace NUMINAMATH_CALUDE_line_direction_vector_l3060_306087

/-- Given a line passing through points (-3, 4) and (4, -1) with direction vector (a, a/2), prove a = -10 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ k * (4 - (-3)) = a ∧ k * (-1 - 4) = a/2) → 
  a = -10 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3060_306087


namespace NUMINAMATH_CALUDE_cone_volume_l3060_306089

/-- The volume of a cone with height equal to its radius, where the radius is √m and m is a rational number -/
theorem cone_volume (m : ℚ) (h : m > 0) : 
  let r : ℝ := Real.sqrt m
  let volume := (1/3 : ℝ) * Real.pi * r^2 * r
  volume = (1/3 : ℝ) * Real.pi * m^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3060_306089

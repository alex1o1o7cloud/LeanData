import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3631_363182

/-- Represents a quadratic function of the form y = x^2 + bx - c -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic function -/
structure Roots where
  m : ℝ
  h : m ≠ 0

theorem quadratic_function_properties (f : QuadraticFunction) (r : Roots) :
  (∀ x, f.b * x + x^2 - f.c = 0 ↔ x = r.m ∨ x = -2 * r.m) →
  f.c = 2 * f.b^2 ∧
  (f.b / 2 = -1 → f.b = 2 ∧ f.c = 8) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l3631_363182


namespace NUMINAMATH_CALUDE_even_function_sum_l3631_363163

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + 1

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a (a + 1) → g x = g (-x)

-- Theorem statement
theorem even_function_sum (a b : ℝ) :
  is_even_function (f a b) a → a + a^b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l3631_363163


namespace NUMINAMATH_CALUDE_not_both_nonstandard_l3631_363161

def IntegerFunction (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def NonStandard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_nonstandard (G : ℤ → ℤ) (h : IntegerFunction G) :
  ¬(NonStandard G 267 ∧ NonStandard G 269) := by
  sorry

end NUMINAMATH_CALUDE_not_both_nonstandard_l3631_363161


namespace NUMINAMATH_CALUDE_consecutive_letters_probability_l3631_363106

/-- The number of cards in the deck -/
def n : ℕ := 5

/-- The number of cards to draw -/
def k : ℕ := 2

/-- The number of ways to choose k cards from n cards -/
def total_outcomes : ℕ := n.choose k

/-- The number of pairs of consecutive letters -/
def favorable_outcomes : ℕ := n - 1

/-- The probability of drawing 2 cards with consecutive letters -/
def probability : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of drawing 2 cards with consecutive letters is 2/5 -/
theorem consecutive_letters_probability :
  probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_letters_probability_l3631_363106


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3631_363180

/-- Given a circle described by the equation x^2 + y^2 - 2x + 4y = 0,
    prove that its center coordinates are (1, -2) and its radius is √5. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 5 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 2*x + 4*y = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3631_363180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_formula_l3631_363166

/-- The sum of an arithmetic sequence of consecutive integers -/
def arithmeticSequenceSum (k : ℕ) : ℕ :=
  let firstTerm := (k - 1)^2 + 1
  let numTerms := 2 * k
  numTerms * (2 * firstTerm + (numTerms - 1)) / 2

theorem arithmetic_sequence_sum_formula (k : ℕ) :
  arithmeticSequenceSum k = 2 * k^3 + k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_formula_l3631_363166


namespace NUMINAMATH_CALUDE_linear_transformation_uniqueness_l3631_363139

theorem linear_transformation_uniqueness (z₁ z₂ w₁ w₂ : ℂ) 
  (h₁ : z₁ ≠ z₂) (h₂ : w₁ ≠ w₂) :
  ∃! (a b : ℂ), (a * z₁ + b = w₁) ∧ (a * z₂ + b = w₂) := by
  sorry

end NUMINAMATH_CALUDE_linear_transformation_uniqueness_l3631_363139


namespace NUMINAMATH_CALUDE_m_equals_eight_m_uniqueness_l3631_363162

/-- The value of m for which the given conditions are satisfied -/
def find_m : ℝ → Prop := λ m =>
  m ≠ 0 ∧
  ∃ A B : ℝ × ℝ,
    -- Circle equation
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    -- Points A and B are on the directrix of the parabola
    A.1 = -m/4 ∧
    B.1 = -m/4 ∧
    -- Distance between A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 ∧
    -- Parabola equation (not directly used, but implied by the directrix)
    ∀ x y, y^2 = m*x → x ≥ -m/4

/-- Theorem stating that m = 8 satisfies the given conditions -/
theorem m_equals_eight : find_m 8 := by sorry

/-- Theorem stating that 8 is the only value of m that satisfies the given conditions -/
theorem m_uniqueness : ∀ m, find_m m → m = 8 := by sorry

end NUMINAMATH_CALUDE_m_equals_eight_m_uniqueness_l3631_363162


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l3631_363156

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l3631_363156


namespace NUMINAMATH_CALUDE_seeds_in_gray_parts_l3631_363189

theorem seeds_in_gray_parts (total_seeds : ℕ) 
  (white_seeds_circle1 : ℕ) (white_seeds_circle2 : ℕ) (white_seeds_each : ℕ)
  (h1 : white_seeds_circle1 = 87)
  (h2 : white_seeds_circle2 = 110)
  (h3 : white_seeds_each = 68) :
  (white_seeds_circle1 - white_seeds_each) + (white_seeds_circle2 - white_seeds_each) = 61 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_gray_parts_l3631_363189


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3631_363141

/-- Given vectors a = (x, 2) and b = (1, y) where x > 0, y > 0, and a ⋅ b = 1,
    the minimum value of 1/x + 2/y is 35/6 -/
theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_dot_product : x * 1 + 2 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * 1 + 2 * y' = 1 → 1 / x + 2 / y ≤ 1 / x' + 2 / y') ∧
  1 / x + 2 / y = 35 / 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3631_363141


namespace NUMINAMATH_CALUDE_regression_line_intercept_l3631_363183

/-- Given a linear regression line ŷ = (1/3)x + a passing through the point (x̄, ȳ),
    where x̄ = 3/8 and ȳ = 5/8, prove that a = 1/2. -/
theorem regression_line_intercept (x_bar y_bar : ℝ) (a : ℝ) 
    (h1 : x_bar = 3/8)
    (h2 : y_bar = 5/8)
    (h3 : y_bar = (1/3) * x_bar + a) : 
  a = 1/2 := by
  sorry

#check regression_line_intercept

end NUMINAMATH_CALUDE_regression_line_intercept_l3631_363183


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l3631_363155

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l3631_363155


namespace NUMINAMATH_CALUDE_toy_distribution_ratio_l3631_363121

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  (total_toys / num_friends) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_toy_distribution_ratio_l3631_363121


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l3631_363186

/-- Given a line intersecting y = x^2 at x₁ and x₂, and the x-axis at x₃ (all non-zero),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0)
  (h_parabola : ∃ (a b : ℝ), x₁^2 = a*x₁ + b ∧ x₂^2 = a*x₂ + b)
  (h_x_axis : ∃ (a b : ℝ), 0 = a*x₃ + b ∧ (x₁^2 = a*x₁ + b ∨ x₂^2 = a*x₁ + b)) :
  1/x₁ + 1/x₂ = 1/x₃ := by sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l3631_363186


namespace NUMINAMATH_CALUDE_polynomial_zeros_product_l3631_363184

theorem polynomial_zeros_product (z₁ z₂ : ℂ) : 
  z₁^2 + 6*z₁ + 11 = 0 → 
  z₂^2 + 6*z₂ + 11 = 0 → 
  (1 + z₁^2*z₂)*(1 + z₁*z₂^2) = 1266 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zeros_product_l3631_363184


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3631_363191

theorem polynomial_remainder (x : ℤ) : (x^15 - 1) % (x + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3631_363191


namespace NUMINAMATH_CALUDE_inequality_and_ln2_bounds_l3631_363172

theorem inequality_and_ln2_bounds (x a : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (2 * x / a < ∫ t in (a - x)..(a + x), 1 / t) ∧
  (∫ t in (a - x)..(a + x), 1 / t < x * (1 / (a + x) + 1 / (a - x))) ∧
  (0.68 < Real.log 2) ∧ (Real.log 2 < 0.71) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_ln2_bounds_l3631_363172


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3631_363127

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h1 : on_hyperbola A) 
  (h2 : on_hyperbola B) 
  (h3 : A.1 < 0 ∧ B.1 < 0)  -- A and B are on the left branch
  (h4 : ∃ t : ℝ, A.1 = (1 - t) * left_focus.1 + t * B.1 ∧ 
               A.2 = (1 - t) * left_focus.2 + t * B.2)  -- AB passes through left focus
  (h5 : distance A B = 5) :
  distance A right_focus + distance B right_focus + distance A B = 26 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3631_363127


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3631_363176

theorem selling_price_ratio 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : profit_percentage = 60) 
  (h2 : loss_percentage = 20) : 
  (cost_price - loss_percentage / 100 * cost_price) / 
  (cost_price + profit_percentage / 100 * cost_price) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3631_363176


namespace NUMINAMATH_CALUDE_square_area_ratio_l3631_363110

theorem square_area_ratio (R : ℝ) (R_pos : R > 0) : 
  let x := Real.sqrt ((4 / 5) * R^2)
  let y := R * Real.sqrt 2
  x^2 / y^2 = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3631_363110


namespace NUMINAMATH_CALUDE_sector_max_area_l3631_363188

theorem sector_max_area (perimeter : ℝ) (h : perimeter = 40) :
  ∃ (area : ℝ), area ≤ 100 ∧ 
  ∀ (r l : ℝ), r > 0 → l > 0 → l + 2 * r = perimeter → 
  (1 / 2) * l * r ≤ area :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3631_363188


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3631_363111

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
theorem wrapping_paper_area (l w h : ℝ) (h_positive : l > 0 ∧ w > 0 ∧ h > 0) 
  (h_different : h ≠ l ∧ h ≠ w) : 
  let side_length := l + w
  (side_length ^ 2 : ℝ) = (l + w) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l3631_363111


namespace NUMINAMATH_CALUDE_at_least_100_triangles_l3631_363148

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Calculates the number of triangular regions formed by the lines -/
def num_triangular_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem stating that for 300 lines with given conditions, there are at least 100 triangular regions -/
theorem at_least_100_triangles (config : LineConfiguration) 
  (h1 : config.num_lines = 300)
  (h2 : config.no_parallel = true)
  (h3 : config.no_triple_intersection = true) :
  num_triangular_regions config ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_at_least_100_triangles_l3631_363148


namespace NUMINAMATH_CALUDE_fraction_numerator_l3631_363199

theorem fraction_numerator (y : ℝ) (n : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 10 + (n / y) * y = 0.5 * y) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l3631_363199


namespace NUMINAMATH_CALUDE_angle_c_is_right_angle_l3631_363165

theorem angle_c_is_right_angle (A B C : ℝ) (a b c : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a > 0) → (b > 0) → (c > 0) →
  (A + B + C = π) →
  (a / Real.sin B + b / Real.sin A = 2 * c) →
  C = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_c_is_right_angle_l3631_363165


namespace NUMINAMATH_CALUDE_sophomore_mean_is_94_l3631_363100

/-- Represents the number of students and their scores in a math competition -/
structure MathCompetition where
  total_students : ℕ
  overall_mean : ℝ
  sophomores : ℕ
  juniors : ℕ
  sophomore_mean : ℝ
  junior_mean : ℝ

/-- The math competition satisfies the given conditions -/
def satisfies_conditions (mc : MathCompetition) : Prop :=
  mc.total_students = 150 ∧
  mc.overall_mean = 85 ∧
  mc.juniors = mc.sophomores - (mc.sophomores / 5) ∧
  mc.sophomore_mean = mc.junior_mean * 1.25

/-- Theorem stating that under the given conditions, the sophomore mean score is 94 -/
theorem sophomore_mean_is_94 (mc : MathCompetition) 
  (h : satisfies_conditions mc) : mc.sophomore_mean = 94 := by
  sorry

#check sophomore_mean_is_94

end NUMINAMATH_CALUDE_sophomore_mean_is_94_l3631_363100


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3631_363164

/-- Given a hyperbola with focal length 2c = 26 and a²/c = 25/13, 
    its standard equation is x²/25 - y²/144 = 1 or y²/25 - x²/144 = 1 -/
theorem hyperbola_equation (c : ℝ) (a : ℝ) (h1 : 2 * c = 26) (h2 : a^2 / c = 25 / 13) :
  (∃ x y : ℝ, x^2 / 25 - y^2 / 144 = 1) ∨ (∃ x y : ℝ, y^2 / 25 - x^2 / 144 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3631_363164


namespace NUMINAMATH_CALUDE_divisibility_by_eighteen_l3631_363170

theorem divisibility_by_eighteen (n : ℕ) : 
  n ≤ 9 → 
  913 * 10 + n ≥ 1000 → 
  913 * 10 + n < 10000 → 
  (913 * 10 + n) % 18 = 0 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_eighteen_l3631_363170


namespace NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l3631_363198

/-- The star operation defined as a ☆ b = ab^2 - ab - 1 -/
def star (a b : ℝ) : ℝ := a * b^2 - a * b - 1

/-- Theorem stating that the equation 1 ☆ x = 0 has two distinct real roots -/
theorem star_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star 1 x = 0 ∧ star 1 y = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l3631_363198


namespace NUMINAMATH_CALUDE_cube_root_function_l3631_363124

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 125, 
    prove that y = 8√3/5 when x = 8 -/
theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → (k * x^(1/3) = 4 * Real.sqrt 3 ↔ x = 125)) → 
  k * 8^(1/3) = 8 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l3631_363124


namespace NUMINAMATH_CALUDE_compare_expressions_l3631_363193

theorem compare_expressions : -|(-5)| < -(-3) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3631_363193


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3631_363142

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 49 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 7 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3631_363142


namespace NUMINAMATH_CALUDE_relay_team_permutations_l3631_363103

theorem relay_team_permutations :
  (Finset.range 4).card.factorial = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l3631_363103


namespace NUMINAMATH_CALUDE_sum_base4_equals_l3631_363147

/-- Convert a base 4 number to its decimal representation -/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Convert a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Addition of two base 4 numbers -/
def addBase4 (a b : List Nat) : List Nat :=
  decimalToBase4 (base4ToDecimal a + base4ToDecimal b)

theorem sum_base4_equals : 
  addBase4 (addBase4 [3, 0, 2] [2, 1, 1]) [0, 3, 3] = [0, 1, 1, 3, 1] := by
  sorry


end NUMINAMATH_CALUDE_sum_base4_equals_l3631_363147


namespace NUMINAMATH_CALUDE_find_a_l3631_363130

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem find_a : ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3 ↔ x ∈ solution_set a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3631_363130


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3631_363160

def daily_differences : List Int := [15, -5, 25, -10, 5, 20, -15]

def days_in_week : Nat := 7

theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / days_in_week = 5 := by sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3631_363160


namespace NUMINAMATH_CALUDE_merchant_scale_problem_merchant_loss_l3631_363159

theorem merchant_scale_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  m / n + n / m > 2 :=
sorry

theorem merchant_loss (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  let x := n / m
  let y := m / n
  x + y > 2 :=
sorry

end NUMINAMATH_CALUDE_merchant_scale_problem_merchant_loss_l3631_363159


namespace NUMINAMATH_CALUDE_equation_solution_l3631_363104

theorem equation_solution : ∃ x : ℚ, (-2*x + 3 - 2*x + 3 = 3*x - 6) ∧ (x = 12/7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3631_363104


namespace NUMINAMATH_CALUDE_point_on_line_l3631_363125

/-- Given a line passing through point M(0, 1) with slope -1,
    prove that any point P(3, m) on this line satisfies m = -2 -/
theorem point_on_line (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P.1 = 3 ∧ P.2 = m ∧ 
   (m - 1) / (3 - 0) = -1) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l3631_363125


namespace NUMINAMATH_CALUDE_vector_operation_l3631_363158

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  2 • a - b = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3631_363158


namespace NUMINAMATH_CALUDE_y_axis_symmetry_l3631_363115

/-- Given a point P(2, 1), its symmetric point P' with respect to the y-axis has coordinates (-2, 1) -/
theorem y_axis_symmetry :
  let P : ℝ × ℝ := (2, 1)
  let P' : ℝ × ℝ := (-P.1, P.2)
  P' = (-2, 1) := by sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_l3631_363115


namespace NUMINAMATH_CALUDE_total_shingles_needed_l3631_363138

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  length : ℕ
  width : ℕ

/-- Represents a roof with two identical slanted sides and shingle requirement -/
structure Roof where
  side : RoofSide
  shingles_per_sqft : ℕ

/-- Calculates the number of shingles needed for a roof -/
def shingles_needed (roof : Roof) : ℕ :=
  2 * roof.side.length * roof.side.width * roof.shingles_per_sqft

/-- The three roofs in the problem -/
def roof_A : Roof := { side := { length := 20, width := 40 }, shingles_per_sqft := 8 }
def roof_B : Roof := { side := { length := 25, width := 35 }, shingles_per_sqft := 10 }
def roof_C : Roof := { side := { length := 30, width := 30 }, shingles_per_sqft := 12 }

/-- Theorem stating the total number of shingles needed for all three roofs -/
theorem total_shingles_needed :
  shingles_needed roof_A + shingles_needed roof_B + shingles_needed roof_C = 51900 := by
  sorry

end NUMINAMATH_CALUDE_total_shingles_needed_l3631_363138


namespace NUMINAMATH_CALUDE_existence_and_pigeonhole_l3631_363197

def is_pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem existence_and_pigeonhole :
  (∃ (S : Finset ℕ), S.card = 1328 ∧ S.toSet ⊆ Finset.range 1993 ∧
    ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → Nat.gcd (Nat.gcd a b) c > 1) ∧
  (∀ (T : Finset ℕ), T.card = 1329 → T.toSet ⊆ Finset.range 1993 →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_pairwise_coprime a b c) :=
sorry

end NUMINAMATH_CALUDE_existence_and_pigeonhole_l3631_363197


namespace NUMINAMATH_CALUDE_division_remainder_l3631_363157

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 4 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3631_363157


namespace NUMINAMATH_CALUDE_max_value_on_interval_l3631_363122

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ I ∧ f c = 6 ∧ ∀ x ∈ I, f x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l3631_363122


namespace NUMINAMATH_CALUDE_rocks_theorem_l3631_363102

def rocks_problem (initial_rocks : ℕ) (eaten_fraction : ℚ) (retrieved_rocks : ℕ) : Prop :=
  let remaining_after_eating := initial_rocks - (initial_rocks * eaten_fraction).floor
  let final_rocks := remaining_after_eating + retrieved_rocks
  initial_rocks = 10 ∧ eaten_fraction = 1/2 ∧ retrieved_rocks = 2 → final_rocks = 7

theorem rocks_theorem : rocks_problem 10 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_rocks_theorem_l3631_363102


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3631_363123

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  (n ≥ 3) → 
  (interior_angle = 144) → 
  (interior_angle = (n - 2) * 180 / n) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3631_363123


namespace NUMINAMATH_CALUDE_correct_subtraction_l3631_363109

theorem correct_subtraction (x : ℤ) (h1 : x - 32 = 25) (h2 : 23 ≠ 32) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l3631_363109


namespace NUMINAMATH_CALUDE_edward_final_earnings_l3631_363169

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l3631_363169


namespace NUMINAMATH_CALUDE_correct_additional_oil_l3631_363194

/-- The amount of oil needed per cylinder in ounces -/
def oil_per_cylinder : ℕ := 8

/-- The number of cylinders in George's car -/
def num_cylinders : ℕ := 6

/-- The amount of oil already added to the engine in ounces -/
def oil_already_added : ℕ := 16

/-- The additional amount of oil needed in ounces -/
def additional_oil_needed : ℕ := oil_per_cylinder * num_cylinders - oil_already_added

theorem correct_additional_oil : additional_oil_needed = 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_additional_oil_l3631_363194


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3631_363150

/-- The curve to which the line is tangent -/
def f (x : ℝ) : ℝ := x^2 * (x + 1)

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * x

/-- A line that passes through (3/5, 0) and is tangent to f at point t -/
def tangent_line (t : ℝ) (x : ℝ) : ℝ :=
  f' t * (x - t) + f t

/-- The point (3/5, 0) lies on the tangent line -/
def point_condition (t : ℝ) : Prop :=
  tangent_line t (3/5) = 0

/-- The possible equations for the tangent line -/
def possible_equations (x : ℝ) : Prop :=
  (∃ t, point_condition t ∧ tangent_line t x = 0) ∨
  (∃ t, point_condition t ∧ tangent_line t x = -3/2 * x + 9/125) ∨
  (∃ t, point_condition t ∧ tangent_line t x = 5 * x - 3)

theorem tangent_line_equations :
  ∀ x, possible_equations x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3631_363150


namespace NUMINAMATH_CALUDE_percentage_failed_both_subjects_l3631_363112

theorem percentage_failed_both_subjects 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (passed_both : Real) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : passed_both = 24) : 
  Real := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_both_subjects_l3631_363112


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l3631_363168

theorem johns_arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) :
  allowance = 3/2 →
  2/3 * (1 - arcade_fraction) * allowance = 2/5 →
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l3631_363168


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_36_l3631_363129

theorem log_216_equals_3_log_36 : Real.log 216 = 3 * Real.log 36 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_36_l3631_363129


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l3631_363153

/-- The value of k for which the line y = kx is tangent to the curve y = e^x -/
theorem tangent_line_to_exp_curve (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = Real.exp x₀ ∧ k = Real.exp x₀) → k = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l3631_363153


namespace NUMINAMATH_CALUDE_prob_at_least_one_three_l3631_363132

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows the target number -/
def non_target_outcomes : ℕ := (num_sides - 1) * (num_sides - 1)

/-- The probability of at least one die showing the target number -/
def prob_at_least_one_target : ℚ := (total_outcomes - non_target_outcomes) / total_outcomes

theorem prob_at_least_one_three :
  prob_at_least_one_target = 15 / 64 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_three_l3631_363132


namespace NUMINAMATH_CALUDE_total_training_hours_endurance_training_hours_l3631_363152

/-- Represents the training schedule for a goalkeeper --/
structure GoalkeeperSchedule where
  diving_catching : ℝ
  strength_conditioning : ℝ
  goalkeeper_specific : ℝ
  footwork : ℝ
  reaction_time : ℝ
  aerial_ball : ℝ
  shot_stopping : ℝ
  defensive_communication : ℝ
  game_simulation : ℝ
  endurance : ℝ

/-- Calculates the total training hours per week --/
def weekly_hours (s : GoalkeeperSchedule) : ℝ :=
  s.diving_catching + s.strength_conditioning + s.goalkeeper_specific +
  s.footwork + s.reaction_time + s.aerial_ball + s.shot_stopping +
  s.defensive_communication + s.game_simulation + s.endurance

/-- Mike's weekly training schedule --/
def mike_schedule : GoalkeeperSchedule :=
  { diving_catching := 2
  , strength_conditioning := 4
  , goalkeeper_specific := 2
  , footwork := 2
  , reaction_time := 1
  , aerial_ball := 3.5
  , shot_stopping := 1.5
  , defensive_communication := 1.5
  , game_simulation := 3
  , endurance := 3
  }

/-- The number of weeks Mike will train --/
def training_weeks : ℕ := 3

/-- Theorem: Mike's total training hours over 3 weeks is 70.5 --/
theorem total_training_hours :
  (weekly_hours mike_schedule) * training_weeks = 70.5 := by sorry

/-- Theorem: Mike's endurance training hours over 3 weeks is 9 --/
theorem endurance_training_hours :
  mike_schedule.endurance * training_weeks = 9 := by sorry

end NUMINAMATH_CALUDE_total_training_hours_endurance_training_hours_l3631_363152


namespace NUMINAMATH_CALUDE_zhuge_liang_army_count_l3631_363135

theorem zhuge_liang_army_count : 
  let n := 8
  let sum := n + n^2 + n^3 + n^4 + n^5 + n^6
  sum = (1 / 7) * (n^7 - n) := by
  sorry

end NUMINAMATH_CALUDE_zhuge_liang_army_count_l3631_363135


namespace NUMINAMATH_CALUDE_leo_current_weight_l3631_363167

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l3631_363167


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l3631_363177

theorem positive_integer_solutions_of_equation :
  {(x, y) : ℕ × ℕ | 2 * x^2 - 7 * x * y + 3 * y^3 = 0 ∧ x > 0 ∧ y > 0} =
  {(3, 1), (3, 2), (4, 2)} :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l3631_363177


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l3631_363126

theorem modulus_of_complex_quotient :
  let z : ℂ := (1 - Complex.I) / (3 + 4 * Complex.I)
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l3631_363126


namespace NUMINAMATH_CALUDE_brown_mushrooms_count_l3631_363175

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms -/
def total_white_spotted : ℕ := 17

theorem brown_mushrooms_count :
  brown_mushrooms = 6 :=
by
  have h1 : (blue_mushrooms / 2 : ℕ) + (2 * red_mushrooms / 3 : ℕ) + brown_mushrooms = total_white_spotted :=
    sorry
  sorry

end NUMINAMATH_CALUDE_brown_mushrooms_count_l3631_363175


namespace NUMINAMATH_CALUDE_min_m_plus_n_l3631_363107

theorem min_m_plus_n (m n : ℕ+) (h : 90 * m = n^3) : 
  ∃ (m' n' : ℕ+), 90 * m' = n'^3 ∧ m' + n' ≤ m + n ∧ m' + n' = 120 :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l3631_363107


namespace NUMINAMATH_CALUDE_new_books_count_l3631_363134

def adventure_books : ℕ := 13
def mystery_books : ℕ := 17
def used_books : ℕ := 15

def total_books : ℕ := adventure_books + mystery_books

theorem new_books_count : total_books - used_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_books_count_l3631_363134


namespace NUMINAMATH_CALUDE_orange_ribbons_l3631_363140

theorem orange_ribbons (total : ℚ) (black : ℕ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + black = total →
  black = 40 →
  (1/6 : ℚ) * total = 80/3 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l3631_363140


namespace NUMINAMATH_CALUDE_animals_left_in_barn_l3631_363179

theorem animals_left_in_barn (pigs cows sold : ℕ) 
  (h1 : pigs = 156)
  (h2 : cows = 267)
  (h3 : sold = 115) :
  pigs + cows - sold = 308 :=
by sorry

end NUMINAMATH_CALUDE_animals_left_in_barn_l3631_363179


namespace NUMINAMATH_CALUDE_conference_attendees_l3631_363178

theorem conference_attendees (total : ℕ) (writers : ℕ) (editors : ℕ) (both : ℕ) 
    (h1 : total = 100)
    (h2 : writers = 40)
    (h3 : editors ≥ 39)
    (h4 : both ≤ 21) :
  total - (writers + editors - both) ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_conference_attendees_l3631_363178


namespace NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l3631_363136

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions) 
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 2 * funcs.g x) = 5) :
  quadratic_min (λ x => (funcs.g x)^2 + 2 * funcs.f x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l3631_363136


namespace NUMINAMATH_CALUDE_complex_moduli_product_l3631_363187

theorem complex_moduli_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l3631_363187


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_or_neg_six_l3631_363185

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem -/
theorem lines_parallel_iff_m_eq_one_or_neg_six (m : ℝ) :
  let l1 : Line2D := ⟨m, 3, -6⟩
  let l2 : Line2D := ⟨2, 5 + m, 2⟩
  parallel l1 l2 ↔ m = 1 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_or_neg_six_l3631_363185


namespace NUMINAMATH_CALUDE_square_area_difference_l3631_363116

theorem square_area_difference (area_A : ℝ) (side_diff : ℝ) : 
  area_A = 25 → side_diff = 4 → 
  let side_A := Real.sqrt area_A
  let side_B := side_A + side_diff
  side_B ^ 2 = 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_difference_l3631_363116


namespace NUMINAMATH_CALUDE_lines_are_parallel_l3631_363146

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem lines_are_parallel (l1 l2 : Line) 
  (h1 : l1 = { slope := 2, intercept := 1 })
  (h2 : l2 = { slope := 2, intercept := 5 }) : 
  parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l3631_363146


namespace NUMINAMATH_CALUDE_circle_radius_from_intersecting_chords_l3631_363151

theorem circle_radius_from_intersecting_chords (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  ∃ (r : ℝ),
    (r = (a/d) * Real.sqrt (a^2 + b^2 - 2*b * Real.sqrt (a^2 - d^2))) ∨
    (r = (a/d) * Real.sqrt (a^2 + b^2 + 2*b * Real.sqrt (a^2 - d^2))) ∨
    (a = d ∧ r = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_intersecting_chords_l3631_363151


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3631_363133

theorem sqrt_product_equality (x : ℝ) : 
  Real.sqrt (x * (x - 6)) = Real.sqrt x * Real.sqrt (x - 6) → x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3631_363133


namespace NUMINAMATH_CALUDE_expression_evaluation_l3631_363174

theorem expression_evaluation : 5 + 7 * (2 + 1/4) = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3631_363174


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_and_minimum_value_l3631_363149

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_and_inequality_and_minimum_value :
  -- 1. The tangent line to f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. f(x) ≥ x - 1 for all x > 0
  (∀ x > 0, f x ≥ x - 1) ∧
  -- 3. The minimum value of a such that f(x) ≥ ax² + 2/a for all x > 0 and a ≠ 0 is -e³
  (∀ a ≠ 0, (∀ x > 0, f x ≥ a * x^2 + 2/a) ↔ a ≥ -Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_and_minimum_value_l3631_363149


namespace NUMINAMATH_CALUDE_anna_rearrangement_time_l3631_363113

def name : String := "Anna"
def letters : ℕ := 4
def repetitions : List ℕ := [2, 2]  -- 'A' repeated twice, 'N' repeated twice
def rearrangements_per_minute : ℕ := 8

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_rearrangements : ℕ :=
  factorial letters / (factorial repetitions[0]! * factorial repetitions[1]!)

def time_in_minutes : ℚ :=
  total_rearrangements / rearrangements_per_minute

theorem anna_rearrangement_time :
  time_in_minutes / 60 = 0.0125 := by sorry

end NUMINAMATH_CALUDE_anna_rearrangement_time_l3631_363113


namespace NUMINAMATH_CALUDE_min_value_of_E_l3631_363173

/-- Given that the minimum value of |x - 4| + |E| + |x - 5| is 11,
    prove that the minimum value of |E| is 10. -/
theorem min_value_of_E (E : ℝ) :
  (∃ (c : ℝ), ∀ (x : ℝ), c ≤ |x - 4| + |E| + |x - 5| ∧ 
   ∃ (x : ℝ), c = |x - 4| + |E| + |x - 5|) →
  (c = 11) →
  (∃ (d : ℝ), ∀ (y : ℝ), d ≤ |y| ∧ 
   ∃ (y : ℝ), d = |y|) →
  (d = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_E_l3631_363173


namespace NUMINAMATH_CALUDE_orange_ribbons_l3631_363143

theorem orange_ribbons (total : ℚ) 
  (yellow_frac : ℚ) (purple_frac : ℚ) (orange_frac : ℚ) (black_count : ℕ) :
  yellow_frac = 1/3 →
  purple_frac = 1/4 →
  orange_frac = 1/6 →
  black_count = 40 →
  (1 - yellow_frac - purple_frac - orange_frac) * total = black_count →
  orange_frac * total = 80/3 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l3631_363143


namespace NUMINAMATH_CALUDE_f_500_equals_39_l3631_363120

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ) : Prop :=
  (∀ x y : ℕ+, f (x * y) = f x + f y) ∧ 
  (f 10 = 14) ∧ 
  (f 40 = 20)

/-- Theorem stating the result for f(500) -/
theorem f_500_equals_39 (f : ℕ+ → ℕ) (h : special_function f) : f 500 = 39 := by
  sorry

end NUMINAMATH_CALUDE_f_500_equals_39_l3631_363120


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3631_363195

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3631_363195


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3631_363118

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ (n : ℕ), a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 7 + a 5 * a 9 = 16)
  (h_mean : (a 5 + a 9) / 2 = 4) :
  ∃ (q : ℝ), q > 0 ∧ (∀ (n : ℕ), a (n + 1) = q * a n) ∧ q = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3631_363118


namespace NUMINAMATH_CALUDE_sector_central_angle_l3631_363196

theorem sector_central_angle (R : ℝ) (α : ℝ) 
  (h1 : 2 * R + α * R = 6)  -- circumference of sector
  (h2 : 1/2 * R^2 * α = 2)  -- area of sector
  : α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3631_363196


namespace NUMINAMATH_CALUDE_total_questions_submitted_l3631_363105

/-- Given the ratio of questions submitted by Rajat, Vikas, and Abhishek,
    and the number of questions submitted by Vikas, calculate the total
    number of questions submitted. -/
theorem total_questions_submitted
  (ratio_rajat : ℕ)
  (ratio_vikas : ℕ)
  (ratio_abhishek : ℕ)
  (vikas_questions : ℕ)
  (h_ratio : ratio_rajat = 7 ∧ ratio_vikas = 3 ∧ ratio_abhishek = 2)
  (h_vikas : vikas_questions = 6) :
  ratio_rajat * vikas_questions / ratio_vikas +
  vikas_questions +
  ratio_abhishek * vikas_questions / ratio_vikas = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_questions_submitted_l3631_363105


namespace NUMINAMATH_CALUDE_or_not_implies_other_l3631_363137

theorem or_not_implies_other (p q : Prop) : (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_or_not_implies_other_l3631_363137


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3631_363145

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (x^2 - 3*x + 2) > x + 5 ↔ x < -23/13 := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3631_363145


namespace NUMINAMATH_CALUDE_lcm_of_150_and_456_l3631_363144

theorem lcm_of_150_and_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_150_and_456_l3631_363144


namespace NUMINAMATH_CALUDE_circle_equation_l3631_363128

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- Tangent line -/
def tangent_line (y : ℝ) : Prop :=
  y = 0

/-- Possible equations of the sought circle -/
def sought_circle (x y : ℝ) : Prop :=
  ((x - 2 - 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 10)^2 + (y - 4)^2 = 16) ∨
  ((x - 2 - 2*Real.sqrt 6)^2 + (y + 4)^2 = 16) ∨
  ((x - 2 + 2*Real.sqrt 6)^2 + (y + 4)^2 = 16)

/-- Theorem stating the properties of the sought circle -/
theorem circle_equation :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = 16) ∧
    (∃ (x y : ℝ), given_circle x y ∧ (x - a)^2 + (y - b)^2 = 36) ∧
    (∃ y : ℝ, tangent_line y ∧ (a - a)^2 + (y - b)^2 = 16) →
    sought_circle a b :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3631_363128


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_equal_intercept_line_standard_form_l3631_363181

/-- A line passing through point (-3, 4) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, 4) -/
  passes_through_point : slope * (-3) + y_intercept = 4
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept

/-- The equation of the line is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = -4/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 1) := by
  sorry

/-- The line equation in standard form is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_standard_form (l : EqualInterceptLine) :
  (∃ (k : ℝ), k ≠ 0 ∧ 4*k*l.slope + 3*k = 0 ∧ k*l.y_intercept = 0) ∨
  (∃ (k : ℝ), k ≠ 0 ∧ k*l.slope + k = 0 ∧ k*l.y_intercept = k) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_equal_intercept_line_standard_form_l3631_363181


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3631_363171

/-- A linear function y = -5ax + b with specific properties -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  increasing : ∀ x₁ x₂, x₁ < x₂ → (-5 * a * x₁ + b) < (-5 * a * x₂ + b)
  ab_positive : a * b > 0

/-- The point A(a, b) -/
def point_A (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- Third quadrant definition -/
def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem stating that point A lies in the third quadrant -/
theorem point_A_in_third_quadrant (f : LinearFunction) :
  third_quadrant (point_A f) := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3631_363171


namespace NUMINAMATH_CALUDE_function_inequality_l3631_363154

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem function_inequality (h1 : IsEven f) (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3631_363154


namespace NUMINAMATH_CALUDE_cryptarithm_solution_is_unique_l3631_363117

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  F : Nat
  R : Nat
  Y : Nat
  H : Nat
  A : Nat
  M : Nat
  digit_constraint : F < 10 ∧ R < 10 ∧ Y < 10 ∧ H < 10 ∧ A < 10 ∧ M < 10
  unique_digits : F ≠ R ∧ F ≠ Y ∧ F ≠ H ∧ F ≠ A ∧ F ≠ M ∧
                  R ≠ Y ∧ R ≠ H ∧ R ≠ A ∧ R ≠ M ∧
                  Y ≠ H ∧ Y ≠ A ∧ Y ≠ M ∧
                  H ≠ A ∧ H ≠ M ∧
                  A ≠ M
  equation_holds : 7 * (100000 * F + 10000 * R + 1000 * Y + 100 * H + 10 * A + M) =
                   6 * (100000 * H + 10000 * A + 1000 * M + 100 * F + 10 * R + Y)

theorem cryptarithm_solution_is_unique : 
  ∀ (sol : CryptarithmSolution), 
    100 * sol.F + 10 * sol.R + sol.Y = 461 ∧ 
    100 * sol.H + 10 * sol.A + sol.M = 538 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_is_unique_l3631_363117


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3631_363114

theorem cube_equation_solution : ∃! x : ℝ, (x - 3)^3 = 27 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3631_363114


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l3631_363101

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l3631_363101


namespace NUMINAMATH_CALUDE_writing_ways_equals_notebooks_l3631_363119

/-- The number of ways to start writing given a ratio of pens to notebooks and their quantities -/
def ways_to_start_writing (pen_ratio : ℕ) (notebook_ratio : ℕ) (num_pens : ℕ) (num_notebooks : ℕ) : ℕ :=
  min num_pens num_notebooks

/-- Theorem: Given the ratio of pens to notebooks is 5:4, with 50 pens and 40 notebooks,
    the number of ways to start writing is equal to the number of notebooks -/
theorem writing_ways_equals_notebooks :
  ways_to_start_writing 5 4 50 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_writing_ways_equals_notebooks_l3631_363119


namespace NUMINAMATH_CALUDE_jackson_souvenir_collection_l3631_363108

/-- Proves that given the conditions in Jackson's souvenir collection, 
    the number of starfish per spiral shell is 2. -/
theorem jackson_souvenir_collection 
  (hermit_crabs : ℕ) 
  (shells_per_crab : ℕ) 
  (total_souvenirs : ℕ) 
  (h1 : hermit_crabs = 45)
  (h2 : shells_per_crab = 3)
  (h3 : total_souvenirs = 450) :
  (total_souvenirs - hermit_crabs - hermit_crabs * shells_per_crab) / (hermit_crabs * shells_per_crab) = 2 :=
by sorry

end NUMINAMATH_CALUDE_jackson_souvenir_collection_l3631_363108


namespace NUMINAMATH_CALUDE_digital_earth_functions_l3631_363131

/-- Represents the Digital Earth system -/
structure DigitalEarth where
  -- Define properties of Digital Earth
  is_huge : Bool
  is_precise : Bool
  is_digital_representation : Bool
  is_information_repository : Bool

/-- Functions that Digital Earth can perform -/
inductive DigitalEarthFunction
  | JointResearch
  | GlobalEducation
  | CrimeTracking
  | SustainableDevelopment

/-- Theorem stating that Digital Earth supports all four functions -/
theorem digital_earth_functions (de : DigitalEarth) : 
  (de.is_huge ∧ de.is_precise ∧ de.is_digital_representation ∧ de.is_information_repository) →
  (∀ f : DigitalEarthFunction, f ∈ [DigitalEarthFunction.JointResearch, 
                                    DigitalEarthFunction.GlobalEducation, 
                                    DigitalEarthFunction.CrimeTracking, 
                                    DigitalEarthFunction.SustainableDevelopment]) :=
by
  sorry


end NUMINAMATH_CALUDE_digital_earth_functions_l3631_363131


namespace NUMINAMATH_CALUDE_investment_growth_rate_l3631_363192

def annual_growth_rate (growth_rate : ℝ) (compounding_periods : ℕ) : ℝ :=
  ((growth_rate ^ (1 / compounding_periods)) ^ compounding_periods - 1) * 100

theorem investment_growth_rate 
  (P : ℝ) 
  (t : ℕ) 
  (h1 : P > 0) 
  (h2 : 1 ≤ t ∧ t ≤ 5) : 
  annual_growth_rate 1.20 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_investment_growth_rate_l3631_363192


namespace NUMINAMATH_CALUDE_amount_lent_to_C_is_correct_l3631_363190

/-- The amount of money A lent to C -/
def amount_lent_to_C : ℝ := 500

/-- The amount of money A lent to B -/
def amount_lent_to_B : ℝ := 5000

/-- The duration of the loan to B in years -/
def duration_B : ℝ := 2

/-- The duration of the loan to C in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.1

/-- The total interest received from both B and C -/
def total_interest : ℝ := 2200

theorem amount_lent_to_C_is_correct :
  amount_lent_to_C * interest_rate * duration_C +
  amount_lent_to_B * interest_rate * duration_B = total_interest :=
sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_is_correct_l3631_363190

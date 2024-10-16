import Mathlib

namespace NUMINAMATH_CALUDE_triangle_tangent_difference_bound_l3833_383324

theorem triangle_tangent_difference_bound (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a*c →  -- Given condition
  1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_difference_bound_l3833_383324


namespace NUMINAMATH_CALUDE_winning_post_at_200m_l3833_383381

/-- Two runners A and B, where A is faster than B but gives B a head start -/
structure RaceScenario where
  /-- The speed ratio of runner A to runner B -/
  speed_ratio : ℚ
  /-- The head start given to runner B in meters -/
  head_start : ℚ

/-- The winning post distance for two runners to arrive simultaneously -/
def winning_post_distance (scenario : RaceScenario) : ℚ :=
  (scenario.speed_ratio * scenario.head_start) / (scenario.speed_ratio - 1)

/-- Theorem stating that for the given scenario, the winning post distance is 200 meters -/
theorem winning_post_at_200m (scenario : RaceScenario) 
  (h1 : scenario.speed_ratio = 5/3)
  (h2 : scenario.head_start = 80) :
  winning_post_distance scenario = 200 := by
  sorry

end NUMINAMATH_CALUDE_winning_post_at_200m_l3833_383381


namespace NUMINAMATH_CALUDE_john_learning_alphabets_l3833_383359

/-- The number of alphabets John is learning in the first group -/
def alphabets_learned : ℕ := 15 / 3

/-- The number of days it takes John to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning the alphabets -/
def total_days : ℕ := 15

theorem john_learning_alphabets :
  alphabets_learned = 5 :=
by sorry

end NUMINAMATH_CALUDE_john_learning_alphabets_l3833_383359


namespace NUMINAMATH_CALUDE_curve_identification_l3833_383399

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := (ρ - 1) * (θ - Real.pi) = 0 ∧ ρ ≥ 0

-- Define the parametric equations
def parametric_equations (x y θ : ℝ) : Prop := x = Real.tan θ ∧ y = 2 / Real.cos θ

-- Theorem statement
theorem curve_identification :
  (∃ (x y : ℝ), x^2 + y^2 = 1) ∧  -- Circle
  (∃ (x y : ℝ), x < 0 ∧ y = 0) ∧  -- Ray
  (∃ (x y : ℝ), y^2 - 4*x^2 = 4)  -- Hyperbola
  :=
sorry

end NUMINAMATH_CALUDE_curve_identification_l3833_383399


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l3833_383325

/-- Two cyclists meet at the starting point on a circular track -/
theorem cyclists_meet_time (circumference : ℝ) (speed1 speed2 : ℝ) 
  (h_circumference : circumference = 600)
  (h_speed1 : speed1 = 7)
  (h_speed2 : speed2 = 8) :
  (circumference / (speed1 + speed2)) = 40 := by
  sorry

#check cyclists_meet_time

end NUMINAMATH_CALUDE_cyclists_meet_time_l3833_383325


namespace NUMINAMATH_CALUDE_min_distance_intersection_points_l3833_383326

open Real

theorem min_distance_intersection_points (a : ℝ) :
  let f (x : ℝ) := (x - exp x - 3) / 2
  ∃ (x₁ x₂ : ℝ), a = 2 * x₁ - 3 ∧ a = x₂ + exp x₂ ∧ 
    ∀ (y₁ y₂ : ℝ), a = 2 * y₁ - 3 → a = y₂ + exp y₂ → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧ |x₂ - x₁| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_intersection_points_l3833_383326


namespace NUMINAMATH_CALUDE_cubic_factorization_l3833_383388

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3833_383388


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l3833_383302

theorem parkway_elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (boys_playing_soccer_percentage : ℚ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : playing_soccer = 250)
  (h4 : boys_playing_soccer_percentage = 86 / 100)
  : ℕ := by
  sorry

#check parkway_elementary_girls_not_playing_soccer

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l3833_383302


namespace NUMINAMATH_CALUDE_vanya_masha_speed_ratio_l3833_383395

/-- Represents the scenario of Vanya and Masha's journey to school -/
structure SchoolJourney where
  d : ℝ  -- Total distance from home to school
  vanya_speed : ℝ  -- Vanya's speed
  masha_speed : ℝ  -- Masha's speed

/-- The theorem stating the relationship between Vanya and Masha's speeds -/
theorem vanya_masha_speed_ratio (journey : SchoolJourney) :
  journey.d > 0 →  -- Ensure the distance is positive
  (2/3 * journey.d) / journey.vanya_speed = (1/6 * journey.d) / journey.masha_speed →  -- Condition from overtaking point
  (1/2 * journey.d) / journey.masha_speed = journey.d / journey.vanya_speed →  -- Condition when Vanya reaches school
  journey.vanya_speed / journey.masha_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_vanya_masha_speed_ratio_l3833_383395


namespace NUMINAMATH_CALUDE_painting_discount_l3833_383385

theorem painting_discount (x : ℝ) (h1 : x / 5 = 15) : x * (1 - 1/3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_painting_discount_l3833_383385


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l3833_383377

theorem missing_digit_divisible_by_nine : ∃ (x : ℕ), 
  x < 10 ∧ 
  (346000 + x * 100 + 92) % 9 = 0 ∧ 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l3833_383377


namespace NUMINAMATH_CALUDE_reading_time_is_fifty_l3833_383319

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total time in hours needed to read the book -/
def reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_is_fifty : reading_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_is_fifty_l3833_383319


namespace NUMINAMATH_CALUDE_problem_2012_l3833_383305

theorem problem_2012 (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (eq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (eq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c*d)^2012 - (a*b)^2012 = 2011 := by
sorry

end NUMINAMATH_CALUDE_problem_2012_l3833_383305


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3833_383315

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 18) -- second term is 18
  (h2 : a * r^2 = 24) -- third term is 24
  : a = 27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3833_383315


namespace NUMINAMATH_CALUDE_correct_calculation_l3833_383321

theorem correct_calculation (y : ℝ) : -8 * y + 3 * y = -5 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3833_383321


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_equals_one_l3833_383300

theorem intersection_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, 2, 5}
  let B : Set ℝ := {a + 4, a}
  A ∩ B = B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_equals_one_l3833_383300


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l3833_383362

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_zero :
  (lg 2)^2 + (lg 2) * (lg 5) + lg 5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  -- Assume the given condition
  have h : lg 2 + lg 5 = 1 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l3833_383362


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_characterization_l3833_383360

/-- A line passing through (1, 3) with equal x and y intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1, 3) -/
  point_condition : 3 = m * 1 + b
  /-- The line has equal x and y intercepts -/
  equal_intercepts : b ≠ 0 → -b / m = b

/-- The equation of a line with equal intercepts passing through (1, 3) -/
def equal_intercept_line_equation (l : EqualInterceptLine) : Prop :=
  (l.m = 3 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 4)

/-- Theorem stating that a line with equal intercepts passing through (1, 3) 
    must have the equation 3x - y = 0 or x + y - 4 = 0 -/
theorem equal_intercept_line_equation_characterization (l : EqualInterceptLine) :
  equal_intercept_line_equation l := by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_characterization_l3833_383360


namespace NUMINAMATH_CALUDE_percentage_relation_l3833_383382

/-- Given the relationships between j, k, l, and m, prove that 150% of k equals 50% of l -/
theorem percentage_relation (j k l m : ℝ) : 
  (1.25 * j = 0.25 * k) →
  (∃ x : ℝ, 0.01 * x * k = 0.5 * l) →
  (1.75 * l = 0.75 * m) →
  (0.2 * m = 7 * j) →
  1.5 * k = 0.5 * l := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3833_383382


namespace NUMINAMATH_CALUDE_line_through_circle_center_l3833_383356

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*x' - 4*y' = 0 → 
   (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l3833_383356


namespace NUMINAMATH_CALUDE_inequality_solution_l3833_383397

theorem inequality_solution (x : ℝ) : 
  (x^(1/4) + 3 / (x^(1/4) + 4) ≤ 1) ↔ 
  (x < (((-3 - Real.sqrt 5) / 2)^4) ∨ x > (((-3 + Real.sqrt 5) / 2)^4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3833_383397


namespace NUMINAMATH_CALUDE_supermarket_pricing_problem_l3833_383383

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 60

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - 10) * (sales_function x)

theorem supermarket_pricing_problem :
  -- 1. The linear function satisfies the given data points
  (sales_function 12 = 36 ∧ sales_function 13 = 34) ∧
  -- 2. When the profit is 192 yuan, the selling price is 18 yuan
  (profit_function 18 = 192) ∧
  -- 3. The maximum profit is 198 yuan when the selling price is 19 yuan, given the constraints
  (∀ x : ℝ, 10 ≤ x ∧ x ≤ 19 → profit_function x ≤ profit_function 19) ∧
  (profit_function 19 = 198) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_pricing_problem_l3833_383383


namespace NUMINAMATH_CALUDE_total_animals_l3833_383354

theorem total_animals (num_pigs num_giraffes : ℕ) : 
  num_pigs = 7 → num_giraffes = 6 → num_pigs + num_giraffes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l3833_383354


namespace NUMINAMATH_CALUDE_line_slope_l3833_383372

def curve (x y : ℝ) : Prop := 5 * y = 2 * x^2 - 9 * x + 10

def line_through_origin (k x y : ℝ) : Prop := y = k * x

theorem line_slope (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line_through_origin k x₁ y₁ ∧
    line_through_origin k x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 77) →
  k = 29 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l3833_383372


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3833_383358

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem: The y-intercept of the tangent line to f(x) at x = 1 is (0, 3)
theorem tangent_line_y_intercept :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  b = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3833_383358


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l3833_383355

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3)
  (hn : |n| = 2)
  (hmn : m < n) :
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l3833_383355


namespace NUMINAMATH_CALUDE_h_is_even_l3833_383365

-- Define k as an even function
def k_even (k : ℝ → ℝ) : Prop :=
  ∀ x, k (-x) = k x

-- Define h using k
def h (k : ℝ → ℝ) (x : ℝ) : ℝ :=
  |k (x^5)|

-- Theorem statement
theorem h_is_even (k : ℝ → ℝ) (h_even : k_even k) :
  ∀ x, h k (-x) = h k x :=
by sorry

end NUMINAMATH_CALUDE_h_is_even_l3833_383365


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l3833_383368

theorem existence_of_n_with_k_prime_factors 
  (k : Nat) 
  (m : Nat) 
  (hk : k ≠ 0) 
  (hm : Odd m) :
  ∃ n : Nat, (Nat.factors (m^n + n^m)).card ≥ k :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l3833_383368


namespace NUMINAMATH_CALUDE_smallest_n_doughnuts_l3833_383310

theorem smallest_n_doughnuts : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m.val - 1) % 11 = 0 → n ≤ m) ∧
  (15 * n.val - 1) % 11 = 0 ∧
  n.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_doughnuts_l3833_383310


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_l3833_383384

theorem sum_of_imaginary_parts (a c d e f : ℂ) : 
  (a + 2*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 4*Complex.I →
  e = -2*a - c →
  d + f = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_l3833_383384


namespace NUMINAMATH_CALUDE_divisibility_condition_l3833_383301

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3833_383301


namespace NUMINAMATH_CALUDE_land_allocation_equations_l3833_383323

/-- Represents the land allocation problem for tea gardens and grain fields. -/
theorem land_allocation_equations (total_area : ℝ) (vegetable_percentage : ℝ) 
  (tea_grain_area : ℝ) (tea_area : ℝ) (grain_area : ℝ) : 
  total_area = 60 ∧ 
  vegetable_percentage = 0.1 ∧ 
  tea_grain_area = total_area - vegetable_percentage * total_area ∧
  tea_area = 2 * grain_area - 3 →
  tea_area + grain_area = 54 ∧ tea_area = 2 * grain_area - 3 :=
by sorry

end NUMINAMATH_CALUDE_land_allocation_equations_l3833_383323


namespace NUMINAMATH_CALUDE_intersection_points_on_line_l3833_383386

/-- The slope of the line containing all intersection points of the given parametric lines -/
def intersection_line_slope : ℚ := 10/31

/-- The first line equation: 2x + 3y = 8u + 4 -/
def line1 (u x y : ℝ) : Prop := 2*x + 3*y = 8*u + 4

/-- The second line equation: 3x - 2y = 5u - 3 -/
def line2 (u x y : ℝ) : Prop := 3*x - 2*y = 5*u - 3

/-- The theorem stating that all intersection points lie on a line with slope 10/31 -/
theorem intersection_points_on_line :
  ∀ (u x y : ℝ), line1 u x y → line2 u x y →
  ∃ (k b : ℝ), y = intersection_line_slope * x + b :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_line_l3833_383386


namespace NUMINAMATH_CALUDE_expression_simplification_l3833_383333

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 4) - 3 * p * 3) * 2 + (5 - 2 / 4) * (4 * p - 6) = 14 * p - 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3833_383333


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3833_383342

/-- Represents a hyperbola with its asymptotic equation coefficient -/
structure Hyperbola where
  k : ℝ
  asymptote_eq : ∀ (x y : ℝ), y = k * x ∨ y = -k * x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (h_asymptote : h.k = 1/2) :
  (eccentricity h = Real.sqrt 5 / 2) ∨ (eccentricity h = Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3833_383342


namespace NUMINAMATH_CALUDE_flag_arrangement_modulo_l3833_383352

/-- The number of distinguishable arrangements of flags on two poles -/
def M : ℕ :=
  let total_flags := 17
  let blue_flags := 9
  let red_flags := 8
  let slots_for_red := blue_flags + 1
  let ways_to_place_red := Nat.choose slots_for_red red_flags
  let initial_arrangements := (blue_flags + 1) * ways_to_place_red
  let invalid_cases := 2 * Nat.choose blue_flags red_flags
  initial_arrangements - invalid_cases

/-- Theorem stating that M is congruent to 432 modulo 1000 -/
theorem flag_arrangement_modulo :
  M % 1000 = 432 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_modulo_l3833_383352


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l3833_383348

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

theorem min_sum_reciprocal_constraint_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : 
  (x + y = 9) ↔ (x = 3 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l3833_383348


namespace NUMINAMATH_CALUDE_circular_binary_arrangement_l3833_383379

/-- A type representing a binary number using only 1 and 2 -/
def BinaryNumber (n : ℕ) := Fin n → Fin 2

/-- A function to check if two binary numbers differ by exactly one digit -/
def differByOneDigit (n : ℕ) (a b : BinaryNumber n) : Prop :=
  ∃! i : Fin n, a i ≠ b i

/-- A type representing an arrangement of binary numbers in a circle -/
def CircularArrangement (n : ℕ) := Fin (2^n) → BinaryNumber n

/-- The main theorem statement -/
theorem circular_binary_arrangement (n : ℕ) :
  ∃ (arrangement : CircularArrangement n),
    (∀ i j : Fin (2^n), i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i : Fin (2^n), differByOneDigit n (arrangement i) (arrangement (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_circular_binary_arrangement_l3833_383379


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3833_383393

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3833_383393


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l3833_383387

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 5)^2 + log10 2 * log10 5 + log10 20 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l3833_383387


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3833_383367

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (a' b' c' : ℕ+), 
    (∃ (k' : ℚ), k' * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (a' * Real.sqrt 6 + b' * Real.sqrt 8) / c') →
    c ≤ c') →
  a.val + b.val + c.val = 106 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3833_383367


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l3833_383306

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

theorem complement_of_A_relative_to_U :
  {x ∈ U | x ∉ A} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l3833_383306


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3833_383328

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 87) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3833_383328


namespace NUMINAMATH_CALUDE_marbles_left_l3833_383311

def initial_marbles : ℕ := 38
def lost_marbles : ℕ := 15

theorem marbles_left : initial_marbles - lost_marbles = 23 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l3833_383311


namespace NUMINAMATH_CALUDE_sisters_gift_l3833_383341

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def additional_money_needed : ℕ := 3214

def job_earnings : ℕ := hourly_wage * hours_worked
def cookie_earnings : ℕ := cookie_price * cookies_sold
def total_earnings : ℕ := job_earnings + cookie_earnings - lottery_ticket_cost + lottery_winnings

theorem sisters_gift (sisters_gift : ℕ) : sisters_gift = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_sisters_gift_l3833_383341


namespace NUMINAMATH_CALUDE_maryville_population_increase_l3833_383351

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating the average annual population increase in Maryville between 2000 and 2005 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l3833_383351


namespace NUMINAMATH_CALUDE_motorcycle_car_profit_difference_l3833_383374

/-- Represents the production and sales data for a vehicle type -/
structure VehicleProduction where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle production -/
def profit (v : VehicleProduction) : ℤ :=
  (v.quantity * v.price : ℤ) - v.materialCost

/-- Proves that the difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_car_profit_difference 
  (car : VehicleProduction)
  (motorcycle : VehicleProduction)
  (h_car : car = { materialCost := 100, quantity := 4, price := 50 })
  (h_motorcycle : motorcycle = { materialCost := 250, quantity := 8, price := 50 }) :
  profit motorcycle - profit car = 50 := by
  sorry

#eval profit { materialCost := 250, quantity := 8, price := 50 } - 
      profit { materialCost := 100, quantity := 4, price := 50 }

end NUMINAMATH_CALUDE_motorcycle_car_profit_difference_l3833_383374


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3833_383307

-- Define a regular pentagon
def RegularPentagon : Type := Unit

-- Define the function to calculate the sum of interior angles of a polygon
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

-- Theorem: The sum of interior angles of a regular pentagon is 540°
theorem sum_interior_angles_pentagon (p : RegularPentagon) :
  sumInteriorAngles 5 = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3833_383307


namespace NUMINAMATH_CALUDE_train_passing_time_l3833_383363

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 180 →
  train_speed_kmh = 36 →
  passing_time = 18 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3833_383363


namespace NUMINAMATH_CALUDE_decreasing_reciprocal_function_l3833_383366

theorem decreasing_reciprocal_function 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, x > 0 → f x = 1 / x) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
sorry

end NUMINAMATH_CALUDE_decreasing_reciprocal_function_l3833_383366


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3833_383308

theorem rectangle_area_problem (A : ℝ) : 
  let square_side : ℝ := 12
  let new_horizontal : ℝ := square_side + 3
  let new_vertical : ℝ := square_side - A
  let new_area : ℝ := 120
  new_horizontal * new_vertical = new_area → A = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3833_383308


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3833_383350

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3833_383350


namespace NUMINAMATH_CALUDE_no_sum_equal_powers_l3833_383339

theorem no_sum_equal_powers : ¬∃ (n m : ℕ), n * (n + 1) / 2 = 2^m + 3^m := by
  sorry

end NUMINAMATH_CALUDE_no_sum_equal_powers_l3833_383339


namespace NUMINAMATH_CALUDE_scientific_notation_1200_l3833_383313

theorem scientific_notation_1200 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1200 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1200_l3833_383313


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3833_383309

/-- Calculates the length of a platform given the length of a train, its speed, and the time it takes to cross the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 160) 
  (h2 : train_speed_kmph = 72) 
  (h3 : crossing_time = 25) : ℝ :=
let train_speed_mps := train_speed_kmph * (1000 / 3600)
let total_distance := train_speed_mps * crossing_time
let platform_length := total_distance - train_length
340

theorem platform_length_proof : platform_length 160 72 25 rfl rfl rfl = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3833_383309


namespace NUMINAMATH_CALUDE_smallest_square_sides_l3833_383320

/-- Represents the configuration of three squares arranged as described in the problem -/
structure SquareArrangement where
  small_side : ℝ
  mid_side : ℝ
  large_side : ℝ
  mid_is_larger : mid_side = small_side + 8
  large_is_50 : large_side = 50

/-- The theorem stating the possible side lengths of the smallest square -/
theorem smallest_square_sides (arr : SquareArrangement) : 
  (arr.small_side = 2 ∨ arr.small_side = 32) ↔ 
  (∃ (x : ℝ), x * (x + 8) * 8 = x * (42 - x) * (x + 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_sides_l3833_383320


namespace NUMINAMATH_CALUDE_product_of_fractions_l3833_383347

theorem product_of_fractions (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3833_383347


namespace NUMINAMATH_CALUDE_interval_intersection_l3833_383340

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l3833_383340


namespace NUMINAMATH_CALUDE_time_addition_theorem_l3833_383391

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (initial : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds -/
def sumTimeComponents (time : Time) : Nat :=
  sorry

theorem time_addition_theorem :
  let initial_time := Time.mk 15 15 30  -- 3:15:30 PM
  let duration_hours := 174
  let duration_minutes := 58
  let duration_seconds := 16
  let final_time := to12Hour (addTime initial_time duration_hours duration_minutes duration_seconds)
  final_time = Time.mk 10 13 46 ∧ sumTimeComponents final_time = 69 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l3833_383391


namespace NUMINAMATH_CALUDE_min_honey_purchase_l3833_383332

def is_valid_purchase (o h : ℕ) : Prop :=
  o ≥ 7 + h / 2 ∧ 
  o ≤ 3 * h ∧ 
  2 * o + 3 * h ≤ 36

theorem min_honey_purchase : 
  (∃ (o h : ℕ), is_valid_purchase o h) ∧ 
  (∀ (o h : ℕ), is_valid_purchase o h → h ≥ 4) ∧
  (∃ (o : ℕ), is_valid_purchase o 4) :=
sorry

end NUMINAMATH_CALUDE_min_honey_purchase_l3833_383332


namespace NUMINAMATH_CALUDE_gcd_of_all_P_is_one_l3833_383396

-- Define P as a function of n, where n represents the first of the three consecutive even integers
def P (n : ℕ) : ℕ := 2 * n * (2 * n + 2) * (2 * n + 4) + 2

-- Theorem stating that the greatest common divisor of all P(n) is 1
theorem gcd_of_all_P_is_one : ∃ (d : ℕ), d > 0 ∧ (∀ (n : ℕ), n > 0 → d ∣ P n) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_all_P_is_one_l3833_383396


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3833_383303

/-- Proves that the percentage of loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage (CP : ℝ) : 
  CP > 720 ∧ 880 = 1.10 * CP → (CP - 720) / CP * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3833_383303


namespace NUMINAMATH_CALUDE_simplify_expression_l3833_383369

theorem simplify_expression (a b : ℝ) : 2*a*(2*a^2 + a*b) - a^2*b = 4*a^3 + a^2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3833_383369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l3833_383375

theorem arithmetic_sequence_sum_mod (a d : ℕ) (n : ℕ) (h : n > 0) :
  let last_term := a + (n - 1) * d
  let sum := n * (a + last_term) / 2
  sum % 17 = 12 :=
by
  sorry

#check arithmetic_sequence_sum_mod 3 5 21

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l3833_383375


namespace NUMINAMATH_CALUDE_range_of_a_l3833_383361

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∃ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ∧ 
            ∀ b : ℝ, (0 ≤ b ∧ b ≤ 1/4) → b ≠ a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3833_383361


namespace NUMINAMATH_CALUDE_total_students_in_schools_l3833_383314

theorem total_students_in_schools (capacity1 capacity2 : ℕ) 
  (h1 : capacity1 = 400) 
  (h2 : capacity2 = 340) : 
  2 * capacity1 + 2 * capacity2 = 1480 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_schools_l3833_383314


namespace NUMINAMATH_CALUDE_marys_candy_count_l3833_383370

/-- Given that Megan has 5 pieces of candy, and Mary has 3 times as much candy as Megan
    plus an additional 10 pieces, prove that Mary's total candy is 25 pieces. -/
theorem marys_candy_count (megan_candy : ℕ) (mary_initial_multiplier : ℕ) (mary_additional_candy : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_initial_multiplier = 3)
  (h3 : mary_additional_candy = 10) :
  megan_candy * mary_initial_multiplier + mary_additional_candy = 25 := by
  sorry

end NUMINAMATH_CALUDE_marys_candy_count_l3833_383370


namespace NUMINAMATH_CALUDE_stationery_gain_percentage_l3833_383318

/-- Represents the gain percentage calculation for pens and pencils -/
theorem stationery_gain_percentage
  (P : ℝ) -- Cost price of a pen pack
  (Q : ℝ) -- Cost price of a pencil pack
  (h1 : P > 0)
  (h2 : Q > 0)
  (h3 : 80 * P = 100 * P - 20 * P) -- Selling 80 packs of pens gains the cost of 20 packs
  (h4 : 120 * Q = 150 * Q - 30 * Q) -- Selling 120 packs of pencils gains the cost of 30 packs
  : (20 * P) / (80 * P) * 100 = 25 ∧ (30 * Q) / (120 * Q) * 100 = 25 :=
sorry


end NUMINAMATH_CALUDE_stationery_gain_percentage_l3833_383318


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3833_383312

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the intersecting line
def intersecting_line (x y : ℝ) : Prop := ∃ t, x = t*y + 4

-- Define the tangent line
def tangent_line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the circle condition
def circle_condition (x₀ k m r : ℝ) : Prop :=
  ∃ x y, tangent_line k m x y ∧
  (2*m^2 - r)*(x₀ - r) + 2*k*m*x₀ + 2*m^2 = 0

theorem parabola_and_line_properties :
  ∀ p : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    intersecting_line x₁ y₁ ∧ intersecting_line x₂ y₂ ∧
    y₁ * y₂ = -8) →
  p = 1 ∧
  (∀ k m r : ℝ,
    (∃ x y, parabola 1 x y ∧ tangent_line k m x y) →
    (∀ x₀, circle_condition x₀ k m r → x₀ = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3833_383312


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l3833_383322

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0 elements -/
def subsets_0 : ℕ := Nat.choose n 0

/-- The number of subsets with 1 element -/
def subsets_1 : ℕ := Nat.choose n 1

/-- The number of subsets with 2 elements -/
def subsets_2 : ℕ := Nat.choose n 2

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_polygons_count : num_polygons = 32647 := by
  sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l3833_383322


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3833_383353

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (min : ℝ), min = 50 ∧ ∀ (a b : ℝ), (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3833_383353


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3833_383378

theorem system_of_equations_solution (a b : ℝ) 
  (eq1 : 2020 * a + 2024 * b = 2040)
  (eq2 : 2022 * a + 2026 * b = 2050)
  (eq3 : 2025 * a + 2028 * b = 2065) :
  a + 2 * b = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3833_383378


namespace NUMINAMATH_CALUDE_tesla_ratio_proof_l3833_383371

/-- The number of Teslas owned by Chris -/
def chris_teslas : ℕ := 6

/-- The number of Teslas owned by Elon -/
def elon_teslas : ℕ := 13

/-- The number of additional Teslas Elon has compared to Sam -/
def elon_sam_difference : ℕ := 10

/-- The number of Teslas owned by Sam -/
def sam_teslas : ℕ := elon_teslas - elon_sam_difference

theorem tesla_ratio_proof :
  (sam_teslas : ℚ) / chris_teslas = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_tesla_ratio_proof_l3833_383371


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_through_reflection_l3833_383343

/-- The minimum distance from a point to a circle through a reflection point on the x-axis -/
theorem min_distance_point_to_circle_through_reflection (A B P : ℝ × ℝ) : 
  A = (-3, 3) →
  P.2 = 0 →
  (B.1 - 1)^2 + (B.2 - 1)^2 = 2 →
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), P'.2 = 0 → 
      Real.sqrt ((P'.1 - A.1)^2 + (P'.2 - A.2)^2) + 
      Real.sqrt ((B.1 - P'.1)^2 + (B.2 - P'.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_through_reflection_l3833_383343


namespace NUMINAMATH_CALUDE_triangle_max_third_side_l3833_383337

theorem triangle_max_third_side (a b x : ℝ) (ha : a = 5) (hb : b = 10) :
  (a + b > x ∧ a + x > b ∧ b + x > a) →
  ∀ n : ℕ, (n : ℝ) > x → n ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_third_side_l3833_383337


namespace NUMINAMATH_CALUDE_delta_problem_l3833_383335

-- Define the Δ operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_problem : delta (5^(delta 6 17)) (2^(delta 7 11)) = 5^38 - 2^38 := by
  sorry

end NUMINAMATH_CALUDE_delta_problem_l3833_383335


namespace NUMINAMATH_CALUDE_marius_darius_score_difference_l3833_383331

/-- The difference in scores between Marius and Darius in a table football game -/
theorem marius_darius_score_difference :
  ∀ (marius_score darius_score matt_score : ℕ),
    darius_score = 10 →
    matt_score = darius_score + 5 →
    marius_score + darius_score + matt_score = 38 →
    marius_score - darius_score = 3 := by
  sorry

end NUMINAMATH_CALUDE_marius_darius_score_difference_l3833_383331


namespace NUMINAMATH_CALUDE_count_valid_n_values_l3833_383376

/-- Represents a way to split a string of 7's into groups --/
structure SevenGrouping where
  ones : ℕ  -- number of single 7's
  tens : ℕ  -- number of 77's
  thousands : ℕ  -- number of 7777's

/-- The total number of 7's used in a grouping --/
def SevenGrouping.total_sevens (g : SevenGrouping) : ℕ :=
  g.ones + 2 * g.tens + 4 * g.thousands

/-- The value of the expression created by a grouping --/
def SevenGrouping.value (g : SevenGrouping) : ℕ :=
  7 * g.ones + 77 * g.tens + 7777 * g.thousands

/-- A grouping is valid if its value is 8000 --/
def SevenGrouping.isValid (g : SevenGrouping) : Prop :=
  g.value = 8000

theorem count_valid_n_values : 
  (∃ (s : Finset ℕ), s.card = 111 ∧ 
    (∀ n : ℕ, n ∈ s ↔ 
      ∃ g : SevenGrouping, g.isValid ∧ g.total_sevens = n)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_n_values_l3833_383376


namespace NUMINAMATH_CALUDE_lyle_percentage_l3833_383329

/-- Given a total number of chips and a ratio for division, 
    calculate the percentage of chips the second person receives. -/
def calculate_percentage (total_chips : ℕ) (ratio1 ratio2 : ℕ) : ℚ :=
  let total_parts := ratio1 + ratio2
  let chips_per_part := total_chips / total_parts
  let second_person_chips := ratio2 * chips_per_part
  (second_person_chips : ℚ) / total_chips * 100

/-- Theorem stating that given 100 chips divided in a 4:6 ratio, 
    the person with the larger share has 60% of the total chips. -/
theorem lyle_percentage : calculate_percentage 100 4 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lyle_percentage_l3833_383329


namespace NUMINAMATH_CALUDE_sequence_ratio_l3833_383304

/-- Given an arithmetic sequence a and a geometric sequence b with specific conditions,
    prove that the ratio of their second terms is 1. -/
theorem sequence_ratio (a b : ℕ → ℚ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- a is arithmetic
  (∀ n : ℕ, b (n + 1) / b n = b 1 / b 0) →  -- b is geometric
  a 0 = -1 →                                -- a₁ = -1
  b 0 = -1 →                                -- b₁ = -1
  a 3 = 8 →                                 -- a₄ = 8
  b 3 = 8 →                                 -- b₄ = 8
  a 1 / b 1 = 1 :=                          -- a₂/b₂ = 1
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3833_383304


namespace NUMINAMATH_CALUDE_fixed_distance_theorem_l3833_383392

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_fixed_distance (p a b : E) : Prop :=
  ∃ (c : ℝ), ∀ (q : E), ‖p - b‖ = 3 * ‖p - a‖ → ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖p - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖

theorem fixed_distance_theorem (a b p : E) :
  ‖p - b‖ = 3 * ‖p - a‖ → is_fixed_distance p a b :=
by sorry

end NUMINAMATH_CALUDE_fixed_distance_theorem_l3833_383392


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3833_383380

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3833_383380


namespace NUMINAMATH_CALUDE_escalator_standing_time_l3833_383345

/-- Represents the time it takes Clea to ride down an escalator under different conditions -/
def escalator_time (non_operating_time walking_time standing_time : ℝ) : Prop :=
  -- Distance of the escalator
  ∃ d : ℝ,
  -- Speed of Clea walking down the escalator
  ∃ c : ℝ,
  -- Speed of the escalator
  ∃ s : ℝ,
  -- Conditions
  (d = 70 * c) ∧  -- Time to walk down non-operating escalator
  (d = 28 * (c + s)) ∧  -- Time to walk down operating escalator
  (standing_time = d / s) ∧  -- Time to stand on operating escalator
  (standing_time = 47)  -- The result we want to prove

/-- Theorem stating that given the conditions, the standing time on the operating escalator is 47 seconds -/
theorem escalator_standing_time :
  escalator_time 70 28 47 :=
sorry

end NUMINAMATH_CALUDE_escalator_standing_time_l3833_383345


namespace NUMINAMATH_CALUDE_gnome_count_l3833_383316

/-- The number of garden gnomes with red hats, small noses, and striped shirts -/
def redHatSmallNoseStripedShirt (totalGnomes redHats bigNoses blueHatBigNoses : ℕ) : ℕ :=
  let blueHats := totalGnomes - redHats
  let smallNoses := totalGnomes - bigNoses
  let redHatSmallNoses := smallNoses - (blueHats - blueHatBigNoses)
  redHatSmallNoses / 2

/-- Theorem stating the number of garden gnomes with red hats, small noses, and striped shirts -/
theorem gnome_count : redHatSmallNoseStripedShirt 28 21 14 6 = 6 := by
  sorry

#eval redHatSmallNoseStripedShirt 28 21 14 6

end NUMINAMATH_CALUDE_gnome_count_l3833_383316


namespace NUMINAMATH_CALUDE_valid_numbers_l3833_383338

def is_valid_number (a b : Nat) : Prop :=
  let n := 201800 + 10 * a + b
  n % 5 = 1 ∧ n % 11 = 8

theorem valid_numbers : 
  ∀ a b : Nat, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  is_valid_number a b ↔ (a = 3 ∧ b = 1) ∨ (a = 8 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3833_383338


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3833_383390

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) → 
  (c = (7 * a) / 3) → 
  (c - a = 40) → 
  c = 70 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3833_383390


namespace NUMINAMATH_CALUDE_exponent_division_l3833_383398

theorem exponent_division (x : ℝ) : x^6 / x^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3833_383398


namespace NUMINAMATH_CALUDE_intersection_nonempty_l3833_383389

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ a ∧
  (∃ y : ℕ, (∃ x : ℕ, y = a^x) ∧ (∃ x : ℕ, y = (a+1)^x + b)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_l3833_383389


namespace NUMINAMATH_CALUDE_cheese_division_l3833_383357

theorem cheese_division (w : Fin 6 → ℝ) (h_positive : ∀ i, w i > 0) 
  (h_distinct : ∀ i j, i ≠ j → w i ≠ w j) : 
  ∃ (a b c d e f : Fin 6), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                           b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                           c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                           d ≠ e ∧ d ≠ f ∧
                           e ≠ f ∧
                           w a + w b + w c = w d + w e + w f := by
  sorry

end NUMINAMATH_CALUDE_cheese_division_l3833_383357


namespace NUMINAMATH_CALUDE_inequality_proof_l3833_383364

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3833_383364


namespace NUMINAMATH_CALUDE_f_value_theorem_l3833_383344

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_value_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 1 / x) :
  f (-5/2) + f 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_theorem_l3833_383344


namespace NUMINAMATH_CALUDE_alexandra_magazines_l3833_383330

theorem alexandra_magazines : 
  let friday_magazines : ℕ := 8
  let saturday_magazines : ℕ := 12
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 4
  friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines = 48
  := by sorry

end NUMINAMATH_CALUDE_alexandra_magazines_l3833_383330


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l3833_383334

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (AM MB : ℝ) : 
  R = 15 →
  OM = 13 →
  AB = 18 →
  AM + MB = AB →
  OM^2 = R^2 - (AB/2)^2 + ((AM - MB)/2)^2 →
  AM = 14 ∧ MB = 4 :=
by sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l3833_383334


namespace NUMINAMATH_CALUDE_diana_earnings_l3833_383346

def july_earnings : ℕ := 150

def august_earnings : ℕ := 3 * july_earnings

def september_earnings : ℕ := 2 * august_earnings

def total_earnings : ℕ := july_earnings + august_earnings + september_earnings

theorem diana_earnings : total_earnings = 1500 := by
  sorry

end NUMINAMATH_CALUDE_diana_earnings_l3833_383346


namespace NUMINAMATH_CALUDE_wilsons_theorem_l3833_383394

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) ≡ -1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l3833_383394


namespace NUMINAMATH_CALUDE_fibonacci_ratio_difference_bound_l3833_383327

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_ratio_difference_bound (n k : ℕ) (hn : n ≥ 1) (hk : k ≥ 1) :
  |((fibonacci (n + 1) : ℝ) / fibonacci n) - ((fibonacci (k + 1) : ℝ) / fibonacci k)| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_ratio_difference_bound_l3833_383327


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l3833_383349

theorem seeds_per_flowerbed 
  (total_seeds : ℕ) 
  (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45) 
  (h2 : num_flowerbeds = 9) 
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l3833_383349


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3833_383373

theorem complex_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3833_383373


namespace NUMINAMATH_CALUDE_max_percent_error_circular_garden_l3833_383336

theorem max_percent_error_circular_garden (diameter : ℝ) (error_rate : ℝ) : 
  diameter = 30 → 
  error_rate = 0.1 → 
  ∃ (max_error : ℝ), max_error = 21 ∧ 
    ∀ (measured_diameter : ℝ), 
      diameter * (1 - error_rate) ≤ measured_diameter ∧ 
      measured_diameter ≤ diameter * (1 + error_rate) → 
      abs ((π * (measured_diameter / 2)^2 - π * (diameter / 2)^2) / (π * (diameter / 2)^2)) * 100 ≤ max_error :=
by sorry

end NUMINAMATH_CALUDE_max_percent_error_circular_garden_l3833_383336


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l3833_383317

/-- Given a geometric sequence where the first term is 4 and the second term is 16/3,
    the 10th term of this sequence is 1048576/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 4
  let a₂ : ℚ := 16/3
  let r : ℚ := a₂ / a₁
  let a₁₀ : ℚ := a₁ * r^9
  a₁₀ = 1048576/19683 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l3833_383317

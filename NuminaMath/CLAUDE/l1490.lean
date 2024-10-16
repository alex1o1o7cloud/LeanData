import Mathlib

namespace NUMINAMATH_CALUDE_seventeen_sum_of_two_primes_l1490_149004

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem seventeen_sum_of_two_primes :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 17 :=
sorry

end NUMINAMATH_CALUDE_seventeen_sum_of_two_primes_l1490_149004


namespace NUMINAMATH_CALUDE_symmetric_linear_factor_implies_quadratic_factor_l1490_149001

-- Define a polynomial in two variables
variable (P : ℝ → ℝ → ℝ)

-- Define the property of being symmetric
def IsSymmetric (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, P x y = P y x

-- Define the property of having (x - y) as a factor
def HasLinearFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ Q : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y) * Q x y

-- Define the property of having (x - y)² as a factor
def HasQuadraticFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ R : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y)^2 * R x y

-- State the theorem
theorem symmetric_linear_factor_implies_quadratic_factor
  (hSymmetric : IsSymmetric P) (hLinearFactor : HasLinearFactor P) :
  HasQuadraticFactor P := by
  sorry

end NUMINAMATH_CALUDE_symmetric_linear_factor_implies_quadratic_factor_l1490_149001


namespace NUMINAMATH_CALUDE_school_boys_count_l1490_149021

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 0.44 →
  hindu_percent = 0.28 →
  sikh_percent = 0.10 →
  other_count = 72 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / total) = 1 ∧
    total = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1490_149021


namespace NUMINAMATH_CALUDE_find_y_value_l1490_149073

theorem find_y_value (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1490_149073


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1490_149023

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5683 : ℤ) ≡ 420 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5683 : ℤ) ≡ 420 [ZMOD 17] → x ≤ y :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1490_149023


namespace NUMINAMATH_CALUDE_inverse_proportion_point_difference_l1490_149046

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -2/x,
where x₁ < 0 < x₂, prove that y₁ - y₂ > 0.
-/
theorem inverse_proportion_point_difference (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 / x₁)
  (h2 : y₂ = -2 / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂) : 
  y₁ - y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_difference_l1490_149046


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1490_149075

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_factorial_sum (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def does_not_contain_five (n : ℕ) : Prop :=
  5 ∉ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ does_not_contain_five n ∧ n = digit_factorial_sum n :=
  by sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1490_149075


namespace NUMINAMATH_CALUDE_star_value_l1490_149011

/-- The operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem: If a + b = 10 and ab = 24, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1490_149011


namespace NUMINAMATH_CALUDE_function_inequality_l1490_149017

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x < -1, (deriv f) x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1490_149017


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l1490_149014

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l1490_149014


namespace NUMINAMATH_CALUDE_square_minus_product_plus_triple_l1490_149005

theorem square_minus_product_plus_triple (x y : ℝ) :
  x - y + 3 = 0 → x^2 - x*y + 3*y = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_triple_l1490_149005


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l1490_149063

theorem simultaneous_inequalities (x : ℝ) :
  x^3 - 11*x^2 + 10*x < 0 ∧ x^3 - 12*x^2 + 32*x > 0 → (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l1490_149063


namespace NUMINAMATH_CALUDE_oliver_bumper_car_rides_solve_oliver_rides_l1490_149038

theorem oliver_bumper_car_rides : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun ferris_rides ride_cost total_tickets bumper_rides =>
    ferris_rides * ride_cost + bumper_rides * ride_cost = total_tickets →
    ferris_rides = 5 →
    ride_cost = 7 →
    total_tickets = 63 →
    bumper_rides = 4

theorem solve_oliver_rides : oliver_bumper_car_rides 5 7 63 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_bumper_car_rides_solve_oliver_rides_l1490_149038


namespace NUMINAMATH_CALUDE_circle_center_sum_l1490_149078

/-- Given a circle with equation x^2 + y^2 = 6x + 4y + 4, prove that the sum of the coordinates of its center is 5. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 4*y + 4 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4)) → 
  h + k = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1490_149078


namespace NUMINAMATH_CALUDE_remainder_theorem_l1490_149056

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 100 * k - 1) :
  (n^2 + 2*n + 3 + n^3) % 100 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1490_149056


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1490_149089

theorem floor_equation_solution (x : ℝ) :
  ⌊⌊3 * x⌋ - 3 / 2⌋ = ⌊x + 3⌋ ↔ 7 / 3 ≤ x ∧ x < 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1490_149089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1490_149068

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence is 1046 -/
theorem arithmetic_sequence_150th_term :
  arithmetic_sequence 3 7 150 = 1046 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1490_149068


namespace NUMINAMATH_CALUDE_yoga_time_l1490_149093

/-- Mandy's exercise routine -/
def exercise_routine (gym bicycle yoga : ℕ) : Prop :=
  -- Gym to bicycle ratio is 2:3
  3 * gym = 2 * bicycle ∧
  -- Yoga to total exercise ratio is 2:3
  3 * yoga = 2 * (gym + bicycle) ∧
  -- Mandy spends 30 minutes doing yoga
  yoga = 30

/-- Theorem stating that given the exercise routine, yoga time is 30 minutes -/
theorem yoga_time (gym bicycle yoga : ℕ) :
  exercise_routine gym bicycle yoga → yoga = 30 := by
  sorry

end NUMINAMATH_CALUDE_yoga_time_l1490_149093


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l1490_149085

theorem rates_sum_of_squares : ∃ (b j s h : ℕ),
  (3 * b + 4 * j + 2 * s + 3 * h = 120) ∧
  (2 * b + 3 * j + 4 * s + 3 * h = 150) ∧
  (b^2 + j^2 + s^2 + h^2 = 1850) :=
by sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l1490_149085


namespace NUMINAMATH_CALUDE_combined_swimming_distance_l1490_149026

/-- Given swimming distances for Jamir, Sarah, and Julien, prove their combined weekly distance --/
theorem combined_swimming_distance
  (julien_daily : ℕ)
  (sarah_daily : ℕ)
  (jamir_daily : ℕ)
  (days_in_week : ℕ)
  (h1 : julien_daily = 50)
  (h2 : sarah_daily = 2 * julien_daily)
  (h3 : jamir_daily = sarah_daily + 20)
  (h4 : days_in_week = 7) :
  julien_daily * days_in_week +
  sarah_daily * days_in_week +
  jamir_daily * days_in_week = 1890 := by
sorry

end NUMINAMATH_CALUDE_combined_swimming_distance_l1490_149026


namespace NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l1490_149006

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of available colors -/
def num_colors : ℕ := 8

/-- Represents the number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- 
Calculates the number of distinguishable ways to paint a cube
given the number of faces, colors, and rotational symmetries
-/
def distinguishable_colorings (faces : ℕ) (colors : ℕ) (symmetries : ℕ) : ℕ :=
  faces * (Nat.factorial (colors - 1)) / symmetries

/-- 
Theorem stating that the number of distinguishable ways to paint a cube
with 8 different colors, where each face is painted a different color, is 1260
-/
theorem cube_coloring_theorem : 
  distinguishable_colorings num_faces num_colors cube_symmetries = 1260 := by
  sorry

end NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l1490_149006


namespace NUMINAMATH_CALUDE_unique_x_with_square_property_l1490_149094

theorem unique_x_with_square_property : ∃! x : ℕ+, 
  (∃ k : ℕ, (2 * x.val + 1 : ℕ) = k^2) ∧ 
  (∀ y : ℕ, (2 * x.val + 2 : ℕ) ≤ y ∧ y ≤ (3 * x.val + 2) → ¬∃ k : ℕ, y = k^2) ∧
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_square_property_l1490_149094


namespace NUMINAMATH_CALUDE_z_value_range_l1490_149099

theorem z_value_range (x y z : ℝ) (sum_eq : x + y + z = 3) (sum_sq_eq : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
sorry

end NUMINAMATH_CALUDE_z_value_range_l1490_149099


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l1490_149086

/-- Given a line l passing through (-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at points P1 and P2, and P being the midpoint of P1P2, 
    if k2 is the slope of OP, then k1 * k2 = -1/2 -/
theorem ellipse_intersection_slope_product (k1 : ℝ) (h1 : k1 ≠ 0) : 
  ∃ (P1 P2 P : ℝ × ℝ) (k2 : ℝ),
    (P1.1^2 + 2*P1.2^2 = 4) ∧ 
    (P2.1^2 + 2*P2.2^2 = 4) ∧
    (P1.2 = k1 * (P1.1 + 2)) ∧ 
    (P2.2 = k1 * (P2.1 + 2)) ∧
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧
    (k2 = P.2 / P.1) →
    k1 * k2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l1490_149086


namespace NUMINAMATH_CALUDE_total_food_is_338_l1490_149082

/-- The maximum amount of food (in pounds) consumed by an individual guest -/
def max_food_per_guest : ℝ := 2

/-- The minimum number of guests that attended the banquet -/
def min_guests : ℕ := 169

/-- The total amount of food consumed by all guests (in pounds) -/
def total_food_consumed : ℝ := min_guests * max_food_per_guest

/-- Theorem: The total amount of food consumed is 338 pounds -/
theorem total_food_is_338 : total_food_consumed = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_food_is_338_l1490_149082


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1490_149034

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Ioo (-1 : ℝ) 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1490_149034


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_3_and_5_l1490_149072

theorem largest_four_digit_multiple_of_3_and_5 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 9990 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_3_and_5_l1490_149072


namespace NUMINAMATH_CALUDE_vorontsova_dashkova_lifespan_l1490_149042

/-- Represents a person's lifespan across two centuries -/
structure Lifespan where
  total : ℕ
  diff_18th_19th : ℕ
  years_19th : ℕ
  birth_year : ℕ
  death_year : ℕ

/-- Theorem about E.P. Vorontsova-Dashkova's lifespan -/
theorem vorontsova_dashkova_lifespan :
  ∃ (l : Lifespan),
    l.total = 66 ∧
    l.diff_18th_19th = 46 ∧
    l.years_19th = 10 ∧
    l.birth_year = 1744 ∧
    l.death_year = 1810 ∧
    l.total = l.years_19th + (l.years_19th + l.diff_18th_19th) ∧
    l.birth_year + l.total = l.death_year ∧
    l.birth_year + (l.total - l.years_19th) = 1800 :=
by
  sorry


end NUMINAMATH_CALUDE_vorontsova_dashkova_lifespan_l1490_149042


namespace NUMINAMATH_CALUDE_count_integers_with_factors_l1490_149077

theorem count_integers_with_factors : 
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 500 ∧ 22 ∣ n ∧ 16 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_factors_l1490_149077


namespace NUMINAMATH_CALUDE_two_numbers_satisfy_conditions_l1490_149090

/-- A function that checks if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A function that checks if a two-digit number is a square --/
def isTwoDigitSquare (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ isPerfectSquare n

/-- A function that checks if a number is a single-digit square (1, 4, or 9) --/
def isSingleDigitSquare (n : ℕ) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 9

/-- A function that returns the first two digits of a five-digit number --/
def firstTwoDigits (n : ℕ) : ℕ :=
  n / 1000

/-- A function that returns the sum of the third and fourth digits of a five-digit number --/
def sumMiddleTwoDigits (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10)

/-- A function that checks if a number satisfies all the given conditions --/
def satisfiesAllConditions (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧  -- five-digit number
  (∀ d, d ∈ [n / 10000, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → d ≠ 0) ∧  -- no digit is zero
  isPerfectSquare n ∧  -- perfect square
  isTwoDigitSquare (firstTwoDigits n) ∧  -- first two digits form a square
  isSingleDigitSquare (sumMiddleTwoDigits n) ∧  -- sum of middle two digits is a single-digit square
  n % 7 = 0  -- divisible by 7

/-- The main theorem stating that exactly two numbers satisfy all conditions --/
theorem two_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesAllConditions n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_numbers_satisfy_conditions_l1490_149090


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1490_149083

/-- Given a line with equation y - 7 = -3(x + 2), 
    prove that the sum of its x-intercept and y-intercept is 4/3 -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 7 = -3 * (x + 2)) →
  ∃ (x_int y_int : ℝ),
    (x_int - 7 = -3 * (x_int + 2)) ∧  -- x-intercept condition
    (0 - 7 = -3 * (x_int + 2)) ∧      -- x-intercept definition
    (y_int - 7 = -3 * (0 + 2)) ∧      -- y-intercept condition
    (x_int + y_int = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1490_149083


namespace NUMINAMATH_CALUDE_arcsin_three_fifths_cos_tan_l1490_149061

theorem arcsin_three_fifths_cos_tan :
  (Real.cos (Real.arcsin (3/5)) = 4/5) ∧ 
  (Real.tan (Real.arcsin (3/5)) = 3/4) := by
sorry

end NUMINAMATH_CALUDE_arcsin_three_fifths_cos_tan_l1490_149061


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l1490_149018

theorem power_multiplication_equality (m : ℝ) : m^2 * (-m)^4 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l1490_149018


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l1490_149037

/-- Calculates the sum S as defined in the problem -/
def calculate_sum : ℚ := sorry

/-- Finds the closest natural number to a given rational number -/
def closest_natural (q : ℚ) : ℕ := sorry

/-- The main theorem stating that the remainder when the closest natural number
    to the sum S is divided by 5 is equal to 4 -/
theorem sum_remainder_theorem : 
  (closest_natural calculate_sum) % 5 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l1490_149037


namespace NUMINAMATH_CALUDE_good_characterization_l1490_149060

def is_good (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → (a + 1) ∣ (n + 1)

theorem good_characterization :
  ∀ n : ℕ, n ≥ 1 → (is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_good_characterization_l1490_149060


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1490_149043

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some real a and b -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1490_149043


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1490_149055

/-- Given a line y = kx tangent to y = ln x and passing through the origin, k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → 
  k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1490_149055


namespace NUMINAMATH_CALUDE_gaochun_population_scientific_notation_l1490_149029

theorem gaochun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    (1 ≤ |a| ∧ |a| < 10) ∧ 
    425000 = a * (10 : ℝ) ^ n ∧
    a = 4.25 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_gaochun_population_scientific_notation_l1490_149029


namespace NUMINAMATH_CALUDE_bus_passengers_after_three_stops_l1490_149020

theorem bus_passengers_after_three_stops : 
  let initial_passengers := 0
  let first_stop_on := 7
  let second_stop_off := 3
  let second_stop_on := 5
  let third_stop_off := 2
  let third_stop_on := 4
  
  let after_first_stop := initial_passengers + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  
  after_third_stop = 11 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_after_three_stops_l1490_149020


namespace NUMINAMATH_CALUDE_tetradecagon_side_length_l1490_149053

/-- A regular tetradecagon is a polygon with 14 sides of equal length -/
def RegularTetradecagon := { n : ℕ // n = 14 }

/-- The perimeter of the tetradecagon table in centimeters -/
def perimeter : ℝ := 154

/-- Theorem: In a regular tetradecagon with a perimeter of 154 cm, the length of each side is 11 cm -/
theorem tetradecagon_side_length (t : RegularTetradecagon) :
  perimeter / t.val = 11 := by sorry

end NUMINAMATH_CALUDE_tetradecagon_side_length_l1490_149053


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l1490_149031

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (female_employees : ℕ)
  (advanced_degree_employees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : advanced_degree_employees = 78)
  (h4 : males_with_college_only = 31) :
  total_employees - female_employees - males_with_college_only - 
  (advanced_degree_employees - (total_employees - female_employees - males_with_college_only)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l1490_149031


namespace NUMINAMATH_CALUDE_counterclockwise_notation_l1490_149002

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with its direction and angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Converts a rotation to its signed angle representation -/
def Rotation.toSignedAngle (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation (angle : ℝ) :
  (Rotation.toSignedAngle { direction := RotationDirection.Counterclockwise, angle := angle }) = -angle :=
by sorry

end NUMINAMATH_CALUDE_counterclockwise_notation_l1490_149002


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l1490_149062

/-- Represents a stationery box with sheets of paper and envelopes -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Joe's usage of the stationery box -/
def joe_usage (box : StationeryBox) : Prop :=
  box.sheets - box.envelopes = 70

/-- Lily's usage of the stationery box -/
def lily_usage (box : StationeryBox) : Prop :=
  4 * (box.envelopes - 20) = box.sheets

theorem stationery_box_sheets : 
  ∃ (box : StationeryBox), joe_usage box ∧ lily_usage box ∧ box.sheets = 120 := by
  sorry


end NUMINAMATH_CALUDE_stationery_box_sheets_l1490_149062


namespace NUMINAMATH_CALUDE_total_books_count_l1490_149015

/-- Given Benny's initial book count, the number of books he gave to Sandy, and Tim's book count,
    prove that the total number of books Benny and Tim have together is 47. -/
theorem total_books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
    (h1 : benny_initial = 24)
    (h2 : given_to_sandy = 10)
    (h3 : tim_books = 33) :
    benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1490_149015


namespace NUMINAMATH_CALUDE_lewis_weekly_earnings_l1490_149050

/-- Lewis's earnings during the harvest -/
def total_earnings : ℕ := 1216

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 76

/-- Weekly earnings of Lewis during the harvest -/
def weekly_earnings : ℚ := total_earnings / harvest_duration

theorem lewis_weekly_earnings : weekly_earnings = 16 := by
  sorry

end NUMINAMATH_CALUDE_lewis_weekly_earnings_l1490_149050


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1490_149008

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60) 
  (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1490_149008


namespace NUMINAMATH_CALUDE_basketball_score_problem_l1490_149081

theorem basketball_score_problem (total_points winning_margin : ℕ) 
  (h1 : total_points = 48) 
  (h2 : winning_margin = 18) : 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_points ∧ 
    sharks_score - dolphins_score = winning_margin ∧ 
    dolphins_score = 15 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_problem_l1490_149081


namespace NUMINAMATH_CALUDE_bear_buns_l1490_149092

theorem bear_buns (x : ℚ) : 
  (x / 8 - 7 / 8 = 0) → x = 7 := by sorry

end NUMINAMATH_CALUDE_bear_buns_l1490_149092


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1490_149022

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 4 ∧ b = 5 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1490_149022


namespace NUMINAMATH_CALUDE_benny_payment_l1490_149058

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people in the group -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_payment : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_benny_payment_l1490_149058


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1490_149041

/-- A regular polygon with perimeter 160 and side length 10 has 16 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side_length : side_length = 10)
  (h_regular : p * side_length = perimeter) : 
  p = 16 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1490_149041


namespace NUMINAMATH_CALUDE_new_device_improvement_l1490_149097

/-- Represents the data for a device's products -/
structure DeviceData where
  mean : ℝ
  variance : ℝ

/-- Criterion for significant improvement -/
def significant_improvement (old new : DeviceData) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement (old new : DeviceData)
  (h_old : old.mean = 10 ∧ old.variance = 0.036)
  (h_new : new.mean = 10.3 ∧ new.variance = 0.04) :
  significant_improvement old new :=
by sorry

end NUMINAMATH_CALUDE_new_device_improvement_l1490_149097


namespace NUMINAMATH_CALUDE_evans_books_multiple_l1490_149045

/-- Proves the multiple of Evan's current books in 5 years --/
theorem evans_books_multiple (books_two_years_ago : ℕ) (books_decrease : ℕ) (books_in_five_years : ℕ) : 
  books_two_years_ago = 200 →
  books_decrease = 40 →
  books_in_five_years = 860 →
  ∃ (current_books : ℕ) (multiple : ℕ),
    current_books = books_two_years_ago - books_decrease ∧
    books_in_five_years = multiple * current_books + 60 ∧
    multiple = 5 := by
  sorry

#check evans_books_multiple

end NUMINAMATH_CALUDE_evans_books_multiple_l1490_149045


namespace NUMINAMATH_CALUDE_population_growth_model_l1490_149052

/-- World population growth model from 1992 to 2000 -/
theorem population_growth_model 
  (initial_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (final_population : ℝ) :
  initial_population = 5.48 →
  years = 8 →
  final_population = initial_population * (1 + growth_rate / 100) ^ years :=
by sorry

end NUMINAMATH_CALUDE_population_growth_model_l1490_149052


namespace NUMINAMATH_CALUDE_binomial_coefficient_1502_1_l1490_149065

theorem binomial_coefficient_1502_1 : Nat.choose 1502 1 = 1502 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1502_1_l1490_149065


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1490_149012

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = (1/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * x^2 - 5 * x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1490_149012


namespace NUMINAMATH_CALUDE_three_digit_number_property_l1490_149028

theorem three_digit_number_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ)^2 + ((n / 10) % 10 : ℕ)^2 + (n % 10 : ℕ)^2 ∧
  n = 550 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l1490_149028


namespace NUMINAMATH_CALUDE_soccer_shoe_price_l1490_149016

theorem soccer_shoe_price (total_pairs : Nat) (total_price : Nat) :
  total_pairs = 99 →
  total_price % 100 = 76 →
  total_price < 20000 →
  ∃ (price_per_pair : Nat), 
    price_per_pair * total_pairs = total_price ∧
    price_per_pair = 124 :=
by sorry

end NUMINAMATH_CALUDE_soccer_shoe_price_l1490_149016


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l1490_149009

theorem infinite_solutions_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l1490_149009


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_45_l1490_149088

theorem gcd_lcm_product_24_45 : Nat.gcd 24 45 * Nat.lcm 24 45 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_45_l1490_149088


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1490_149069

/-- An ellipse with foci F₁ and F₂, and a point P on the ellipse. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : (P.1^2 / a^2) + (P.2^2 / b^2) = 1  -- P is on the ellipse

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

theorem ellipse_eccentricity (e : Ellipse) 
  (h₄ : angle (e.P - e.F₁) (e.F₂ - e.F₁) = 75 * π / 180)
  (h₅ : angle (e.P - e.F₂) (e.F₁ - e.F₂) = 15 * π / 180) :
  eccentricity e = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1490_149069


namespace NUMINAMATH_CALUDE_no_valid_mapping_divisible_by_1010_l1490_149080

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Fin 10

/-- Checks if a mapping is valid for the word INNOPOLIS -/
def is_valid_mapping (m : LetterToDigitMap) : Prop :=
  m 'I' ≠ m 'N' ∧ m 'I' ≠ m 'O' ∧ m 'I' ≠ m 'P' ∧ m 'I' ≠ m 'L' ∧ m 'I' ≠ m 'S' ∧
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'P' ∧ m 'N' ≠ m 'L' ∧ m 'N' ≠ m 'S' ∧
  m 'O' ≠ m 'P' ∧ m 'O' ≠ m 'L' ∧ m 'O' ≠ m 'S' ∧
  m 'P' ≠ m 'L' ∧ m 'P' ≠ m 'S' ∧
  m 'L' ≠ m 'S'

/-- Converts the word INNOPOLIS to a number using the given mapping -/
def word_to_number (m : LetterToDigitMap) : ℕ :=
  m 'I' * 100000000 + m 'N' * 10000000 + m 'N' * 1000000 + 
  m 'O' * 100000 + m 'P' * 10000 + m 'O' * 1000 + 
  m 'L' * 100 + m 'I' * 10 + m 'S'

/-- The main theorem stating that no valid mapping exists that makes the number divisible by 1010 -/
theorem no_valid_mapping_divisible_by_1010 :
  ¬ ∃ (m : LetterToDigitMap), is_valid_mapping m ∧ (word_to_number m % 1010 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_mapping_divisible_by_1010_l1490_149080


namespace NUMINAMATH_CALUDE_johns_dog_walking_earnings_l1490_149095

/-- Proves that John earns $10 per day for walking the dog -/
theorem johns_dog_walking_earnings :
  ∀ (days_in_april : ℕ) (sundays : ℕ) (total_spent : ℕ) (money_left : ℕ),
    days_in_april = 30 →
    sundays = 4 →
    total_spent = 100 →
    money_left = 160 →
    (days_in_april - sundays) * 10 = total_spent + money_left :=
by
  sorry

end NUMINAMATH_CALUDE_johns_dog_walking_earnings_l1490_149095


namespace NUMINAMATH_CALUDE_negative_215_in_fourth_quadrant_l1490_149051

-- Define a function to convert degrees to the equivalent angle in the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem stating that -215° is in the fourth quadrant
theorem negative_215_in_fourth_quadrant :
  getQuadrant (-215) = 4 := by sorry

end NUMINAMATH_CALUDE_negative_215_in_fourth_quadrant_l1490_149051


namespace NUMINAMATH_CALUDE_twentieth_group_number_l1490_149071

/-- Represents the total number of students -/
def total_students : ℕ := 400

/-- Represents the number of groups -/
def num_groups : ℕ := 20

/-- Represents the first group's drawn number -/
def first_group_number : ℕ := 11

/-- Calculates the drawn number for a given group -/
def drawn_number (group : ℕ) : ℕ :=
  first_group_number + (group - 1) * num_groups

/-- Theorem stating that the 20th group's drawn number is 391 -/
theorem twentieth_group_number :
  drawn_number num_groups = 391 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_group_number_l1490_149071


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1490_149047

theorem square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1490_149047


namespace NUMINAMATH_CALUDE_smallest_union_size_l1490_149025

theorem smallest_union_size (A B : Finset ℕ) 
  (hA : A.card = 30)
  (hB : B.card = 20)
  (hInter : (A ∩ B).card ≥ 10) :
  (A ∪ B).card ≥ 40 ∧ ∃ (C D : Finset ℕ), C.card = 30 ∧ D.card = 20 ∧ (C ∩ D).card ≥ 10 ∧ (C ∪ D).card = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_union_size_l1490_149025


namespace NUMINAMATH_CALUDE_divisibility_32xy76_l1490_149000

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_32xy76 (x y : ℕ) : ℕ := 320000 + 10000 * x + 1000 * y + 76

theorem divisibility_32xy76 (x y : ℕ) (hx : is_digit x) (hy : is_digit y) :
  ∃ k : ℕ, number_32xy76 x y = 4 * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_32xy76_l1490_149000


namespace NUMINAMATH_CALUDE_zack_traveled_18_countries_l1490_149003

/-- The number of countries George traveled to -/
def george_countries : ℕ := 6

/-- The number of countries Joseph traveled to -/
def joseph_countries : ℕ := george_countries / 2

/-- The number of countries Patrick traveled to -/
def patrick_countries : ℕ := 3 * joseph_countries

/-- The number of countries Zack traveled to -/
def zack_countries : ℕ := 2 * patrick_countries

/-- Proof that Zack traveled to 18 countries -/
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_18_countries_l1490_149003


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1490_149030

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (∀ x y, (x + 4) * (x + 3) ≥ 0 → x^2 + y^2 + 4*x + 3 ≤ 0) ∧ 
  (∃ x y, x^2 + y^2 + 4*x + 3 ≤ 0 ∧ (x + 4) * (x + 3) < 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1490_149030


namespace NUMINAMATH_CALUDE_expression_simplification_l1490_149013

theorem expression_simplification (y : ℝ) :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1490_149013


namespace NUMINAMATH_CALUDE_chemistry_class_size_l1490_149070

theorem chemistry_class_size 
  (total_students : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_students = 52) 
  (h2 : both_subjects = 8) 
  (h3 : ∃ (biology_only chemistry_only : ℕ), 
    total_students = biology_only + chemistry_only + both_subjects ∧
    chemistry_only + both_subjects = 2 * (biology_only + both_subjects)) :
  ∃ (chemistry_class : ℕ), chemistry_class = 40 ∧ 
    chemistry_class = (total_students - both_subjects) / 3 * 2 + both_subjects :=
by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l1490_149070


namespace NUMINAMATH_CALUDE_f_properties_l1490_149036

noncomputable def f : ℝ → ℝ := fun x => if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x) ∧
  (StrictMonoOn f (Set.Ioo (-1) 0)) ∧
  (StrictMonoOn f (Set.Ioi 1)) ∧
  (StrictAntiOn f (Set.Iic (-1))) ∧
  (StrictAntiOn f (Set.Ioo 0 1)) ∧
  (Set.range f = Set.Ici (-1)) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l1490_149036


namespace NUMINAMATH_CALUDE_students_in_all_three_activities_l1490_149084

/-- Represents the number of students in each activity and their intersections -/
structure ActivityCounts where
  total : ℕ
  meditation : ℕ
  chess : ℕ
  sculpture : ℕ
  exactlyTwo : ℕ
  allThree : ℕ

/-- The conditions of the problem -/
def problemConditions : ActivityCounts where
  total := 25
  meditation := 15
  chess := 18
  sculpture := 11
  exactlyTwo := 6
  allThree := 0  -- This is what we need to prove

theorem students_in_all_three_activities :
  ∃ (c : ActivityCounts), c.total = 25 ∧
    c.meditation = 15 ∧
    c.chess = 18 ∧
    c.sculpture = 11 ∧
    c.exactlyTwo = 6 ∧
    c.allThree = 7 ∧
    c.total = (c.meditation + c.chess + c.sculpture - 2 * c.exactlyTwo - 3 * c.allThree) :=
  sorry


end NUMINAMATH_CALUDE_students_in_all_three_activities_l1490_149084


namespace NUMINAMATH_CALUDE_f_has_three_distinct_roots_l1490_149040

/-- The polynomial function whose roots we're counting -/
def f (x : ℝ) : ℝ := (x - 8) * (x^2 + 4*x + 3)

/-- The theorem stating that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : 
  ∃ (r₁ r₂ r₃ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃) ∧
  (∀ x : ℝ, f x = 0 → x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_distinct_roots_l1490_149040


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1490_149044

theorem gcf_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1490_149044


namespace NUMINAMATH_CALUDE_sum_of_digits_1948_base9_l1490_149024

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1948_base9 :
  sum (toBase9 1948) = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_1948_base9_l1490_149024


namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l1490_149032

theorem fraction_equality_implies_equality (a b : ℝ) : 
  a / (-5 : ℝ) = b / (-5 : ℝ) → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l1490_149032


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_9x_l1490_149010

theorem factorization_3x_squared_minus_9x (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_9x_l1490_149010


namespace NUMINAMATH_CALUDE_N_mod_five_l1490_149054

def base_nine_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def N : Nat :=
  base_nine_to_decimal [2, 5, 0, 0, 0, 0, 0, 6, 0, 0, 7, 2]

theorem N_mod_five : N % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_N_mod_five_l1490_149054


namespace NUMINAMATH_CALUDE_bob_and_bill_transfer_probability_l1490_149048

theorem bob_and_bill_transfer_probability (total_students : ℕ) (transfer_students : ℕ) (num_classes : ℕ) :
  total_students = 32 →
  transfer_students = 2 →
  num_classes = 2 →
  (1 : ℚ) / (Nat.choose total_students transfer_students * num_classes) = 1 / 992 :=
by sorry

end NUMINAMATH_CALUDE_bob_and_bill_transfer_probability_l1490_149048


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l1490_149039

/-- Calculates the simple interest for a loan -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem annual_interest_calculation :
  let principal : ℝ := 9000
  let rate : ℝ := 0.09
  let time : ℝ := 1
  simple_interest principal rate time = 810 := by
sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l1490_149039


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_zero_l1490_149098

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The square of the distance between two points in 2D space -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Determines if a point lies on a circle -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distanceSquared p c.center = c.radius^2

/-- The main theorem: The square of the distance between intersection points of two specific circles is 0 -/
theorem intersection_distance_squared_is_zero (c1 c2 : Circle)
    (h1 : c1 = { center := (3, -2), radius := 5 })
    (h2 : c2 = { center := (3, 6), radius := 3 }) :
    ∀ p1 p2 : ℝ × ℝ, isOnCircle p1 c1 ∧ isOnCircle p1 c2 ∧ isOnCircle p2 c1 ∧ isOnCircle p2 c2 →
    distanceSquared p1 p2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_squared_is_zero_l1490_149098


namespace NUMINAMATH_CALUDE_marbles_left_l1490_149007

def marbles_in_box : ℕ := 50
def white_marbles : ℕ := 20

def red_blue_marbles : ℕ := marbles_in_box - white_marbles
def blue_marbles : ℕ := red_blue_marbles / 2
def red_marbles : ℕ := red_blue_marbles / 2

def marbles_removed : ℕ := 2 * (white_marbles - blue_marbles)

theorem marbles_left : marbles_in_box - marbles_removed = 40 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l1490_149007


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1490_149087

theorem smallest_integer_solution (x : ℤ) : (∀ y : ℤ, 7 + 3 * y < 26 → x ≤ y) ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1490_149087


namespace NUMINAMATH_CALUDE_payroll_threshold_proof_l1490_149074

/-- Proves that the payroll threshold is $200,000 given the problem conditions --/
theorem payroll_threshold_proof 
  (total_payroll : ℝ) 
  (tax_paid : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_payroll = 400000)
  (h2 : tax_paid = 400)
  (h3 : tax_rate = 0.002) : 
  ∃ threshold : ℝ, 
    threshold = 200000 ∧ 
    tax_rate * (total_payroll - threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_payroll_threshold_proof_l1490_149074


namespace NUMINAMATH_CALUDE_max_distance_is_1375_l1490_149096

/-- Represents the boat trip scenario -/
structure BoatTrip where
  totalTime : Real
  rowingTime : Real
  restTime : Real
  boatSpeed : Real
  currentSpeed : Real

/-- Calculates the maximum distance the boat can travel from the starting point -/
def maxDistance (trip : BoatTrip) : Real :=
  sorry

/-- Theorem stating that the maximum distance is 1.375 km for the given conditions -/
theorem max_distance_is_1375 :
  let trip : BoatTrip := {
    totalTime := 120,
    rowingTime := 30,
    restTime := 10,
    boatSpeed := 3,
    currentSpeed := 1.5
  }
  maxDistance trip = 1.375 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_is_1375_l1490_149096


namespace NUMINAMATH_CALUDE_pipe_laying_efficiency_l1490_149019

theorem pipe_laying_efficiency 
  (n : ℕ) 
  (sequential_length : ℝ) 
  (h1 : n = 7) 
  (h2 : sequential_length = 60) :
  let individual_work_time := sequential_length / (6 * n)
  let total_time := n * individual_work_time
  let simultaneous_rate := n * (sequential_length / total_time)
  simultaneous_rate * total_time = 130 := by
sorry

end NUMINAMATH_CALUDE_pipe_laying_efficiency_l1490_149019


namespace NUMINAMATH_CALUDE_carrot_count_l1490_149035

theorem carrot_count (olivia_carrots : ℕ) (mom_carrots : ℕ) : 
  olivia_carrots = 20 → mom_carrots = 14 → olivia_carrots + mom_carrots = 34 := by
sorry

end NUMINAMATH_CALUDE_carrot_count_l1490_149035


namespace NUMINAMATH_CALUDE_prob_even_sum_is_14_27_l1490_149067

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- Ensures probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Ensures even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- Represents the result of rolling the die three times -/
def ThreeRolls := Fin 3 → Bool

/-- The probability of getting an even sum when rolling the unfair die three times -/
def prob_even_sum (d : UnfairDie) : ℝ :=
  (d.even_prob^3) + 3 * (d.even_prob * d.odd_prob^2)

theorem prob_even_sum_is_14_27 (d : UnfairDie) : prob_even_sum d = 14/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_14_27_l1490_149067


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1490_149057

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) ≥ 10 := by
  sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1490_149057


namespace NUMINAMATH_CALUDE_problem_statement_l1490_149033

theorem problem_statement : (-3)^7 / 3^5 + 5^5 - 8^2 = 3052 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1490_149033


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l1490_149066

theorem smallest_perfect_square_divisible_by_5_and_7 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ 5 ∣ n ∧ 7 ∣ n ∧
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, m = j^2) → 5 ∣ m → 7 ∣ m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l1490_149066


namespace NUMINAMATH_CALUDE_binary_sequence_eventually_periodic_l1490_149079

/-- A sequence of 0s and 1s -/
def BinarySequence := ℕ → Fin 2

/-- A block of n consecutive terms in a sequence -/
def Block (s : BinarySequence) (n : ℕ) (start : ℕ) : Fin n → Fin 2 :=
  fun i => s (start + i)

/-- A sequence is eventually periodic if there exist positive integers p and N such that
    for all k ≥ N, s(k + p) = s(k) -/
def EventuallyPeriodic (s : BinarySequence) : Prop :=
  ∃ (p N : ℕ), p > 0 ∧ ∀ k ≥ N, s (k + p) = s k

/-- The main theorem: if a binary sequence contains only n different blocks of
    n consecutive terms, where n is a positive integer, then it is eventually periodic -/
theorem binary_sequence_eventually_periodic
  (s : BinarySequence) (n : ℕ) (hn : n > 0)
  (h_blocks : ∃ (blocks : Finset (Fin n → Fin 2)),
    blocks.card = n ∧
    ∀ k, ∃ b ∈ blocks, Block s n k = b) :
  EventuallyPeriodic s :=
sorry

end NUMINAMATH_CALUDE_binary_sequence_eventually_periodic_l1490_149079


namespace NUMINAMATH_CALUDE_salary_increase_after_five_years_l1490_149059

theorem salary_increase_after_five_years (annual_raise : Real) 
  (h1 : annual_raise = 0.15) : 
  (1 + annual_raise)^5 > 2 := by
  sorry

#check salary_increase_after_five_years

end NUMINAMATH_CALUDE_salary_increase_after_five_years_l1490_149059


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1490_149049

theorem inequality_equivalence (x : ℝ) : 
  5 - 3 / (3 * x - 2) < 7 ↔ x < 1/6 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1490_149049


namespace NUMINAMATH_CALUDE_max_spheres_in_cube_l1490_149064

/-- Represents a three-dimensional cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cube -/
def maxSpheres (c : Cube) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating the maximum number of spheres in the given cube -/
theorem max_spheres_in_cube :
  ∃ (c : Cube) (s : Sphere),
    c.edgeLength = 4 ∧ s.diameter = 1 ∧ maxSpheres c s = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_cube_l1490_149064


namespace NUMINAMATH_CALUDE_blue_shirt_percentage_l1490_149027

/-- Proves that the percentage of students wearing blue shirts is 45% -/
theorem blue_shirt_percentage
  (total_students : ℕ)
  (red_shirt_percentage : ℚ)
  (green_shirt_percentage : ℚ)
  (other_colors_count : ℕ)
  (h1 : total_students = 600)
  (h2 : red_shirt_percentage = 23 / 100)
  (h3 : green_shirt_percentage = 15 / 100)
  (h4 : other_colors_count = 102)
  : (1 : ℚ) - (red_shirt_percentage + green_shirt_percentage + (other_colors_count : ℚ) / (total_students : ℚ)) = 45 / 100 := by
  sorry

#check blue_shirt_percentage

end NUMINAMATH_CALUDE_blue_shirt_percentage_l1490_149027


namespace NUMINAMATH_CALUDE_defective_items_count_l1490_149076

def total_products : ℕ := 100
def defective_items : ℕ := 2
def items_to_draw : ℕ := 3

def ways_with_defective : ℕ := Nat.choose total_products items_to_draw - Nat.choose (total_products - defective_items) items_to_draw

theorem defective_items_count : ways_with_defective = 9472 := by
  sorry

end NUMINAMATH_CALUDE_defective_items_count_l1490_149076


namespace NUMINAMATH_CALUDE_cyclists_speed_l1490_149091

/-- Cyclist's trip problem -/
theorem cyclists_speed (v : ℝ) : 
  v > 0 → -- The speed is positive
  (9 / v + 12 / 9 : ℝ) = 21 / 10.08 → -- Total time equation
  v = 12 := by
sorry

end NUMINAMATH_CALUDE_cyclists_speed_l1490_149091

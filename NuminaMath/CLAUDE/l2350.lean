import Mathlib

namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2350_235077

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is $50,
    prove that the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  P * 0.05 * 2 = 50 → P * (1 + 0.05)^2 - P = 51.25 := by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2350_235077


namespace NUMINAMATH_CALUDE_same_heads_probability_l2350_235051

def num_pennies_keiko : ℕ := 2
def num_pennies_ephraim : ℕ := 3

def total_outcomes : ℕ := 2^num_pennies_keiko * 2^num_pennies_ephraim

def favorable_outcomes : ℕ := 6

theorem same_heads_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_l2350_235051


namespace NUMINAMATH_CALUDE_geometric_sequence_from_formula_l2350_235078

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_from_formula (c q : ℝ) (hcq : c * q ≠ 0) :
  is_geometric_sequence (fun n => c * q ^ n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_formula_l2350_235078


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l2350_235054

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let total_shirts : ℕ := 245
  let total_revenue : ℚ := 2205
  let profit_per_shirt : ℚ := total_revenue / total_shirts
  profit_per_shirt = 9 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l2350_235054


namespace NUMINAMATH_CALUDE_jerry_tips_problem_l2350_235024

/-- The amount Jerry needs to earn on the fifth night to achieve an average of $50 per night -/
theorem jerry_tips_problem (
  days_per_week : ℕ)
  (target_average : ℝ)
  (past_earnings : List ℝ)
  (h1 : days_per_week = 5)
  (h2 : target_average = 50)
  (h3 : past_earnings = [20, 60, 15, 40]) :
  target_average * days_per_week - past_earnings.sum = 115 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tips_problem_l2350_235024


namespace NUMINAMATH_CALUDE_height_area_ratio_not_always_equal_l2350_235001

-- Define the properties of an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ
  base_positive : 0 < base
  height_positive : 0 < height
  side_positive : 0 < side
  perimeter_eq : perimeter = base + 2 * side
  area_eq : area = (1/2) * base * height

-- Theorem statement
theorem height_area_ratio_not_always_equal : 
  ∃ (t1 t2 : IsoscelesTriangle), t1.height ≠ t2.height ∧ 
    (t1.height / t2.height ≠ t1.area / t2.area) := by
  sorry


end NUMINAMATH_CALUDE_height_area_ratio_not_always_equal_l2350_235001


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2350_235027

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/3, -2)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x - 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 4 = -6 * x

theorem intersection_point_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2350_235027


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2350_235014

/-- A quadratic function with vertex (h, k) and y-intercept (0, y0) -/
def QuadraticFunction (a b c h k y0 : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k ∧
  c = y0 ∧
  -b / (2 * a) = h ∧
  a * h^2 + b * h + c = k

theorem quadratic_function_uniqueness (a b c : ℝ) : 
  QuadraticFunction a b c 2 (-1) 11 → a = 3 ∧ b = -12 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2350_235014


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2350_235059

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, is_two_digit n → digit_sum n = 9 → ¬ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2350_235059


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2350_235005

theorem square_sum_reciprocal (x : ℝ) (h : 18 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2350_235005


namespace NUMINAMATH_CALUDE_regular_pyramid_volume_l2350_235055

theorem regular_pyramid_volume (b : ℝ) (h : b = 2) :
  ∀ V : ℝ, V ≤ (16 * Real.pi) / (9 * Real.sqrt 3) → V < 3.25 := by sorry

end NUMINAMATH_CALUDE_regular_pyramid_volume_l2350_235055


namespace NUMINAMATH_CALUDE_total_remaining_value_l2350_235043

/-- Represents the types of gift cards Jack has --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents the initial number and value of each type of gift card --/
def initial_gift_cards : List (GiftCardType × Nat × Nat) :=
  [(GiftCardType.BestBuy, 5, 500),
   (GiftCardType.Target, 3, 250),
   (GiftCardType.Walmart, 7, 100),
   (GiftCardType.Amazon, 2, 1000)]

/-- Represents the number of gift cards Jack sent codes for --/
def sent_gift_cards : List (GiftCardType × Nat) :=
  [(GiftCardType.BestBuy, 1),
   (GiftCardType.Walmart, 2),
   (GiftCardType.Amazon, 1)]

/-- Calculates the total value of remaining gift cards --/
def remaining_value (initial : List (GiftCardType × Nat × Nat)) (sent : List (GiftCardType × Nat)) : Nat :=
  sorry

/-- Theorem stating that the total value of gift cards Jack can still return is $4250 --/
theorem total_remaining_value : 
  remaining_value initial_gift_cards sent_gift_cards = 4250 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_value_l2350_235043


namespace NUMINAMATH_CALUDE_complete_quadrilateral_l2350_235029

/-- A point in the projective plane -/
structure ProjPoint where
  x : ℝ
  y : ℝ
  z : ℝ
  nontrivial : (x, y, z) ≠ (0, 0, 0)

/-- A line in the projective plane -/
structure ProjLine where
  a : ℝ
  b : ℝ
  c : ℝ
  nontrivial : (a, b, c) ≠ (0, 0, 0)

/-- The cross ratio of four collinear points -/
def cross_ratio (A B C D : ProjPoint) : ℝ := sorry

/-- Intersection of two lines -/
def intersect (l1 l2 : ProjLine) : ProjPoint := sorry

/-- Line passing through two points -/
def line_through (A B : ProjPoint) : ProjLine := sorry

theorem complete_quadrilateral 
  (A B C D : ProjPoint) 
  (P : ProjPoint := intersect (line_through A B) (line_through C D))
  (Q : ProjPoint := intersect (line_through A D) (line_through B C))
  (R : ProjPoint := intersect (line_through A C) (line_through B D))
  (K : ProjPoint := intersect (line_through Q R) (line_through A B))
  (L : ProjPoint := intersect (line_through Q R) (line_through C D)) :
  cross_ratio Q R K L = -1 := by
  sorry

end NUMINAMATH_CALUDE_complete_quadrilateral_l2350_235029


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_thirds_l2350_235068

theorem tan_thirteen_pi_thirds : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_thirds_l2350_235068


namespace NUMINAMATH_CALUDE_distance_traveled_l2350_235018

/-- Calculates the actual distance traveled given the conditions of the problem -/
def actual_distance (actual_speed hours_walked : ℝ) : ℝ :=
  actual_speed * hours_walked

/-- Represents the additional distance that would be covered at the higher speed -/
def additional_distance (actual_speed higher_speed hours_walked : ℝ) : ℝ :=
  (higher_speed - actual_speed) * hours_walked

theorem distance_traveled (actual_speed higher_speed additional : ℝ) 
  (h1 : actual_speed = 12)
  (h2 : higher_speed = 20)
  (h3 : additional = 30) :
  ∃ (hours_walked : ℝ), 
    additional_distance actual_speed higher_speed hours_walked = additional ∧ 
    actual_distance actual_speed hours_walked = 45 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2350_235018


namespace NUMINAMATH_CALUDE_gcd_6051_10085_l2350_235050

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6051_10085_l2350_235050


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l2350_235058

/-- Definition of the quadratic function g(x) -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that under the given conditions, k = 0 -/
theorem quadratic_function_k_value
  (a b c : ℤ)
  (h1 : g a b c 2 = 0)
  (h2 : 60 < g a b c 6 ∧ g a b c 6 < 70)
  (h3 : 90 < g a b c 9 ∧ g a b c 9 < 100)
  (k : ℤ)
  (h4 : 10000 * ↑k < g a b c 50 ∧ g a b c 50 < 10000 * ↑(k + 1)) :
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l2350_235058


namespace NUMINAMATH_CALUDE_perpendicular_lines_min_value_l2350_235053

theorem perpendicular_lines_min_value (b : ℝ) (a : ℝ) (h1 : b > 1) :
  ((b^2 + 1) * (-1 / a) * (b - 1) = -1) →
  (∀ a' : ℝ, ((b^2 + 1) * (-1 / a') * (b - 1) = -1) → a ≤ a') →
  a = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_min_value_l2350_235053


namespace NUMINAMATH_CALUDE_flower_planting_cost_l2350_235031

theorem flower_planting_cost (flower_cost soil_cost clay_pot_cost total_cost : ℕ) : 
  flower_cost = 9 →
  clay_pot_cost = flower_cost + 20 →
  soil_cost < flower_cost →
  total_cost = 45 →
  total_cost = flower_cost + clay_pot_cost + soil_cost →
  flower_cost - soil_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_flower_planting_cost_l2350_235031


namespace NUMINAMATH_CALUDE_solve_system_l2350_235022

theorem solve_system (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2350_235022


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2350_235036

/-- The ratio of the real semi-axis length to the imaginary axis length of the hyperbola 2x^2 - y^2 = 8 -/
theorem hyperbola_axis_ratio : ∃ (a b : ℝ), 
  (∀ x y : ℝ, 2 * x^2 - y^2 = 8 ↔ x^2 / (2 * a^2) - y^2 / (2 * b^2) = 1) ∧ 
  (a / (2 * b) = Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2350_235036


namespace NUMINAMATH_CALUDE_fractional_factorial_max_test_points_l2350_235042

/-- The number of experiments in the fractional factorial design. -/
def num_experiments : ℕ := 6

/-- The maximum number of test points that can be handled. -/
def max_test_points : ℕ := 20

/-- Theorem stating that given 6 experiments in a fractional factorial design,
    the maximum number of test points that can be handled is 20. -/
theorem fractional_factorial_max_test_points :
  ∀ n : ℕ, n ≤ 2^num_experiments - 1 → n ≤ max_test_points :=
by sorry

end NUMINAMATH_CALUDE_fractional_factorial_max_test_points_l2350_235042


namespace NUMINAMATH_CALUDE_new_person_weight_l2350_235096

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 87 kg -/
theorem new_person_weight :
  weight_of_new_person 8 (5/2) 67 = 87 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2350_235096


namespace NUMINAMATH_CALUDE_triangle_area_l2350_235008

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (7, 1)
def C : ℝ × ℝ := (5, 6)

-- State the theorem
theorem triangle_area : 
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 13 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2350_235008


namespace NUMINAMATH_CALUDE_repeating_decimal_bounds_l2350_235023

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℕ
  fractionalPart : List ℕ
  repeatingPart : List ℕ

/-- Converts a RepeatingDecimal to a real number -/
noncomputable def RepeatingDecimal.toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Generates all possible repeating decimals from a given decimal string -/
def generateRepeatingDecimals (s : String) : List RepeatingDecimal :=
  sorry

/-- Finds the maximum repeating decimal from a list of repeating decimals -/
noncomputable def findMaxRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

/-- Finds the minimum repeating decimal from a list of repeating decimals -/
noncomputable def findMinRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

theorem repeating_decimal_bounds :
  let decimals := generateRepeatingDecimals "0.20120415"
  let maxDecimal := findMaxRepeatingDecimal decimals
  let minDecimal := findMinRepeatingDecimal decimals
  maxDecimal = { integerPart := 0, fractionalPart := [2, 0, 1, 2, 0, 4, 1], repeatingPart := [5] } ∧
  minDecimal = { integerPart := 0, fractionalPart := [2], repeatingPart := [0, 1, 2, 0, 4, 1, 5] } :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_bounds_l2350_235023


namespace NUMINAMATH_CALUDE_sin_shift_stretch_l2350_235093

/-- Given a function f(x) = sin(2x), prove that shifting it right by π/12 and
    stretching x-coordinates by a factor of 2 results in g(x) = sin(x - π/6) -/
theorem sin_shift_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x)
  let shift : ℝ → ℝ := λ x => x - π / 12
  let stretch : ℝ → ℝ := λ x => x / 2
  let g : ℝ → ℝ := λ x => Real.sin (x - π / 6)
  (f ∘ shift ∘ stretch) x = g x :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_stretch_l2350_235093


namespace NUMINAMATH_CALUDE_star_operation_example_l2350_235004

def star_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_example :
  let A : Set ℕ := {1,3,5,7}
  let B : Set ℕ := {2,3,5}
  star_operation A B = {1,7} := by
sorry

end NUMINAMATH_CALUDE_star_operation_example_l2350_235004


namespace NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2350_235085

theorem percentage_of_boys_from_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 450)
  (h2 : boys_A_not_science = 63)
  (h3 : (30 : ℚ) / 100 = 1 - (boys_A_not_science : ℚ) / ((20 : ℚ) / 100 * total_boys)) :
  (20 : ℚ) / 100 = (boys_A_not_science : ℚ) / (0.7 * total_boys) :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2350_235085


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2350_235025

theorem floor_ceil_sum : ⌊(3.998 : ℝ)⌋ + ⌈(7.002 : ℝ)⌉ = 11 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2350_235025


namespace NUMINAMATH_CALUDE_product_of_solutions_l2350_235070

theorem product_of_solutions (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (5 * x₁) + 4 = abs (40 - 5)) ∧ 
   (abs (5 * x₂) + 4 = abs (40 - 5)) ∧
   x₁ * x₂ = -961 / 25) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2350_235070


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l2350_235002

theorem prime_square_minus_one_divisible_by_thirty (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l2350_235002


namespace NUMINAMATH_CALUDE_shopping_trip_proof_l2350_235019

def shopping_trip (initial_amount bag_price lunch_price : ℚ) : Prop :=
  let shoe_price : ℚ := 45
  let remaining : ℚ := 78
  initial_amount = 158 ∧
  bag_price = shoe_price - 17 ∧
  initial_amount = shoe_price + bag_price + lunch_price + remaining ∧
  lunch_price / bag_price = 1/4

theorem shopping_trip_proof : ∃ bag_price lunch_price : ℚ, shopping_trip 158 bag_price lunch_price :=
sorry

end NUMINAMATH_CALUDE_shopping_trip_proof_l2350_235019


namespace NUMINAMATH_CALUDE_inequality_abc_l2350_235082

theorem inequality_abc (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2*(a*b + b*c + c*a) := by
sorry

end NUMINAMATH_CALUDE_inequality_abc_l2350_235082


namespace NUMINAMATH_CALUDE_same_color_probability_l2350_235083

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 3 blue balls, with replacement. -/
theorem same_color_probability (green red blue : ℕ) (total : ℕ) :
  green = 8 → red = 5 → blue = 3 → total = green + red + blue →
  (green^2 + red^2 + blue^2 : ℚ) / total^2 = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2350_235083


namespace NUMINAMATH_CALUDE_shortest_distance_to_x_axis_l2350_235039

/-- Two points on a parabola -/
structure PointsOnParabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  on_parabola₁ : x₁^2 = 4*y₁
  on_parabola₂ : x₂^2 = 4*y₂
  distance : Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 6

/-- Theorem: The shortest distance from the midpoint of AB to the x-axis is 2 -/
theorem shortest_distance_to_x_axis (p : PointsOnParabola) :
  (p.y₁ + p.y₂) / 2 ≥ 2 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_x_axis_l2350_235039


namespace NUMINAMATH_CALUDE_target_is_largest_in_column_and_smallest_in_row_l2350_235067

/-- The matrix represented as a 4x4 array of integers -/
def matrix : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![5, -2, 3, 7],
    ![8, 0, 2, -1],
    ![1, -3, 6, 0],
    ![9, 1, 4, 2]]

/-- The element we're proving to be both largest in column and smallest in row -/
def target_element : ℤ := 1

/-- The position of the target element in the matrix -/
def target_position : Fin 4 × Fin 4 := (3, 1)

theorem target_is_largest_in_column_and_smallest_in_row :
  (∀ i : Fin 4, matrix i (target_position.2) ≤ target_element) ∧
  (∀ j : Fin 4, target_element ≤ matrix (target_position.1) j) := by
  sorry

#check target_is_largest_in_column_and_smallest_in_row

end NUMINAMATH_CALUDE_target_is_largest_in_column_and_smallest_in_row_l2350_235067


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l2350_235076

theorem polynomial_decomposition (x y : ℝ) :
  x^7 + x^6*y + x^5*y^2 + x^4*y^3 + x^3*y^4 + x^2*y^5 + x*y^6 + y^7 = (x + y)*(x^2 + y^2)*(x^4 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l2350_235076


namespace NUMINAMATH_CALUDE_two_bagels_solution_l2350_235044

/-- Represents the number of items bought in a week -/
structure WeeklyPurchase where
  bagels : ℕ
  muffins : ℕ
  donuts : ℕ

/-- Checks if the weekly purchase is valid (totals to 6 days) -/
def isValidPurchase (wp : WeeklyPurchase) : Prop :=
  wp.bagels + wp.muffins + wp.donuts = 6

/-- Calculates the total cost in cents -/
def totalCost (wp : WeeklyPurchase) : ℕ :=
  60 * wp.bagels + 45 * wp.muffins + 30 * wp.donuts

/-- Checks if the total cost is a whole number of dollars -/
def isWholeDollarAmount (wp : WeeklyPurchase) : Prop :=
  totalCost wp % 100 = 0

/-- Main theorem: There exists a valid purchase with 2 bagels that costs a whole dollar amount -/
theorem two_bagels_solution :
  ∃ (wp : WeeklyPurchase), wp.bagels = 2 ∧ isValidPurchase wp ∧ isWholeDollarAmount wp :=
sorry

end NUMINAMATH_CALUDE_two_bagels_solution_l2350_235044


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2350_235046

theorem consecutive_page_numbers_sum (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧
  x * y = 20412 →
  x + y + z = 429 := by
sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2350_235046


namespace NUMINAMATH_CALUDE_sqrt_undefined_for_positive_integer_l2350_235071

theorem sqrt_undefined_for_positive_integer (x : ℕ+) :
  (¬ ∃ (y : ℝ), y ^ 2 = (x : ℝ) - 3) ↔ (x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_undefined_for_positive_integer_l2350_235071


namespace NUMINAMATH_CALUDE_xiaojuan_savings_l2350_235097

/-- Xiaojuan's original savings in yuan -/
def original_savings : ℝ := 12.4

/-- Amount Xiaojuan's mother gave her in yuan -/
def mother_gift : ℝ := 5

/-- Amount spent on dictionary in addition to half of mother's gift -/
def extra_dictionary_cost : ℝ := 0.4

/-- Amount left after all purchases -/
def remaining_amount : ℝ := 5.2

theorem xiaojuan_savings :
  original_savings / 2 + (mother_gift / 2 + extra_dictionary_cost) + remaining_amount = mother_gift + original_savings := by
  sorry

#check xiaojuan_savings

end NUMINAMATH_CALUDE_xiaojuan_savings_l2350_235097


namespace NUMINAMATH_CALUDE_least_number_to_add_or_subtract_l2350_235066

def original_number : ℕ := 856324

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def divisible_by_three_digit_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_three_digit_prime p ∧ p ∣ n

theorem least_number_to_add_or_subtract :
  ∀ k : ℕ, k < 46 →
    ¬(divisible_by_three_digit_prime (original_number + k) ∨
      divisible_by_three_digit_prime (original_number - k)) ∧
    (divisible_by_three_digit_prime (original_number - 46)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_to_add_or_subtract_l2350_235066


namespace NUMINAMATH_CALUDE_species_x_count_day_6_l2350_235045

/-- Represents the number of days passed -/
def days : ℕ := 6

/-- The population growth factor for Species X per day -/
def species_x_growth : ℕ := 2

/-- The population growth factor for Species Y per day -/
def species_y_growth : ℕ := 4

/-- The total number of ants on Day 0 -/
def initial_total : ℕ := 40

/-- The total number of ants on Day 6 -/
def final_total : ℕ := 21050

/-- Theorem stating that the number of Species X ants on Day 6 is 2304 -/
theorem species_x_count_day_6 : ℕ := by
  sorry

end NUMINAMATH_CALUDE_species_x_count_day_6_l2350_235045


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_specific_remainders_l2350_235069

theorem two_digit_numbers_with_specific_remainders :
  let S := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 4 = 3 ∧ n % 3 = 2}
  S = {11, 23, 35, 47, 59, 71, 83, 95} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_specific_remainders_l2350_235069


namespace NUMINAMATH_CALUDE_q_polynomial_expression_l2350_235016

theorem q_polynomial_expression (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 8 * x^2) = (5 * x^4 + 18 * x^3 + 20 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + x^4 + 18 * x^3 + 12 * x^2 + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_expression_l2350_235016


namespace NUMINAMATH_CALUDE_two_variable_data_representable_by_scatter_plot_l2350_235065

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- Define the structure of two-variable data
  -- (We don't need to specify the exact structure for this problem)

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- Define the structure of a scatter plot
  -- (We don't need to specify the exact structure for this problem)

/-- Creates a scatter plot from two-variable data -/
def create_scatter_plot (data : TwoVariableData) : ScatterPlot :=
  sorry -- The actual implementation is not important for this statement

/-- Theorem: Any two-variable statistical data can be represented by a scatter plot -/
theorem two_variable_data_representable_by_scatter_plot (data : TwoVariableData) :
  ∃ (plot : ScatterPlot), plot = create_scatter_plot data :=
sorry

end NUMINAMATH_CALUDE_two_variable_data_representable_by_scatter_plot_l2350_235065


namespace NUMINAMATH_CALUDE_jerry_debt_payment_l2350_235098

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℝ) (remaining_debt : ℝ) (extra_payment : ℝ) :
  total_debt = 50 ∧ 
  remaining_debt = 23 ∧ 
  extra_payment = 3 →
  ∃ (payment_two_months_ago : ℝ),
    payment_two_months_ago = 12 ∧
    total_debt = remaining_debt + payment_two_months_ago + (payment_two_months_ago + extra_payment) :=
by sorry

end NUMINAMATH_CALUDE_jerry_debt_payment_l2350_235098


namespace NUMINAMATH_CALUDE_line_parametric_equation_l2350_235074

/-- Parametric equation of a line passing through (1, 5) with slope angle π/3 -/
theorem line_parametric_equation :
  let M : ℝ × ℝ := (1, 5)
  let slope_angle : ℝ := π / 3
  let parametric_equation (t : ℝ) : ℝ × ℝ :=
    (M.1 + t * Real.cos slope_angle, M.2 + t * Real.sin slope_angle)
  ∀ t : ℝ, parametric_equation t = (1 + (1/2) * t, 5 + (Real.sqrt 3 / 2) * t) :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l2350_235074


namespace NUMINAMATH_CALUDE_contest_awards_l2350_235017

theorem contest_awards (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (n.factorial / (n - k).factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contest_awards_l2350_235017


namespace NUMINAMATH_CALUDE_proportionality_check_l2350_235048

-- Define the type of proportionality
inductive Proportionality
  | Direct
  | Inverse
  | Neither

-- Define a function to check proportionality
def check_proportionality (eq : ℝ → ℝ → Prop) : Proportionality :=
  sorry

-- Theorem statement
theorem proportionality_check :
  (check_proportionality (fun x y => 2*x + y = 5) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => 4*x*y = 15) = Proportionality.Inverse) ∧
  (check_proportionality (fun x y => x = 7*y) = Proportionality.Direct) ∧
  (check_proportionality (fun x y => 2*x + 3*y = 12) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => x/y = 4) = Proportionality.Direct) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_check_l2350_235048


namespace NUMINAMATH_CALUDE_least_integer_with_digit_removal_property_l2350_235064

theorem least_integer_with_digit_removal_property : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n % 10 = 5 ∧ n / 10 = 9) ∧
  n = 19 * (n % 10) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    (m % 10 ≠ 19 * (m / 10) ∨ m / 10 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_digit_removal_property_l2350_235064


namespace NUMINAMATH_CALUDE_bookmarks_end_of_march_l2350_235075

/-- Represents the number of pages bookmarked on each day of the week -/
def weekly_bookmarks : Fin 7 → ℕ
| 0 => 25  -- Monday
| 1 => 30  -- Tuesday
| 2 => 35  -- Wednesday
| 3 => 40  -- Thursday
| 4 => 45  -- Friday
| 5 => 50  -- Saturday
| _ => 55  -- Sunday

/-- The current number of bookmarked pages -/
def current_bookmarks : ℕ := 400

/-- The number of days in March -/
def march_days : ℕ := 31

/-- March starts on a Monday (represented by 0) -/
def march_start : Fin 7 := 0

/-- Calculates the total number of bookmarked pages at the end of March -/
def total_bookmarks_end_of_march : ℕ :=
  current_bookmarks +
  (march_days / 7 * (Finset.sum Finset.univ weekly_bookmarks)) +
  (Finset.sum (Finset.range (march_days % 7)) (λ i => weekly_bookmarks ((i + march_start) % 7)))

/-- Theorem stating that the total number of bookmarked pages at the end of March is 1610 -/
theorem bookmarks_end_of_march :
  total_bookmarks_end_of_march = 1610 := by sorry

end NUMINAMATH_CALUDE_bookmarks_end_of_march_l2350_235075


namespace NUMINAMATH_CALUDE_four_adjacent_squares_l2350_235084

/-- A square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two vertices are adjacent if they are consecutive in the cyclic order of the square -/
def adjacent (s : Square) (i j : Fin 4) : Prop :=
  (j = i + 1) ∨ (i = 3 ∧ j = 0)

/-- A square shares two adjacent vertices with another square -/
def shares_adjacent_vertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), adjacent s1 i j ∧ s1.vertices i = s2.vertices 0 ∧ s1.vertices j = s2.vertices 1

/-- The main theorem: there are exactly 4 squares sharing adjacent vertices with a given square -/
theorem four_adjacent_squares (s : Square) :
  ∃! (squares : Finset Square), squares.card = 4 ∧
    ∀ s' ∈ squares, shares_adjacent_vertices s s' :=
  sorry

end NUMINAMATH_CALUDE_four_adjacent_squares_l2350_235084


namespace NUMINAMATH_CALUDE_jack_sent_three_bestbuy_cards_l2350_235021

def total_requested : ℕ := 6 * 500 + 9 * 200

def walmart_sent : ℕ := 2

def walmart_value : ℕ := 200

def bestbuy_value : ℕ := 500

def remaining_value : ℕ := 3900

def bestbuy_sent : ℕ := 3

theorem jack_sent_three_bestbuy_cards :
  total_requested - remaining_value = walmart_sent * walmart_value + bestbuy_sent * bestbuy_value :=
by sorry

end NUMINAMATH_CALUDE_jack_sent_three_bestbuy_cards_l2350_235021


namespace NUMINAMATH_CALUDE_de_morgan_laws_l2350_235049

theorem de_morgan_laws (A B : Prop) : 
  (¬(A ∧ B) ↔ ¬A ∨ ¬B) ∧ (¬(A ∨ B) ↔ ¬A ∧ ¬B) := by
  sorry

end NUMINAMATH_CALUDE_de_morgan_laws_l2350_235049


namespace NUMINAMATH_CALUDE_integral_roots_system_l2350_235038

theorem integral_roots_system : ∃! (x y z : ℤ),
  (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
  (2 : ℝ) ^ (z : ℝ) = 2 * (4 : ℝ) ^ (x : ℝ) ∧
  x + y + z = 16 ∧
  x = 4 ∧ y = 3 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_integral_roots_system_l2350_235038


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2350_235034

/-- The perimeter of the shaded region formed by four touching circles -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  let arc_length := c / 4
  4 * arc_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2350_235034


namespace NUMINAMATH_CALUDE_tire_price_proof_l2350_235057

/-- The regular price of a single tire -/
def regular_price : ℝ := 104.17

/-- The discounted price of three tires -/
def discounted_price (p : ℝ) : ℝ := 3 * (0.8 * p)

/-- The price of the fourth tire -/
def fourth_tire_price : ℝ := 5

/-- The total price paid for four tires -/
def total_price : ℝ := 255

/-- Theorem stating that the regular price of a tire is approximately 104.17 dollars 
    given the discount and total price conditions -/
theorem tire_price_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  discounted_price regular_price + fourth_tire_price = total_price - ε :=
sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2350_235057


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2350_235006

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area (perimeter : ℝ) (width_ratio : ℝ) : 
  perimeter = 72 ∧ width_ratio = 1/3 → 
  (perimeter / (4 * (1 + width_ratio))) * (perimeter * width_ratio / (4 * (1 + width_ratio))) = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2350_235006


namespace NUMINAMATH_CALUDE_chip_credit_card_balance_l2350_235091

/-- Calculates the balance on a credit card after two months, given an initial balance,
    monthly interest rate, and an additional charge in the second month. -/
def balance_after_two_months (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating that given the specific conditions of Chip's credit card,
    the balance after two months is $48.00. -/
theorem chip_credit_card_balance :
  balance_after_two_months 50 0.2 20 = 48 :=
by sorry

end NUMINAMATH_CALUDE_chip_credit_card_balance_l2350_235091


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2350_235020

theorem greatest_divisor_with_remainders : Nat.gcd (60 - 6) (190 - 10) = 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2350_235020


namespace NUMINAMATH_CALUDE_triangle_count_in_square_with_inscribed_circle_l2350_235094

structure SquareWithInscribedCircle where
  square : Set (ℝ × ℝ)
  circle : Set (ℝ × ℝ)
  midpoints : Set (ℝ × ℝ)
  diagonals : Set (Set (ℝ × ℝ))
  midpoint_segments : Set (Set (ℝ × ℝ))

/-- Given a square with an inscribed circle touching the midpoints of each side,
    with diagonals and segments joining midpoints of opposite sides drawn,
    the total number of triangles formed is 16. -/
theorem triangle_count_in_square_with_inscribed_circle
  (config : SquareWithInscribedCircle) : Nat :=
  16

#check triangle_count_in_square_with_inscribed_circle

end NUMINAMATH_CALUDE_triangle_count_in_square_with_inscribed_circle_l2350_235094


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2350_235030

/-- Simple interest calculation -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_principal_calculation (rate time interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 7200) :
  ∃ principal : ℚ, simpleInterest principal rate time = interest ∧ principal = 20000 := by
  sorry

#check loan_principal_calculation

end NUMINAMATH_CALUDE_loan_principal_calculation_l2350_235030


namespace NUMINAMATH_CALUDE_age_difference_proof_l2350_235089

theorem age_difference_proof (ramesh_age mahesh_age : ℝ) : 
  ramesh_age / mahesh_age = 2 / 5 →
  (ramesh_age + 10) / (mahesh_age + 10) = 10 / 15 →
  mahesh_age - ramesh_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2350_235089


namespace NUMINAMATH_CALUDE_alice_quarters_l2350_235047

/-- Represents the number of quarters Alice had initially -/
def initial_quarters : ℕ := 20

/-- Represents the number of nickels Alice received after exchange -/
def total_nickels : ℕ := 100

/-- Represents the value of a regular nickel in dollars -/
def regular_nickel_value : ℚ := 1/20

/-- Represents the value of an iron nickel in dollars -/
def iron_nickel_value : ℚ := 3

/-- Represents the proportion of iron nickels -/
def iron_nickel_proportion : ℚ := 1/5

/-- Represents the proportion of regular nickels -/
def regular_nickel_proportion : ℚ := 4/5

/-- Represents the total value of all nickels in dollars -/
def total_value : ℚ := 64

theorem alice_quarters :
  (iron_nickel_proportion * total_nickels * iron_nickel_value + 
   regular_nickel_proportion * total_nickels * regular_nickel_value = total_value) ∧
  (initial_quarters * 5 = total_nickels) := by
  sorry

end NUMINAMATH_CALUDE_alice_quarters_l2350_235047


namespace NUMINAMATH_CALUDE_product_of_complex_magnitudes_l2350_235081

theorem product_of_complex_magnitudes : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_complex_magnitudes_l2350_235081


namespace NUMINAMATH_CALUDE_beth_graphic_novels_l2350_235087

theorem beth_graphic_novels (total : ℕ) (novel_percent : ℚ) (comic_percent : ℚ) 
  (h_total : total = 120)
  (h_novel : novel_percent = 65 / 100)
  (h_comic : comic_percent = 20 / 100) :
  total - (novel_percent * total).floor - (comic_percent * total).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_beth_graphic_novels_l2350_235087


namespace NUMINAMATH_CALUDE_stamp_distribution_l2350_235003

theorem stamp_distribution (total : ℕ) (x y : ℕ) 
  (h1 : total = 70)
  (h2 : x = 4 * y + 5)
  (h3 : x + y = total) :
  x = 57 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_stamp_distribution_l2350_235003


namespace NUMINAMATH_CALUDE_third_grade_class_size_l2350_235060

/-- Represents the number of students in each third grade class -/
def third_grade_students : ℕ := sorry

/-- Represents the total number of classes -/
def total_classes : ℕ := 5 + 4 + 4

/-- Represents the total number of students in fourth and fifth grades -/
def fourth_fifth_students : ℕ := 4 * 28 + 4 * 27

/-- Represents the cost of lunch per student in cents -/
def lunch_cost_per_student : ℕ := 210 + 50 + 20

/-- Represents the total cost of all lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem third_grade_class_size :
  third_grade_students = 30 ∧
  third_grade_students * 5 * lunch_cost_per_student +
  fourth_fifth_students * lunch_cost_per_student = total_lunch_cost :=
sorry

end NUMINAMATH_CALUDE_third_grade_class_size_l2350_235060


namespace NUMINAMATH_CALUDE_petting_zoo_theorem_l2350_235090

theorem petting_zoo_theorem (total_animals : ℕ) (carrot_eaters : ℕ) (hay_eaters : ℕ) (both_eaters : ℕ) :
  total_animals = 75 →
  carrot_eaters = 26 →
  hay_eaters = 56 →
  both_eaters = 14 →
  total_animals - (carrot_eaters + hay_eaters - both_eaters) = 7 :=
by sorry

end NUMINAMATH_CALUDE_petting_zoo_theorem_l2350_235090


namespace NUMINAMATH_CALUDE_consecutive_integers_product_mod_three_l2350_235092

theorem consecutive_integers_product_mod_three (n : ℤ) : 
  (n * (n + 1) / 2) % 3 = 0 ∨ (n * (n + 1) / 2) % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_mod_three_l2350_235092


namespace NUMINAMATH_CALUDE_trig_identity_l2350_235011

theorem trig_identity (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.tan θ + (Real.tan θ)⁻¹ = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2350_235011


namespace NUMINAMATH_CALUDE_euclidean_division_123456789_by_37_l2350_235012

theorem euclidean_division_123456789_by_37 :
  ∃ (q r : ℤ), 123456789 = 37 * q + r ∧ 0 ≤ r ∧ r < 37 ∧ q = 3336669 ∧ r = 36 := by
  sorry

end NUMINAMATH_CALUDE_euclidean_division_123456789_by_37_l2350_235012


namespace NUMINAMATH_CALUDE_expense_difference_l2350_235061

def road_trip_expenses (alex_paid bob_paid carol_paid : ℚ) 
                       (a b : ℚ) : Prop :=
  let total := alex_paid + bob_paid + carol_paid
  let share := total / 3
  let alex_owes := share - alex_paid
  let bob_receives := bob_paid - share
  let carol_receives := carol_paid - share
  (alex_owes = a) ∧ (bob_receives + b = carol_receives) ∧ (a - b = 30)

theorem expense_difference :
  road_trip_expenses 120 150 210 40 10 := by sorry

end NUMINAMATH_CALUDE_expense_difference_l2350_235061


namespace NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l2350_235041

/-- The system of equations has at least one solution for any value of parameter a 
    if and only if b is in the specified range -/
theorem system_solution_iff_b_in_range (b : ℝ) : 
  (∀ a : ℝ, ∃ x y : ℝ, 
    x * Real.cos a + y * Real.sin a + 3 ≤ 0 ∧ 
    x^2 + y^2 + 8*x - 4*y - b^2 + 6*b + 11 = 0) ↔ 
  (b ≤ -2 * Real.sqrt 5 ∨ b ≥ 6 + 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l2350_235041


namespace NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l2350_235009

/-- The focus of a parabola y^2 = 12x --/
def focus_of_parabola : ℝ × ℝ := (3, 0)

/-- The equation of the parabola --/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- Theorem: The focus of the parabola y^2 = 12x is at the point (3, 0) --/
theorem focus_of_parabola_is_correct :
  let (a, b) := focus_of_parabola
  ∀ x y : ℝ, parabola_equation x y → (x - a)^2 + y^2 = (x + a)^2 :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l2350_235009


namespace NUMINAMATH_CALUDE_escalator_travel_time_l2350_235010

/-- Proves that a person walking on a moving escalator takes 8 seconds to cover its length --/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) 
  (h1 : escalator_speed = 10) 
  (h2 : escalator_length = 112) 
  (h3 : person_speed = 4) : 
  escalator_length / (escalator_speed + person_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l2350_235010


namespace NUMINAMATH_CALUDE_unique_n_solution_l2350_235000

def is_not_divisible_by_cube_of_prime (x : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^3 ∣ x)

theorem unique_n_solution :
  ∃! n : ℕ, n ≥ 1 ∧
    ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧
      is_not_divisible_by_cube_of_prime (a^2 + b + 3) ∧
      n = (a * b + 3 * b + 8) / (a^2 + b + 3) ∧
      n = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_n_solution_l2350_235000


namespace NUMINAMATH_CALUDE_arithmetic_progression_bijection_l2350_235032

theorem arithmetic_progression_bijection (f : ℕ → ℕ) (hf : Function.Bijective f) :
  ∃ a b c : ℕ, (b - a = c - b) ∧ (f a < f b) ∧ (f b < f c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_bijection_l2350_235032


namespace NUMINAMATH_CALUDE_napoleon_has_17_beans_l2350_235095

/-- The number of jelly beans Napoleon has -/
def napoleon_beans : ℕ := sorry

/-- The number of jelly beans Sedrich has -/
def sedrich_beans : ℕ := napoleon_beans + 4

/-- The number of jelly beans Mikey has -/
def mikey_beans : ℕ := 19

theorem napoleon_has_17_beans : napoleon_beans = 17 := by
  have h1 : sedrich_beans = napoleon_beans + 4 := rfl
  have h2 : 2 * (napoleon_beans + sedrich_beans) = 4 * mikey_beans := sorry
  have h3 : mikey_beans = 19 := rfl
  sorry

end NUMINAMATH_CALUDE_napoleon_has_17_beans_l2350_235095


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2350_235033

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2350_235033


namespace NUMINAMATH_CALUDE_zero_in_interval_l2350_235052

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2350_235052


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2350_235035

-- Define the complex number z
def z : ℂ := 4 - 8 * Complex.I

-- State the theorem
theorem imaginary_part_of_z_times_i : Complex.im (z * Complex.I) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2350_235035


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2350_235013

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : a 3 * a 7 = 3 ∧ a 3 + a 7 = 4) :
  a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2350_235013


namespace NUMINAMATH_CALUDE_inscribed_cube_sphere_surface_area_l2350_235072

theorem inscribed_cube_sphere_surface_area (cube_surface_area : ℝ) (sphere_surface_area : ℝ) :
  cube_surface_area = 6 →
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    cube_surface_area = 6 * cube_edge^2 ∧
    sphere_radius = (cube_edge * Real.sqrt 3) / 2 ∧
    sphere_surface_area = 4 * Real.pi * sphere_radius^2 ∧
    sphere_surface_area = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_sphere_surface_area_l2350_235072


namespace NUMINAMATH_CALUDE_min_value_of_f_l2350_235080

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2350_235080


namespace NUMINAMATH_CALUDE_find_b_l2350_235063

theorem find_b : ∃ b : ℝ,
  let p : ℝ → ℝ := λ x ↦ 2 * x - 3
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 13 → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2350_235063


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l2350_235073

/-- Ellipse C with center at origin, right focus at (√3, 0), and eccentricity √3/2 -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line l: y = kx + √2 -/
def line_l (k x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- Points A and B are distinct intersections of ellipse C and line l -/
def distinct_intersections (k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- Dot product of OA and OB is greater than 2 -/
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

/-- The range of k satisfies the given conditions -/
theorem ellipse_intersection_range :
  ∀ k : ℝ,
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      distinct_intersections k x₁ y₁ x₂ y₂ →
      dot_product_condition x₁ y₁ x₂ y₂) →
    (k ∈ Set.Ioo (-Real.sqrt 3 / 3) (-1/2) ∪ Set.Ioo (1/2) (Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l2350_235073


namespace NUMINAMATH_CALUDE_gold_coin_count_l2350_235099

theorem gold_coin_count (c n : ℕ) (h1 : n = 8 * (c - 3))
  (h2 : n = 5 * c + 4) (h3 : c ≥ 10) : n = 54 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_count_l2350_235099


namespace NUMINAMATH_CALUDE_base_conversion_537_8_to_7_l2350_235062

def base_8_to_10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

def base_10_to_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion_537_8_to_7 :
  base_10_to_7 (base_8_to_10 537) = [1, 1, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_537_8_to_7_l2350_235062


namespace NUMINAMATH_CALUDE_cubic_geometric_roots_l2350_235079

theorem cubic_geometric_roots (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + a*x^2 + b*x + c = 0 ∧
    y^3 + a*y^2 + b*y + c = 0 ∧
    z^3 + a*z^2 + b*z + c = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ y = x*q ∧ z = x*q^2) ↔
  (b^3 = a^3*c ∧
   c ≠ 0 ∧
   ∃ m : ℝ, m^3 = -c ∧ a < m ∧ m < -a/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_geometric_roots_l2350_235079


namespace NUMINAMATH_CALUDE_age_difference_l2350_235007

theorem age_difference (anand_age_10_years_ago bala_age_10_years_ago : ℕ) : 
  anand_age_10_years_ago = bala_age_10_years_ago / 3 →
  anand_age_10_years_ago + 10 = 15 →
  (bala_age_10_years_ago + 10) - (anand_age_10_years_ago + 10) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2350_235007


namespace NUMINAMATH_CALUDE_rent_during_harvest_l2350_235026

/-- The total rent paid during the harvest season -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $526,692 -/
theorem rent_during_harvest : total_rent 388 1359 = 526692 := by
  sorry

end NUMINAMATH_CALUDE_rent_during_harvest_l2350_235026


namespace NUMINAMATH_CALUDE_quadratic_solution_proof_l2350_235040

theorem quadratic_solution_proof (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_proof_l2350_235040


namespace NUMINAMATH_CALUDE_banana_permutations_proof_l2350_235086

def banana_permutations : ℕ := 60

theorem banana_permutations_proof :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  banana_permutations = (Nat.factorial total_letters) / (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) :=
by sorry

end NUMINAMATH_CALUDE_banana_permutations_proof_l2350_235086


namespace NUMINAMATH_CALUDE_shop_distance_is_500_l2350_235088

/-- Represents the configuration of camps and shop -/
structure CampConfig where
  girls_distance : ℝ  -- perpendicular distance from girls' camp to road
  boys_distance : ℝ   -- distance along road from perpendicular to boys' camp
  shop_distance : ℝ   -- distance from shop to each camp

/-- The shop is equidistant from both camps -/
def is_equidistant (config : CampConfig) : Prop :=
  config.shop_distance^2 = config.girls_distance^2 + (config.shop_distance - config.boys_distance)^2

/-- The theorem stating that given the conditions, the shop is 500 rods from each camp -/
theorem shop_distance_is_500 (config : CampConfig) 
    (h1 : config.girls_distance = 400)
    (h2 : config.boys_distance = 800)
    (h3 : is_equidistant config) : 
  config.shop_distance = 500 := by
  sorry

#check shop_distance_is_500

end NUMINAMATH_CALUDE_shop_distance_is_500_l2350_235088


namespace NUMINAMATH_CALUDE_largest_band_formation_l2350_235015

/-- Represents a rectangular band formation -/
structure BandFormation where
  m : ℕ  -- Total number of band members
  r : ℕ  -- Number of rows
  x : ℕ  -- Number of members in each row

/-- Checks if a band formation is valid according to the problem conditions -/
def isValidFormation (f : BandFormation) : Prop :=
  f.r * f.x + 5 = f.m ∧
  (f.r - 3) * (f.x + 2) = f.m ∧
  f.m < 100

/-- The theorem stating the largest possible number of band members -/
theorem largest_band_formation :
  ∃ (f : BandFormation), isValidFormation f ∧
    ∀ (g : BandFormation), isValidFormation g → g.m ≤ f.m :=
by sorry

end NUMINAMATH_CALUDE_largest_band_formation_l2350_235015


namespace NUMINAMATH_CALUDE_factor_divisibility_l2350_235056

theorem factor_divisibility : ∃ (n m : ℕ), (4 ∣ 24) ∧ (9 ∣ 180) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisibility_l2350_235056


namespace NUMINAMATH_CALUDE_intersection_and_range_l2350_235028

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

theorem intersection_and_range :
  (A ∩ B = {x : ℝ | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3}) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ C m → x ∈ A) ∧ (∃ x : ℝ, x ∈ C m ∧ x ∉ A) ↔ -3 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_range_l2350_235028


namespace NUMINAMATH_CALUDE_emily_necklaces_l2350_235037

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 16) (h2 : beads_per_necklace = 8) :
  total_beads / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2350_235037

import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_remainder_l3833_383385

theorem polynomial_remainder (x : ℝ) : 
  (x^11 + 1) % (x + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3833_383385


namespace NUMINAMATH_CALUDE_recipe_calculation_l3833_383309

/-- The amount of flour Julia uses in mL -/
def flour_amount : ℕ := 800

/-- The base amount of flour in mL for the recipe ratio -/
def base_flour : ℕ := 200

/-- The amount of milk in mL needed for the base amount of flour -/
def milk_per_base : ℕ := 60

/-- The number of eggs needed for the base amount of flour -/
def eggs_per_base : ℕ := 1

/-- The amount of milk needed for Julia's recipe -/
def milk_needed : ℕ := (flour_amount / base_flour) * milk_per_base

/-- The number of eggs needed for Julia's recipe -/
def eggs_needed : ℕ := (flour_amount / base_flour) * eggs_per_base

theorem recipe_calculation : 
  milk_needed = 240 ∧ eggs_needed = 4 := by sorry

end NUMINAMATH_CALUDE_recipe_calculation_l3833_383309


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3833_383336

theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℚ) 
  (h1 : original_buckets = 25)
  (h2 : reduced_buckets = 62.5)
  (h3 : original_buckets * original_capacity = reduced_buckets * reduced_capacity) :
  reduced_capacity / original_capacity = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3833_383336


namespace NUMINAMATH_CALUDE_jake_ball_count_l3833_383305

/-- The number of balls each person has -/
structure BallCount where
  jake : ℕ
  audrey : ℕ
  charlie : ℕ

/-- The conditions of the problem -/
def problem_conditions (bc : BallCount) : Prop :=
  bc.audrey = bc.jake + 34 ∧
  bc.audrey = 2 * bc.charlie ∧
  bc.charlie + 7 = 41

/-- The theorem to be proved -/
theorem jake_ball_count (bc : BallCount) : 
  problem_conditions bc → bc.jake = 62 := by
  sorry

end NUMINAMATH_CALUDE_jake_ball_count_l3833_383305


namespace NUMINAMATH_CALUDE_intersection_sum_l3833_383374

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x : ℝ | b < x ∧ x < 2}

-- State the theorem
theorem intersection_sum (a b : ℝ) (h : A ∩ B a = C b) : a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3833_383374


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3833_383348

theorem scientific_notation_equivalence : 26900000 = 2.69 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3833_383348


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_ten_l3833_383395

-- Define the variables
variable (a b c : ℝ)

-- State the theorem
theorem sqrt_sum_equals_sqrt_ten
  (h1 : (2*a + 2)^(1/3) = 2)
  (h2 : b^(1/2) = 2)
  (h3 : ⌊Real.sqrt 15⌋ = c)
  : Real.sqrt (a + b + c) = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_ten_l3833_383395


namespace NUMINAMATH_CALUDE_average_mile_time_l3833_383382

theorem average_mile_time (mile1 mile2 mile3 mile4 : ℕ) 
  (h1 : mile1 = 6)
  (h2 : mile2 = 5)
  (h3 : mile3 = 5)
  (h4 : mile4 = 4) :
  (mile1 + mile2 + mile3 + mile4) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_mile_time_l3833_383382


namespace NUMINAMATH_CALUDE_range_of_a_correct_l3833_383302

/-- Proposition p: The solution set of a^x > 1 (a > 0 and a ≠ 1) is {x | x < 0} -/
def p (a : ℝ) : Prop :=
  0 < a ∧ a ≠ 1 ∧ ∀ x, a^x > 1 ↔ x < 0

/-- Proposition q: The domain of y = log(x^2 - x + a) is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x, x^2 - x + a > 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ :=
  {a | (0 < a ∧ a ≤ 1/4) ∨ a ≥ 1}

theorem range_of_a_correct :
  ∀ a, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_correct_l3833_383302


namespace NUMINAMATH_CALUDE_fraction_simplification_l3833_383388

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (a - b) / (b - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3833_383388


namespace NUMINAMATH_CALUDE_cos_11pi_3_plus_tan_neg_3pi_4_l3833_383331

theorem cos_11pi_3_plus_tan_neg_3pi_4 :
  Real.cos (11 * π / 3) + Real.tan (-3 * π / 4) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_11pi_3_plus_tan_neg_3pi_4_l3833_383331


namespace NUMINAMATH_CALUDE_square_root_squared_sqrt_987654_squared_l3833_383326

theorem square_root_squared (n : ℝ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem sqrt_987654_squared : (Real.sqrt 987654) ^ 2 = 987654 := by sorry

end NUMINAMATH_CALUDE_square_root_squared_sqrt_987654_squared_l3833_383326


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3833_383327

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 2 / b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x * y ≥ 8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x + 2 * y ≥ 9) :=
by sorry

#check min_values_ab_and_a_plus_2b

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3833_383327


namespace NUMINAMATH_CALUDE_leapYears105_l3833_383328

/-- Calculates the maximum number of leap years in a given period under a system where
    leap years occur every 4 years and every 5th year. -/
def maxLeapYears (period : ℕ) : ℕ :=
  (period / 4) + (period / 5) - (period / 20)

/-- Theorem stating that in a 105-year period, the maximum number of leap years is 42
    under the given leap year system. -/
theorem leapYears105 : maxLeapYears 105 = 42 := by
  sorry

#eval maxLeapYears 105  -- Should output 42

end NUMINAMATH_CALUDE_leapYears105_l3833_383328


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_1_and_2_l3833_383370

theorem smallest_value_for_x_between_1_and_2 (x : ℝ) (h : 1 < x ∧ x < 2) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_1_and_2_l3833_383370


namespace NUMINAMATH_CALUDE_total_spent_on_souvenirs_l3833_383383

/-- The amount spent on t-shirts -/
def t_shirts : ℝ := 201

/-- The amount spent on key chains and bracelets -/
def key_chains_and_bracelets : ℝ := 347

/-- The difference between key_chains_and_bracelets and t_shirts -/
def difference : ℝ := 146

theorem total_spent_on_souvenirs :
  key_chains_and_bracelets = t_shirts + difference →
  t_shirts + key_chains_and_bracelets = 548 :=
by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_souvenirs_l3833_383383


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_sum_l3833_383358

/-- Theorem about perpendicular distances in an ellipse -/
theorem ellipse_perpendicular_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_ge_b : a ≥ b) :
  let e := Real.sqrt (a^2 - b^2)
  ∀ (x₀ y₀ : ℝ), x₀^2 / a^2 + y₀^2 / b^2 = 1 →
    let d₁ := |y₀ - b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    let d₂ := |y₀ + b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    d₁^2 + d₂^2 = 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_sum_l3833_383358


namespace NUMINAMATH_CALUDE_sum_of_digits_oneOver99Squared_l3833_383362

/-- Represents a repeating decimal expansion -/
structure RepeatingDecimal where
  digits : List Nat
  period : Nat

/-- The repeating decimal expansion of 1/(99^2) -/
def oneOver99Squared : RepeatingDecimal :=
  { digits := sorry
    period := sorry }

/-- The sum of digits in one period of the repeating decimal expansion of 1/(99^2) -/
def sumOfDigits (rd : RepeatingDecimal) : Nat :=
  (rd.digits.take rd.period).sum

theorem sum_of_digits_oneOver99Squared :
  sumOfDigits oneOver99Squared = 883 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_oneOver99Squared_l3833_383362


namespace NUMINAMATH_CALUDE_max_value_of_a_l3833_383345

theorem max_value_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (prod_sum_eq : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 ∧ 
  ∃ (b' c' : ℝ), b' + c' = 7 - (7 + Real.sqrt 46) / 3 ∧ 
                 ((7 + Real.sqrt 46) / 3) * b' + ((7 + Real.sqrt 46) / 3) * c' + b' * c' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3833_383345


namespace NUMINAMATH_CALUDE_samia_walk_distance_l3833_383393

def average_bike_speed : ℝ := 20
def bike_distance : ℝ := 2
def walk_speed : ℝ := 4
def total_time_minutes : ℝ := 78

theorem samia_walk_distance :
  let total_time_hours : ℝ := total_time_minutes / 60
  let bike_time : ℝ := bike_distance / average_bike_speed
  let walk_time : ℝ := total_time_hours - bike_time
  let walk_distance : ℝ := walk_time * walk_speed
  walk_distance = 4.8 := by sorry

end NUMINAMATH_CALUDE_samia_walk_distance_l3833_383393


namespace NUMINAMATH_CALUDE_estimated_red_balls_l3833_383397

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the number of draws -/
def total_draws : ℕ := 100

/-- Represents the number of white balls drawn -/
def white_draws : ℕ := 40

/-- Theorem: Given the conditions, the estimated number of red balls is 6 -/
theorem estimated_red_balls :
  total_balls * (total_draws - white_draws) / total_draws = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimated_red_balls_l3833_383397


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3833_383306

theorem right_triangle_sides : ∃ (a b c : ℕ), a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3833_383306


namespace NUMINAMATH_CALUDE_positive_solution_to_equation_l3833_383384

theorem positive_solution_to_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1 / x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_to_equation_l3833_383384


namespace NUMINAMATH_CALUDE_frame_254_width_2_l3833_383355

/-- The number of cells in a square frame with given side length and width -/
def frame_cells (side_length : ℕ) (width : ℕ) : ℕ :=
  side_length ^ 2 - (side_length - 2 * width) ^ 2

/-- Theorem: A 254 × 254 frame with width 2 has 2016 cells -/
theorem frame_254_width_2 :
  frame_cells 254 2 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_frame_254_width_2_l3833_383355


namespace NUMINAMATH_CALUDE_factorial_division_l3833_383342

theorem factorial_division (h : Nat.factorial 9 = 362880) :
  Nat.factorial 9 / Nat.factorial 4 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3833_383342


namespace NUMINAMATH_CALUDE_union_A_B_complement_intersection_A_B_l3833_383319

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

-- Define set B
def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_intersection_A_B : (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_intersection_A_B_l3833_383319


namespace NUMINAMATH_CALUDE_distance_rode_bus_l3833_383310

/-- The distance Craig walked from the bus stop to home, in miles -/
def distance_walked : ℝ := 0.17

/-- The difference between the distance Craig rode the bus and the distance he walked, in miles -/
def distance_difference : ℝ := 3.67

/-- Theorem: The distance Craig rode the bus is 3.84 miles -/
theorem distance_rode_bus : distance_walked + distance_difference = 3.84 := by
  sorry

end NUMINAMATH_CALUDE_distance_rode_bus_l3833_383310


namespace NUMINAMATH_CALUDE_saree_stripe_theorem_l3833_383394

/-- Represents the stripes on a Saree --/
structure SareeStripes where
  brown : ℕ
  gold : ℕ
  blue : ℕ

/-- Represents the properties of the Saree's stripe pattern --/
def SareeProperties (s : SareeStripes) : Prop :=
  s.gold = 3 * s.brown ∧
  s.blue = 5 * s.gold ∧
  s.brown = 4 ∧
  s.brown + s.gold + s.blue = 100

/-- Calculates the number of complete patterns on the Saree --/
def patternCount (s : SareeStripes) : ℕ :=
  (s.brown + s.gold + s.blue) / 3

theorem saree_stripe_theorem (s : SareeStripes) 
  (h : SareeProperties s) : s.blue = 84 ∧ patternCount s = 33 := by
  sorry

#check saree_stripe_theorem

end NUMINAMATH_CALUDE_saree_stripe_theorem_l3833_383394


namespace NUMINAMATH_CALUDE_exist_consecutive_sum_digits_div_13_l3833_383367

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Theorem: There exist two consecutive natural numbers such that
    the sum of the digits of each of them is divisible by 13 -/
theorem exist_consecutive_sum_digits_div_13 :
  ∃ n : ℕ, 13 ∣ sum_of_digits n ∧ 13 ∣ sum_of_digits (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_sum_digits_div_13_l3833_383367


namespace NUMINAMATH_CALUDE_correct_payments_l3833_383399

/-- Represents the payment information for the gardeners' plot plowing problem. -/
structure PlowingPayment where
  totalPayment : ℕ
  rectangularPlotArea : ℕ
  rectangularPlotSide : ℕ
  squarePlot1Side : ℕ
  squarePlot2Side : ℕ

/-- Calculates the payments for each gardener based on the given information. -/
def calculatePayments (info : PlowingPayment) : (ℕ × ℕ × ℕ) :=
  let rectangularPlotWidth := info.rectangularPlotArea / info.rectangularPlotSide
  let squarePlot1Area := info.squarePlot1Side * info.squarePlot1Side
  let squarePlot2Area := info.squarePlot2Side * info.squarePlot2Side
  let totalArea := info.rectangularPlotArea + squarePlot1Area + squarePlot2Area
  let pricePerArea := info.totalPayment / totalArea
  let payment1 := info.rectangularPlotArea * pricePerArea
  let payment2 := squarePlot1Area * pricePerArea
  let payment3 := squarePlot2Area * pricePerArea
  (payment1, payment2, payment3)

/-- Theorem stating that the calculated payments match the expected values. -/
theorem correct_payments (info : PlowingPayment) 
  (h1 : info.totalPayment = 570)
  (h2 : info.rectangularPlotArea = 600)
  (h3 : info.rectangularPlotSide = 20)
  (h4 : info.squarePlot1Side = info.rectangularPlotSide)
  (h5 : info.squarePlot2Side = info.rectangularPlotArea / info.rectangularPlotSide) :
  calculatePayments info = (180, 120, 270) := by
  sorry

end NUMINAMATH_CALUDE_correct_payments_l3833_383399


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3833_383380

theorem right_triangle_hypotenuse : 
  ∀ (longer_side shorter_side hypotenuse : ℝ),
  longer_side > 0 →
  shorter_side > 0 →
  hypotenuse > 0 →
  hypotenuse = longer_side + 2 →
  shorter_side = longer_side - 7 →
  shorter_side ^ 2 + longer_side ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3833_383380


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3833_383364

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let side_area1 := 2 * (length * depth)
  let side_area2 := 2 * (width * depth)
  bottom_area + side_area1 + side_area2

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 m² --/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

#eval total_wet_surface_area 4 8 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3833_383364


namespace NUMINAMATH_CALUDE_fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l3833_383347

/-- The Fibonacci sequence modulo 139 -/
def fib_mod (n : ℕ) : Fin 139 :=
  if n = 0 then 0
  else if n = 1 then 1
  else (fib_mod (n - 1) + fib_mod (n - 2))

/-- The closed form expression for the Fibonacci sequence modulo 139 -/
def fib_closed_form (n : ℕ) : Fin 139 :=
  58 * (76^n - 64^n)

/-- Theorem stating that the Fibonacci sequence modulo 139 is equivalent to the closed form expression -/
theorem fib_mod_eq_closed_form (n : ℕ) : fib_mod n = fib_closed_form n := by
  sorry

/-- 12 is a solution of y² ≡ 5 (mod 139) -/
theorem twelve_squared_eq_five : (12 : Fin 139)^2 = 5 := by
  sorry

/-- 64 and 76 are solutions of x² - x - 1 ≡ 0 (mod 139) -/
theorem solutions_of_quadratic : 
  ((64 : Fin 139)^2 - 64 - 1 = 0) ∧ ((76 : Fin 139)^2 - 76 - 1 = 0) := by
  sorry

/-- 58 is the modular multiplicative inverse of 12 modulo 139 -/
theorem inverse_of_twelve : (12 : Fin 139) * 58 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l3833_383347


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3833_383344

def repeating_decimal : ℚ := 2 + 35 / 99

theorem repeating_decimal_as_fraction :
  repeating_decimal = 233 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3833_383344


namespace NUMINAMATH_CALUDE_garden_perimeter_l3833_383325

/-- The perimeter of a rectangle given its length and breadth -/
def rectangle_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 140 m and breadth 100 m is 480 m -/
theorem garden_perimeter :
  rectangle_perimeter 140 100 = 480 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3833_383325


namespace NUMINAMATH_CALUDE_min_value_expression_l3833_383363

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 ∧
  ((x^4 / (y - 1)) + (y^4 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3833_383363


namespace NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_for_abs_sum_equality_l3833_383376

theorem xy_positive_sufficient_not_necessary_for_abs_sum_equality (x y : ℝ) :
  (∀ x y : ℝ, x * y > 0 → |x + y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x + y| = |x| + |y| ∧ ¬(x * y > 0)) :=
sorry

end NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_for_abs_sum_equality_l3833_383376


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3833_383396

/-- The polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^1000 + a*x^2 + 9

/-- Theorem: P(x) is divisible by (x + 1) iff a = -10 -/
theorem polynomial_divisibility (a : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, P a x = (x + 1) * q x) ↔ a = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3833_383396


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l3833_383386

theorem power_of_power_of_three : (3^3)^3 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l3833_383386


namespace NUMINAMATH_CALUDE_max_im_part_is_sin_90_deg_l3833_383377

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^3 - z + 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | p z = 0}

-- Define the imaginary part function
def imPart (z : ℂ) : ℝ := z.im

-- Define the theorem
theorem max_im_part_is_sin_90_deg :
  ∃ (z : ℂ), z ∈ roots ∧ 
  (∀ (w : ℂ), w ∈ roots → imPart w ≤ imPart z) ∧
  imPart z = Real.sin (π / 2) :=
sorry

end NUMINAMATH_CALUDE_max_im_part_is_sin_90_deg_l3833_383377


namespace NUMINAMATH_CALUDE_square_eq_product_sum_seven_l3833_383304

theorem square_eq_product_sum_seven (a b : ℕ) : 
  a * a = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end NUMINAMATH_CALUDE_square_eq_product_sum_seven_l3833_383304


namespace NUMINAMATH_CALUDE_blocks_given_by_theresa_l3833_383338

theorem blocks_given_by_theresa (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : final_blocks = 83) :
  final_blocks - initial_blocks = 79 := by
  sorry

end NUMINAMATH_CALUDE_blocks_given_by_theresa_l3833_383338


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3833_383313

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧ 
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧ 
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 := by
  sorry

#check least_number_divisible_by_five_primes

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3833_383313


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3833_383381

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the square boundary -/
def square_boundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4 ∨ p.y = 0 ∨ p.y = 4

/-- Represents reaching a vertical side of the square -/
def vertical_side (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4

/-- Probability of ending on a vertical side when starting from a given point -/
noncomputable def prob_vertical_end (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  prob_vertical_end ⟨1, 2⟩ = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3833_383381


namespace NUMINAMATH_CALUDE_product_one_sum_squares_and_products_inequality_l3833_383300

theorem product_one_sum_squares_and_products_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_squares_and_products_inequality_l3833_383300


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l3833_383390

theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
  (h1 : total = 390)
  (h2 : afternoon = 260) :
  (afternoon : ℚ) / (total - afternoon : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l3833_383390


namespace NUMINAMATH_CALUDE_expression_value_l3833_383365

theorem expression_value (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 4) :
  (a^2 + b^2) / (2 - c) + (b^2 + c^2) / (2 - a) + (c^2 + a^2) / (2 - b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3833_383365


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3833_383323

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / (4 * x^2 - 1) = 3 / (2 * x + 1) - 4 / (4 * x - 2) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3833_383323


namespace NUMINAMATH_CALUDE_v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l3833_383350

-- Define the set v as cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^3}

-- Theorem stating that v is closed under multiplication and division
theorem v_closed_under_mult_and_div :
  (∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v) ∧
  (∀ a b : ℕ, a ∈ v → b ∈ v → b ≠ 0 → (a / b) ∈ v) :=
sorry

-- Theorem stating that v is not closed under addition
theorem v_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (a + b) ∉ v :=
sorry

-- Theorem stating that v is not closed under negative powers
theorem v_not_closed_under_negative_powers :
  ∃ a : ℕ, a ∈ v ∧ a ≠ 0 ∧ (1 / a) ∉ v :=
sorry

end NUMINAMATH_CALUDE_v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l3833_383350


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l3833_383337

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Theorem stating the ratio of canoes to kayaks is 3:1 given the conditions -/
theorem canoe_kayak_ratio (rb : RentalBusiness) :
  rb.canoe_cost = 14 →
  rb.kayak_cost = 15 →
  rb.total_revenue = 288 →
  rb.canoe_count = rb.kayak_count + 4 →
  rb.canoe_count = 3 * rb.kayak_count →
  rb.canoe_count / rb.kayak_count = 3 := by
  sorry


end NUMINAMATH_CALUDE_canoe_kayak_ratio_l3833_383337


namespace NUMINAMATH_CALUDE_prism_volume_with_inscribed_sphere_l3833_383314

/-- The volume of a regular triangular prism with an inscribed sphere -/
theorem prism_volume_with_inscribed_sphere (r : ℝ) (h : r > 0) :
  let sphere_volume : ℝ := (4 / 3) * Real.pi * r^3
  let prism_side : ℝ := 2 * Real.sqrt 3 * r
  let prism_height : ℝ := 2 * r
  let prism_volume : ℝ := (Real.sqrt 3 / 4) * prism_side^2 * prism_height
  sphere_volume = 36 * Real.pi → prism_volume = 162 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_with_inscribed_sphere_l3833_383314


namespace NUMINAMATH_CALUDE_line_equation_from_intersections_and_midpoint_l3833_383357

/-- The equation of line l given its intersections with two other lines and its midpoint -/
theorem line_equation_from_intersections_and_midpoint 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ l₁ ↔ 4 * x + y + 3 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ 3 * x - 5 * y - 5 = 0) →
  P = (-1, 2) →
  ∃ A B : ℝ × ℝ, A ∈ l₁ ∧ B ∈ l₂ ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ l : Set (ℝ × ℝ), (∀ x y, (x, y) ∈ l ↔ 3 * x + y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intersections_and_midpoint_l3833_383357


namespace NUMINAMATH_CALUDE_linear_function_through_point_l3833_383340

def f (x : ℝ) : ℝ := x + 1

theorem linear_function_through_point :
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_point_l3833_383340


namespace NUMINAMATH_CALUDE_f_properties_l3833_383334

def f (x : ℝ) := |2*x + 1| - |x - 2|

theorem f_properties :
  (∀ x : ℝ, f x > 2 ↔ (x > 1 ∨ x < -5)) ∧
  (∀ t : ℝ, (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3833_383334


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3833_383353

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 15 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ Finset.card s = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3833_383353


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3833_383320

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3833_383320


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l3833_383308

theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ = Real.sqrt 2)
  (h₂ : Complex.abs z₂ = Real.sqrt 2)
  (h₃ : Complex.abs z₃ = Real.sqrt 2) :
  Complex.abs (1 / z₁ + 1 / z₂ + 1 / z₃) / Complex.abs (z₁ + z₂ + z₃) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l3833_383308


namespace NUMINAMATH_CALUDE_total_packs_is_108_l3833_383312

/-- The number of people buying baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- Theorem: The total number of packs for all people is 108 -/
theorem total_packs_is_108 : 
  (num_people * cards_per_person) / cards_per_pack = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_is_108_l3833_383312


namespace NUMINAMATH_CALUDE_set_operations_l3833_383361

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2}) ∧
  (Aᶜ ∩ Bᶜ = {x | x < 3 ∨ x ≥ 4}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3833_383361


namespace NUMINAMATH_CALUDE_new_figure_has_five_sides_l3833_383307

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  sideLength_eq_one : sideLength = 1

/-- The new figure formed by connecting a hexagon and triangle -/
def NewFigure (hexagon triangle : RegularPolygon) : ℕ :=
  hexagon.sides + triangle.sides - 2

/-- Theorem stating that the new figure has 5 sides -/
theorem new_figure_has_five_sides
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (hexagon_is_hexagon : hexagon.sides = 6)
  (triangle_is_triangle : triangle.sides = 3) :
  NewFigure hexagon triangle = 5 := by
  sorry

#eval NewFigure ⟨6, 1, rfl⟩ ⟨3, 1, rfl⟩

end NUMINAMATH_CALUDE_new_figure_has_five_sides_l3833_383307


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_set_l3833_383343

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

-- Theorem statement
theorem M_intersect_N_equals_zero_set : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_set_l3833_383343


namespace NUMINAMATH_CALUDE_equal_projections_l3833_383391

/-- A circle divided into 42 equal arcs -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (points : Fin 42 → ℝ × ℝ)

/-- Projection of a point onto a line segment -/
def project (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem equal_projections (c : Circle) :
  let A₀ := c.points 0
  let A₃ := c.points 3
  let A₆ := c.points 6
  let A₇ := c.points 7
  let A₉ := c.points 9
  let A₂₁ := c.points 21
  let A'₃ := project A₃ A₀ A₂₁
  let A'₆ := project A₆ A₀ A₂₁
  let A'₇ := project A₇ A₀ A₂₁
  let A'₉ := project A₉ A₀ A₂₁
  distance A'₃ A'₆ = distance A'₇ A'₉ := by
    sorry

end NUMINAMATH_CALUDE_equal_projections_l3833_383391


namespace NUMINAMATH_CALUDE_prob_each_student_gets_book_l3833_383351

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of books --/
def num_books : ℕ := 5

/-- The total number of possible distributions --/
def total_distributions : ℕ := num_students ^ num_books

/-- The number of valid distributions where each student gets at least one book --/
def valid_distributions : ℕ := 
  num_students ^ num_books - 
  num_students * (num_students - 1) ^ num_books + 
  (num_students.choose 2) * (num_students - 2) ^ num_books - 
  num_students

/-- The probability that each student receives at least one book --/
theorem prob_each_student_gets_book : 
  (valid_distributions : ℚ) / total_distributions = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_each_student_gets_book_l3833_383351


namespace NUMINAMATH_CALUDE_largest_power_of_seven_divisor_l3833_383317

theorem largest_power_of_seven_divisor : ∃ (n : ℕ), 
  (∀ (k : ℕ), 7^k ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30)) → k ≤ n) ∧
  (7^n ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30))) ∧
  n = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_seven_divisor_l3833_383317


namespace NUMINAMATH_CALUDE_value_range_of_f_l3833_383311

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The domain of f is [0, +∞) -/
def domain : Set ℝ := { x | x ≥ 0 }

theorem value_range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | y ≥ -1 } := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3833_383311


namespace NUMINAMATH_CALUDE_y_share_l3833_383303

theorem y_share (total : ℝ) (x_share y_share z_share : ℝ) : 
  total = 210 →
  y_share = 0.45 * x_share →
  z_share = 0.30 * x_share →
  total = x_share + y_share + z_share →
  y_share = 54 := by
  sorry

end NUMINAMATH_CALUDE_y_share_l3833_383303


namespace NUMINAMATH_CALUDE_twenty_three_percent_of_200_is_46_l3833_383372

theorem twenty_three_percent_of_200_is_46 : 
  ∃ x : ℝ, (23 / 100) * x = 46 ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_percent_of_200_is_46_l3833_383372


namespace NUMINAMATH_CALUDE_series_sum_equals_five_l3833_383332

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n + 1) / k^n = 5) : k = 1.2 + 0.2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_five_l3833_383332


namespace NUMINAMATH_CALUDE_trevor_eggs_end_wednesday_l3833_383301

/-- Represents the number of eggs laid by a chicken on a given day -/
structure ChickenEggs :=
  (monday : ℕ)
  (tuesday : ℕ)
  (wednesday : ℕ)

/-- Represents the egg-laying data for all chickens -/
def chicken_data : List ChickenEggs := [
  ⟨4, 6, 4⟩,  -- Gertrude
  ⟨3, 3, 3⟩,  -- Blanche
  ⟨2, 1, 2⟩,  -- Nancy
  ⟨3, 4, 3⟩,  -- Martha
  ⟨5, 3, 5⟩,  -- Ophelia
  ⟨1, 3, 1⟩,  -- Penelope
  ⟨3, 1, 3⟩,  -- Quinny
  ⟨4, 0, 4⟩   -- Rosie
]

def eggs_eaten_per_day : ℕ := 2
def eggs_dropped_monday : ℕ := 3
def eggs_dropped_wednesday : ℕ := 3

def total_eggs_collected (data : List ChickenEggs) : ℕ :=
  (data.map (·.monday)).sum + (data.map (·.tuesday)).sum + (data.map (·.wednesday)).sum

def eggs_eaten_total (days : ℕ) : ℕ :=
  eggs_eaten_per_day * days

def eggs_dropped_total : ℕ :=
  eggs_dropped_monday + eggs_dropped_wednesday

def eggs_sold (data : List ChickenEggs) : ℕ :=
  (data.map (·.tuesday)).sum / 2

theorem trevor_eggs_end_wednesday :
  total_eggs_collected chicken_data - 
  eggs_eaten_total 3 - 
  eggs_dropped_total - 
  eggs_sold chicken_data = 49 := by
  sorry

end NUMINAMATH_CALUDE_trevor_eggs_end_wednesday_l3833_383301


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3833_383359

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3833_383359


namespace NUMINAMATH_CALUDE_cost_of_pens_l3833_383375

/-- Given that a box of 150 pens costs $45, prove that 4500 pens cost $1350 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (num_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  num_pens = 4500 →
  (num_pens : ℚ) * (box_cost / box_size) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_pens_l3833_383375


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3833_383324

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3833_383324


namespace NUMINAMATH_CALUDE_order_relationship_l3833_383371

theorem order_relationship (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : c < d) 
  (h3 : a + b < c + d) 
  (h4 : a * b = c * d) 
  (h5 : c * d < 0) : 
  a < c ∧ c < b ∧ b < d := by
sorry

end NUMINAMATH_CALUDE_order_relationship_l3833_383371


namespace NUMINAMATH_CALUDE_local_face_value_difference_l3833_383333

def number : ℕ := 96348621

def digit_position (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).reverse.indexOf d

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  d * (10 ^ (digit_position n d))

def face_value (d : ℕ) : ℕ := d

theorem local_face_value_difference :
  local_value number 8 - face_value 8 = 7992 := by
  sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l3833_383333


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3833_383330

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3833_383330


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l3833_383378

theorem integer_ratio_problem (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  (A + B + C + D) / 4 = 16 →
  ∃ k : ℕ, A = k * B →
  B = C - 2 →
  D = 2 →
  A / B = 28 := by
sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l3833_383378


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3833_383335

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.363636

/-- The fraction 4/11 -/
def fraction : ℚ := 4 / 11

/-- Theorem stating that the repeating decimal 0.363636... equals 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3833_383335


namespace NUMINAMATH_CALUDE_outfit_combinations_l3833_383354

theorem outfit_combinations (tshirts pants hats : ℕ) 
  (h1 : tshirts = 8) 
  (h2 : pants = 6) 
  (h3 : hats = 3) : 
  tshirts * pants * hats = 144 := by
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3833_383354


namespace NUMINAMATH_CALUDE_sin_pi_6_plus_cos_pi_3_simplification_l3833_383366

theorem sin_pi_6_plus_cos_pi_3_simplification (α : ℝ) : 
  Real.sin (π/6 + α) + Real.cos (π/3 + α) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_6_plus_cos_pi_3_simplification_l3833_383366


namespace NUMINAMATH_CALUDE_existence_of_special_set_l3833_383321

theorem existence_of_special_set : ∃ (A : Set ℕ), 
  ∀ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) → (Set.Infinite S) →
    ∃ (k : ℕ) (m n : ℕ),
      k ≥ 2 ∧
      m ∈ A ∧
      n ∉ A ∧
      (∃ (factors_m factors_n : Finset ℕ),
        factors_m.card = k ∧
        factors_n.card = k ∧
        (∀ p ∈ factors_m, p ∈ S) ∧
        (∀ p ∈ factors_n, p ∈ S) ∧
        (Finset.prod factors_m id = m) ∧
        (Finset.prod factors_n id = n)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l3833_383321


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3833_383373

theorem a_gt_b_necessary_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3833_383373


namespace NUMINAMATH_CALUDE_broken_marbles_percentage_l3833_383316

theorem broken_marbles_percentage (total_broken : ℕ) (set1_count : ℕ) (set2_count : ℕ) (set2_broken_percent : ℚ) :
  total_broken = 17 →
  set1_count = 50 →
  set2_count = 60 →
  set2_broken_percent = 20 / 100 →
  ∃ (set1_broken_percent : ℚ),
    set1_broken_percent = 10 / 100 ∧
    total_broken = set1_broken_percent * set1_count + set2_broken_percent * set2_count :=
by sorry

end NUMINAMATH_CALUDE_broken_marbles_percentage_l3833_383316


namespace NUMINAMATH_CALUDE_proportion_solution_l3833_383349

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/8) : x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3833_383349


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3833_383398

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3833_383398


namespace NUMINAMATH_CALUDE_line_perp_plane_from_conditions_l3833_383329

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_from_conditions 
  (α β : Plane) (m n : Line) 
  (h1 : perp_plane_line α n) 
  (h2 : perp_plane_line β n) 
  (h3 : perp_plane_line α m) : 
  perp_plane_line β m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_from_conditions_l3833_383329


namespace NUMINAMATH_CALUDE_sports_only_count_l3833_383360

theorem sports_only_count (total employees : ℕ) (sports_fans : ℕ) (art_fans : ℕ) (neither_fans : ℕ) :
  total = 60 →
  sports_fans = 28 →
  art_fans = 26 →
  neither_fans = 12 →
  sports_fans - (total - neither_fans - art_fans) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_only_count_l3833_383360


namespace NUMINAMATH_CALUDE_pigeonhole_disks_l3833_383387

/-- The number of distinct labels -/
def n : ℕ := 50

/-- The function that maps a label to the number of disks with that label -/
def f (i : ℕ) : ℕ := i

/-- The total number of disks -/
def total_disks : ℕ := n * (n + 1) / 2

/-- The minimum number of disks to guarantee at least 10 of the same label -/
def min_disks : ℕ := 415

theorem pigeonhole_disks :
  ∀ (S : Finset ℕ), S.card = min_disks →
  ∃ (i : ℕ), i ∈ Finset.range n ∧ (S.filter (λ x => x = i)).card ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_disks_l3833_383387


namespace NUMINAMATH_CALUDE_ink_covered_term_l3833_383341

variables {a b : ℝ}

theorem ink_covered_term (h : ∃ x, x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3) :
  ∃ x, x = 2 - b ^ 2 ∧ x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3 := by
sorry

end NUMINAMATH_CALUDE_ink_covered_term_l3833_383341


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3833_383379

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3833_383379


namespace NUMINAMATH_CALUDE_stair_climbing_comparison_l3833_383322

/-- Given two people climbing stairs at different speeds, this theorem calculates
    how many steps the faster person climbs when the slower person reaches a certain height. -/
theorem stair_climbing_comparison
  (matt_speed : ℕ)  -- Matt's speed in steps per minute
  (tom_speed_diff : ℕ)  -- How many more steps per minute Tom climbs compared to Matt
  (matt_steps : ℕ)  -- Number of steps Matt has climbed
  (matt_speed_pos : 0 < matt_speed)  -- Matt's speed is positive
  (h_matt_speed : matt_speed = 20)  -- Matt's actual speed
  (h_tom_speed_diff : tom_speed_diff = 5)  -- Tom's speed difference
  (h_matt_steps : matt_steps = 220)  -- Steps Matt has climbed
  : (matt_steps + (matt_steps / matt_speed) * tom_speed_diff : ℕ) = 275 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_comparison_l3833_383322


namespace NUMINAMATH_CALUDE_tech_group_selection_l3833_383368

theorem tech_group_selection (total : ℕ) (select : ℕ) (ways_with_girl : ℕ) :
  total = 6 →
  select = 3 →
  ways_with_girl = 16 →
  (Nat.choose total select - Nat.choose (total - (total - (Nat.choose total select - ways_with_girl))) select = ways_with_girl) →
  total - (Nat.choose total select - ways_with_girl) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tech_group_selection_l3833_383368


namespace NUMINAMATH_CALUDE_max_side_length_24_perimeter_l3833_383352

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different_sides : a ≠ b ∧ b ≠ c ∧ a ≠ c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  perimeter_24 : a + b + c = 24

/-- The maximum length of any side in a triangle with perimeter 24 and different integer side lengths is 12 -/
theorem max_side_length_24_perimeter (t : Triangle) : t.a ≤ 12 ∧ t.b ≤ 12 ∧ t.c ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_24_perimeter_l3833_383352


namespace NUMINAMATH_CALUDE_pentagon_to_squares_ratio_is_one_eighth_l3833_383339

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with given side length and bottom-left corner -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Configuration of three squares as described in the problem -/
structure SquareConfiguration :=
  (square1 : Square)
  (square2 : Square)
  (square3 : Square)

/-- The ratio of the area of pentagon PAWSR to the total area of three squares -/
def pentagonToSquaresRatio (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the ratio is 1/8 for the given configuration -/
theorem pentagon_to_squares_ratio_is_one_eighth
  (config : SquareConfiguration)
  (h1 : config.square1.sideLength = 1)
  (h2 : config.square2.sideLength = 1)
  (h3 : config.square3.sideLength = 1)
  (h4 : config.square1.bottomLeft.x = config.square2.bottomLeft.x)
  (h5 : config.square1.bottomLeft.y + 1 = config.square2.bottomLeft.y)
  (h6 : config.square2.bottomLeft.x + 1 = config.square3.bottomLeft.x)
  (h7 : config.square2.bottomLeft.y = config.square3.bottomLeft.y) :
  pentagonToSquaresRatio config = 1/8 :=
sorry

end NUMINAMATH_CALUDE_pentagon_to_squares_ratio_is_one_eighth_l3833_383339


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3833_383392

theorem geometric_sequence_first_term (a r : ℝ) : 
  a * r = 5 → a * r^3 = 45 → a = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3833_383392


namespace NUMINAMATH_CALUDE_sector_central_angle_l3833_383315

/-- Given a circular sector with arc length 2 and area 4, prove that its central angle is 1/2 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 4) :
  let r := 2 * area / arc_length
  (arc_length / r) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3833_383315


namespace NUMINAMATH_CALUDE_P_120_l3833_383318

/-- 
P(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def P (n : ℕ) : ℕ := sorry

/-- The prime factorization of 120 -/
def primeFactors120 : List ℕ := [2, 2, 2, 3, 5]

/-- 120 is the product of its prime factors -/
axiom is120 : (primeFactors120.prod = 120)

/-- All elements in primeFactors120 are prime numbers -/
axiom allPrime : ∀ p ∈ primeFactors120, Nat.Prime p

theorem P_120 : P 120 = 29 := by sorry

end NUMINAMATH_CALUDE_P_120_l3833_383318


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l3833_383356

def initial_stickers (total : ℝ) (store : ℝ) (birthday : ℝ) (sister : ℝ) (mother : ℝ) : ℝ :=
  total - (store + birthday + sister + mother)

theorem mika_initial_stickers :
  initial_stickers 130 26 20 6 58 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_initial_stickers_l3833_383356


namespace NUMINAMATH_CALUDE_maria_earnings_l3833_383346

/-- The cost of brushes in dollars -/
def brush_cost : ℕ := 20

/-- The cost of canvas in dollars -/
def canvas_cost : ℕ := 3 * brush_cost

/-- The cost of paint per liter in dollars -/
def paint_cost_per_liter : ℕ := 8

/-- The minimum number of liters of paint needed -/
def paint_liters : ℕ := 5

/-- The selling price of the painting in dollars -/
def selling_price : ℕ := 200

/-- Maria's earnings from selling the painting -/
def earnings : ℕ := selling_price - (brush_cost + canvas_cost + paint_cost_per_liter * paint_liters)

theorem maria_earnings : earnings = 80 := by
  sorry

end NUMINAMATH_CALUDE_maria_earnings_l3833_383346


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3833_383369

theorem inequality_solution_set (x : ℝ) :
  (((1 - x) / (x + 1) ≤ 0) ∧ (x ≠ -1)) ↔ (x < -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3833_383369


namespace NUMINAMATH_CALUDE_kelly_apples_l3833_383389

/-- The number of apples Kelly has altogether, given her initial apples and the apples she picked. -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly has 161.0 apples altogether. -/
theorem kelly_apples : total_apples 56.0 105.0 = 161.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l3833_383389

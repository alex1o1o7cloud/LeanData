import Mathlib

namespace NUMINAMATH_CALUDE_sum_37_29_base5_l2150_215027

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_37_29_base5 :
  addBase5 (toBase5 37) (toBase5 29) = [2, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_sum_37_29_base5_l2150_215027


namespace NUMINAMATH_CALUDE_cannot_form_desired_rectangle_l2150_215014

-- Define the tile sizes
def tile_size_1 : ℕ := 3
def tile_size_2 : ℕ := 4

-- Define the initial rectangles
def rect1_width : ℕ := 2
def rect1_height : ℕ := 6
def rect2_width : ℕ := 7
def rect2_height : ℕ := 8

-- Define the desired rectangle
def desired_width : ℕ := 12
def desired_height : ℕ := 5

-- Theorem statement
theorem cannot_form_desired_rectangle :
  ∀ (removed_tile1 removed_tile2 : ℕ),
  (removed_tile1 = tile_size_1 ∨ removed_tile1 = tile_size_2) →
  (removed_tile2 = tile_size_1 ∨ removed_tile2 = tile_size_2) →
  (rect1_width * rect1_height + rect2_width * rect2_height - removed_tile1 - removed_tile2) >
  (desired_width * desired_height) :=
by sorry

end NUMINAMATH_CALUDE_cannot_form_desired_rectangle_l2150_215014


namespace NUMINAMATH_CALUDE_infinitely_many_special_integers_l2150_215080

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A function that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- A function that checks if a number is a perfect fifth power -/
def isPerfectFifthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

/-- The main theorem stating that there are infinitely many integers satisfying the conditions -/
theorem infinitely_many_special_integers :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
    ∀ k : ℕ, 
      isPerfectSquare (2 * f k) ∧ 
      isPerfectCube (3 * f k) ∧ 
      isPerfectFifthPower (5 * f k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_integers_l2150_215080


namespace NUMINAMATH_CALUDE_inequality_proof_l2150_215072

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + ((1 / b + 6 * c) ^ (1/3 : ℝ)) + ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2150_215072


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2150_215051

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 25 * Real.pi → d = (2 * (A / Real.pi).sqrt) → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2150_215051


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_additive_l2150_215025

/-- A function satisfying the given condition for tangential quadrilaterals is additive. -/
theorem tangential_quadrilateral_additive 
  (f : ℝ → ℝ) 
  (h : ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → 
    (∃ (r : ℝ), r > 0 ∧ a + c = b + d ∧ a * b = r * (a + b) ∧ b * c = r * (b + c) ∧ 
      c * d = r * (c + d) ∧ d * a = r * (d + a)) → 
    f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x + y) = f x + f y :=
by sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_additive_l2150_215025


namespace NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_B_l2150_215095

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 14 < 0}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3*a - 2}

-- Theorem 1: A ∪ B when a = 4
theorem union_A_B_when_a_4 : 
  A ∪ B 4 = {x | -2 < x ∧ x ≤ 10} := by sorry

-- Theorem 2: Range of a when A ∩ B = B
theorem intersection_A_B_equals_B : 
  ∀ a : ℝ, A ∩ B a = B a ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_B_l2150_215095


namespace NUMINAMATH_CALUDE_tom_seashells_count_l2150_215007

/-- The number of seashells Tom and Fred found together -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := total_seashells - fred_seashells

theorem tom_seashells_count : tom_seashells = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_count_l2150_215007


namespace NUMINAMATH_CALUDE_two_number_problem_l2150_215065

theorem two_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : 3 * y - 4 * x = 9) :
  |y - x| = 129 / 21 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l2150_215065


namespace NUMINAMATH_CALUDE_parabola_equation_l2150_215008

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y - 24 = 0

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y, p.equation x y ↔ x^2 = 2 * y) →  -- Standard form of parabola with vertex at origin and y-axis as axis of symmetry
  (∃ x y, p.equation x y ∧ line x y) →     -- Focus lies on the given line
  (∀ x y, p.equation x y ↔ x^2 = -24 * y)  -- Conclusion: The standard equation is x² = -24y
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2150_215008


namespace NUMINAMATH_CALUDE_equation_solution_l2150_215092

theorem equation_solution :
  ∃! x : ℚ, 6 * (3 * x - 1) + 7 = -3 * (2 - 5 * x) - 4 :=
by
  use -11/3
  constructor
  · -- Proof that -11/3 satisfies the equation
    sorry
  · -- Proof of uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2150_215092


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2150_215075

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2150_215075


namespace NUMINAMATH_CALUDE_complex_division_result_l2150_215021

theorem complex_division_result : (5 + Complex.I) / (1 - Complex.I) = 2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2150_215021


namespace NUMINAMATH_CALUDE_retirement_savings_l2150_215081

/-- Calculates the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the final amount after 15 years is 1,640,000 rubles -/
theorem retirement_savings : 
  let principal : ℝ := 800000
  let rate : ℝ := 0.07
  let time : ℝ := 15
  simpleInterest principal rate time = 1640000 := by
  sorry

end NUMINAMATH_CALUDE_retirement_savings_l2150_215081


namespace NUMINAMATH_CALUDE_bread_roll_flour_usage_l2150_215036

theorem bread_roll_flour_usage
  (original_rolls : ℕ) (original_flour_per_roll : ℚ)
  (new_rolls : ℕ) (new_flour_per_roll : ℚ)
  (h1 : original_rolls = 24)
  (h2 : original_flour_per_roll = 1 / 8)
  (h3 : new_rolls = 16)
  (h4 : original_rolls * original_flour_per_roll = new_rolls * new_flour_per_roll) :
  new_flour_per_roll = 3 / 16 := by
sorry

end NUMINAMATH_CALUDE_bread_roll_flour_usage_l2150_215036


namespace NUMINAMATH_CALUDE_trig_sum_equality_l2150_215084

theorem trig_sum_equality : 
  Real.sin (2 * π / 3) ^ 2 + Real.cos π + Real.tan (π / 4) - 
  Real.cos (-11 * π / 6) ^ 2 + Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l2150_215084


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2150_215030

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 8 * k) → -- n is divisible by 8
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 15) → -- last two digits sum to 15
  (n % 10) * ((n / 10) % 10) = 54 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2150_215030


namespace NUMINAMATH_CALUDE_square_sum_diff_l2150_215058

theorem square_sum_diff (a b : ℝ) 
  (h1 : (a + b)^2 = 8) 
  (h2 : (a - b)^2 = 12) : 
  a^2 + b^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_diff_l2150_215058


namespace NUMINAMATH_CALUDE_sixDigitPermutations_eq_60_l2150_215082

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 9 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 9 is equal to 60 -/
theorem sixDigitPermutations_eq_60 : sixDigitPermutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutations_eq_60_l2150_215082


namespace NUMINAMATH_CALUDE_unrestricted_x_l2150_215002

theorem unrestricted_x (x y z w : ℤ) 
  (h1 : (x + 2) / (y - 1) < -(z + 3) / (w - 2))
  (h2 : (y - 1) * (w - 2) ≠ 0) :
  ∃ (x_pos x_neg x_zero : ℤ), 
    (x_pos > 0 ∧ (x_pos + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_neg < 0 ∧ (x_neg + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_zero = 0 ∧ (x_zero + 2) / (y - 1) < -(z + 3) / (w - 2)) :=
by sorry

end NUMINAMATH_CALUDE_unrestricted_x_l2150_215002


namespace NUMINAMATH_CALUDE_square_of_binomial_b_value_l2150_215053

/-- If 9x^2 + 27x + b is the square of a binomial, then b = 81/4 -/
theorem square_of_binomial_b_value (b : ℝ) : 
  (∃ (c : ℝ), ∀ (x : ℝ), 9*x^2 + 27*x + b = (3*x + c)^2) → 
  b = 81/4 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_b_value_l2150_215053


namespace NUMINAMATH_CALUDE_min_value_expression_l2150_215091

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∀ x y : ℝ, x - y^2 = 1 → m^2 + 2*n^2 + 4*m - 1 ≤ x^2 + 2*y^2 + 4*x - 1 ∧
  ∃ a b : ℝ, a - b^2 = 1 ∧ a^2 + 2*b^2 + 4*a - 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2150_215091


namespace NUMINAMATH_CALUDE_multiple_properties_l2150_215086

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a - b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 5 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l2150_215086


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2150_215023

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The initial center of circle U -/
def initial_center : ℝ × ℝ := (3, -4)

/-- The transformation applied to the center of circle U -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_vertical (reflect_y (reflect_x p)) (-10)

theorem circle_center_transformation :
  transform initial_center = (-3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2150_215023


namespace NUMINAMATH_CALUDE_relationship_implies_function_l2150_215070

-- Define the relationship between x and y
def relationship (x y : ℝ) : Prop :=
  y = 2*x - 1 - Real.sqrt (y^2 - 2*x*y + 3*x - 2)

-- Define the function we want to prove
def function (x : ℝ) : Set ℝ :=
  if x ≠ 1 then {2*x - 1.5}
  else {y : ℝ | y ≤ 1}

-- Theorem statement
theorem relationship_implies_function :
  ∀ x y : ℝ, relationship x y → y ∈ function x :=
sorry

end NUMINAMATH_CALUDE_relationship_implies_function_l2150_215070


namespace NUMINAMATH_CALUDE_unique_number_product_sum_digits_l2150_215074

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the only number satisfying the condition -/
theorem unique_number_product_sum_digits : 
  ∃! n : ℕ, n * sum_of_digits n = 2008 ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_unique_number_product_sum_digits_l2150_215074


namespace NUMINAMATH_CALUDE_mrs_hilt_total_miles_l2150_215001

/-- Mrs. Hilt's fitness schedule for a week --/
structure FitnessSchedule where
  monday_run : ℕ
  monday_swim : ℕ
  wednesday_run : ℕ
  wednesday_bike : ℕ
  friday_run : ℕ
  friday_swim : ℕ
  friday_bike : ℕ
  sunday_bike : ℕ

/-- Calculate the total miles for a given fitness schedule --/
def total_miles (schedule : FitnessSchedule) : ℕ :=
  schedule.monday_run + schedule.monday_swim +
  schedule.wednesday_run + schedule.wednesday_bike +
  schedule.friday_run + schedule.friday_swim + schedule.friday_bike +
  schedule.sunday_bike

/-- Mrs. Hilt's actual fitness schedule --/
def mrs_hilt_schedule : FitnessSchedule := {
  monday_run := 3
  monday_swim := 1
  wednesday_run := 2
  wednesday_bike := 6
  friday_run := 7
  friday_swim := 2
  friday_bike := 3
  sunday_bike := 10
}

/-- Theorem: Mrs. Hilt's total miles for the week is 34 --/
theorem mrs_hilt_total_miles :
  total_miles mrs_hilt_schedule = 34 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_miles_l2150_215001


namespace NUMINAMATH_CALUDE_min_socks_for_twenty_pairs_l2150_215044

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (requiredPairs : Nat) : Nat :=
  5 + 2 * (requiredPairs - 1)

/-- Theorem stating the minimum number of socks needed for 20 pairs -/
theorem min_socks_for_twenty_pairs (drawer : SockDrawer) 
  (h1 : drawer.red = 120)
  (h2 : drawer.green = 100)
  (h3 : drawer.blue = 80)
  (h4 : drawer.black = 50) :
  minSocksForPairs drawer 20 = 43 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 80, black := 50 } 20

end NUMINAMATH_CALUDE_min_socks_for_twenty_pairs_l2150_215044


namespace NUMINAMATH_CALUDE_x_range_given_inequality_l2150_215005

theorem x_range_given_inequality (a : ℝ) (h_a : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) →
  {x : ℝ | x < 1 ∨ x > 3}.Nonempty :=
by sorry

end NUMINAMATH_CALUDE_x_range_given_inequality_l2150_215005


namespace NUMINAMATH_CALUDE_shortest_side_length_l2150_215015

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the side divided by the point of tangency -/
  side : ℝ
  /-- The length of the shorter segment of the divided side -/
  segment1 : ℝ
  /-- The length of the longer segment of the divided side -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.r = 3)
  (h2 : t.segment1 = 5)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), shortest_side = 12 ∧ 
  (∀ (other_side : ℝ), other_side ≥ shortest_side) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_length_l2150_215015


namespace NUMINAMATH_CALUDE_tv_sale_effect_l2150_215097

-- Define the price reduction percentage
def price_reduction : ℝ := 0.18

-- Define the sales increase percentage
def sales_increase : ℝ := 0.88

-- Define the net effect on sale value
def net_effect : ℝ := 0.5416

-- Theorem statement
theorem tv_sale_effect :
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  (new_price_factor * new_sales_factor - 1) = net_effect := by sorry

end NUMINAMATH_CALUDE_tv_sale_effect_l2150_215097


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_three_composites_l2150_215034

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_prime_sum_of_three_composites : 
  ∀ p : ℕ, Prime p → 
    (∃ a b c : ℕ, is_composite a ∧ is_composite b ∧ is_composite c ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a + b + c) → 
    p ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_three_composites_l2150_215034


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2150_215037

-- Define the complex function f(x) = x^2
def f (x : ℂ) : ℂ := x^2

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_number_in_second_quadrant :
  let z := f (1 + i) / (3 + i)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2150_215037


namespace NUMINAMATH_CALUDE_cary_needs_14_weekends_l2150_215057

/-- Calculates the number of weekends Cary needs to mow lawns to afford discounted shoes --/
def weekends_needed (
  normal_cost : ℚ
  ) (discount_percent : ℚ
  ) (saved : ℚ
  ) (bus_expense : ℚ
  ) (earnings_per_lawn : ℚ
  ) (lawns_per_weekend : ℕ
  ) : ℕ :=
  sorry

/-- Theorem stating that Cary needs 14 weekends to afford the discounted shoes --/
theorem cary_needs_14_weekends :
  weekends_needed 120 20 30 10 5 3 = 14 :=
  sorry

end NUMINAMATH_CALUDE_cary_needs_14_weekends_l2150_215057


namespace NUMINAMATH_CALUDE_sin_210_plus_cos_60_equals_zero_l2150_215045

theorem sin_210_plus_cos_60_equals_zero :
  Real.sin (210 * π / 180) + Real.cos (60 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_plus_cos_60_equals_zero_l2150_215045


namespace NUMINAMATH_CALUDE_koschei_coins_count_l2150_215068

theorem koschei_coins_count :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end NUMINAMATH_CALUDE_koschei_coins_count_l2150_215068


namespace NUMINAMATH_CALUDE_square_root_meaningful_l2150_215052

theorem square_root_meaningful (x : ℝ) : 
  x ≥ 5 → (x = 6 ∧ x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l2150_215052


namespace NUMINAMATH_CALUDE_room_width_calculation_l2150_215098

/-- Given a rectangular room with known length, flooring cost per square meter, 
    and total flooring cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l2150_215098


namespace NUMINAMATH_CALUDE_tan_product_values_l2150_215085

theorem tan_product_values (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b - 1) = 0) : 
  (Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 - Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 - Real.sqrt 133) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_tan_product_values_l2150_215085


namespace NUMINAMATH_CALUDE_A_in_second_quadrant_implies_x_gt_5_l2150_215039

/-- A point in the second quadrant of the rectangular coordinate system -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The point A with coordinates (6-2x, x-5) -/
def A (x : ℝ) : ℝ × ℝ := (6 - 2*x, x - 5)

/-- Theorem: If A(6-2x, x-5) is in the second quadrant, then x > 5 -/
theorem A_in_second_quadrant_implies_x_gt_5 :
  ∀ x : ℝ, (∃ p : SecondQuadrantPoint, A x = (p.x, p.y)) → x > 5 := by
  sorry

end NUMINAMATH_CALUDE_A_in_second_quadrant_implies_x_gt_5_l2150_215039


namespace NUMINAMATH_CALUDE_train_crossing_time_l2150_215009

/-- Represents the problem of a train crossing a stationary train -/
theorem train_crossing_time
  (train_speed : Real)
  (pole_passing_time : Real)
  (stationary_train_length : Real)
  (h1 : train_speed = 72 * 1000 / 3600) -- 72 km/h converted to m/s
  (h2 : pole_passing_time = 10)
  (h3 : stationary_train_length = 500)
  : (train_speed * pole_passing_time + stationary_train_length) / train_speed = 35 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l2150_215009


namespace NUMINAMATH_CALUDE_work_completion_time_equivalence_l2150_215033

/-- Represents the work rate of a single worker per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, work rate, and days -/
def work_done (workers : ℕ) (rate : ℝ) (days : ℕ) : ℝ :=
  (workers : ℝ) * rate * (days : ℝ)

/-- Theorem stating that if the work is completed in 40 days with varying workforce,
    it would take 45 days with a constant workforce -/
theorem work_completion_time_equivalence :
  let total_work := work_done 100 work_rate 35 + work_done 200 work_rate 5
  ∃ (days : ℕ), days = 45 ∧ work_done 100 work_rate days = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_equivalence_l2150_215033


namespace NUMINAMATH_CALUDE_petya_lives_in_sixth_entrance_l2150_215017

/-- Represents the layout of the houses -/
structure HouseLayout where
  num_entrances : ℕ
  petya_entrance : ℕ
  vasya_entrance : ℕ

/-- Calculates the distance between two entrances -/
def distance (layout : HouseLayout) (entrance1 entrance2 : ℕ) : ℝ :=
  sorry

/-- Represents the shortest path around Petya's house -/
def shortest_path (layout : HouseLayout) (side : Bool) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem petya_lives_in_sixth_entrance (layout : HouseLayout) :
  layout.vasya_entrance = 4 →
  shortest_path layout true = shortest_path layout false →
  layout.petya_entrance = 6 :=
sorry

end NUMINAMATH_CALUDE_petya_lives_in_sixth_entrance_l2150_215017


namespace NUMINAMATH_CALUDE_don_raise_is_880_l2150_215088

/-- Calculates Don's raise given the conditions of the problem -/
def calculate_don_raise (wife_raise : ℚ) (salary_difference : ℚ) : ℚ :=
  let wife_salary := wife_raise / 0.08
  let don_salary := (wife_salary + salary_difference + wife_raise) / 1.08
  0.08 * don_salary

/-- Theorem stating that Don's raise is 880 given the problem conditions -/
theorem don_raise_is_880 :
  calculate_don_raise 840 540 = 880 := by
  sorry

end NUMINAMATH_CALUDE_don_raise_is_880_l2150_215088


namespace NUMINAMATH_CALUDE_lindas_nickels_l2150_215069

/-- The number of nickels Linda initially has -/
def initial_nickels : ℕ := 5

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

/-- The number of initial dimes -/
def initial_dimes : ℕ := 2

/-- The number of initial quarters -/
def initial_quarters : ℕ := 6

/-- The number of additional dimes given by her mother -/
def additional_dimes : ℕ := 2

/-- The number of additional quarters given by her mother -/
def additional_quarters : ℕ := 10

theorem lindas_nickels :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end NUMINAMATH_CALUDE_lindas_nickels_l2150_215069


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2150_215059

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = x + (x + 1) + (x + 2) + (x + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2150_215059


namespace NUMINAMATH_CALUDE_work_completion_time_l2150_215010

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 55
def work_rate_C : ℚ := 1 / 45

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the time taken to complete the work together
def time_to_complete : ℚ := 1 / combined_work_rate

-- Theorem statement
theorem work_completion_time :
  time_to_complete = 55 / 4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2150_215010


namespace NUMINAMATH_CALUDE_dividend_percentage_l2150_215076

theorem dividend_percentage 
  (face_value : ℝ) 
  (purchase_price : ℝ) 
  (return_on_investment : ℝ) 
  (h1 : face_value = 50) 
  (h2 : purchase_price = 31) 
  (h3 : return_on_investment = 0.25) : 
  (return_on_investment * purchase_price) / face_value * 100 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_l2150_215076


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2150_215071

theorem cubic_sum_minus_product (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (sum_product_eq : x * y + x * z + y * z = 30) :
  x^3 + y^3 + z^3 - 3*x*y*z = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2150_215071


namespace NUMINAMATH_CALUDE_inequality_proof_l2150_215042

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2150_215042


namespace NUMINAMATH_CALUDE_min_containers_for_85_units_l2150_215087

/-- Represents the possible container sizes for snacks -/
inductive ContainerSize
  | small : ContainerSize  -- 5 units
  | medium : ContainerSize -- 10 units
  | large : ContainerSize  -- 20 units

/-- Returns the number of units in a given container size -/
def containerUnits (size : ContainerSize) : Nat :=
  match size with
  | .small => 5
  | .medium => 10
  | .large => 20

/-- Represents a combination of containers -/
structure ContainerCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of units in a combination of containers -/
def totalUnits (combo : ContainerCombination) : Nat :=
  combo.small * containerUnits ContainerSize.small +
  combo.medium * containerUnits ContainerSize.medium +
  combo.large * containerUnits ContainerSize.large

/-- Calculates the total number of containers in a combination -/
def totalContainers (combo : ContainerCombination) : Nat :=
  combo.small + combo.medium + combo.large

/-- Theorem: The minimum number of containers to get exactly 85 units is 5 -/
theorem min_containers_for_85_units :
  ∃ (combo : ContainerCombination),
    totalUnits combo = 85 ∧
    totalContainers combo = 5 ∧
    (∀ (other : ContainerCombination),
      totalUnits other = 85 → totalContainers other ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_min_containers_for_85_units_l2150_215087


namespace NUMINAMATH_CALUDE_accommodation_arrangements_theorem_l2150_215043

/-- The number of ways to arrange 5 people in 3 rooms with constraints -/
def accommodationArrangements (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 5 people in 3 rooms with A and B not sharing -/
def accommodationArrangementsWithConstraint (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

theorem accommodation_arrangements_theorem :
  accommodationArrangementsWithConstraint 5 3 2 = 72 :=
sorry

end NUMINAMATH_CALUDE_accommodation_arrangements_theorem_l2150_215043


namespace NUMINAMATH_CALUDE_sand_loss_l2150_215040

theorem sand_loss (initial_sand final_sand : ℝ) 
  (h_initial : initial_sand = 4.1)
  (h_final : final_sand = 1.7) : 
  initial_sand - final_sand = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sand_loss_l2150_215040


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2150_215062

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, its real axis has length 4 -/
theorem hyperbola_real_axis_length :
  ∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (3*a^2) = 1 ∧ 2 * a = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2150_215062


namespace NUMINAMATH_CALUDE_selina_shirt_cost_l2150_215046

/-- Represents the price and quantity of an item of clothing --/
structure ClothingItem where
  price : ℕ
  quantity : ℕ

/-- Calculates the total money Selina got from selling her clothes --/
def totalSalesMoney (pants shorts shirts : ClothingItem) : ℕ :=
  pants.price * pants.quantity + shorts.price * shorts.quantity + shirts.price * shirts.quantity

/-- Represents the problem of finding the cost of each shirt Selina bought --/
theorem selina_shirt_cost (pants shorts shirts : ClothingItem)
  (bought_shirts : ℕ) (money_left : ℕ)
  (h_pants : pants = ⟨5, 3⟩)
  (h_shorts : shorts = ⟨3, 5⟩)
  (h_shirts : shirts = ⟨4, 5⟩)
  (h_bought_shirts : bought_shirts = 2)
  (h_money_left : money_left = 30) :
  (totalSalesMoney pants shorts shirts - money_left) / bought_shirts = 10 := by
  sorry

#check selina_shirt_cost

end NUMINAMATH_CALUDE_selina_shirt_cost_l2150_215046


namespace NUMINAMATH_CALUDE_circle_segment_ratio_l2150_215022

theorem circle_segment_ratio : 
  ∀ (r : ℝ) (S₁ S₂ : ℝ), 
  r > 0 → 
  S₁ = (1 / 12) * r^2 * (4 * π - 3 * Real.sqrt 3) →
  S₂ = (1 / 12) * r^2 * (8 * π + 3 * Real.sqrt 3) →
  S₁ / S₂ = (4 * π - 3 * Real.sqrt 3) / (8 * π + 3 * Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_circle_segment_ratio_l2150_215022


namespace NUMINAMATH_CALUDE_inequality_range_l2150_215056

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2150_215056


namespace NUMINAMATH_CALUDE_nathaniel_win_probability_l2150_215050

/-- A fair six-sided die -/
def FairDie : Type := Fin 6

/-- The game state -/
structure GameState :=
  (sum : ℕ)
  (currentPlayer : Bool)  -- true for Nathaniel, false for Obediah

/-- Check if a number is a multiple of 7 -/
def isMultipleOf7 (n : ℕ) : Bool :=
  n % 7 = 0

/-- The probability of Nathaniel winning the game -/
noncomputable def nathanielWinProbability : ℝ :=
  5 / 11

/-- Theorem: The probability of Nathaniel winning the game is 5/11 -/
theorem nathaniel_win_probability :
  nathanielWinProbability = 5 / 11 := by
  sorry

#check nathaniel_win_probability

end NUMINAMATH_CALUDE_nathaniel_win_probability_l2150_215050


namespace NUMINAMATH_CALUDE_sum_of_cubes_counterexample_l2150_215083

theorem sum_of_cubes_counterexample : ¬∀ a : ℝ, (a + 1) * (a^2 - a + 1) = a^3 + 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_counterexample_l2150_215083


namespace NUMINAMATH_CALUDE_window_installation_time_l2150_215041

theorem window_installation_time 
  (total_windows : ℕ) 
  (installed_windows : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_windows = 9)
  (h2 : installed_windows = 6)
  (h3 : remaining_time = 18)
  (h4 : installed_windows < total_windows) :
  (remaining_time : ℚ) / (total_windows - installed_windows : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_window_installation_time_l2150_215041


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2150_215029

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC, if 2b * sin(2A) = a * sin(B) and c = 2b, then a/b = 2 -/
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.sin (2 * t.A) = t.a * Real.sin t.B)
  (h2 : t.c = 2 * t.b) :
  t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2150_215029


namespace NUMINAMATH_CALUDE_mindy_tax_rate_is_25_percent_l2150_215079

/-- Calculates Mindy's tax rate given Mork's tax rate, their income ratio, and combined tax rate -/
def mindyTaxRate (morkTaxRate : ℚ) (incomeRatio : ℚ) (combinedTaxRate : ℚ) : ℚ :=
  (combinedTaxRate * (1 + incomeRatio) - morkTaxRate) / incomeRatio

/-- Proves that Mindy's tax rate is 25% given the specified conditions -/
theorem mindy_tax_rate_is_25_percent :
  mindyTaxRate (40 / 100) 4 (28 / 100) = 25 / 100 := by
  sorry

#eval mindyTaxRate (40 / 100) 4 (28 / 100)

end NUMINAMATH_CALUDE_mindy_tax_rate_is_25_percent_l2150_215079


namespace NUMINAMATH_CALUDE_fraction_equality_l2150_215089

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 2 / 3)
  (h2 : (x + y) / y = 3) :
  w / x = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2150_215089


namespace NUMINAMATH_CALUDE_min_edges_after_operations_l2150_215019

/-- A complete graph with n vertices. -/
structure CompleteGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  complete : ∀ i j : Fin n, i ≠ j → (i, j) ∈ edges

/-- An elementary operation on a graph. -/
def elementaryOperation (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The result of applying any number of elementary operations. -/
def resultGraph (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The number of edges in a graph. -/
def numEdges (G : CompleteGraph n) : ℕ :=
  G.edges.card

theorem min_edges_after_operations (n : ℕ) (G : CompleteGraph n) (H : CompleteGraph n) :
  resultGraph G H → numEdges H ≥ n :=
  sorry

end NUMINAMATH_CALUDE_min_edges_after_operations_l2150_215019


namespace NUMINAMATH_CALUDE_debate_team_count_l2150_215066

/-- The number of girls in the debate club -/
def num_girls : ℕ := 4

/-- The number of boys in the debate club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for each team -/
def girls_per_team : ℕ := 3

/-- The number of boys to be chosen for each team -/
def boys_per_team : ℕ := 3

/-- Theorem stating the total number of possible debate teams -/
theorem debate_team_count : 
  (Nat.choose num_girls girls_per_team) * (Nat.choose num_boys boys_per_team) = 80 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_count_l2150_215066


namespace NUMINAMATH_CALUDE_spotlight_detection_l2150_215018

/-- Represents the spotlight's properties -/
structure Spotlight where
  illumination_length : ℝ  -- Length of illuminated segment in km
  rotation_period : ℝ      -- Time for one complete rotation in minutes

/-- Represents the boat's properties -/
structure Boat where
  speed : ℝ  -- Speed in km/min

/-- Determines if a boat can approach undetected given a spotlight -/
def can_approach_undetected (s : Spotlight) (b : Boat) : Prop :=
  b.speed ≥ 48.6 / 60  -- Convert 48.6 km/h to km/min

theorem spotlight_detection (s : Spotlight) (b : Boat) :
  s.illumination_length = 1 ∧ s.rotation_period = 1 →
  (b.speed < 800 / 1000 → ¬can_approach_undetected s b) ∧
  (b.speed ≥ 48.6 / 60 → can_approach_undetected s b) := by
  sorry

#check spotlight_detection

end NUMINAMATH_CALUDE_spotlight_detection_l2150_215018


namespace NUMINAMATH_CALUDE_f_composition_half_l2150_215003

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half : f (f (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l2150_215003


namespace NUMINAMATH_CALUDE_price_difference_is_80_cents_l2150_215000

/-- Represents the price calculation methods in Lintonville Fashion Store --/
def price_calculation (original_price discount_rate tax_rate coupon : ℝ) : ℝ × ℝ := 
  let bob_total := (original_price * (1 + tax_rate) * (1 - discount_rate)) - coupon
  let alice_total := (original_price * (1 - discount_rate) - coupon) * (1 + tax_rate)
  (bob_total, alice_total)

/-- The difference between Bob's and Alice's calculations is $0.80 --/
theorem price_difference_is_80_cents 
  (h_original_price : ℝ) 
  (h_discount_rate : ℝ) 
  (h_tax_rate : ℝ) 
  (h_coupon : ℝ) 
  (h_op : h_original_price = 120)
  (h_dr : h_discount_rate = 0.15)
  (h_tr : h_tax_rate = 0.08)
  (h_c : h_coupon = 10) : 
  let (bob_total, alice_total) := price_calculation h_original_price h_discount_rate h_tax_rate h_coupon
  bob_total - alice_total = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_is_80_cents_l2150_215000


namespace NUMINAMATH_CALUDE_die_game_expected_value_l2150_215011

/-- A fair 8-sided die game where you win the rolled amount if it's a multiple of 3 -/
def die_game : ℝ := by sorry

/-- The expected value of the die game -/
theorem die_game_expected_value : die_game = 2.25 := by sorry

end NUMINAMATH_CALUDE_die_game_expected_value_l2150_215011


namespace NUMINAMATH_CALUDE_sequence_bounds_l2150_215004

theorem sequence_bounds (n : ℕ+) (a : ℕ → ℚ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bounds_l2150_215004


namespace NUMINAMATH_CALUDE_problem_solution_l2150_215013

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 30)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2150_215013


namespace NUMINAMATH_CALUDE_complex_calculation_l2150_215099

theorem complex_calculation : 
  let z : ℂ := 1 + I
  z^2 - 2/z = -1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l2150_215099


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_length_l2150_215060

noncomputable def angleBisectorLength (PQ PR : ℝ) (cosP : ℝ) : ℝ :=
  let QR := Real.sqrt (PQ^2 + PR^2 - 2 * PQ * PR * cosP)
  let cosHalfP := Real.sqrt ((1 + cosP) / 2)
  let QT := (5 * Real.sqrt 73) / 13
  Real.sqrt (PQ^2 + QT^2 - 2 * PQ * QT * cosHalfP)

theorem triangle_angle_bisector_length :
  ∀ (ε : ℝ), ε > 0 → 
  |angleBisectorLength 5 8 (1/5) - 5.05| < ε :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_length_l2150_215060


namespace NUMINAMATH_CALUDE_shipping_percentage_above_50_l2150_215063

def flat_rate_shipping : Real := 5.00
def min_purchase_for_percentage : Real := 50.00

def shirt_price : Real := 12.00
def shirt_quantity : Nat := 3
def socks_price : Real := 5.00
def shorts_price : Real := 15.00
def shorts_quantity : Nat := 2
def swim_trunks_price : Real := 14.00

def total_purchase : Real := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def total_bill : Real := 102.00

theorem shipping_percentage_above_50 :
  total_purchase > min_purchase_for_percentage →
  (total_bill - total_purchase) / total_purchase * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_shipping_percentage_above_50_l2150_215063


namespace NUMINAMATH_CALUDE_min_value_when_a_neg_one_max_value_case1_max_value_case2_l2150_215035

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem for the minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∀ x ∈ Set.Icc 0 2, f (-1) x ≥ -2 :=
sorry

-- Theorem for the maximum value when -2 ≤ a ≤ -1/4
theorem max_value_case1 (a : ℝ) (h : a ∈ Set.Icc (-2) (-1/4)) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ -1 / (4 * a) :=
sorry

-- Theorem for the maximum value when -1/4 < a ≤ 0
theorem max_value_case2 (a : ℝ) (h : a ∈ Set.Ioo (-1/4) 0) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ 4 * a + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_neg_one_max_value_case1_max_value_case2_l2150_215035


namespace NUMINAMATH_CALUDE_connor_date_cost_l2150_215032

/-- The cost of Connor's movie date -/
def movie_date_cost (ticket_price : ℚ) (ticket_quantity : ℕ) (combo_meal_price : ℚ) (candy_price : ℚ) (candy_quantity : ℕ) : ℚ :=
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem: Connor's movie date costs $36.00 -/
theorem connor_date_cost :
  movie_date_cost 10 2 11 (5/2) 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_connor_date_cost_l2150_215032


namespace NUMINAMATH_CALUDE_triangle_angle_a_value_l2150_215047

theorem triangle_angle_a_value (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = Real.sin B + Real.cos B ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  a < b →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_a_value_l2150_215047


namespace NUMINAMATH_CALUDE_product_equals_sum_solutions_l2150_215016

theorem product_equals_sum_solutions (x y : ℤ) :
  x * y = x + y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_solutions_l2150_215016


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_calculation_l2150_215064

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_speed * crossing_time / 3600 * 1000
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_calculation : 
  bridge_length 110 45 30 = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_calculation_l2150_215064


namespace NUMINAMATH_CALUDE_consecutive_primes_square_sum_prime_l2150_215090

/-- Definition of consecutive primes -/
def ConsecutivePrimes (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ∃ (x y : Nat), p < x ∧ ¬Nat.Prime x ∧ x < q ∧
                 q < y ∧ ¬Nat.Prime y ∧ y < r

theorem consecutive_primes_square_sum_prime :
  ∀ p q r : Nat,
    ConsecutivePrimes p q r →
    Nat.Prime (p^2 + q^2 + r^2) →
    p = 3 ∧ q = 5 ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_square_sum_prime_l2150_215090


namespace NUMINAMATH_CALUDE_investment_total_l2150_215096

/-- Represents the investment scenario with two parts at different interest rates -/
structure Investment where
  total : ℝ
  part1 : ℝ
  part2 : ℝ
  rate1 : ℝ
  rate2 : ℝ
  total_interest : ℝ

/-- The investment satisfies the given conditions -/
def valid_investment (i : Investment) : Prop :=
  i.total = i.part1 + i.part2 ∧
  i.part1 = 2800 ∧
  i.rate1 = 0.03 ∧
  i.rate2 = 0.05 ∧
  i.total_interest = 144 ∧
  i.part1 * i.rate1 + i.part2 * i.rate2 = i.total_interest

/-- Theorem: Given the conditions, the total amount divided is 4000 -/
theorem investment_total (i : Investment) (h : valid_investment i) : i.total = 4000 := by
  sorry

end NUMINAMATH_CALUDE_investment_total_l2150_215096


namespace NUMINAMATH_CALUDE_digits_of_2_pow_100_l2150_215031

theorem digits_of_2_pow_100 (h : ∃ n : ℕ, 10^(n-1) ≤ 2^200 ∧ 2^200 < 10^n ∧ n = 61) :
  ∃ m : ℕ, 10^(m-1) ≤ 2^100 ∧ 2^100 < 10^m ∧ m = 31 :=
by sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_100_l2150_215031


namespace NUMINAMATH_CALUDE_inequality_proof_l2150_215026

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2150_215026


namespace NUMINAMATH_CALUDE_ladder_problem_l2150_215093

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l2150_215093


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2150_215054

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_population_size_pos : 0 < population_size
  h_sample_size_pos : 0 < sample_size
  h_sample_size_le_population : sample_size ≤ population_size
  h_first_element_in_range : first_element ≤ population_size

/-- Check if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * (s.population_size / s.sample_size) ∧ n ≤ s.population_size

/-- The main theorem to be proved -/
theorem systematic_sample_theorem (s : SystematicSample)
  (h_pop_size : s.population_size = 60)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_33 : s.contains 33)
  (h_contains_48 : s.contains 48) :
  s.contains 18 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_theorem_l2150_215054


namespace NUMINAMATH_CALUDE_max_police_officers_l2150_215020

theorem max_police_officers (n : ℕ) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 :=
sorry

end NUMINAMATH_CALUDE_max_police_officers_l2150_215020


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2150_215006

/-- Represents a repeating decimal with a single-digit repetend -/
def SingleDigitRepeatDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def TwoDigitRepeatDecimal (n : ℕ) : ℚ := n / 99

/-- Represents a repeating decimal with a three-digit repetend -/
def ThreeDigitRepeatDecimal (n : ℕ) : ℚ := n / 999

/-- The sum of 0.1̅, 0.02̅, and 0.003̅ is equal to 164/1221 -/
theorem sum_of_repeating_decimals :
  SingleDigitRepeatDecimal 1 + TwoDigitRepeatDecimal 2 + ThreeDigitRepeatDecimal 3 = 164 / 1221 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2150_215006


namespace NUMINAMATH_CALUDE_part1_part2_l2150_215049

-- Part 1
def f (x : ℝ) : ℝ := |2*x - 2| + 2

theorem part1 : {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
def g (x : ℝ) : ℝ := |2*x - 1|

def h (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

theorem part2 : {a : ℝ | ∀ x : ℝ, h a x + g x ≥ 3} = {a : ℝ | 2 ≤ a} := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2150_215049


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l2150_215048

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l2150_215048


namespace NUMINAMATH_CALUDE_teacher_wang_pen_purchase_l2150_215067

/-- Given that Teacher Wang has enough money to buy 72 pens at 5 yuan each,
    prove that he can buy 60 pens when the price increases to 6 yuan each. -/
theorem teacher_wang_pen_purchase (initial_pens : ℕ) (initial_price : ℕ) (new_price : ℕ) :
  initial_pens = 72 → initial_price = 5 → new_price = 6 →
  (initial_pens * initial_price) / new_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_teacher_wang_pen_purchase_l2150_215067


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2150_215061

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2150_215061


namespace NUMINAMATH_CALUDE_cape_may_sharks_l2150_215077

theorem cape_may_sharks (daytona_sharks : ℕ) (cape_may_sharks : ℕ) : 
  daytona_sharks = 12 → 
  cape_may_sharks = 2 * daytona_sharks + 8 → 
  cape_may_sharks = 32 := by
sorry

end NUMINAMATH_CALUDE_cape_may_sharks_l2150_215077


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2150_215012

theorem balls_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (k : ℕ) ^ n = 64 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2150_215012


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2150_215094

/-- A function f: ℝ₊ → ℝ₊ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f y / f x + 1) = f (x + y / x + 1) - f x

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = ax for some positive constant a -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2150_215094


namespace NUMINAMATH_CALUDE_units_digit_7_pow_6_5_l2150_215078

/-- The units digit of 7^n for n > 0 -/
def unitsDigit7Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | 0 => 1
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

/-- 6^5 mod 4 equals 0 -/
axiom pow6_5_mod4 : (6^5) % 4 = 0

/-- The main theorem: The units digit of 7^(6^5) is 1 -/
theorem units_digit_7_pow_6_5 : unitsDigit7Power (6^5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_7_pow_6_5_l2150_215078


namespace NUMINAMATH_CALUDE_hiker_distance_problem_l2150_215073

theorem hiker_distance_problem (v t d : ℝ) :
  v > 0 ∧ t > 0 ∧ d > 0 ∧
  d = v * t ∧
  d = (v + 1) * (3 * t / 4) ∧
  d = (v - 1) * (t + 3) →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_hiker_distance_problem_l2150_215073


namespace NUMINAMATH_CALUDE_prize_money_problem_l2150_215028

/-- The prize money problem -/
theorem prize_money_problem (total_students : Nat) (team_members : Nat) (member_prize : Nat) (extra_prize : Nat) :
  total_students = 10 →
  team_members = 9 →
  member_prize = 200 →
  extra_prize = 90 →
  ∃ (captain_prize : Nat),
    captain_prize = extra_prize + (captain_prize + team_members * member_prize) / total_students ∧
    captain_prize = 300 := by
  sorry

end NUMINAMATH_CALUDE_prize_money_problem_l2150_215028


namespace NUMINAMATH_CALUDE_polo_shirt_cost_l2150_215038

/-- Calculates the total cost of two discounted polo shirts with sales tax -/
theorem polo_shirt_cost : 
  let regular_price : ℝ := 50
  let discount1 : ℝ := 0.4
  let discount2 : ℝ := 0.3
  let sales_tax : ℝ := 0.08
  let discounted_price1 := regular_price * (1 - discount1)
  let discounted_price2 := regular_price * (1 - discount2)
  let total_before_tax := discounted_price1 + discounted_price2
  let total_with_tax := total_before_tax * (1 + sales_tax)
  total_with_tax = 70.20 := by sorry

end NUMINAMATH_CALUDE_polo_shirt_cost_l2150_215038


namespace NUMINAMATH_CALUDE_negation_existential_l2150_215024

theorem negation_existential (f : ℝ → Prop) :
  (¬ ∃ x₀ > -1, x₀^2 + x₀ - 2018 > 0) ↔ (∀ x > -1, x^2 + x - 2018 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existential_l2150_215024


namespace NUMINAMATH_CALUDE_product_of_arithmetic_sequences_l2150_215055

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The product sequence of two arithmetic sequences -/
def product_seq (a b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = a n * b n

theorem product_of_arithmetic_sequences
  (a b : ℕ → ℝ) (c : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hb : arithmetic_seq b)
  (hc : product_seq a b c)
  (h1 : c 1 = 1440)
  (h2 : c 2 = 1716)
  (h3 : c 3 = 1848) :
  c 8 = 348 := by
  sorry

end NUMINAMATH_CALUDE_product_of_arithmetic_sequences_l2150_215055

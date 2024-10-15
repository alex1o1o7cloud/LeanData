import Mathlib

namespace NUMINAMATH_CALUDE_no_prime_pair_sum_51_l282_28202

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_pair_sum_51 :
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 51 :=
sorry

end NUMINAMATH_CALUDE_no_prime_pair_sum_51_l282_28202


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l282_28228

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) ↔ (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l282_28228


namespace NUMINAMATH_CALUDE_min_value_expression_l282_28274

theorem min_value_expression (a x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hx : -a < x ∧ x < a) 
  (hy : -a < y ∧ y < a) 
  (hz : -a < z ∧ z < a) : 
  (∀ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 / (1 - a^2)^3) ∧ 
  (∃ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2 / (1 - a^2)^3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l282_28274


namespace NUMINAMATH_CALUDE_puppy_sleep_ratio_l282_28248

theorem puppy_sleep_ratio (connor_sleep : ℕ) (luke_extra_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_extra_sleep = 2 →
  puppy_sleep = 16 →
  (puppy_sleep : ℚ) / (connor_sleep + luke_extra_sleep) = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sleep_ratio_l282_28248


namespace NUMINAMATH_CALUDE_optimal_height_minimizes_surface_area_l282_28294

/-- Represents a rectangular box with a lid -/
structure Box where
  x : ℝ  -- Length of one side of the base
  y : ℝ  -- Height of the box

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := 2 * b.x^2 * b.y

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 4 * b.x^2 + 6 * b.x * b.y

/-- States that the volume of the box is 72 -/
def volumeConstraint (b : Box) : Prop := volume b = 72

/-- Finds the height that minimizes the surface area -/
def optimalHeight : ℝ := 4

theorem optimal_height_minimizes_surface_area :
  ∃ (b : Box), volumeConstraint b ∧
    ∀ (b' : Box), volumeConstraint b' → surfaceArea b ≤ surfaceArea b' ∧
    b.y = optimalHeight := by sorry

end NUMINAMATH_CALUDE_optimal_height_minimizes_surface_area_l282_28294


namespace NUMINAMATH_CALUDE_stacy_height_last_year_l282_28222

/-- Represents Stacy's height measurements and growth --/
structure StacyHeight where
  current : ℕ
  brother_growth : ℕ
  growth_difference : ℕ

/-- Calculates Stacy's height last year given her current measurements --/
def height_last_year (s : StacyHeight) : ℕ :=
  s.current - (s.brother_growth + s.growth_difference)

/-- Theorem stating Stacy's height last year was 50 inches --/
theorem stacy_height_last_year :
  let s : StacyHeight := {
    current := 57,
    brother_growth := 1,
    growth_difference := 6
  }
  height_last_year s = 50 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_last_year_l282_28222


namespace NUMINAMATH_CALUDE_function_inequality_l282_28282

/-- Given a function f(x) = 2^((a-x)^2) where a is a real number,
    if f(1) > f(3) and f(2) > f(3), then |a-1| > |a-2|. -/
theorem function_inequality (a : ℝ) :
  let f : ℝ → ℝ := λ x => 2^((a-x)^2)
  (f 1 > f 3) → (f 2 > f 3) → |a-1| > |a-2| := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l282_28282


namespace NUMINAMATH_CALUDE_select_staff_eq_36_l282_28204

/-- The number of ways to select staff for an event -/
def select_staff : ℕ :=
  let n_volunteers : ℕ := 5
  let n_translators : ℕ := 2
  let n_guides : ℕ := 2
  let n_flexible : ℕ := 1
  let n_abc : ℕ := 3  -- number of volunteers named A, B, or C

  -- Definition: Ways to choose at least one from A, B, C for translators and guides
  let ways_abc : ℕ := n_abc.choose n_translators

  -- Definition: Ways to arrange remaining volunteers
  let ways_arrange : ℕ := (n_volunteers - n_translators - n_guides).factorial

  ways_abc * ways_arrange

/-- Theorem: The number of ways to select staff is 36 -/
theorem select_staff_eq_36 : select_staff = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_staff_eq_36_l282_28204


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_400_l282_28203

theorem greatest_sum_consecutive_integers_product_less_400 :
  (∀ n : ℤ, n * (n + 1) < 400 → n + (n + 1) ≤ 39) ∧
  (∃ n : ℤ, n * (n + 1) < 400 ∧ n + (n + 1) = 39) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_400_l282_28203


namespace NUMINAMATH_CALUDE_triangle_existence_l282_28216

/-- Given an angle and two segments representing differences between sides,
    prove the existence of a triangle with these properties. -/
theorem triangle_existence (A : Real) (d e : ℝ) : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
  a - c = d ∧              -- given difference d
  b - c = e ∧              -- given difference e
  0 < A ∧ A < π ∧          -- valid angle measure
  -- The angle A is the smallest in the triangle
  A ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
  A ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l282_28216


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l282_28275

theorem arctan_tan_difference (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  Real.arctan (Real.tan x - 2 * Real.tan y) = 25 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l282_28275


namespace NUMINAMATH_CALUDE_line_equation_proof_l282_28251

/-- Given a line defined by (-3, 4) · ((x, y) - (2, -6)) = 0, 
    prove that its slope-intercept form is y = (3/4)x - 7.5 
    and consequently, m = 3/4 and b = -7.5 -/
theorem line_equation_proof (x y : ℝ) : 
  (-3 : ℝ) * (x - 2) + 4 * (y + 6) = 0 → 
  y = (3/4 : ℝ) * x - (15/2 : ℝ) ∧ 
  (3/4 : ℝ) = (3/4 : ℝ) ∧ 
  -(15/2 : ℝ) = -(15/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l282_28251


namespace NUMINAMATH_CALUDE_blue_square_area_ratio_l282_28298

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  /-- Side length of the square flag -/
  side : ℝ
  /-- Side length of the blue square in the center -/
  blue_side : ℝ
  /-- The cross (red arms + blue center) occupies 36% of the flag's area -/
  cross_area_ratio : side ^ 2 * 0.36 = (4 * blue_side * (side - blue_side) + blue_side ^ 2)

/-- The blue square in the center occupies 9% of the flag's area -/
theorem blue_square_area_ratio (flag : CrossFlag) : 
  flag.blue_side ^ 2 / flag.side ^ 2 = 0.09 := by sorry

end NUMINAMATH_CALUDE_blue_square_area_ratio_l282_28298


namespace NUMINAMATH_CALUDE_rational_abs_eq_neg_l282_28235

theorem rational_abs_eq_neg (a : ℚ) (h : |a| = -a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_abs_eq_neg_l282_28235


namespace NUMINAMATH_CALUDE_prime_squared_plus_two_l282_28232

theorem prime_squared_plus_two (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 2) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_squared_plus_two_l282_28232


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_1500_by_20_percent_l282_28223

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) : 
  initial + (initial * percentage) = initial * (1 + percentage) := by sorry

theorem increase_1500_by_20_percent : 
  1500 + (1500 * (20 / 100)) = 1800 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_1500_by_20_percent_l282_28223


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l282_28237

/-- Proves that the percentage of alcohol in solution y is 30% -/
theorem alcohol_percentage_solution_y :
  let solution_x_volume : ℝ := 200
  let solution_y_volume : ℝ := 600
  let solution_x_percentage : ℝ := 10
  let final_mixture_percentage : ℝ := 25
  let total_volume : ℝ := solution_x_volume + solution_y_volume
  let solution_y_percentage : ℝ := 
    ((final_mixture_percentage / 100) * total_volume - (solution_x_percentage / 100) * solution_x_volume) / 
    solution_y_volume * 100
  solution_y_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l282_28237


namespace NUMINAMATH_CALUDE_louise_teddy_bears_louise_teddy_bears_correct_l282_28288

theorem louise_teddy_bears (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) : ℕ :=
  let remaining_money := total_money - initial_toys * initial_toy_cost
  remaining_money / teddy_bear_cost

theorem louise_teddy_bears_correct 
  (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) :
  louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost = 20 ∧
  initial_toys = 28 ∧
  initial_toy_cost = 10 ∧
  total_money = 580 ∧
  teddy_bear_cost = 15 ∧
  total_money = initial_toys * initial_toy_cost + 
    (louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost) * teddy_bear_cost :=
by
  sorry

end NUMINAMATH_CALUDE_louise_teddy_bears_louise_teddy_bears_correct_l282_28288


namespace NUMINAMATH_CALUDE_sequence_max_ratio_l282_28214

theorem sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n = (n + 1) / 2 * a n) →
  (∃ M : ℝ, ∀ n : ℕ, n > 1 → a n / a (n - 1) ≤ M) ∧
  (∀ ε > 0, ∃ n : ℕ, n > 1 ∧ a n / a (n - 1) > 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_max_ratio_l282_28214


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l282_28289

theorem simplify_sqrt_expression :
  (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 147 / Real.sqrt 63) = (42 - 7 * Real.sqrt 21) / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l282_28289


namespace NUMINAMATH_CALUDE_mr_callen_loss_l282_28277

def paintings_count : ℕ := 15
def paintings_price : ℚ := 60
def wooden_toys_count : ℕ := 12
def wooden_toys_price : ℚ := 25
def hats_count : ℕ := 20
def hats_price : ℚ := 15

def paintings_loss_percentage : ℚ := 18 / 100
def wooden_toys_loss_percentage : ℚ := 25 / 100
def hats_loss_percentage : ℚ := 10 / 100

def total_cost : ℚ := 
  paintings_count * paintings_price + 
  wooden_toys_count * wooden_toys_price + 
  hats_count * hats_price

def total_selling_price : ℚ := 
  paintings_count * paintings_price * (1 - paintings_loss_percentage) +
  wooden_toys_count * wooden_toys_price * (1 - wooden_toys_loss_percentage) +
  hats_count * hats_price * (1 - hats_loss_percentage)

def total_loss : ℚ := total_cost - total_selling_price

theorem mr_callen_loss : total_loss = 267 := by
  sorry

end NUMINAMATH_CALUDE_mr_callen_loss_l282_28277


namespace NUMINAMATH_CALUDE_two_colonies_growth_time_l282_28212

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitatLimitDays : ℕ := 25

/-- Represents the daily growth factor of a bacteria colony -/
def dailyGrowthFactor : ℕ := 2

/-- Theorem stating that two simultaneously growing bacteria colonies 
    will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_growth_time (initialSize : ℕ) (habitatLimit : ℕ) :
  initialSize > 0 →
  habitatLimit > 0 →
  habitatLimit = initialSize * dailyGrowthFactor ^ habitatLimitDays →
  (2 * initialSize) * dailyGrowthFactor ^ habitatLimitDays = 2 * habitatLimit :=
by
  sorry

end NUMINAMATH_CALUDE_two_colonies_growth_time_l282_28212


namespace NUMINAMATH_CALUDE_max_blocks_fit_l282_28211

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The dimensions of the large box -/
def largeBox : BoxDimensions := ⟨3, 3, 2⟩

/-- The dimensions of the small block -/
def smallBlock : BoxDimensions := ⟨1, 2, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Represents the number of small blocks that can fit in the large box -/
def maxBlocks : ℕ := 3

/-- Theorem stating that the maximum number of small blocks that can fit in the large box is 3 -/
theorem max_blocks_fit :
  maxBlocks = 3 ∧
  maxBlocks * volume smallBlock ≤ volume largeBox ∧
  ∀ n : ℕ, n > maxBlocks → n * volume smallBlock > volume largeBox :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l282_28211


namespace NUMINAMATH_CALUDE_smallest_n_for_good_sequence_2014_l282_28233

/-- A sequence of real numbers is good if it satisfies certain conditions. -/
def IsGoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a k = 2014) ∧
  (∃ n : ℕ+, a 0 = n) ∧
  ∀ i : ℕ, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

/-- The smallest positive integer n such that there exists a good sequence with aₙ = 2014 is 60. -/
theorem smallest_n_for_good_sequence_2014 :
  ∃ (a : ℕ → ℝ), IsGoodSequence a ∧ a 60 = 2014 ∧
  ∀ (b : ℕ → ℝ) (m : ℕ), m < 60 → IsGoodSequence b → b m ≠ 2014 := by
  sorry

#check smallest_n_for_good_sequence_2014

end NUMINAMATH_CALUDE_smallest_n_for_good_sequence_2014_l282_28233


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l282_28215

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 ∧ 5 ∣ n → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l282_28215


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l282_28291

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 15*x + 54 = (x + a) * (x + b)) →
  (∀ x, x^2 - 17*x + 72 = (x - b) * (x - c)) →
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l282_28291


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l282_28272

def A : Set ℕ := {1, 3}
def B : Set ℕ := {0, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l282_28272


namespace NUMINAMATH_CALUDE_sally_takes_home_17_pens_l282_28236

/-- Calculates the number of pens Sally takes home --/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_left := total_pens - pens_given
  let pens_in_locker := pens_left / 2
  pens_left - pens_in_locker

/-- Proves that Sally takes home 17 pens --/
theorem sally_takes_home_17_pens :
  pens_taken_home 342 44 7 = 17 := by
  sorry

#eval pens_taken_home 342 44 7

end NUMINAMATH_CALUDE_sally_takes_home_17_pens_l282_28236


namespace NUMINAMATH_CALUDE_calculate_Y_l282_28250

theorem calculate_Y : ∀ P Q Y : ℚ,
  P = 4050 / 5 →
  Q = P / 4 →
  Y = P - Q →
  Y = 607.5 := by
sorry

end NUMINAMATH_CALUDE_calculate_Y_l282_28250


namespace NUMINAMATH_CALUDE_correct_arrangement_satisfies_conditions_l282_28265

-- Define the solutions
inductive Solution
| CuSO4
| CuCl2
| BaCl2
| AgNO3
| NH4OH
| HNO3
| HCl
| H2SO4

-- Define the test tubes
def TestTube := Fin 8 → Solution

-- Define the precipitation relation
def precipitates (s1 s2 : Solution) : Prop := sorry

-- Define the solubility relation
def soluble_in (s1 s2 s3 : Solution) : Prop := sorry

-- Define the correct arrangement
def correct_arrangement : TestTube :=
  fun i => match i with
  | 0 => Solution.CuSO4
  | 1 => Solution.CuCl2
  | 2 => Solution.BaCl2
  | 3 => Solution.AgNO3
  | 4 => Solution.NH4OH
  | 5 => Solution.HNO3
  | 6 => Solution.HCl
  | 7 => Solution.H2SO4

-- State the theorem
theorem correct_arrangement_satisfies_conditions (t : TestTube) :
  (t = correct_arrangement) →
  (precipitates (t 0) (t 2)) ∧
  (precipitates (t 0) (t 4)) ∧
  (precipitates (t 0) (t 3)) ∧
  (soluble_in (t 0) (t 3) (t 4)) ∧
  (soluble_in (t 0) (t 4) (t 4)) ∧
  (soluble_in (t 0) (t 4) (t 5)) ∧
  (soluble_in (t 0) (t 4) (t 6)) ∧
  (soluble_in (t 0) (t 4) (t 7)) ∧
  (precipitates (t 1) (t 3)) ∧
  (precipitates (t 1) (t 4)) ∧
  (soluble_in (t 1) (t 3) (t 4)) ∧
  (soluble_in (t 1) (t 4) (t 4)) ∧
  (soluble_in (t 1) (t 4) (t 5)) ∧
  (soluble_in (t 1) (t 4) (t 6)) ∧
  (soluble_in (t 1) (t 4) (t 7)) ∧
  (precipitates (t 2) (t 0)) ∧
  (precipitates (t 2) (t 3)) ∧
  (precipitates (t 2) (t 7)) ∧
  (soluble_in (t 2) (t 3) (t 4)) ∧
  (precipitates (t 3) (t 1)) ∧
  (precipitates (t 3) (t 4)) ∧
  (precipitates (t 3) (t 6)) ∧
  (precipitates (t 3) (t 0)) ∧
  (precipitates (t 3) (t 7)) ∧
  (∀ i, soluble_in (t 3) (t i) (t 4)) :=
by sorry


end NUMINAMATH_CALUDE_correct_arrangement_satisfies_conditions_l282_28265


namespace NUMINAMATH_CALUDE_hypotenuse_product_equals_area_l282_28269

/-- A right-angled triangle with an incircle -/
structure RightTriangleWithIncircle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the incircle -/
  incircle_radius : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The first part of the hypotenuse divided by the incircle's point of contact -/
  x : ℝ
  /-- The second part of the hypotenuse divided by the incircle's point of contact -/
  y : ℝ
  /-- The sum of x and y is equal to the hypotenuse -/
  hypotenuse_division : x + y = hypotenuse
  /-- All lengths are positive -/
  all_positive : 0 < area ∧ 0 < incircle_radius ∧ 0 < hypotenuse ∧ 0 < x ∧ 0 < y

/-- The theorem stating that the product of the two parts of the hypotenuse 
    is equal to the area of the right-angled triangle with an incircle -/
theorem hypotenuse_product_equals_area (t : RightTriangleWithIncircle) : t.x * t.y = t.area := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_equals_area_l282_28269


namespace NUMINAMATH_CALUDE_last_segment_speed_l282_28221

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_segment_speed = 50 →
  second_segment_speed = 70 →
  ∃ (last_segment_speed : ℝ),
    last_segment_speed = 60 ∧
    (first_segment_speed * (total_time / 3) + 
     second_segment_speed * (total_time / 3) + 
     last_segment_speed * (total_time / 3)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_last_segment_speed_l282_28221


namespace NUMINAMATH_CALUDE_solve_for_y_l282_28255

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l282_28255


namespace NUMINAMATH_CALUDE_opposite_of_one_minus_sqrt_two_l282_28299

theorem opposite_of_one_minus_sqrt_two :
  ∃ x : ℝ, (1 - Real.sqrt 2) + x = 0 ∧ x = -1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_minus_sqrt_two_l282_28299


namespace NUMINAMATH_CALUDE_system_solution_l282_28229

theorem system_solution :
  let f (x y : ℝ) := x^2 - 5*x*y + 6*y^2
  let g (x y : ℝ) := x^2 + y^2
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (f x₁ y₁ = 0 ∧ g x₁ y₁ = 40) ∧
    (f x₂ y₂ = 0 ∧ g x₂ y₂ = 40) ∧
    (f x₃ y₃ = 0 ∧ g x₃ y₃ = 40) ∧
    (f x₄ y₄ = 0 ∧ g x₄ y₄ = 40) ∧
    x₁ = 4 * Real.sqrt 2 ∧ y₁ = 2 * Real.sqrt 2 ∧
    x₂ = -4 * Real.sqrt 2 ∧ y₂ = -2 * Real.sqrt 2 ∧
    x₃ = 6 ∧ y₃ = 2 ∧
    x₄ = -6 ∧ y₄ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l282_28229


namespace NUMINAMATH_CALUDE_hatcher_students_l282_28287

/-- Calculates the total number of students Ms. Hatcher taught -/
def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders

/-- Theorem stating that Ms. Hatcher taught 70 students -/
theorem hatcher_students : total_students 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hatcher_students_l282_28287


namespace NUMINAMATH_CALUDE_xy_values_l282_28256

theorem xy_values (x y : ℝ) 
  (eq1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (eq2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -(1 / Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_xy_values_l282_28256


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_A_iff_l282_28249

def A : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by sorry

theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_A_iff_l282_28249


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l282_28280

/-- The area of a rectangle inscribed in an ellipse -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a^2 / 4 + b^2 / 8 = 1) →  -- Rectangle vertices satisfy ellipse equation
  (2 * a = b) →             -- Length along x-axis is twice the length along y-axis
  (4 * a * b = 16 / 3) :=   -- Area of the rectangle is 16/3
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l282_28280


namespace NUMINAMATH_CALUDE_sam_puppies_count_l282_28241

/-- The number of puppies Sam originally had with spots -/
def original_puppies : ℕ := 8

/-- The number of puppies Sam gave to his friends -/
def given_away_puppies : ℕ := 2

/-- The number of puppies Sam has now -/
def remaining_puppies : ℕ := original_puppies - given_away_puppies

theorem sam_puppies_count : remaining_puppies = 6 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_count_l282_28241


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l282_28264

/-- Converts a base 4 number to decimal --/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number --/
def number : List Nat := [1, 2, 0, 1, 0, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base4ToDecimal number

theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ decimalNumber ∧ ∀ (q : Nat), Nat.Prime q → q ∣ decimalNumber → q ≤ p ∧ p = 181 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l282_28264


namespace NUMINAMATH_CALUDE_units_digit_of_23_times_51_squared_l282_28258

theorem units_digit_of_23_times_51_squared : ∃ n : ℕ, 23 * 51^2 = 10 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_23_times_51_squared_l282_28258


namespace NUMINAMATH_CALUDE_football_team_throwers_l282_28286

/-- Represents the number of throwers on a football team given specific conditions -/
def number_of_throwers (total_players : ℕ) (right_handed : ℕ) : ℕ :=
  total_players - (3 * (total_players - (right_handed - (total_players - right_handed))) / 2)

/-- Theorem stating that under given conditions, there are 28 throwers on the team -/
theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed : ℕ := 56
  number_of_throwers total_players right_handed = 28 := by
  sorry

end NUMINAMATH_CALUDE_football_team_throwers_l282_28286


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l282_28219

theorem regular_polygon_interior_angle (n : ℕ) : 
  (n ≥ 3) → (((n - 2) * 180 : ℝ) / n = 144) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l282_28219


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l282_28253

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l282_28253


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l282_28268

/-- Given that i is the imaginary unit and i · z = 1 - 2i, 
    prove that z is located in the third quadrant of the complex plane. -/
theorem z_in_third_quadrant (i z : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  i * z = 1 - 2*i →  -- given equation
  z.re < 0 ∧ z.im < 0  -- z is in the third quadrant
  := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l282_28268


namespace NUMINAMATH_CALUDE_sum_of_roots_l282_28210

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 9*a^2 + 26*a - 40 = 0)
  (hb : 2*b^3 - 18*b^2 + 22*b - 30 = 0) : 
  a + b = Real.rpow 45 (1/3) + Real.rpow 22.5 (1/3) + 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l282_28210


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l282_28247

theorem rectangle_length_proof (square_perimeter : ℝ) (rectangle_width : ℝ) :
  square_perimeter = 256 →
  rectangle_width = 32 →
  (square_perimeter / 4) ^ 2 = 2 * (rectangle_width * (square_perimeter / 4)) →
  square_perimeter / 4 = 64 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l282_28247


namespace NUMINAMATH_CALUDE_spinner_prob_C_or_D_l282_28278

/-- Represents a circular spinner with four parts -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of landing on either C or D -/
def probCorD (s : Spinner) : ℚ := s.probC + s.probD

theorem spinner_prob_C_or_D (s : Spinner) 
  (h1 : s.probA = 1/4)
  (h2 : s.probB = 1/3)
  (h3 : s.probA + s.probB + s.probC + s.probD = 1) :
  probCorD s = 5/12 := by
    sorry

end NUMINAMATH_CALUDE_spinner_prob_C_or_D_l282_28278


namespace NUMINAMATH_CALUDE_library_visitors_average_l282_28259

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 570) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) :
  let sundays := (days_in_month + 6) / 7
  let other_days := days_in_month - sundays
  let total_visitors := sundays * sunday_visitors + other_days * other_day_visitors
  total_visitors / days_in_month = 295 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_average_l282_28259


namespace NUMINAMATH_CALUDE_sierra_crest_trail_length_l282_28295

/-- Represents the Sierra Crest Trail hike -/
structure HikeData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The Sierra Crest Trail hike theorem -/
theorem sierra_crest_trail_length (h : HikeData) : 
  h.day1 + h.day2 + h.day3 = 36 →
  (h.day2 + h.day4) / 2 = 15 →
  h.day4 + h.day5 = 38 →
  h.day1 + h.day4 = 32 →
  h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 74 := by
  sorry


end NUMINAMATH_CALUDE_sierra_crest_trail_length_l282_28295


namespace NUMINAMATH_CALUDE_inequality_proof_l282_28261

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (a * c / (b + a * c)) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l282_28261


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l282_28283

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l282_28283


namespace NUMINAMATH_CALUDE_min_additions_for_54_l282_28226

/-- A type representing a way to split the number 123456789 into parts -/
def Splitting := List Nat

/-- The original number we're working with -/
def originalNumber : Nat := 123456789

/-- Function to check if a splitting is valid (uses all digits in order) -/
def isValidSplitting (s : Splitting) : Prop :=
  s.foldl (· * 10 + ·) 0 = originalNumber

/-- Function to calculate the sum of a splitting -/
def sumOfSplitting (s : Splitting) : Nat :=
  s.sum

/-- The target sum we want to achieve -/
def targetSum : Nat := 54

/-- Theorem stating that the minimum number of addition signs needed is 7 -/
theorem min_additions_for_54 :
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum) ∧
  (∀ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum → s.length ≥ 8) ∧
  (∃ (s : Splitting), isValidSplitting s ∧ sumOfSplitting s = targetSum ∧ s.length = 8) :=
sorry

end NUMINAMATH_CALUDE_min_additions_for_54_l282_28226


namespace NUMINAMATH_CALUDE_december_november_difference_l282_28218

def october_visitors : ℕ := 100

def november_visitors : ℕ := (october_visitors * 115) / 100

def total_visitors : ℕ := 345

theorem december_november_difference :
  ∃ (december_visitors : ℕ),
    december_visitors > november_visitors ∧
    october_visitors + november_visitors + december_visitors = total_visitors ∧
    december_visitors - november_visitors = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_december_november_difference_l282_28218


namespace NUMINAMATH_CALUDE_fraction_equality_l282_28284

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l282_28284


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l282_28276

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s₁ s₂ : ℝ),
  r > 0 → s₁ > 0 → s₂ > 0 →
  2 * π * r = 4 * s₁ →  -- Circle and first square have same perimeter
  2 * r = s₂ * Real.sqrt 2 →  -- Diameter of circle is diagonal of second square
  (s₂^2) / (s₁^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l282_28276


namespace NUMINAMATH_CALUDE_perfect_square_3_6_4_5_5_4_l282_28279

theorem perfect_square_3_6_4_5_5_4 : ∃ n : ℕ, n ^ 2 = 3^6 * 4^5 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_3_6_4_5_5_4_l282_28279


namespace NUMINAMATH_CALUDE_root_sum_ratio_l282_28213

theorem root_sum_ratio (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 4 * x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ / x₂ + x₂ / x₁ = 6) := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l282_28213


namespace NUMINAMATH_CALUDE_range_of_a_l282_28225

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  a < 0 ∧
  (∀ x, p x a → q x) ∧
  (∃ x, q x ∧ ¬p x a) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l282_28225


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l282_28220

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) - 2 * a) / 5 = (Complex.I * (a + Complex.I) / (1 + 2 * Complex.I))) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l282_28220


namespace NUMINAMATH_CALUDE_prism_volume_l282_28257

/-- 
Given a right rectangular prism with face areas 10, 15, and 6 square inches,
prove that its volume is 30 cubic inches.
-/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 10)
  (area2 : w * h = 15)
  (area3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l282_28257


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l282_28238

theorem fraction_product_theorem :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (9 / 6 : ℚ) * (10 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (15 / 45 : ℚ) * (32 / 16 : ℚ) * (50 / 100 : ℚ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l282_28238


namespace NUMINAMATH_CALUDE_range_of_m_l282_28285

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line in the form 2x + y + m = 0 -/
def Line (m : ℝ) (p : Point) : Prop :=
  2 * p.x + p.y + m = 0

/-- Defines when two points are on opposite sides of a line -/
def OppositesSides (m : ℝ) (p1 p2 : Point) : Prop :=
  (2 * p1.x + p1.y + m) * (2 * p2.x + p2.y + m) < 0

/-- The main theorem -/
theorem range_of_m (p1 p2 : Point) (h : OppositesSides m p1 p2) 
  (h1 : p1 = ⟨1, 3⟩) (h2 : p2 = ⟨-4, -2⟩) : 
  -5 < m ∧ m < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l282_28285


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l282_28208

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (x + 1) = 3 * x - 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (y + 1) = 3 * y - 1 → x ≤ y :=
by
  -- The solution is x = 7/9
  use 7/9
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l282_28208


namespace NUMINAMATH_CALUDE_abs_value_inequality_l282_28252

theorem abs_value_inequality (x : ℝ) : |x| < 5 ↔ -5 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l282_28252


namespace NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l282_28292

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 300 ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l282_28292


namespace NUMINAMATH_CALUDE_biased_coin_prob_sum_l282_28200

/-- The probability of getting heads for a biased coin -/
def h : ℚ :=
  3 / 7

/-- The condition that the probability of 2 heads equals the probability of 3 heads in 6 flips -/
axiom prob_equality : 15 * h^2 * (1 - h)^4 = 20 * h^3 * (1 - h)^3

/-- The probability of getting exactly 4 heads in 6 flips -/
def prob_4_heads : ℚ :=
  15 * h^4 * (1 - h)^2

/-- The numerator and denominator of prob_4_heads in lowest terms -/
def p : ℕ := 19440
def q : ℕ := 117649

theorem biased_coin_prob_sum :
  prob_4_heads = p / q ∧ p + q = 137089 := by sorry

end NUMINAMATH_CALUDE_biased_coin_prob_sum_l282_28200


namespace NUMINAMATH_CALUDE_square_area_ratio_l282_28296

theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (2 * x)^2 / (6 * x)^2 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l282_28296


namespace NUMINAMATH_CALUDE_trash_can_ratio_l282_28262

/-- Represents the number of trash cans added to the streets -/
def street_cans : ℕ := 14

/-- Represents the total number of trash cans -/
def total_cans : ℕ := 42

/-- Represents the number of trash cans added to the back of stores -/
def store_cans : ℕ := total_cans - street_cans

/-- The ratio of trash cans added to the back of stores to trash cans added to the streets -/
theorem trash_can_ratio : (store_cans : ℚ) / street_cans = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_trash_can_ratio_l282_28262


namespace NUMINAMATH_CALUDE_min_square_area_for_rectangles_l282_28240

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.height) (r1.height + r2.width)

/-- Theorem: The smallest square area to fit a 3x4 and a 4x5 rectangle with one rotated is 81 -/
theorem min_square_area_for_rectangles :
  let r1 : Rectangle := ⟨3, 4⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 4⟩ ⟨4, 5⟩) ^ 2

end NUMINAMATH_CALUDE_min_square_area_for_rectangles_l282_28240


namespace NUMINAMATH_CALUDE_simplify_fraction_difference_quotient_l282_28201

theorem simplify_fraction_difference_quotient (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (1 / (a + 2) - 1 / (a - 2)) / (1 / (a - 2)) = -4 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_difference_quotient_l282_28201


namespace NUMINAMATH_CALUDE_unique_quadruple_l282_28260

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d)^3 = 8 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_l282_28260


namespace NUMINAMATH_CALUDE_gaochun_population_eq_scientific_l282_28209

/-- The population of Gaochun County -/
def gaochun_population : ℕ := 425000

/-- Scientific notation representation of Gaochun County's population -/
def gaochun_population_scientific : ℝ := 4.25 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is equal to the actual population -/
theorem gaochun_population_eq_scientific : ↑gaochun_population = gaochun_population_scientific := by
  sorry

end NUMINAMATH_CALUDE_gaochun_population_eq_scientific_l282_28209


namespace NUMINAMATH_CALUDE_elise_remaining_money_l282_28254

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenditures. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_cost : ℕ) (puzzle_cost : ℕ) : ℕ :=
  initial + savings - comic_cost - puzzle_cost

/-- Proves that Elise is left with $1 given her initial amount, savings, and expenditures. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l282_28254


namespace NUMINAMATH_CALUDE_smallest_number_l282_28273

theorem smallest_number : 
  let a := (2010 : ℝ) ^ (1 / 209)
  let b := (2009 : ℝ) ^ (1 / 200)
  let c := (2010 : ℝ)
  let d := (2010 : ℝ) / 2009
  let e := (2009 : ℝ) / 2010
  (e ≤ a) ∧ (e ≤ b) ∧ (e ≤ c) ∧ (e ≤ d) ∧ (e ≤ e) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l282_28273


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l282_28242

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem irrationality_of_sqrt_two_and_rationality_of_others :
  IsIrrational (Real.sqrt 2) ∧ 
  IsRational 3.14 ∧ 
  IsRational (22 / 7) ∧ 
  IsRational 0 :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l282_28242


namespace NUMINAMATH_CALUDE_potato_peeling_time_l282_28243

theorem potato_peeling_time (julie_rate ted_rate initial_time : ℝ) 
  (h1 : julie_rate = 1 / 10)  -- Julie's peeling rate per hour
  (h2 : ted_rate = 1 / 8)     -- Ted's peeling rate per hour
  (h3 : initial_time = 4)     -- Time they work together
  : (1 - (julie_rate + ted_rate) * initial_time) / julie_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_time_l282_28243


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l282_28227

theorem sufficient_condition_range (m : ℝ) : m > 0 →
  (∀ x : ℝ, x^2 - 8*x - 20 ≤ 0 → (1 - m ≤ x ∧ x ≤ 1 + m)) ∧ 
  (∃ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m ∧ x^2 - 8*x - 20 > 0) ↔ 
  m ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l282_28227


namespace NUMINAMATH_CALUDE_min_sum_of_prime_factors_l282_28281

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The sum of n consecutive integers starting from x -/
def consecutiveSum (x n : ℕ) : ℕ :=
  n * (2 * x + n - 1) / 2

theorem min_sum_of_prime_factors (a b c d : ℕ) :
  isPrime a → isPrime b → isPrime c → isPrime d →
  (∃ x : ℕ, a * b * c * d = consecutiveSum x 35) →
  22 ≤ a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_prime_factors_l282_28281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l282_28290

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  product_condition : a 7 * a 11 = 6
  sum_condition : a 4 + a 14 = 5

/-- The common difference of an arithmetic sequence is either 1/4 or -1/4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  (∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ (d = 1/4 ∨ d = -1/4)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l282_28290


namespace NUMINAMATH_CALUDE_carpet_width_l282_28245

/-- Proves that a rectangular carpet covering 75% of a 48 sq ft room with a length of 9 ft has a width of 4 ft -/
theorem carpet_width (room_area : ℝ) (carpet_length : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_length = 9 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l282_28245


namespace NUMINAMATH_CALUDE_total_earnings_l282_28271

theorem total_earnings (jerusha_earnings lottie_earnings : ℕ) :
  jerusha_earnings = 68 →
  jerusha_earnings = 4 * lottie_earnings →
  jerusha_earnings + lottie_earnings = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l282_28271


namespace NUMINAMATH_CALUDE_sequence_even_terms_l282_28217

def sequence_property (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ x (n-1) ∧ x n = x (n-1) + d

theorem sequence_even_terms (x : ℕ → ℕ) (h : sequence_property x) :
  (∃ n : ℕ, Even (x n)) ∧ (∀ m : ℕ, ∃ n : ℕ, n > m ∧ Even (x n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_even_terms_l282_28217


namespace NUMINAMATH_CALUDE_complex_argument_range_l282_28239

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  let arg := Complex.arg z
  arg ∈ (Set.Icc (Real.arccos (Real.sqrt 2 / 4)) (Real.pi - Real.arccos (Real.sqrt 2 / 4))) ∪
           (Set.Icc (Real.pi + Real.arccos (Real.sqrt 2 / 4)) (2 * Real.pi - Real.arccos (Real.sqrt 2 / 4))) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_range_l282_28239


namespace NUMINAMATH_CALUDE_count_ordered_pairs_3255_l282_28297

theorem count_ordered_pairs_3255 : 
  let n : ℕ := 3255
  let prime_factorization : List ℕ := [5, 13, 17]
  ∀ (x y : ℕ), x * y = n → x > 0 ∧ y > 0 →
  (∃! (pairs : List (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) ∧
    pairs.length = 8) :=
by sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_3255_l282_28297


namespace NUMINAMATH_CALUDE_luke_trips_l282_28244

/-- The number of trays Luke can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 20

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 16

/-- The total number of trips Luke will make -/
def total_trips : ℕ := (trays_table1 / trays_per_trip) + (trays_table2 / trays_per_trip)

theorem luke_trips : total_trips = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_trips_l282_28244


namespace NUMINAMATH_CALUDE_roberts_markers_count_l282_28231

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := total_markers - initial_markers

theorem roberts_markers_count : roberts_markers = 109 := by
  sorry

end NUMINAMATH_CALUDE_roberts_markers_count_l282_28231


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l282_28207

theorem polynomial_division_degree (f q d r : Polynomial ℝ) :
  Polynomial.degree f = 17 →
  Polynomial.degree q = 10 →
  Polynomial.degree r = 5 →
  f = d * q + r →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l282_28207


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l282_28270

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (8, 16)
  let p₂ : ℝ × ℝ := (-2, -8)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l282_28270


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l282_28266

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  p^4 + 8*p^3 + 16*p^2 + 5*p + 2 = 0 →
  q^4 + 8*q^3 + 16*q^2 + 5*q + 2 = 0 →
  r^4 + 8*r^3 + 16*r^2 + 5*r + 2 = 0 →
  s^4 + 8*s^3 + 16*s^2 + 5*s + 2 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 8 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l282_28266


namespace NUMINAMATH_CALUDE_factorial_equality_l282_28267

theorem factorial_equality : 7 * 6 * 4 * 2160 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l282_28267


namespace NUMINAMATH_CALUDE_equation_proof_l282_28293

theorem equation_proof : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l282_28293


namespace NUMINAMATH_CALUDE_scrap_rate_cost_increase_l282_28205

/-- The regression equation for cost of cast iron based on scrap rate -/
def cost_equation (x : ℝ) : ℝ := 56 + 8 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_increase (x : ℝ) :
  cost_equation (x + 1) - cost_equation x = 8 := by
  sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_increase_l282_28205


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l282_28206

/-- An isosceles triangle with perimeter 16 and one side length 3 has its other side length equal to 6.5 -/
theorem isosceles_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 16 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 6.5 ∧ b = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (b = 6.5 ∧ c = 6.5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l282_28206


namespace NUMINAMATH_CALUDE_not_perfect_power_l282_28230

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ¬ ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 10^k - 1 = m^n := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_power_l282_28230


namespace NUMINAMATH_CALUDE_product_of_fractions_l282_28234

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions :
  (fraction 7) * (fraction 8) * (fraction 9) * (fraction 10) * (fraction 11) = 133 / 946 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l282_28234


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l282_28246

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem set_intersection_theorem : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l282_28246


namespace NUMINAMATH_CALUDE_expression_evaluation_l282_28224

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 1) / (y - 3)) * ((z + 9) / (z + 7)) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l282_28224


namespace NUMINAMATH_CALUDE_dara_half_jane_age_l282_28263

/-- The problem statement about Dara and Jane's ages -/
theorem dara_half_jane_age :
  let min_age : ℕ := 25  -- Minimum age for employment
  let jane_age : ℕ := 28  -- Jane's current age
  let years_to_min : ℕ := 14  -- Years until Dara reaches minimum age
  let dara_age : ℕ := min_age - years_to_min  -- Dara's current age
  let x : ℕ := 6  -- Years until Dara is half Jane's age
  dara_age + x = (jane_age + x) / 2 := by sorry

end NUMINAMATH_CALUDE_dara_half_jane_age_l282_28263

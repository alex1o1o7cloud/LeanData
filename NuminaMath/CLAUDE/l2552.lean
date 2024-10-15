import Mathlib

namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2552_255221

/-- The area of a stripe wrapped around a cylindrical object -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 30 →
  stripe_width = 4 →
  revolutions = 3 →
  stripe_width * revolutions * (π * diameter) = 360 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2552_255221


namespace NUMINAMATH_CALUDE_factorialLastNonzeroDigitSeq_not_periodic_l2552_255288

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10

/-- The sequence of last nonzero digits of factorials -/
def factorialLastNonzeroDigitSeq : ℕ → ℕ :=
  fun n => lastNonzeroDigit (Nat.factorial n)

/-- The sequence of last nonzero digits of factorials is not periodic -/
theorem factorialLastNonzeroDigitSeq_not_periodic :
  ¬ ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), factorialLastNonzeroDigitSeq (n + p) = factorialLastNonzeroDigitSeq n :=
sorry

end NUMINAMATH_CALUDE_factorialLastNonzeroDigitSeq_not_periodic_l2552_255288


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2552_255270

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + m*x₁ - 8 = 0) ∧ (x₂^2 + m*x₂ - 8 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2552_255270


namespace NUMINAMATH_CALUDE_cube_root_to_square_l2552_255252

theorem cube_root_to_square (y : ℝ) : 
  (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_to_square_l2552_255252


namespace NUMINAMATH_CALUDE_offer_price_per_year_is_half_l2552_255261

/-- Represents a magazine subscription offer -/
structure MagazineOffer where
  regularYearlyFee : ℕ
  offerYears : ℕ
  offerPrice : ℕ
  issuesPerYear : ℕ

/-- The Parents magazine offer -/
def parentsOffer : MagazineOffer :=
  { regularYearlyFee := 12
  , offerYears := 2
  , offerPrice := 12
  , issuesPerYear := 12
  }

/-- Theorem stating that the offer price per year is half of the regular price per year -/
theorem offer_price_per_year_is_half (o : MagazineOffer) 
    (h1 : o.offerYears = 2)
    (h2 : o.offerPrice = o.regularYearlyFee) :
    o.offerPrice / o.offerYears = o.regularYearlyFee / 2 := by
  sorry

#check offer_price_per_year_is_half parentsOffer

end NUMINAMATH_CALUDE_offer_price_per_year_is_half_l2552_255261


namespace NUMINAMATH_CALUDE_cone_surface_area_and_volume_l2552_255256

/-- Represents a cone with given height and sector angle -/
structure Cone where
  height : ℝ
  sectorAngle : ℝ

/-- Calculates the surface area of a cone -/
def surfaceArea (c : Cone) : ℝ := sorry

/-- Calculates the volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c : Cone := { height := 12, sectorAngle := 100.8 * π / 180 }
  surfaceArea c = 56 * π ∧ volume c = 49 * π := by sorry

end NUMINAMATH_CALUDE_cone_surface_area_and_volume_l2552_255256


namespace NUMINAMATH_CALUDE_snack_combinations_l2552_255292

def num_items : ℕ := 4
def items_to_choose : ℕ := 2

theorem snack_combinations : 
  Nat.choose num_items items_to_choose = 6 := by sorry

end NUMINAMATH_CALUDE_snack_combinations_l2552_255292


namespace NUMINAMATH_CALUDE_xiaoming_money_l2552_255224

/-- Proves that Xiaoming brought 108 yuan to the supermarket -/
theorem xiaoming_money (fresh_milk_cost yogurt_cost : ℕ) 
  (fresh_milk_cartons yogurt_cartons total_money : ℕ) : 
  fresh_milk_cost = 6 →
  yogurt_cost = 9 →
  fresh_milk_cost * fresh_milk_cartons = total_money →
  yogurt_cost * yogurt_cartons = total_money →
  fresh_milk_cartons = yogurt_cartons + 6 →
  total_money = 108 := by
  sorry

#check xiaoming_money

end NUMINAMATH_CALUDE_xiaoming_money_l2552_255224


namespace NUMINAMATH_CALUDE_constant_value_l2552_255273

theorem constant_value (x y z : ℝ) : 
  ∃ (c : ℝ), ∀ (x y z : ℝ), 
    ((x - y)^3 + (y - z)^3 + (z - x)^3) / (c * (x - y) * (y - z) * (z - x)) = 0.2 → c = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l2552_255273


namespace NUMINAMATH_CALUDE_intersection_slope_l2552_255227

/-- Given two lines p and q that intersect at (-4, -7), 
    prove that the slope of line q is 2.5 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y, y = 3 * x + 5 → y = k * x + 3 → x = -4 ∧ y = -7) → 
  k = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l2552_255227


namespace NUMINAMATH_CALUDE_complex_product_real_l2552_255209

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + x * Complex.I
  (z₁ * z₂).im = 0 → x = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2552_255209


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2552_255274

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2552_255274


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2552_255210

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x/y > 1) ∧
  ∃ a b : ℝ, a/b > 1 ∧ ¬(a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2552_255210


namespace NUMINAMATH_CALUDE_semicircle_in_square_l2552_255236

theorem semicircle_in_square (d m n : ℝ) : 
  d > 0 →                           -- d is positive (diameter)
  8 > 0 →                           -- square side length is positive
  d ≤ 8 →                           -- semicircle fits in square
  d ≤ m - Real.sqrt n →             -- maximum value of d
  m - Real.sqrt n ≤ 8 →             -- maximum value fits in square
  (∀ x, x > 0 → x - Real.sqrt (4 * x) < m - Real.sqrt n) →  -- m - √n is indeed the maximum
  m + n = 544 := by
sorry

end NUMINAMATH_CALUDE_semicircle_in_square_l2552_255236


namespace NUMINAMATH_CALUDE_bacteria_growth_problem_l2552_255213

/-- Bacteria growth problem -/
theorem bacteria_growth_problem (initial_count : ℕ) : 
  (∀ (period : ℕ), initial_count * (4 ^ period) = initial_count * 4 ^ period) →
  initial_count * 4 ^ 4 = 262144 →
  initial_count = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_problem_l2552_255213


namespace NUMINAMATH_CALUDE_charitable_gentleman_proof_l2552_255279

def charitable_donation (initial : ℕ) : Prop :=
  let after_first := initial - (initial / 2 + 1)
  let after_second := after_first - (after_first / 2 + 2)
  let after_third := after_second - (after_second / 2 + 3)
  after_third = 1

theorem charitable_gentleman_proof :
  ∃ (initial : ℕ), charitable_donation initial ∧ initial = 42 := by
  sorry

end NUMINAMATH_CALUDE_charitable_gentleman_proof_l2552_255279


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l2552_255248

/-- A triangle with side lengths 3, 2x+1, and 10 exists if and only if 3 < x < 6 -/
theorem triangle_existence_condition (x : ℝ) :
  (3 : ℝ) < x ∧ x < 6 ↔ 
  (3 : ℝ) + (2*x + 1) > 10 ∧
  (3 : ℝ) + 10 > 2*x + 1 ∧
  10 + (2*x + 1) > 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l2552_255248


namespace NUMINAMATH_CALUDE_equation_solutions_l2552_255223

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 = 0 ↔ x = 1 + Real.sqrt 2 / 2 ∨ x = 1 - Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2552_255223


namespace NUMINAMATH_CALUDE_ice_machine_cubes_l2552_255290

/-- The number of ice chests -/
def num_chests : ℕ := 7

/-- The number of ice cubes per chest -/
def cubes_per_chest : ℕ := 42

/-- The total number of ice cubes in the ice machine -/
def total_cubes : ℕ := num_chests * cubes_per_chest

/-- Theorem stating that the total number of ice cubes is 294 -/
theorem ice_machine_cubes : total_cubes = 294 := by
  sorry

end NUMINAMATH_CALUDE_ice_machine_cubes_l2552_255290


namespace NUMINAMATH_CALUDE_point_coordinates_l2552_255291

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the second quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (-3, 2) -/
theorem point_coordinates (p : Point) 
    (h1 : SecondQuadrant p) 
    (h2 : DistanceToXAxis p = 2) 
    (h3 : DistanceToYAxis p = 3) : 
    p = Point.mk (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2552_255291


namespace NUMINAMATH_CALUDE_rotate90_neg_6_minus_3i_l2552_255201

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg_6_minus_3i :
  rotate90 (-6 - 3 * Complex.I) = (3 : ℂ) - 6 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotate90_neg_6_minus_3i_l2552_255201


namespace NUMINAMATH_CALUDE_peters_pond_depth_l2552_255265

theorem peters_pond_depth :
  ∀ (mark_depth peter_depth : ℝ),
    mark_depth = 3 * peter_depth + 4 →
    mark_depth = 19 →
    peter_depth = 5 := by
  sorry

end NUMINAMATH_CALUDE_peters_pond_depth_l2552_255265


namespace NUMINAMATH_CALUDE_correct_calculation_l2552_255245

theorem correct_calculation (a : ℝ) : 2 * a * (1 - a) = 2 * a - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2552_255245


namespace NUMINAMATH_CALUDE_three_numbers_average_l2552_255226

theorem three_numbers_average : 
  ∀ (x y z : ℝ), 
    x = 18 ∧ 
    y = 4 * x ∧ 
    z = 2 * y → 
    (x + y + z) / 3 = 78 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_average_l2552_255226


namespace NUMINAMATH_CALUDE_smallest_s_for_E_l2552_255283

/-- Definition of the function E --/
def E (a b c : ℕ) : ℕ := a * b^c

/-- The smallest positive integer s that satisfies E(s, s, 4) = 2401 is 7 --/
theorem smallest_s_for_E : (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ ∀ t : ℕ, t > 0 → E t t 4 = 2401 → s ≤ t) ∧ 
                           (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ s = 7) := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_for_E_l2552_255283


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l2552_255207

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 → not_enrolled = 462 → 
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l2552_255207


namespace NUMINAMATH_CALUDE_profit_at_80_max_profit_profit_range_l2552_255255

/-- Represents the clothing sale scenario with given constraints -/
structure ClothingSale where
  cost : ℝ
  demand : ℝ → ℝ
  profit_function : ℝ → ℝ
  max_profit_percentage : ℝ

/-- The specific clothing sale scenario from the problem -/
def sale : ClothingSale :=
  { cost := 60
  , demand := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , max_profit_percentage := 0.4 }

/-- Theorem stating the profit when selling price is 80 -/
theorem profit_at_80 (s : ClothingSale) (h : s = sale) :
  s.profit_function 80 = 800 :=
sorry

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit (s : ClothingSale) (h : s = sale) :
  ∃ x, x ≤ (1 + s.max_profit_percentage) * s.cost ∧
      s.profit_function x = 864 ∧
      ∀ y, y ≤ (1 + s.max_profit_percentage) * s.cost →
        s.profit_function y ≤ s.profit_function x :=
sorry

/-- Theorem stating the range of selling prices for profit not less than 500 -/
theorem profit_range (s : ClothingSale) (h : s = sale) :
  ∀ x, s.cost ≤ x ∧ x ≤ (1 + s.max_profit_percentage) * s.cost →
    (s.profit_function x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 84) :=
sorry

end NUMINAMATH_CALUDE_profit_at_80_max_profit_profit_range_l2552_255255


namespace NUMINAMATH_CALUDE_ending_number_proof_l2552_255298

/-- The ending number for a sequence of even numbers -/
def ending_number : ℕ := 20

/-- The average of the sequence -/
def average : ℕ := 16

/-- The starting point of the sequence -/
def start : ℕ := 11

theorem ending_number_proof :
  ∀ n : ℕ,
  n > start →
  n ≤ ending_number →
  n % 2 = 0 →
  2 * average = 12 + ending_number :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l2552_255298


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2552_255237

theorem rational_equation_solution (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x^2 - 3*x - 4) / (x - 4) = 3*x + k → x = (1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2552_255237


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2552_255217

/-- Given a train of length 1200 m that crosses a tree in 80 seconds,
    prove that the time it takes to pass a platform of length 1000 m is 146.67 seconds. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 146.67 := by
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2552_255217


namespace NUMINAMATH_CALUDE_smallest_N_for_P_less_than_half_l2552_255258

/-- The probability that at least 2/3 of the green balls are on the same side of either of the red balls -/
def P (N : ℕ) : ℚ :=
  sorry

/-- N is a multiple of 6 -/
def is_multiple_of_six (N : ℕ) : Prop :=
  ∃ k : ℕ, N = 6 * k

theorem smallest_N_for_P_less_than_half :
  (is_multiple_of_six 18) ∧
  (P 18 < 1/2) ∧
  (∀ N : ℕ, is_multiple_of_six N → N < 18 → P N ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_smallest_N_for_P_less_than_half_l2552_255258


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2552_255253

theorem largest_solution_of_equation : 
  ∃ (x : ℝ), x = 6 ∧ 3 * x^2 + 18 * x - 84 = x * (x + 10) ∧
  ∀ (y : ℝ), 3 * y^2 + 18 * y - 84 = y * (y + 10) → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2552_255253


namespace NUMINAMATH_CALUDE_special_sequence_theorem_l2552_255230

/-- A sequence satisfying certain properties -/
def SpecialSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  c > 1 ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ m n, a (m * n) = a m * a n) ∧
  (∀ m n, a (m + n) ≤ c * (a m + a n))

/-- The main theorem: if a sequence satisfies the SpecialSequence properties,
    then a_n = n for all natural numbers n -/
theorem special_sequence_theorem (a : ℕ → ℝ) (c : ℝ) 
    (h : SpecialSequence a c) : ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_theorem_l2552_255230


namespace NUMINAMATH_CALUDE_wall_building_theorem_l2552_255286

/-- The number of men in the first group that can build a 112-metre wall in 6 days,
    given that 40 men can build a similar wall in 3 days. -/
def number_of_men : ℕ := 80

/-- The length of the wall in metres. -/
def wall_length : ℕ := 112

/-- The number of days it takes the first group to build the wall. -/
def days_first_group : ℕ := 6

/-- The number of men in the second group. -/
def men_second_group : ℕ := 40

/-- The number of days it takes the second group to build the wall. -/
def days_second_group : ℕ := 3

theorem wall_building_theorem :
  number_of_men * days_second_group = men_second_group * days_first_group :=
sorry

end NUMINAMATH_CALUDE_wall_building_theorem_l2552_255286


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2552_255293

/-- A right triangle with perimeter 30 and height to hypotenuse 6 has sides 10, 7.5, and 12.5 -/
theorem right_triangle_sides (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 30 →   -- perimeter condition
  a * b = 6 * c →    -- height to hypotenuse condition
  ((a = 10 ∧ b = 7.5 ∧ c = 12.5) ∨ (a = 7.5 ∧ b = 10 ∧ c = 12.5)) := by
  sorry

#check right_triangle_sides

end NUMINAMATH_CALUDE_right_triangle_sides_l2552_255293


namespace NUMINAMATH_CALUDE_min_area_two_rectangles_l2552_255254

/-- Given a wire of length l, cut into two pieces x and (l-x), forming two rectangles
    with length-to-width ratios of 2:1 and 3:2 respectively, the minimum value of 
    the sum of their areas is 3/104 * l^2 --/
theorem min_area_two_rectangles (l : ℝ) (h : l > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < l ∧
  (∀ (y : ℝ), 0 < y → y < l →
    x^2 / 18 + 3 * (l - x)^2 / 50 ≤ y^2 / 18 + 3 * (l - y)^2 / 50) ∧
  x^2 / 18 + 3 * (l - x)^2 / 50 = 3 * l^2 / 104 :=
sorry

end NUMINAMATH_CALUDE_min_area_two_rectangles_l2552_255254


namespace NUMINAMATH_CALUDE_square_area_error_l2552_255203

theorem square_area_error (S : ℝ) (S' : ℝ) (A : ℝ) (A' : ℝ) : 
  S > 0 →
  S' = S * 1.04 →
  A = S^2 →
  A' = S'^2 →
  (A' - A) / A * 100 = 8.16 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2552_255203


namespace NUMINAMATH_CALUDE_derivative_exponential_plus_sine_l2552_255229

theorem derivative_exponential_plus_sine (x : ℝ) :
  let y := fun x => Real.exp x + Real.sin x
  HasDerivAt y (Real.exp x + Real.cos x) x :=
by sorry

end NUMINAMATH_CALUDE_derivative_exponential_plus_sine_l2552_255229


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2552_255284

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_arith : arithmetic_sequence (λ n => match n with
    | 1 => 4 * (a 1)
    | 2 => 2 * (a 2)
    | 3 => a 3
    | _ => 0
  )) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2552_255284


namespace NUMINAMATH_CALUDE_total_gift_wrapping_combinations_l2552_255266

/-- The number of different gift wrapping combinations -/
def gift_wrapping_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_tag : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_tag

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem total_gift_wrapping_combinations :
  gift_wrapping_combinations 10 5 6 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_gift_wrapping_combinations_l2552_255266


namespace NUMINAMATH_CALUDE_tan_difference_l2552_255278

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l2552_255278


namespace NUMINAMATH_CALUDE_expression_value_l2552_255208

-- Define opposite numbers
def opposite (m n : ℝ) : Prop := m + n = 0

-- Define reciprocal numbers
def reciprocal (p q : ℝ) : Prop := p * q = 1

-- Theorem statement
theorem expression_value 
  (m n p q : ℝ) 
  (h1 : opposite m n) 
  (h2 : m ≠ n) 
  (h3 : reciprocal p q) : 
  (m + n) / m + 2 * p * q - m / n = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2552_255208


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2552_255219

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 6 * x₁ = 7) ∧ (2 * x₂^2 - 6 * x₂ = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2552_255219


namespace NUMINAMATH_CALUDE_linear_inequality_m_value_l2552_255269

/-- If 3m - 5x^(3+m) > 4 is a linear inequality in x, then m = -2 -/
theorem linear_inequality_m_value (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, 3*m - 5*x^(3+m) > 4 ↔ a*x + b > 0) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_linear_inequality_m_value_l2552_255269


namespace NUMINAMATH_CALUDE_dave_lisa_slices_l2552_255220

/-- Represents the number of slices in a pizza -/
structure Pizza where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased -/
structure PizzaOrder where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person -/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ
  ann : ℕ
  kelly : ℕ

def pizza_sizes : Pizza := ⟨4, 8⟩
def george_order : PizzaOrder := ⟨4, 3⟩
def slices_eaten : SlicesEaten := ⟨3, 4, 2, 3, 3, 3, 2, 4⟩

def total_slices (p : Pizza) (o : PizzaOrder) : ℕ :=
  p.small * o.small + p.large * o.large

def total_eaten (s : SlicesEaten) : ℕ :=
  s.george + s.bob + s.susie + s.bill + s.fred + s.mark + s.ann + s.kelly

theorem dave_lisa_slices :
  (total_slices pizza_sizes george_order - total_eaten slices_eaten) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_dave_lisa_slices_l2552_255220


namespace NUMINAMATH_CALUDE_distance_after_skating_l2552_255267

/-- Calculates the distance between two skaters moving in opposite directions -/
def distance_between_skaters (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between Ann and Glenda after skating for 3 hours -/
theorem distance_after_skating :
  let ann_speed : ℝ := 6
  let glenda_speed : ℝ := 8
  let skating_time : ℝ := 3
  distance_between_skaters ann_speed glenda_speed skating_time = 42 := by
  sorry

#check distance_after_skating

end NUMINAMATH_CALUDE_distance_after_skating_l2552_255267


namespace NUMINAMATH_CALUDE_asha_win_probability_l2552_255257

theorem asha_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 7/12 → win_prob + lose_prob = 1 → win_prob = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2552_255257


namespace NUMINAMATH_CALUDE_expected_asthma_cases_l2552_255225

theorem expected_asthma_cases (total_sample : ℕ) (asthma_rate : ℚ) 
  (h1 : total_sample = 320) 
  (h2 : asthma_rate = 1 / 8) : 
  ⌊total_sample * asthma_rate⌋ = 40 := by
  sorry

end NUMINAMATH_CALUDE_expected_asthma_cases_l2552_255225


namespace NUMINAMATH_CALUDE_power_product_equals_6300_l2552_255285

theorem power_product_equals_6300 : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_6300_l2552_255285


namespace NUMINAMATH_CALUDE_samuel_initial_skittles_l2552_255260

/-- The number of friends Samuel gave Skittles to -/
def num_friends : ℕ := 4

/-- The number of Skittles each person (including Samuel) ate -/
def skittles_per_person : ℕ := 3

/-- The initial number of Skittles Samuel had -/
def initial_skittles : ℕ := num_friends * skittles_per_person + skittles_per_person

/-- Theorem stating that Samuel initially had 15 Skittles -/
theorem samuel_initial_skittles : initial_skittles = 15 := by
  sorry

end NUMINAMATH_CALUDE_samuel_initial_skittles_l2552_255260


namespace NUMINAMATH_CALUDE_award_sequences_eq_sixteen_l2552_255299

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Calculates the number of possible award sequences -/
def award_sequences : ℕ := 2^num_rounds

/-- Theorem stating that the number of award sequences is 16 -/
theorem award_sequences_eq_sixteen : award_sequences = 16 := by
  sorry

end NUMINAMATH_CALUDE_award_sequences_eq_sixteen_l2552_255299


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2552_255204

-- Define the fourth quadrant
def fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 2 < α ∧ α < (2 * n + 1) * Real.pi

-- Define the fourth quadrant
def fourth_quadrant' (α : Real) : Prop :=
  ∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < α ∧ α < 2 * n * Real.pi

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  fourth_quadrant α → (second_quadrant (α/2) ∨ fourth_quadrant' (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2552_255204


namespace NUMINAMATH_CALUDE_decrypt_ciphertext_l2552_255276

/-- Represents the encryption rule --/
def encrypt (a b c d : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  (a + 2*b, 2*b + c, 2*c + 3*d, 4*d)

/-- Represents the given ciphertext --/
def ciphertext : ℤ × ℤ × ℤ × ℤ := (14, 9, 23, 28)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the given ciphertext --/
theorem decrypt_ciphertext :
  encrypt 6 4 1 7 = ciphertext := by sorry

end NUMINAMATH_CALUDE_decrypt_ciphertext_l2552_255276


namespace NUMINAMATH_CALUDE_equation_solutions_l2552_255249

theorem equation_solutions :
  (∀ x : ℝ, (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2) ∧
  (∀ x : ℝ, 2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2552_255249


namespace NUMINAMATH_CALUDE_hit_frequency_l2552_255202

theorem hit_frequency (total_shots : ℕ) (hits : ℕ) (h1 : total_shots = 20) (h2 : hits = 15) :
  (hits : ℚ) / total_shots = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hit_frequency_l2552_255202


namespace NUMINAMATH_CALUDE_max_value_of_s_l2552_255289

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2552_255289


namespace NUMINAMATH_CALUDE_sum_of_squares_on_sides_l2552_255244

/-- Given a right triangle XYZ with XY = 8 and YZ = 17, 
    the sum of the areas of squares constructed on sides YZ and XZ is 514. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 8^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 17^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →
  17^2 + ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 514 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_on_sides_l2552_255244


namespace NUMINAMATH_CALUDE_lowest_possible_score_l2552_255228

def exam_max_score : ℕ := 120
def num_exams : ℕ := 5
def goal_average : ℕ := 100
def current_scores : List ℕ := [90, 108, 102]

theorem lowest_possible_score :
  let total_needed : ℕ := goal_average * num_exams
  let current_total : ℕ := current_scores.sum
  let remaining_total : ℕ := total_needed - current_total
  let max_score_one_exam : ℕ := min exam_max_score remaining_total
  ∃ (lowest : ℕ), 
    lowest = remaining_total - max_score_one_exam ∧
    lowest = 80 :=
sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l2552_255228


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2552_255296

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 2014, then n is 672 -/
theorem arithmetic_sequence_nth_term (n : ℕ) :
  arithmetic_sequence n = 2014 → n = 672 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2552_255296


namespace NUMINAMATH_CALUDE_correct_calculation_l2552_255250

theorem correct_calculation (x : ℝ) : 2 * (x + 6) = 28 → 6 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2552_255250


namespace NUMINAMATH_CALUDE_f_properties_l2552_255259

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let e := Real.exp 1
  (∀ x ∈ Set.Ioo 0 e, ∀ y ∈ Set.Ioo 0 e, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi e, ∀ y ∈ Set.Ioi e, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f x ≤ f e) ∧
  (∀ x ∈ Set.Ioi (Real.exp 1), f x < f e) ∧
  (∀ a : ℝ, (∀ x ≥ 1, f x ≤ a * (1 - 1 / x^2)) ↔ a ≥ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l2552_255259


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2552_255211

theorem binomial_divisibility (p m : ℕ) (hp : Prime p) (hm : m > 0) :
  p^m ∣ (Nat.choose (p^m) p - p^(m-1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2552_255211


namespace NUMINAMATH_CALUDE_rectangle_D_max_sum_l2552_255242

-- Define the rectangle structure
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the rectangles
def rectangleA : Rectangle := ⟨9, 3, 5, 7⟩
def rectangleB : Rectangle := ⟨8, 2, 4, 6⟩
def rectangleC : Rectangle := ⟨7, 1, 3, 5⟩
def rectangleD : Rectangle := ⟨10, 0, 6, 8⟩
def rectangleE : Rectangle := ⟨6, 4, 2, 0⟩

-- Define the list of all rectangles
def rectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Function to check if a value is unique in a list
def isUnique (n : ℕ) (l : List ℕ) : Bool :=
  (l.filter (· = n)).length = 1

-- Theorem: Rectangle D has the maximum sum of w + z where z is unique
theorem rectangle_D_max_sum : 
  ∀ r ∈ rectangles, 
    isUnique r.z (rectangles.map Rectangle.z) → 
      r.w + r.z ≤ rectangleD.w + rectangleD.z :=
sorry

end NUMINAMATH_CALUDE_rectangle_D_max_sum_l2552_255242


namespace NUMINAMATH_CALUDE_log_inequality_l2552_255240

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 2*x) < 2*x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2552_255240


namespace NUMINAMATH_CALUDE_chord_length_l2552_255232

/-- In a circle with radius 15 units, a chord that is a perpendicular bisector of the radius has a length of 26√3 units. -/
theorem chord_length (r : ℝ) (c : ℝ) : 
  r = 15 → -- The radius is 15 units
  c^2 = 4 * (r^2 - (r/2)^2) → -- The chord is a perpendicular bisector of the radius
  c = 26 * Real.sqrt 3 := by -- The length of the chord is 26√3 units
sorry

end NUMINAMATH_CALUDE_chord_length_l2552_255232


namespace NUMINAMATH_CALUDE_correct_exponent_calculation_l2552_255238

theorem correct_exponent_calculation (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_exponent_calculation_l2552_255238


namespace NUMINAMATH_CALUDE_power_three_405_mod_13_l2552_255263

theorem power_three_405_mod_13 : 3^405 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_three_405_mod_13_l2552_255263


namespace NUMINAMATH_CALUDE_expression_evaluation_l2552_255294

theorem expression_evaluation : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753/4648 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2552_255294


namespace NUMINAMATH_CALUDE_roots_shifted_l2552_255206

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the roots of the original polynomial
def roots_exist (a b c : ℝ) : Prop := 
  original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x + 10

-- Theorem statement
theorem roots_shifted (a b c : ℝ) : 
  roots_exist a b c → 
  (new_poly (a - 3) = 0 ∧ new_poly (b - 3) = 0 ∧ new_poly (c - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_l2552_255206


namespace NUMINAMATH_CALUDE_used_computer_cost_l2552_255218

/-- Proves the cost of each used computer given the conditions of the problem -/
theorem used_computer_cost
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_lifespan : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_lifespan = 3)
  (h4 : 2 * used_computer_lifespan = new_computer_lifespan)
  (h5 : savings = 200)
  (h6 : ∃ (used_computer_cost : ℕ),
        new_computer_cost = 2 * used_computer_cost + savings) :
  ∃ (used_computer_cost : ℕ), used_computer_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_used_computer_cost_l2552_255218


namespace NUMINAMATH_CALUDE_mini_cupcakes_count_l2552_255247

theorem mini_cupcakes_count (students : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) :
  students = 13 →
  donut_holes = 12 →
  desserts_per_student = 2 →
  ∃ (mini_cupcakes : ℕ), 
    mini_cupcakes + donut_holes = students * desserts_per_student ∧
    mini_cupcakes = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_mini_cupcakes_count_l2552_255247


namespace NUMINAMATH_CALUDE_emily_quiz_score_l2552_255295

/-- Emily's quiz scores -/
def emily_scores : List ℕ := [85, 88, 90, 94, 96, 92]

/-- The required arithmetic mean -/
def required_mean : ℕ := 91

/-- The number of quizzes including the new one -/
def total_quizzes : ℕ := 7

/-- The score Emily needs on her seventh quiz -/
def seventh_score : ℕ := 92

theorem emily_quiz_score :
  (emily_scores.sum + seventh_score) / total_quizzes = required_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_quiz_score_l2552_255295


namespace NUMINAMATH_CALUDE_two_sevenths_as_distinct_unit_fractions_l2552_255268

theorem two_sevenths_as_distinct_unit_fractions :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (2 : ℚ) / 7 = 1 / a + 1 / b + 1 / c :=
sorry

end NUMINAMATH_CALUDE_two_sevenths_as_distinct_unit_fractions_l2552_255268


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2552_255212

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (s : ℝ) : 
  (∀ k < n, ¬ ∃ (t : ℝ) (l : ℕ), t > 0 ∧ t < 1/500 ∧ l^(1/3 : ℝ) = k + t) →
  s > 0 → 
  s < 1/500 → 
  m^(1/3 : ℝ) = n + s → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2552_255212


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2552_255271

theorem negation_of_proposition (x y : ℝ) : 
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ ((x ≤ 0 ∨ y ≤ 0) → x * y ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2552_255271


namespace NUMINAMATH_CALUDE_select_five_from_fifteen_l2552_255275

theorem select_five_from_fifteen (n : Nat) (r : Nat) : n = 15 ∧ r = 5 →
  Nat.choose n r = 3003 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_fifteen_l2552_255275


namespace NUMINAMATH_CALUDE_f_inverse_property_implies_c_plus_d_eq_nine_halves_l2552_255214

-- Define the piecewise function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

-- State the theorem
theorem f_inverse_property_implies_c_plus_d_eq_nine_halves
  (c d : ℝ)
  (h : ∀ x, f c d (f c d x) = x) :
  c + d = 9/2 := by
sorry

end NUMINAMATH_CALUDE_f_inverse_property_implies_c_plus_d_eq_nine_halves_l2552_255214


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l2552_255233

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l2552_255233


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2552_255277

/-- If x^2 + 6x + k^2 is exactly the square of a polynomial, then k = ±3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → k = 3 ∨ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2552_255277


namespace NUMINAMATH_CALUDE_total_eggs_needed_l2552_255243

def eggs_from_andrew : ℕ := 155
def eggs_to_buy : ℕ := 67

theorem total_eggs_needed : 
  eggs_from_andrew + eggs_to_buy = 222 := by sorry

end NUMINAMATH_CALUDE_total_eggs_needed_l2552_255243


namespace NUMINAMATH_CALUDE_parallel_equidistant_lines_theorem_l2552_255200

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents three parallel, equidistant line segments -/
structure ParallelEquidistantLines where
  line1 : LineSegment
  line2 : LineSegment
  line3 : LineSegment

/-- Given three parallel, equidistant lines where the first line is 120 cm and the second is 80 cm,
    the length of the third line is 160/3 cm -/
theorem parallel_equidistant_lines_theorem (lines : ParallelEquidistantLines) 
    (h1 : lines.line1.length = 120)
    (h2 : lines.line2.length = 80) :
    lines.line3.length = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_equidistant_lines_theorem_l2552_255200


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2552_255281

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x = 2 → ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2552_255281


namespace NUMINAMATH_CALUDE_christinas_walking_speed_l2552_255216

/-- Prove that Christina's walking speed is 8 feet per second given the initial conditions and the total distance traveled by Lindy. -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jack_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : lindy_speed = 10)
  (h4 : lindy_total_distance = 100) :
  ∃ christina_speed : ℝ, christina_speed = 8 ∧ 
    (lindy_total_distance / lindy_speed) * (jack_speed + christina_speed) = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_christinas_walking_speed_l2552_255216


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l2552_255282

theorem discount_clinic_savings (normal_cost : ℚ) (discount_percentage : ℚ) (discount_visits : ℕ) : 
  normal_cost = 200 →
  discount_percentage = 70 →
  discount_visits = 2 →
  normal_cost - (discount_visits * (normal_cost * (1 - discount_percentage / 100))) = 80 := by
sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l2552_255282


namespace NUMINAMATH_CALUDE_seventeen_in_both_competitions_l2552_255262

/-- The number of students who participated in both math and physics competitions -/
def students_in_both_competitions (total : ℕ) (math : ℕ) (physics : ℕ) (none : ℕ) : ℕ :=
  math + physics + none - total

/-- Theorem stating that 17 students participated in both competitions -/
theorem seventeen_in_both_competitions :
  students_in_both_competitions 37 30 20 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_both_competitions_l2552_255262


namespace NUMINAMATH_CALUDE_sum_of_squares_for_specific_conditions_l2552_255272

theorem sum_of_squares_for_specific_conditions : 
  ∃ (S : Finset ℕ), 
    (∀ s ∈ S, ∃ x y z : ℕ, 
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      x + y + z = 30 ∧ 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 ∧
      s = x^2 + y^2 + z^2) ∧
    (∀ x y z : ℕ, 
      x > 0 → y > 0 → z > 0 →
      x + y + z = 30 → 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 →
      (x^2 + y^2 + z^2) ∈ S) ∧
    S.sum id = 710 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_for_specific_conditions_l2552_255272


namespace NUMINAMATH_CALUDE_vector_on_line_l2552_255241

/-- Given two complex numbers z₁ and z₂, representing points A and B in the complex plane,
    we define z as the vector from A to B, and prove that when z lies on the line y = 1/2 x,
    we can determine the value of parameter a. -/
theorem vector_on_line (a : ℝ) :
  let z₁ : ℂ := 2 * a + 6 * Complex.I
  let z₂ : ℂ := -1 + Complex.I
  let z : ℂ := z₂ - z₁
  z.im = (1/2 : ℝ) * z.re →
  z = -1 - 2 * a - 5 * Complex.I ∧ a = (9/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_l2552_255241


namespace NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l2552_255231

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l2552_255231


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2552_255234

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 70

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Beta T-Shirt Company's setup fee -/
def beta_setup : ℕ := 10

/-- Beta T-Shirt Company's per-shirt cost -/
def beta_per_shirt : ℕ := 15

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_for_acme < 
  beta_setup + beta_per_shirt * min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → 
    acme_setup + acme_per_shirt * n ≥ beta_setup + beta_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2552_255234


namespace NUMINAMATH_CALUDE_can_capacity_l2552_255222

/-- The capacity of a can given specific milk-water ratios --/
theorem can_capacity (initial_milk : ℝ) (initial_water : ℝ) (added_milk : ℝ) : 
  initial_water = 5 * initial_milk →
  added_milk = 2 →
  (initial_milk + added_milk) / initial_water = 2.00001 / 5.00001 →
  initial_milk + initial_water + added_milk = 14 := by
  sorry

end NUMINAMATH_CALUDE_can_capacity_l2552_255222


namespace NUMINAMATH_CALUDE_unknown_towel_rate_unknown_towel_rate_solution_l2552_255235

/-- Proves that the unknown rate of two towels is 300, given the conditions of the problem -/
theorem unknown_towel_rate : ℕ → Prop :=
  fun (x : ℕ) ↦
    let total_towels : ℕ := 3 + 5 + 2
    let known_cost : ℕ := 3 * 100 + 5 * 150
    let total_cost : ℕ := 165 * total_towels
    (known_cost + 2 * x = total_cost) → (x = 300)

/-- Solution to the unknown_towel_rate theorem -/
theorem unknown_towel_rate_solution : unknown_towel_rate 300 := by
  sorry

end NUMINAMATH_CALUDE_unknown_towel_rate_unknown_towel_rate_solution_l2552_255235


namespace NUMINAMATH_CALUDE_lottery_probability_l2552_255246

/-- The number of people participating in the lottery drawing -/
def num_people : ℕ := 4

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 4

/-- The number of winning tickets -/
def winning_tickets : ℕ := 2

/-- The probability that the event ends right after the third person has finished drawing -/
def prob_end_after_third : ℚ := 1/3

theorem lottery_probability :
  (num_people = 4) →
  (total_tickets = 4) →
  (winning_tickets = 2) →
  (prob_end_after_third = 1/3) := by
  sorry

#check lottery_probability

end NUMINAMATH_CALUDE_lottery_probability_l2552_255246


namespace NUMINAMATH_CALUDE_apple_picking_fraction_l2552_255239

theorem apple_picking_fraction (total_apples : ℕ) (remaining_apples : ℕ) : 
  total_apples = 200 →
  remaining_apples = 20 →
  ∃ f : ℚ, 
    f > 0 ∧ 
    f < 1 ∧
    (f * total_apples : ℚ) + (2 * f * total_apples : ℚ) + (f * total_apples + 20 : ℚ) = total_apples - remaining_apples ∧
    f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_fraction_l2552_255239


namespace NUMINAMATH_CALUDE_max_y_over_x_l2552_255205

theorem max_y_over_x (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≥ 0) (h3 : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (k : ℝ), ∀ (x' y' : ℝ), x' ≠ 0 → y' ≥ 0 → x'^2 + y'^2 - 4*x' + 1 = 0 → y'/x' ≤ k ∧ k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l2552_255205


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l2552_255264

def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x, 2, x^2],
    ![3, y, 4],
    ![z, 3, z^2]]

def B (x y z k l m n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![-8, k, -x^3],
    ![l, -y^2, m],
    ![3, n, z^3]]

theorem inverse_matrices_sum (x y z k l m n : ℝ) :
  A x y z * B x y z k l m n = 1 →
  x + y + z + k + l + m + n = -1/3 := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l2552_255264


namespace NUMINAMATH_CALUDE_set_problem_l2552_255280

theorem set_problem (U A B : Finset ℕ) (h1 : U.card = 190) (h2 : B.card = 49)
  (h3 : (U \ (A ∪ B)).card = 59) (h4 : (A ∩ B).card = 23) :
  A.card = 105 := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l2552_255280


namespace NUMINAMATH_CALUDE_marked_price_calculation_l2552_255297

theorem marked_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 100 →
  discount_rate = 0.2 →
  profit_rate = 0.2 →
  ∃ (marked_price : ℝ), 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    marked_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l2552_255297


namespace NUMINAMATH_CALUDE_polynomial_properties_l2552_255287

theorem polynomial_properties (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ Even q ∧ Odd p) ∧ 
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ Odd q ∧ Odd p) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l2552_255287


namespace NUMINAMATH_CALUDE_certification_cost_coverage_percentage_l2552_255215

/-- Calculates the percentage of certification cost covered by insurance for a seeing-eye dog. -/
theorem certification_cost_coverage_percentage
  (adoption_fee : ℕ)
  (training_cost_per_week : ℕ)
  (training_weeks : ℕ)
  (certification_cost : ℕ)
  (total_out_of_pocket : ℕ)
  (h1 : adoption_fee = 150)
  (h2 : training_cost_per_week = 250)
  (h3 : training_weeks = 12)
  (h4 : certification_cost = 3000)
  (h5 : total_out_of_pocket = 3450) :
  (100 * (certification_cost - (total_out_of_pocket - adoption_fee - training_cost_per_week * training_weeks))) / certification_cost = 90 :=
by sorry

end NUMINAMATH_CALUDE_certification_cost_coverage_percentage_l2552_255215


namespace NUMINAMATH_CALUDE_relay_race_total_time_l2552_255251

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_time athlete3_time athlete4_time : ℕ) : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : 
  ∀ (athlete1_time : ℕ),
    athlete1_time = 55 →
    ∀ (athlete2_time : ℕ),
      athlete2_time = athlete1_time + 10 →
      ∀ (athlete3_time : ℕ),
        athlete3_time = athlete2_time - 15 →
        ∀ (athlete4_time : ℕ),
          athlete4_time = athlete1_time - 25 →
          relay_race_time athlete1_time athlete2_time athlete3_time athlete4_time = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l2552_255251

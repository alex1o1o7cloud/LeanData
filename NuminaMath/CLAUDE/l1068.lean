import Mathlib

namespace NUMINAMATH_CALUDE_equal_distance_to_axes_l1068_106887

theorem equal_distance_to_axes (m : ℝ) : 
  let P : ℝ × ℝ := (3*m + 1, 2*m - 5)
  (|P.1| = |P.2|) ↔ (m = -6 ∨ m = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_equal_distance_to_axes_l1068_106887


namespace NUMINAMATH_CALUDE_intersection_solution_set_l1068_106894

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l1068_106894


namespace NUMINAMATH_CALUDE_youngest_not_first_or_last_l1068_106811

def number_of_people : ℕ := 5

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) : ℕ := Nat.factorial n

-- Define a function to calculate the number of valid arrangements
def valid_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * permutations (n - 1)

-- Theorem statement
theorem youngest_not_first_or_last :
  valid_arrangements number_of_people = 72 := by
  sorry

end NUMINAMATH_CALUDE_youngest_not_first_or_last_l1068_106811


namespace NUMINAMATH_CALUDE_calen_lost_pencils_l1068_106857

theorem calen_lost_pencils (p_candy p_caleb p_calen_original p_calen_after_loss : ℕ) :
  p_candy = 9 →
  p_caleb = 2 * p_candy - 3 →
  p_calen_original = p_caleb + 5 →
  p_calen_after_loss = 10 →
  p_calen_original - p_calen_after_loss = 10 := by
sorry

end NUMINAMATH_CALUDE_calen_lost_pencils_l1068_106857


namespace NUMINAMATH_CALUDE_divisibility_property_l1068_106807

def sequence_a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (2 * (n + 2) * sequence_a n - (n + 1) - 2) / (n + 1)

theorem divisibility_property (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ m : ℕ, p ∣ sequence_a m ∧ p ∣ sequence_a (m + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l1068_106807


namespace NUMINAMATH_CALUDE_no_double_square_sum_l1068_106866

theorem no_double_square_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), a^2 = x^2 + y ∧ b^2 = y^2 + x) := by
  sorry

end NUMINAMATH_CALUDE_no_double_square_sum_l1068_106866


namespace NUMINAMATH_CALUDE_fruit_eating_permutations_l1068_106801

theorem fruit_eating_permutations :
  let total_fruits : ℕ := 4 + 2 + 1
  let apple_count : ℕ := 4
  let orange_count : ℕ := 2
  let banana_count : ℕ := 1
  (Nat.factorial total_fruits) / 
  (Nat.factorial apple_count * Nat.factorial orange_count * Nat.factorial banana_count) = 105 := by
sorry

end NUMINAMATH_CALUDE_fruit_eating_permutations_l1068_106801


namespace NUMINAMATH_CALUDE_bug_flower_consumption_l1068_106829

theorem bug_flower_consumption (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 := by
  sorry

end NUMINAMATH_CALUDE_bug_flower_consumption_l1068_106829


namespace NUMINAMATH_CALUDE_taylor_family_reunion_tables_l1068_106893

theorem taylor_family_reunion_tables (num_kids : ℕ) (num_adults : ℕ) (people_per_table : ℕ) : 
  num_kids = 45 → num_adults = 123 → people_per_table = 12 → 
  (num_kids + num_adults) / people_per_table = 14 := by
sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_tables_l1068_106893


namespace NUMINAMATH_CALUDE_juggling_balls_average_l1068_106862

/-- Represents a juggling sequence -/
def JugglingSequence (n : ℕ) := Fin n → ℕ

/-- The number of balls in a juggling sequence -/
def numberOfBalls (n : ℕ) (j : JugglingSequence n) : ℚ :=
  (Finset.sum Finset.univ (fun i => j i)) / n

theorem juggling_balls_average (n : ℕ) (j : JugglingSequence n) :
  numberOfBalls n j = (Finset.sum Finset.univ (fun i => j i)) / n :=
by sorry

end NUMINAMATH_CALUDE_juggling_balls_average_l1068_106862


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l1068_106896

/-- Calculates the number of free donuts earned in a summer reading challenge -/
def free_donuts (books_per_week : ℕ) (weeks : ℕ) (books_per_coupon : ℕ) : ℕ :=
  (books_per_week * weeks) / books_per_coupon

/-- Proves that Quinn is eligible for 4 free donuts -/
theorem quinn_free_donuts : free_donuts 2 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l1068_106896


namespace NUMINAMATH_CALUDE_samara_detailing_cost_samara_detailing_cost_proof_l1068_106808

/-- Proves that Samara's spending on detailing equals $79 given the problem conditions -/
theorem samara_detailing_cost : ℕ → Prop :=
  fun (detailing_cost : ℕ) =>
    let alberto_total : ℕ := 2457
    let samara_oil : ℕ := 25
    let samara_tires : ℕ := 467
    let difference : ℕ := 1886
    alberto_total = samara_oil + samara_tires + detailing_cost + difference →
    detailing_cost = 79

/-- The proof of the theorem -/
theorem samara_detailing_cost_proof : samara_detailing_cost 79 := by
  sorry

end NUMINAMATH_CALUDE_samara_detailing_cost_samara_detailing_cost_proof_l1068_106808


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1068_106859

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 120) :
  ab + bc + cd ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1068_106859


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1068_106823

theorem sum_of_reciprocals (a b : ℝ) 
  (ha : a^2 + 2*a = 2) 
  (hb : b^2 + 2*b = 2) : 
  (1/a + 1/b = 1) ∨ 
  (1/a + 1/b = Real.sqrt 3 + 1) ∨ 
  (1/a + 1/b = -Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1068_106823


namespace NUMINAMATH_CALUDE_divisible_by_eleven_iff_d_equals_three_l1068_106825

/-- A function that constructs a six-digit number from its digits -/
def sixDigitNumber (a b c d e f : ℕ) : ℕ := 
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

/-- Proposition: The six-digit number 54321d is divisible by 11 if and only if d = 3 -/
theorem divisible_by_eleven_iff_d_equals_three : 
  ∀ d : ℕ, d < 10 → (sixDigitNumber 5 4 3 2 1 d) % 11 = 0 ↔ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_iff_d_equals_three_l1068_106825


namespace NUMINAMATH_CALUDE_divisible_numbers_in_range_l1068_106812

theorem divisible_numbers_in_range : ∃! n : ℕ, 
  1000 < n ∧ n < 2500 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisible_numbers_in_range_l1068_106812


namespace NUMINAMATH_CALUDE_rectangle_tiling_divisibility_l1068_106879

/-- An L-shaped piece made of 4 unit squares -/
structure LPiece :=
  (squares : Fin 4 → (Nat × Nat))

/-- A tiling of an m × n rectangle with L-shaped pieces -/
def Tiling (m n : Nat) := List LPiece

/-- Predicate to check if a tiling is valid for an m × n rectangle -/
def IsValidTiling (t : Tiling m n) : Prop := sorry

theorem rectangle_tiling_divisibility (m n : Nat) (t : Tiling m n) :
  IsValidTiling t → (m * n) % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_divisibility_l1068_106879


namespace NUMINAMATH_CALUDE_root_line_tangent_to_discriminant_parabola_l1068_106841

/-- The discriminant parabola in the Opq plane -/
def discriminant_parabola (p q : ℝ) : Prop := p^2 - 4*q = 0

/-- The root line for a given real number a in the Opq plane -/
def root_line (a p q : ℝ) : Prop := a^2 + a*p + q = 0

/-- A line is tangent to the discriminant parabola -/
def is_tangent_line (p q : ℝ → ℝ) : Prop :=
  ∃ (x : ℝ), discriminant_parabola (p x) (q x) ∧
    ∀ (y : ℝ), y ≠ x → ¬discriminant_parabola (p y) (q y)

theorem root_line_tangent_to_discriminant_parabola :
  (∀ a : ℝ, ∃ p q : ℝ → ℝ, is_tangent_line p q ∧ ∀ x : ℝ, root_line a (p x) (q x)) ∧
  (∀ p q : ℝ → ℝ, is_tangent_line p q → ∃ a : ℝ, ∀ x : ℝ, root_line a (p x) (q x)) :=
sorry

end NUMINAMATH_CALUDE_root_line_tangent_to_discriminant_parabola_l1068_106841


namespace NUMINAMATH_CALUDE_total_potatoes_eq_sum_l1068_106830

/-- The number of potatoes mother bought -/
def total_potatoes : ℕ := sorry

/-- The number of potatoes used for salads -/
def salad_potatoes : ℕ := 15

/-- The number of potatoes used for mashed potatoes -/
def mashed_potatoes : ℕ := 24

/-- The number of leftover potatoes -/
def leftover_potatoes : ℕ := 13

/-- Theorem stating that the total number of potatoes is equal to the sum of
    potatoes used for salads, mashed potatoes, and leftover potatoes -/
theorem total_potatoes_eq_sum :
  total_potatoes = salad_potatoes + mashed_potatoes + leftover_potatoes := by sorry

end NUMINAMATH_CALUDE_total_potatoes_eq_sum_l1068_106830


namespace NUMINAMATH_CALUDE_gunther_typing_capacity_l1068_106890

/-- Gunther's typing rate in words per 3 minutes -/
def typing_rate : ℕ := 160

/-- Number of minutes in 3 minutes -/
def minutes_per_unit : ℕ := 3

/-- Number of minutes Gunther works per day -/
def working_minutes : ℕ := 480

/-- Number of words Gunther can type in a working day -/
def words_per_day : ℕ := 25598

theorem gunther_typing_capacity :
  (typing_rate : ℚ) / minutes_per_unit * working_minutes = words_per_day := by
  sorry

end NUMINAMATH_CALUDE_gunther_typing_capacity_l1068_106890


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l1068_106869

theorem factorial_equation_solution :
  ∃ (n : ℕ), (4 * 3 * 2 * 1) / (Nat.factorial (4 - n)) = 24 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l1068_106869


namespace NUMINAMATH_CALUDE_sin_75_degrees_l1068_106876

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l1068_106876


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_l1068_106824

def total_runs : ℕ := 120
def boundaries : ℕ := 5
def sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem percentage_runs_by_running (total_runs boundaries sixes runs_per_boundary runs_per_six : ℕ) 
  (h1 : total_runs = 120)
  (h2 : boundaries = 5)
  (h3 : sixes = 5)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6) :
  (total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100 = 
  (120 - (5 * 4 + 5 * 6)) / 120 * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_l1068_106824


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l1068_106897

theorem divisible_by_eleven (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∃ k : ℤ, (n^2 + 4^n + 7^n : ℤ) = k * n) : 
  ∃ m : ℤ, (n^2 + 4^n + 7^n : ℤ) / n = 11 * m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l1068_106897


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1068_106805

theorem unique_solution_for_equation (N : ℕ+) :
  ∃! (m n : ℕ+), m + (1/2 : ℚ) * (m + n - 1) * (m + n - 2) = N := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1068_106805


namespace NUMINAMATH_CALUDE_box_height_proof_l1068_106800

/-- Proves that the height of boxes is 12 inches given the specified conditions. -/
theorem box_height_proof (box_length : ℝ) (box_width : ℝ) (total_volume : ℝ) 
  (cost_per_box : ℝ) (min_spend : ℝ) (h : ℝ) : 
  box_length = 20 → 
  box_width = 20 → 
  total_volume = 2160000 → 
  cost_per_box = 0.4 → 
  min_spend = 180 → 
  (total_volume / (box_length * box_width * h)) * cost_per_box = min_spend → 
  h = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l1068_106800


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l1068_106892

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the points
def P₁ : ℝ × ℝ := (-1, f (-1))
def P₂ : ℝ × ℝ := (2, f 2)
def P₃ : ℝ × ℝ := (5, f 5)

-- Theorem statement
theorem quadratic_points_ordering :
  P₂.2 > P₁.2 ∧ P₁.2 > P₃.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l1068_106892


namespace NUMINAMATH_CALUDE_common_tangents_of_specific_circles_l1068_106878

/-- The number of common tangents to two intersecting circles -/
def num_common_tangents (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : ℕ :=
  sorry

/-- The theorem stating that the number of common tangents to the given circles is 2 -/
theorem common_tangents_of_specific_circles : 
  num_common_tangents (2, 1) 2 (-1, 2) 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_tangents_of_specific_circles_l1068_106878


namespace NUMINAMATH_CALUDE_chess_team_shirt_numbers_l1068_106816

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 
  10 ≤ n ∧ n ≤ 99

theorem chess_team_shirt_numbers 
  (d e f : ℕ) 
  (h1 : isPrime d ∧ isPrime e ∧ isPrime f)
  (h2 : isTwoDigit d ∧ isTwoDigit e ∧ isTwoDigit f)
  (h3 : d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) :
  f = 19 := by
sorry

end NUMINAMATH_CALUDE_chess_team_shirt_numbers_l1068_106816


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l1068_106860

/-- Represents a solid constructed from unit cubes -/
structure CubeSolid where
  bottomRow : ℕ
  middleColumn : ℕ
  leftColumns : ℕ
  leftColumnHeight : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surfaceArea (solid : CubeSolid) : ℕ :=
  let bottomArea := solid.bottomRow + 2 * (solid.bottomRow + 1)
  let middleColumnArea := 4 + (solid.middleColumn - 1)
  let leftColumnsArea := 2 * (2 * solid.leftColumnHeight + 1)
  bottomArea + middleColumnArea + leftColumnsArea

/-- The specific solid described in the problem -/
def problemSolid : CubeSolid :=
  { bottomRow := 5
  , middleColumn := 5
  , leftColumns := 2
  , leftColumnHeight := 3 }

theorem problem_solid_surface_area :
  surfaceArea problemSolid = 34 := by
  sorry

#eval surfaceArea problemSolid

end NUMINAMATH_CALUDE_problem_solid_surface_area_l1068_106860


namespace NUMINAMATH_CALUDE_range_of_m_l1068_106840

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂) →
  m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1068_106840


namespace NUMINAMATH_CALUDE_circle_tape_length_16_strips_l1068_106835

/-- The total length of a circle-shaped tape made from overlapping strips -/
def circle_tape_length (num_strips : ℕ) (strip_length : ℝ) (overlap_length : ℝ) : ℝ :=
  num_strips * strip_length - num_strips * overlap_length

/-- Theorem: The length of a circle-shaped tape made from 16 strips of 10.4 cm
    with 3.5 cm overlaps is 110.4 cm -/
theorem circle_tape_length_16_strips :
  circle_tape_length 16 10.4 3.5 = 110.4 := by
  sorry

#eval circle_tape_length 16 10.4 3.5

end NUMINAMATH_CALUDE_circle_tape_length_16_strips_l1068_106835


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1068_106845

/-- The positive slope of an asymptote of the hyperbola defined by 
    √((x-2)² + (y-3)²) - √((x-8)² + (y-3)²) = 4 -/
theorem hyperbola_asymptote_slope : ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4 →
    m = Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1068_106845


namespace NUMINAMATH_CALUDE_solution_condition_l1068_106817

theorem solution_condition (m : ℚ) : (∀ x, m * x = m ↔ x = 1) ↔ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_condition_l1068_106817


namespace NUMINAMATH_CALUDE_natalia_comics_count_l1068_106863

/-- The number of novels Natalia has -/
def novels : ℕ := 145

/-- The number of documentaries Natalia has -/
def documentaries : ℕ := 419

/-- The number of albums Natalia has -/
def albums : ℕ := 209

/-- The number of items each crate can hold -/
def items_per_crate : ℕ := 9

/-- The number of crates Natalia will use -/
def num_crates : ℕ := 116

/-- The number of comics Natalia has -/
def comics : ℕ := 271

theorem natalia_comics_count : 
  novels + documentaries + albums + comics = num_crates * items_per_crate := by
  sorry

end NUMINAMATH_CALUDE_natalia_comics_count_l1068_106863


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_nine_l1068_106818

def number (B : Nat) : Nat := 5 * 100000 + B * 10000 + 4 * 1000 + 8 * 100 + 6 * 10 + 1

theorem largest_B_divisible_by_nine :
  ∀ B : Nat, B < 10 →
    (∃ m : Nat, number B = 9 * m) →
    B ≤ 9 ∧
    (∀ C : Nat, C < 10 → C > B → ¬∃ n : Nat, number C = 9 * n) :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_nine_l1068_106818


namespace NUMINAMATH_CALUDE_april_roses_unsold_l1068_106891

/-- The number of roses left unsold after a sale --/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Theorem: Given the conditions of April's rose sale, prove that 4 roses were left unsold --/
theorem april_roses_unsold : roses_left_unsold 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_roses_unsold_l1068_106891


namespace NUMINAMATH_CALUDE_max_base_eight_digit_sum_l1068_106846

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base-eight representation --/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-eight representation --/
def digitSum (rep : BaseEightRepresentation) : Nat :=
  sorry

/-- Theorem stating the maximum digit sum for numbers less than 1729 in base 8 --/
theorem max_base_eight_digit_sum :
  (∃ (n : Nat), n < 1729 ∧ 
    digitSum (toBaseEight n) = 19 ∧ 
    ∀ (m : Nat), m < 1729 → digitSum (toBaseEight m) ≤ 19) :=
  sorry

end NUMINAMATH_CALUDE_max_base_eight_digit_sum_l1068_106846


namespace NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l1068_106873

/-- If positive numbers a, b, c form an arithmetic sequence with non-zero common difference,
    then their reciprocals cannot form an arithmetic sequence. -/
theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h_arith : ∃ d ≠ 0, b - a = d ∧ c - b = d) : 
    ¬∃ k : ℝ, (1 / b - 1 / a = k) ∧ (1 / c - 1 / b = k) := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l1068_106873


namespace NUMINAMATH_CALUDE_roberto_cost_per_dozen_approx_l1068_106898

/-- Represents the chicken and egg scenario for Roberto --/
structure ChickenScenario where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  break_even_weeks : ℕ

/-- Calculates the cost per dozen eggs given a ChickenScenario --/
def cost_per_dozen (scenario : ChickenScenario) : ℚ :=
  let total_cost := scenario.num_chickens * scenario.chicken_cost + 
                    scenario.weekly_feed_cost * scenario.break_even_weeks
  let total_eggs := scenario.num_chickens * scenario.eggs_per_chicken_per_week * 
                    scenario.break_even_weeks
  let total_dozens := total_eggs / 12
  total_cost / total_dozens

/-- Roberto's specific scenario --/
def roberto_scenario : ChickenScenario :=
  { num_chickens := 4
  , chicken_cost := 20
  , weekly_feed_cost := 1
  , eggs_per_chicken_per_week := 3
  , break_even_weeks := 81 }

/-- Theorem stating that Roberto's cost per dozen eggs is approximately $1.99 --/
theorem roberto_cost_per_dozen_approx :
  abs (cost_per_dozen roberto_scenario - 1.99) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_roberto_cost_per_dozen_approx_l1068_106898


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1068_106802

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 ≤ x ∧ x < 1) ↔ (x < -1 ∨ x ≥ 1 → x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1068_106802


namespace NUMINAMATH_CALUDE_area_triangle_AEF_l1068_106809

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ :=
  sorry

/-- Divides a line segment in a given ratio -/
def divideLineSegment (A B : Point) (ratio : ℚ) : Point :=
  sorry

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ :=
  sorry

theorem area_triangle_AEF (ABCD : Quadrilateral) 
  (hParallelogram : isParallelogram ABCD)
  (hArea : areaQuadrilateral ABCD = 50)
  (E : Point) (hE : E = divideLineSegment ABCD.A ABCD.B (2/5))
  (F : Point) (hF : F = divideLineSegment ABCD.C ABCD.D (3/5))
  (G : Point) (hG : G = divideLineSegment ABCD.B ABCD.C (1/2)) :
  areaTriangle ⟨ABCD.A, E, F⟩ = 12 :=
sorry

end NUMINAMATH_CALUDE_area_triangle_AEF_l1068_106809


namespace NUMINAMATH_CALUDE_circulation_ratio_l1068_106842

/-- Represents the circulation of a magazine over time -/
structure MagazineCirculation where
  /-- Circulation in 1962 -/
  C_1962 : ℝ
  /-- Growth rate per year (as a decimal) -/
  r : ℝ
  /-- Average yearly circulation from 1962 to 1970 -/
  A : ℝ

/-- Theorem stating the ratio of circulation in 1961 to total circulation 1961-1970 -/
theorem circulation_ratio (P : MagazineCirculation) :
  /- Circulation in 1961 is 4 times the average from 1962-1970 -/
  (4 * P.A) / 
  /- Total circulation from 1961-1970 -/
  (4 * P.A + 9 * P.A) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_circulation_ratio_l1068_106842


namespace NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l1068_106888

/-- For a cube with side length s, if the volume is equal to the surface area, then s = 6. -/
theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l1068_106888


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1068_106855

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (x < 4 → x < 0) ∧ ¬(x < 0 → x < 4) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1068_106855


namespace NUMINAMATH_CALUDE_heather_walk_distance_l1068_106820

theorem heather_walk_distance : 
  let car_to_entrance : Float := 0.645
  let to_carnival : Float := 1.235
  let to_animals : Float := 0.875
  let to_food : Float := 1.537
  let food_to_car : Float := 0.932
  car_to_entrance + to_carnival + to_animals + to_food + food_to_car = 5.224 := by
  sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l1068_106820


namespace NUMINAMATH_CALUDE_box_side_length_l1068_106877

/-- Given the total volume needed, cost per box, and minimum total cost,
    calculate the length of one side of a cubic box. -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ) 
  (h1 : total_volume = 1920000) 
  (h2 : cost_per_box = 0.5) 
  (h3 : min_total_cost = 200) : 
  ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry

#check box_side_length

end NUMINAMATH_CALUDE_box_side_length_l1068_106877


namespace NUMINAMATH_CALUDE_derivative_of_x_exp_x_l1068_106858

noncomputable def f (x : ℝ) := x * Real.exp x

theorem derivative_of_x_exp_x :
  deriv f = fun x ↦ (1 + x) * Real.exp x := by sorry

end NUMINAMATH_CALUDE_derivative_of_x_exp_x_l1068_106858


namespace NUMINAMATH_CALUDE_domain_intersection_and_union_range_of_p_l1068_106839

def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

theorem domain_intersection_and_union :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3}) ∧
  (A ∪ B = Set.univ) :=
sorry

theorem range_of_p (p : ℝ) :
  (C p ⊆ A) → p ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_and_union_range_of_p_l1068_106839


namespace NUMINAMATH_CALUDE_solve_for_t_l1068_106828

theorem solve_for_t (a b d x y t : ℕ) 
  (h1 : a + b = x)
  (h2 : x + d = t)
  (h3 : t + a = y)
  (h4 : b + d + y = 16)
  (ha : a > 0)
  (hb : b > 0)
  (hd : d > 0)
  (hx : x > 0)
  (hy : y > 0)
  (ht : t > 0) :
  t = 8 := by
sorry


end NUMINAMATH_CALUDE_solve_for_t_l1068_106828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1068_106844

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = 10 →
  a 4 = a 3 + 2 →
  a 3 + a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1068_106844


namespace NUMINAMATH_CALUDE_segment_length_from_perpendicular_lines_and_midpoint_l1068_106822

/-- Given two perpendicular lines and a midpoint, prove the length of the segment. -/
theorem segment_length_from_perpendicular_lines_and_midpoint
  (A B : ℝ × ℝ) -- Points A and B
  (a : ℝ) -- Parameter in the equation of the second line
  (h1 : (2 * A.1 - A.2 = 0)) -- A is on the line 2x - y = 0
  (h2 : (B.1 + a * B.2 = 0)) -- B is on the line x + ay = 0
  (h3 : (2 : ℝ) * A.1 + (-1 : ℝ) * a = 0) -- Perpendicularity condition
  (h4 : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 10 / a) -- Midpoint condition
  : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_from_perpendicular_lines_and_midpoint_l1068_106822


namespace NUMINAMATH_CALUDE_quadratic_sum_l1068_106880

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -3 and 5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 36) ∧
  QuadraticFunction a b c (-3) = 0 ∧
  QuadraticFunction a b c 5 = 0 →
  a + b + c = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1068_106880


namespace NUMINAMATH_CALUDE_exists_non_increasing_exponential_l1068_106856

theorem exists_non_increasing_exponential : 
  ∃ (a : ℝ), a > 0 ∧ ¬(∀ x y : ℝ, x < y → (a^(-x) : ℝ) < a^(-y)) :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_exponential_l1068_106856


namespace NUMINAMATH_CALUDE_bruce_remaining_eggs_l1068_106810

def bruce_initial_eggs : ℕ := 75
def eggs_lost : ℕ := 70

theorem bruce_remaining_eggs :
  bruce_initial_eggs - eggs_lost = 5 :=
by sorry

end NUMINAMATH_CALUDE_bruce_remaining_eggs_l1068_106810


namespace NUMINAMATH_CALUDE_mork_mindy_tax_rate_l1068_106884

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) :
  mork_tax_rate = 0.45 →
  mindy_tax_rate = 0.15 →
  let mindy_income := 4 * mork_income
  let combined_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_income := mork_income + mindy_income
  combined_tax / combined_income = 0.21 :=
by
  sorry

#check mork_mindy_tax_rate

end NUMINAMATH_CALUDE_mork_mindy_tax_rate_l1068_106884


namespace NUMINAMATH_CALUDE_false_proposition_l1068_106848

def p1 : Prop := ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0

def p2 : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem false_proposition : ¬((¬p1) ∧ (¬p2)) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l1068_106848


namespace NUMINAMATH_CALUDE_total_courses_is_200_l1068_106843

/-- The number of college courses Max attended -/
def max_courses : ℕ := 40

/-- The number of college courses Sid attended relative to Max -/
def sid_multiplier : ℕ := 4

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_multiplier * max_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_courses_is_200_l1068_106843


namespace NUMINAMATH_CALUDE_f_properties_l1068_106851

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (3^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 2 ∧ ∀ x y, x < y → f 2 x < f 2 y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1068_106851


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1068_106852

theorem min_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                      2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                      (x1 - x2)^2 + (y1 - y2)^2 = 16) →
  (4/a + 1/b ≥ 9 ∧ ∃ a0 b0 : ℝ, a0 > 0 ∧ b0 > 0 ∧ 4/a0 + 1/b0 = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1068_106852


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l1068_106875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment (A B A' : Point) :
  A.x = -4 ∧ A.y = -1 ∧
  B.x = 1 ∧ B.y = 1 ∧
  A'.x = -2 ∧ A'.y = 2 →
  ∃ (t : Translation), translatePoint A t = A' ∧ translatePoint B t = { x := 3, y := 4 } := by
  sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l1068_106875


namespace NUMINAMATH_CALUDE_ball_box_distribution_l1068_106895

def num_balls : ℕ := 5
def num_boxes : ℕ := 5

/-- The number of ways to put all balls into boxes -/
def total_ways : ℕ := num_boxes ^ num_balls

/-- The number of ways to put balls into boxes with exactly one box left empty -/
def one_empty : ℕ := Nat.choose num_boxes 2 * Nat.factorial (num_balls - 1)

/-- The number of ways to put balls into boxes with exactly two boxes left empty -/
def two_empty : ℕ := 
  (Nat.choose num_boxes 2 * Nat.choose 3 2 * Nat.factorial (num_balls - 2) +
   Nat.choose num_boxes 3 * Nat.choose 2 1 * Nat.factorial (num_balls - 2)) * 
  Nat.factorial num_boxes / (Nat.factorial 2)

theorem ball_box_distribution :
  total_ways = 3125 ∧ one_empty = 1200 ∧ two_empty = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_distribution_l1068_106895


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1068_106821

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  let original_area := l * w
  let new_area := (2 * l) * (2 * w)
  (new_area - original_area) / original_area = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1068_106821


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1068_106806

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1068_106806


namespace NUMINAMATH_CALUDE_packing_problem_l1068_106899

theorem packing_problem :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end NUMINAMATH_CALUDE_packing_problem_l1068_106899


namespace NUMINAMATH_CALUDE_inequality_range_l1068_106833

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Set.Ioo (-10) 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1068_106833


namespace NUMINAMATH_CALUDE_recurrence_sequence_eventually_periodic_l1068_106865

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → u n = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

/-- A sequence is bounded if there exist m and M such that m ≤ u_n ≤ M for all n -/
def IsBounded (u : ℕ → ℤ) : Prop :=
  ∃ m M : ℤ, ∀ n : ℕ, m ≤ u n ∧ u n ≤ M

/-- A sequence is eventually periodic if there exist N and p such that u_{n+p} = u_n for all n ≥ N -/
def EventuallyPeriodic (u : ℕ → ℤ) : Prop :=
  ∃ N p : ℕ, p > 0 ∧ ∀ n : ℕ, n ≥ N → u (n + p) = u n

/-- The main theorem: a bounded recurrence sequence is eventually periodic -/
theorem recurrence_sequence_eventually_periodic (u : ℕ → ℤ) 
  (h_recurrence : RecurrenceSequence u) (h_bounded : IsBounded u) : 
  EventuallyPeriodic u :=
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_eventually_periodic_l1068_106865


namespace NUMINAMATH_CALUDE_characterization_theorem_l1068_106837

/-- A function that checks if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

/-- A function that checks if a number is a square of a prime -/
def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = p * p

/-- The main theorem statement -/
theorem characterization_theorem (n : ℕ) (h : n ≥ 2) :
  (∀ d : ℕ, d ≥ 2 → d ∣ n → (d - 1) ∣ (n - 1)) ↔ (is_prime n ∨ is_prime_square n) :=
sorry

end NUMINAMATH_CALUDE_characterization_theorem_l1068_106837


namespace NUMINAMATH_CALUDE_translation_proof_l1068_106868

-- Define a translation of the complex plane
def translation (z w : ℂ) := z + w

-- Theorem statement
theorem translation_proof (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z : ℂ, t z = translation z w) →
  (t (1 + 3*I) = 4 + 7*I) →
  (t (2 - I) = 5 + 3*I) :=
by sorry

end NUMINAMATH_CALUDE_translation_proof_l1068_106868


namespace NUMINAMATH_CALUDE_find_number_l1068_106874

theorem find_number (x : ℝ) : ((x - 1.9) * 1.5 + 32) / 2.5 = 20 → x = 13.9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1068_106874


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l1068_106847

theorem sum_a_b_equals_one (a b : ℝ) : 
  Real.sqrt (a - b - 3) + abs (2 * a - 4) = 0 → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l1068_106847


namespace NUMINAMATH_CALUDE_symmetric_points_count_l1068_106889

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are distinct and non-collinear -/
def distinct_non_collinear (M G T : Point2D) : Prop :=
  M ≠ G ∧ M ≠ T ∧ G ≠ T ∧
  (G.x - M.x) * (T.y - M.y) ≠ (T.x - M.x) * (G.y - M.y)

/-- Check if a figure has an axis of symmetry -/
def has_axis_of_symmetry (points : List Point2D) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Count the number of distinct points U that create a figure with symmetry -/
def count_symmetric_points (M G T : Point2D) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem symmetric_points_count 
  (M G T : Point2D) 
  (h1 : distinct_non_collinear M G T) 
  (h2 : ¬ has_axis_of_symmetry [M, G, T]) :
  count_symmetric_points M G T = 5 ∨ count_symmetric_points M G T = 6 :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l1068_106889


namespace NUMINAMATH_CALUDE_expression_equals_one_l1068_106886

theorem expression_equals_one (b : ℝ) (hb : b ≠ 0) :
  ∀ x : ℝ, x ≠ b ∧ x ≠ -b →
    (b / (b - x) - x / (b + x)) / (b / (b + x) + x / (b - x)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1068_106886


namespace NUMINAMATH_CALUDE_max_value_fraction_l1068_106872

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : b^2 + 2*(a + c)*b - a*c = 0) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → y^2 + 2*(x + z)*y - x*z = 0 → 
  y / (x + z) ≤ b / (a + c) → b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1068_106872


namespace NUMINAMATH_CALUDE_red_peach_count_l1068_106853

/-- Represents the count of peaches of different colors in a basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Given a basket of peaches with 8 green peaches and 1 more green peach than red peaches,
    prove that there are 7 red peaches -/
theorem red_peach_count (basket : PeachBasket) 
    (green_count : basket.green = 8)
    (green_red_diff : basket.green = basket.red + 1) : 
  basket.red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peach_count_l1068_106853


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1068_106813

def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / (10 - a) + y^2 / (a - 2) = 1

theorem ellipse_focal_length (a : ℝ) : 
  (∃ x y : ℝ, ellipse_equation x y a) → 
  (∃ c : ℝ, c = 2) →
  (a = 4 ∨ a = 8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1068_106813


namespace NUMINAMATH_CALUDE_factorization_equality_l1068_106864

theorem factorization_equality (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1068_106864


namespace NUMINAMATH_CALUDE_inequality_interval_length_l1068_106831

/-- Given an inequality a ≤ 3x + 4 ≤ b, if the length of the interval of solutions is 8, then b - a = 24 -/
theorem inequality_interval_length (a b : ℝ) : 
  (∃ (l : ℝ), l = 8 ∧ l = (b - 4) / 3 - (a - 4) / 3) → b - a = 24 :=
by sorry

end NUMINAMATH_CALUDE_inequality_interval_length_l1068_106831


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l1068_106881

/-- The number of cats Susan has initially -/
def susan_initial_cats : ℕ := 21

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives to Robert -/
def cats_given_to_robert : ℕ := 4

/-- Theorem stating the difference between Susan's remaining cats and Bob's cats -/
theorem susan_bob_cat_difference : 
  susan_initial_cats - cats_given_to_robert - bob_cats = 14 := by
  sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l1068_106881


namespace NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1068_106838

theorem quadratic_expression_evaluation :
  let x : ℝ := 2
  (x^2 - 3*x + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1068_106838


namespace NUMINAMATH_CALUDE_lioness_age_l1068_106854

theorem lioness_age (hyena_age lioness_age : ℕ) : 
  lioness_age = 2 * hyena_age →
  (hyena_age / 2 + 5) + (lioness_age / 2 + 5) = 19 →
  lioness_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_lioness_age_l1068_106854


namespace NUMINAMATH_CALUDE_nested_sqrt_solution_l1068_106815

theorem nested_sqrt_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_solution_l1068_106815


namespace NUMINAMATH_CALUDE_slope_condition_implies_m_zero_l1068_106814

theorem slope_condition_implies_m_zero (m : ℝ) : 
  (4 - m^2) / (m - (-2)) = 2 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_m_zero_l1068_106814


namespace NUMINAMATH_CALUDE_fraction_equality_l1068_106870

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1068_106870


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1068_106867

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 15 - 20 * I ∧ Complex.abs w = 5 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1068_106867


namespace NUMINAMATH_CALUDE_only_setA_cannot_form_triangle_l1068_106861

-- Define a function to check if three line segments can form a triangle
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def setA : List ℝ := [4, 4, 9]
def setB : List ℝ := [3, 5, 6]
def setC : List ℝ := [6, 8, 10]
def setD : List ℝ := [5, 12, 13]

-- State the theorem
theorem only_setA_cannot_form_triangle :
  (¬ canFormTriangle setA[0] setA[1] setA[2]) ∧
  (canFormTriangle setB[0] setB[1] setB[2]) ∧
  (canFormTriangle setC[0] setC[1] setC[2]) ∧
  (canFormTriangle setD[0] setD[1] setD[2]) := by
  sorry

end NUMINAMATH_CALUDE_only_setA_cannot_form_triangle_l1068_106861


namespace NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l1068_106849

structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

def similar_triangle (T1 T2 : Triangle) (ratio : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), 
    (T2.A.1 - center.1 = ratio * (T1.A.1 - center.1)) ∧
    (T2.A.2 - center.2 = ratio * (T1.A.2 - center.2)) ∧
    (T2.B.1 - center.1 = ratio * (T1.B.1 - center.1)) ∧
    (T2.B.2 - center.2 = ratio * (T1.B.2 - center.2))

theorem similar_triangle_point_coordinates 
  (a : ℝ) 
  (OAB : Triangle) 
  (OCD : Triangle) 
  (h1 : OAB.O = (0, 0)) 
  (h2 : OAB.A = (4, 3)) 
  (h3 : OAB.B = (3, a)) 
  (h4 : similar_triangle OAB OCD (1/3)) 
  (h5 : OCD.O = (0, 0)) :
  OCD.A = (4/3, 1) ∨ OCD.A = (-4/3, -1) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l1068_106849


namespace NUMINAMATH_CALUDE_exists_arrangement_with_more_than_five_holes_l1068_106883

/-- Represents a strange ring, which is a circle with a square hole in the middle. -/
structure StrangeRing where
  circle_radius : ℝ
  square_side : ℝ
  center : ℝ × ℝ
  h_square_fits : square_side ≤ 2 * circle_radius

/-- Represents an arrangement of two strange rings on a table. -/
structure StrangeRingArrangement where
  ring1 : StrangeRing
  ring2 : StrangeRing
  placement : ℝ × ℝ  -- Relative placement of ring2 with respect to ring1

/-- Counts the number of holes in a given arrangement of strange rings. -/
def count_holes (arrangement : StrangeRingArrangement) : ℕ :=
  sorry

/-- Theorem stating that there exists an arrangement of two strange rings
    that results in more than 5 holes. -/
theorem exists_arrangement_with_more_than_five_holes :
  ∃ (arrangement : StrangeRingArrangement), count_holes arrangement > 5 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_with_more_than_five_holes_l1068_106883


namespace NUMINAMATH_CALUDE_factorization_equality_l1068_106871

theorem factorization_equality (m n : ℝ) : m^2 * n - 16 * n = n * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1068_106871


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l1068_106834

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l1068_106834


namespace NUMINAMATH_CALUDE_probability_B_given_A_l1068_106826

/-- Represents the number of people in the research study group -/
def group_size : ℕ := 6

/-- Represents the number of halls in the exhibition -/
def num_halls : ℕ := 3

/-- Represents the event A: In the first hour, each hall has exactly 2 people -/
def event_A : Prop := True

/-- Represents the event B: In the second hour, there are exactly 2 people in Hall A -/
def event_B : Prop := True

/-- Represents the number of ways event B can occur given event A has occurred -/
def ways_B_given_A : ℕ := 3

/-- Represents the total number of possible distributions in the second hour -/
def total_distributions : ℕ := 8

/-- The probability of event B given event A -/
def P_B_given_A : ℚ := ways_B_given_A / total_distributions

theorem probability_B_given_A : 
  P_B_given_A = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_B_given_A_l1068_106826


namespace NUMINAMATH_CALUDE_factorial_difference_l1068_106804

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1068_106804


namespace NUMINAMATH_CALUDE_pizza_diameter_increase_l1068_106850

theorem pizza_diameter_increase (d : ℝ) (D : ℝ) (h : d > 0) (h' : D > 0) :
  (π * (D / 2)^2 = 1.96 * π * (d / 2)^2) →
  (D = 1.4 * d) := by
sorry

end NUMINAMATH_CALUDE_pizza_diameter_increase_l1068_106850


namespace NUMINAMATH_CALUDE_quarter_orbit_distance_l1068_106882

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ  -- Distance of nearest point from focus
  apogee : ℝ   -- Distance of farthest point from focus

/-- Calculates the distance from a point on the orbit to the focus -/
def distance_to_focus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  sorry

theorem quarter_orbit_distance (orbit : EllipticalOrbit) 
  (h1 : orbit.perigee = 3)
  (h2 : orbit.apogee = 15) :
  distance_to_focus orbit 0.25 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_quarter_orbit_distance_l1068_106882


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1068_106885

/-- A quadratic function with two distinct zeros -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  has_distinct_zeros : ∃ (x y : ℝ), x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- Four distinct roots in arithmetic progression -/
structure FourRoots where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  distinct : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
  arithmetic : ∃ (d : ℝ), x₂ - x₁ = d ∧ x₃ - x₂ = d ∧ x₄ - x₃ = d

/-- The main theorem -/
theorem quadratic_function_range (f : QuadraticFunction) (roots : FourRoots) 
  (h : ∀ x, (x^2 + 2*x - 1)^2 + f.a*(x^2 + 2*x - 1) + f.b = 0 ↔ 
           x = roots.x₁ ∨ x = roots.x₂ ∨ x = roots.x₃ ∨ x = roots.x₄) :
  ∀ x, x ≤ 25/9 ∧ (∃ y, f.a - f.b = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1068_106885


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l1068_106819

/-- If a quadratic equation with real coefficients has 5 + 3i as a root, then its constant term is 34 -/
theorem quadratic_constant_term (b c : ℝ) : 
  (∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 5 + 3*Complex.I) →
  c = 34 := by sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l1068_106819


namespace NUMINAMATH_CALUDE_square_floor_tiles_l1068_106803

theorem square_floor_tiles (n : ℕ) : 
  (2 * n - 1 = 37) → n^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l1068_106803


namespace NUMINAMATH_CALUDE_tangent_triangle_angle_theorem_l1068_106827

-- Define the circle
variable (O : Point)

-- Define the triangle
variable (P A B : Point)

-- Define the property that PAB is formed by tangents to circle O
def is_tangent_triangle (O P A B : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem tangent_triangle_angle_theorem 
  (h_tangent : is_tangent_triangle O P A B)
  (h_angle : angle_measure A P B = 50) :
  angle_measure A O B = 65 := by sorry

end NUMINAMATH_CALUDE_tangent_triangle_angle_theorem_l1068_106827


namespace NUMINAMATH_CALUDE_chocolate_milk_consumption_l1068_106832

theorem chocolate_milk_consumption (milk_per_glass : ℝ) (syrup_per_glass : ℝ) 
  (total_milk : ℝ) (total_syrup : ℝ) : 
  milk_per_glass = 6.5 → 
  syrup_per_glass = 1.5 → 
  total_milk = 130 → 
  total_syrup = 60 → 
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses_made := min glasses_from_milk glasses_from_syrup
  let total_consumption := glasses_made * (milk_per_glass + syrup_per_glass)
  total_consumption = 160 := by
  sorry

#check chocolate_milk_consumption

end NUMINAMATH_CALUDE_chocolate_milk_consumption_l1068_106832


namespace NUMINAMATH_CALUDE_symmetry_composition_is_translation_l1068_106836

/-- Central symmetry with respect to a point -/
def central_symmetry {V : Type*} [AddCommGroup V] (center : V) (point : V) : V :=
  2 • center - point

/-- Composition of two central symmetries -/
def compose_symmetries {V : Type*} [AddCommGroup V] (O₁ O₂ : V) (point : V) : V :=
  central_symmetry O₂ (central_symmetry O₁ point)

/-- Translation by a vector -/
def translate {V : Type*} [AddCommGroup V] (v : V) (point : V) : V :=
  point + v

theorem symmetry_composition_is_translation {V : Type*} [AddCommGroup V] (O₁ O₂ : V) :
  ∀ (point : V), compose_symmetries O₁ O₂ point = translate (2 • (O₂ - O₁)) point := by
  sorry

end NUMINAMATH_CALUDE_symmetry_composition_is_translation_l1068_106836

import Mathlib

namespace NUMINAMATH_CALUDE_multiples_of_10_average_l17_1744

theorem multiples_of_10_average : 
  let first := 10
  let last := 600
  let step := 10
  let count := (last - first) / step + 1
  let sum := count * (first + last) / 2
  sum / count = 305 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_10_average_l17_1744


namespace NUMINAMATH_CALUDE_airplane_passengers_l17_1784

theorem airplane_passengers (total : ℕ) (children : ℕ) : 
  total = 80 → children = 20 → ∃ (men women : ℕ), 
    men = women ∧ 
    men + women + children = total ∧ 
    men = 30 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l17_1784


namespace NUMINAMATH_CALUDE_weight_difference_l17_1768

/-- Given the combined weights of Annette and Caitlin, and Caitlin and Sara,
    prove that Annette weighs 8 pounds more than Sara. -/
theorem weight_difference (a c s : ℝ) 
  (h1 : a + c = 95) 
  (h2 : c + s = 87) : 
  a - s = 8 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l17_1768


namespace NUMINAMATH_CALUDE_sum_equals_336_l17_1710

theorem sum_equals_336 : 237 + 45 + 36 + 18 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_336_l17_1710


namespace NUMINAMATH_CALUDE_leo_current_weight_l17_1793

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- Theorem stating that Leo's current weight is 92 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 160) →
  leo_weight = 92 := by
  sorry

end NUMINAMATH_CALUDE_leo_current_weight_l17_1793


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l17_1797

theorem square_sum_given_conditions (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 4)
  (h2 : a + b + c = 17) : 
  a^2 + b^2 + c^2 = 281 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l17_1797


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l17_1737

/-- Converts a base 9 number to base 10 --/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10To9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 9 number --/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase9 n ∧ 
             (base9To10 n) % 7 = 0 ∧
             ∀ (m : ℕ), isThreeDigitBase9 m ∧ (base9To10 m) % 7 = 0 → m ≤ n ∧
             n = 888 := by
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l17_1737


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l17_1798

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b,
    then the x-coordinate of a is -2/3. -/
theorem perpendicular_vectors_x_coord
  (a b : ℝ × ℝ)
  (h1 : a.1 = x ∧ a.2 = x + 1)
  (h2 : b = (1, 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l17_1798


namespace NUMINAMATH_CALUDE_remaining_pills_l17_1726

/-- Calculates the total number of pills left after using supplements for a specified number of days. -/
def pillsLeft (largeBottles smallBottles : ℕ) (largePillCount smallPillCount daysUsed : ℕ) : ℕ :=
  (largeBottles * (largePillCount - daysUsed)) + (smallBottles * (smallPillCount - daysUsed))

/-- Theorem stating that given the specific supplement configuration and usage, 350 pills remain. -/
theorem remaining_pills :
  pillsLeft 3 2 120 30 14 = 350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pills_l17_1726


namespace NUMINAMATH_CALUDE_distance_AB_distance_AB_value_l17_1734

def path_north : ℝ := 30 - 15 + 10
def path_east : ℝ := 80 - 30

theorem distance_AB : ℝ :=
  let north_south_distance := path_north
  let east_west_distance := path_east
  Real.sqrt (north_south_distance ^ 2 + east_west_distance ^ 2)

theorem distance_AB_value : distance_AB = 25 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_distance_AB_value_l17_1734


namespace NUMINAMATH_CALUDE_orange_juice_consumption_l17_1753

theorem orange_juice_consumption (initial_amount : ℚ) (alex_fraction : ℚ) (pat_fraction : ℚ) :
  initial_amount = 3/4 →
  alex_fraction = 1/2 →
  pat_fraction = 1/3 →
  pat_fraction * (initial_amount - alex_fraction * initial_amount) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_consumption_l17_1753


namespace NUMINAMATH_CALUDE_remainder_is_six_l17_1770

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^48 + x^36 + x^24 + x^12 + 1

/-- Theorem stating that the remainder is 6 -/
theorem remainder_is_six : ∃ q : ℂ → ℂ, ∀ x : ℂ, 
  dividend x = (divisor x) * (q x) + 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_six_l17_1770


namespace NUMINAMATH_CALUDE_discount_problem_l17_1715

theorem discount_problem (original_price : ℝ) : 
  original_price > 0 → 
  0.7 * original_price + 0.8 * original_price = 50 → 
  original_price = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l17_1715


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l17_1774

theorem maximize_x_cubed_y_fourth :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 27 →
  x^3 * y^4 ≤ (81/7)^3 * (108/7)^4 :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l17_1774


namespace NUMINAMATH_CALUDE_triangle_problem_l17_1756

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (2 * (Real.cos (A / 2))^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1) →
  (c = 2) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3) →
  -- Conclusions to prove
  (C = π/3) ∧ (a = 2) ∧ (b = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l17_1756


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l17_1749

/-- Given two circles with centers 25 inches apart, where one circle has a radius of 7 inches
    and the other has a radius of 10 inches, the length of their common internal tangent
    is √336 inches. -/
theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 25)
  (h2 : small_radius = 7)
  (h3 : large_radius = 10) :
  Real.sqrt (center_distance ^ 2 - (small_radius + large_radius) ^ 2) = Real.sqrt 336 :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l17_1749


namespace NUMINAMATH_CALUDE_interior_angle_sum_l17_1746

/-- 
Given a convex polygon where the sum of interior angles is 1800°,
prove that the sum of interior angles of a polygon with 3 fewer sides is 1260°.
-/
theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n - 3) - 2) = 1260) := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l17_1746


namespace NUMINAMATH_CALUDE_stone_number_150_l17_1701

/-- Represents the counting pattern for each round -/
def countingPattern : List Nat := [12, 10, 8, 6, 4, 2]

/-- Calculates the sum of a list of natural numbers -/
def sumList (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

/-- Represents the total count in one complete cycle -/
def cycleCount : Nat := sumList countingPattern

/-- Calculates the number of complete cycles before reaching the target count -/
def completeCycles (target : Nat) : Nat :=
  target / cycleCount

/-- Calculates the remaining count after complete cycles -/
def remainingCount (target : Nat) : Nat :=
  target % cycleCount

/-- Finds the original stone number corresponding to the target count -/
def findStoneNumber (target : Nat) : Nat :=
  let remainingCount := remainingCount target
  let rec findInPattern (count : Nat) (pattern : List Nat) : Nat :=
    match pattern with
    | [] => 0  -- Should not happen if the input is valid
    | h :: t =>
      if count <= h then
        12 - (h - count) - (6 - pattern.length) * 2
      else
        findInPattern (count - h) t
  findInPattern remainingCount countingPattern

theorem stone_number_150 :
  findStoneNumber 150 = 4 := by sorry

end NUMINAMATH_CALUDE_stone_number_150_l17_1701


namespace NUMINAMATH_CALUDE_f_properties_l17_1755

def f (a x : ℝ) : ℝ := x * |x - a|

theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y ↔ -1 ≤ a ∧ a ≤ 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1 ↔ 3/2 < a ∧ a < 2) ∧
  (a ≥ 2 →
    (∀ x : ℝ, x ∈ Set.Icc 2 4 → 
      (a > 8 → f a x ∈ Set.Icc (2*a-4) (4*a-16)) ∧
      (4 ≤ a ∧ a < 6 → f a x ∈ Set.Icc (4*a-16) (a^2/4)) ∧
      (6 ≤ a ∧ a ≤ 8 → f a x ∈ Set.Icc (2*a-4) (a^2/4)) ∧
      (2 ≤ a ∧ a < 10/3 → f a x ∈ Set.Icc 0 (16-4*a)) ∧
      (10/3 ≤ a ∧ a < 4 → f a x ∈ Set.Icc 0 (2*a-4)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l17_1755


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l17_1717

/-- Given real numbers a, b, c, and a positive number m satisfying the condition,
    the quadratic equation has a root between 0 and 1. -/
theorem quadratic_root_in_unit_interval (a b c m : ℝ) (hm : m > 0) 
    (h : a / (m + 2) + b / (m + 1) + c / m = 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l17_1717


namespace NUMINAMATH_CALUDE_tan_negative_1140_degrees_l17_1732

theorem tan_negative_1140_degrees : Real.tan (-(1140 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_1140_degrees_l17_1732


namespace NUMINAMATH_CALUDE_a_minus_2ab_plus_b_eq_zero_l17_1778

theorem a_minus_2ab_plus_b_eq_zero 
  (a b : ℝ) 
  (h1 : a + b = 2) 
  (h2 : a * b = 1) : 
  a - 2 * a * b + b = 0 := by
sorry

end NUMINAMATH_CALUDE_a_minus_2ab_plus_b_eq_zero_l17_1778


namespace NUMINAMATH_CALUDE_partially_symmetric_iff_l17_1705

/-- A function is partially symmetric if it satisfies three specific conditions. -/
def PartiallySymmetric (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂)

/-- Theorem: A function is partially symmetric if and only if it satisfies the three conditions. -/
theorem partially_symmetric_iff (f : ℝ → ℝ) :
  PartiallySymmetric f ↔
    (f 0 = 0) ∧
    (∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0) ∧
    (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ abs x₁ < abs x₂ → f x₁ < f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_partially_symmetric_iff_l17_1705


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l17_1782

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℝ).sqrt = k := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l17_1782


namespace NUMINAMATH_CALUDE_batsman_average_l17_1759

/-- Represents a batsman's performance over multiple innings -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman after their latest innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.latestScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 58 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.latestScore = 80)
  (h3 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + 2) :
  calculateAverage b = 58 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l17_1759


namespace NUMINAMATH_CALUDE_square_equality_implies_four_l17_1794

theorem square_equality_implies_four (x : ℝ) : (8 - x)^2 = x^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_implies_four_l17_1794


namespace NUMINAMATH_CALUDE_lost_to_remaining_ratio_l17_1730

def initial_amount : ℕ := 5000
def motorcycle_cost : ℕ := 2800
def final_amount : ℕ := 825

def amount_after_motorcycle : ℕ := initial_amount - motorcycle_cost
def concert_ticket_cost : ℕ := amount_after_motorcycle / 2
def amount_after_concert : ℕ := amount_after_motorcycle - concert_ticket_cost
def amount_lost : ℕ := amount_after_concert - final_amount

theorem lost_to_remaining_ratio :
  (amount_lost : ℚ) / (amount_after_concert : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_lost_to_remaining_ratio_l17_1730


namespace NUMINAMATH_CALUDE_male_cattle_percentage_l17_1766

/-- Represents the farmer's cattle statistics -/
structure CattleStats where
  total_milk : ℕ
  milk_per_cow : ℕ
  male_count : ℕ

/-- Calculates the percentage of male cattle -/
def male_percentage (stats : CattleStats) : ℚ :=
  let female_count := stats.total_milk / stats.milk_per_cow
  let total_cattle := stats.male_count + female_count
  (stats.male_count : ℚ) / (total_cattle : ℚ) * 100

/-- Theorem stating that the percentage of male cattle is 40% -/
theorem male_cattle_percentage (stats : CattleStats) 
  (h1 : stats.total_milk = 150)
  (h2 : stats.milk_per_cow = 2)
  (h3 : stats.male_count = 50) :
  male_percentage stats = 40 := by
  sorry

#eval male_percentage { total_milk := 150, milk_per_cow := 2, male_count := 50 }

end NUMINAMATH_CALUDE_male_cattle_percentage_l17_1766


namespace NUMINAMATH_CALUDE_prob_sum_less_than_one_l17_1765

theorem prob_sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_one_l17_1765


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l17_1754

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Digit
  b : Digit
  c : Digit
  d : Digit

def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.a.val + 100 * n.b.val + 10 * n.c.val + n.d.val

def adjacent (x y : Digit) : Prop :=
  x.val + 1 = y.val ∨ y.val + 1 = x.val

theorem cryptarithm_solution :
  ∃! (n : FourDigitNumber),
    adjacent n.a n.c
    ∧ (n.b.val + 2 = n.d.val ∨ n.d.val + 2 = n.b.val)
    ∧ (∃ (e f g h i j : Digit),
        g.val * 10 + h.val = 19
        ∧ f.val + j.val = 14
        ∧ e.val + i.val = 10
        ∧ n.value = 5240) := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l17_1754


namespace NUMINAMATH_CALUDE_square_area_l17_1775

/-- The area of a square with side length 13 cm is 169 square centimeters. -/
theorem square_area (side_length : ℝ) (h : side_length = 13) : side_length ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l17_1775


namespace NUMINAMATH_CALUDE_rational_solution_exists_l17_1724

theorem rational_solution_exists : ∃ (a b : ℚ), (a ≠ 0) ∧ (a + b ≠ 0) ∧ ((a + b) / a + a / (a + b) = b) := by
  sorry

end NUMINAMATH_CALUDE_rational_solution_exists_l17_1724


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l17_1779

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h_sum : a + b + c + d + e + f = 10) :
  2/a + 3/b + 9/c + 16/d + 25/e + 36/f ≥ (329 + 38 * Real.sqrt 6) / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l17_1779


namespace NUMINAMATH_CALUDE_stating_max_squares_specific_cases_l17_1780

/-- 
Given a rectangular grid of dimensions m × n, this function calculates 
the maximum number of squares that can be cut along the grid lines.
-/
def max_squares (m n : ℕ) : ℕ := sorry

/--
Theorem stating that for specific grid dimensions (8, 11) and (8, 12),
the maximum number of squares that can be cut is 5.
-/
theorem max_squares_specific_cases : 
  (max_squares 8 11 = 5) ∧ (max_squares 8 12 = 5) := by sorry

end NUMINAMATH_CALUDE_stating_max_squares_specific_cases_l17_1780


namespace NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l17_1769

theorem tan_plus_3sin_30_deg :
  Real.tan (30 * π / 180) + 3 * Real.sin (30 * π / 180) = (1 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l17_1769


namespace NUMINAMATH_CALUDE_complex_equation_solution_l17_1772

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l17_1772


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l17_1722

/-- The total surface area of a pyramid with a regular hexagonal base -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  let base_area := (3 * a^2 * Real.sqrt 3) / 2
  let perp_edge_length := a
  let side_triangle_area := a^2 / 2
  let side_triangle_area2 := a^2
  let side_triangle_area3 := (a^2 * Real.sqrt 7) / 4
  base_area + 2 * side_triangle_area + 2 * side_triangle_area2 + 2 * side_triangle_area3 =
    (a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l17_1722


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l17_1706

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem: The profit percent for the given values is 25% -/
theorem retailer_profit_percent :
  profit_percent 225 15 300 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l17_1706


namespace NUMINAMATH_CALUDE_smallest_soldier_arrangement_l17_1783

theorem smallest_soldier_arrangement : ∃ (n : ℕ), n > 0 ∧
  (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 → (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), m % k = k - 1) → m ≥ n) ∧
  n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_smallest_soldier_arrangement_l17_1783


namespace NUMINAMATH_CALUDE_bird_count_l17_1741

/-- The number of birds in a park -/
theorem bird_count (blackbirds_per_tree : ℕ) (num_trees : ℕ) (num_magpies : ℕ) :
  blackbirds_per_tree = 3 →
  num_trees = 7 →
  num_magpies = 13 →
  blackbirds_per_tree * num_trees + num_magpies = 34 := by
sorry


end NUMINAMATH_CALUDE_bird_count_l17_1741


namespace NUMINAMATH_CALUDE_cubic_factorization_l17_1713

theorem cubic_factorization (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l17_1713


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l17_1787

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 3 - m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := m - 1

/-- If point P(3-m, m-1) is in the second quadrant, then m > 3 -/
theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (x_coord m) (y_coord m) → m > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l17_1787


namespace NUMINAMATH_CALUDE_givenPointInSecondQuadrant_l17_1742

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point we want to prove is in the second quadrant -/
def givenPoint : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that the given point is in the second quadrant -/
theorem givenPointInSecondQuadrant : isInSecondQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInSecondQuadrant_l17_1742


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l17_1727

/-- An arithmetic sequence with its first term and sum function -/
structure ArithmeticSequence where
  a₁ : ℤ
  S : ℕ → ℤ

/-- The specific arithmetic sequence from the problem -/
def problemSequence : ArithmeticSequence where
  a₁ := -2012
  S := sorry  -- Definition of S is left as sorry as it's not explicitly given in the conditions

/-- The main theorem to prove -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h : seq.a₁ = -2012)
  (h_sum_diff : seq.S 2012 / 2012 - seq.S 10 / 10 = 2002) :
  seq.S 2017 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l17_1727


namespace NUMINAMATH_CALUDE_phase_shift_cos_l17_1750

theorem phase_shift_cos (b c : ℝ) : 
  let phase_shift := -c / b
  b = 2 ∧ c = π / 2 → phase_shift = -π / 4 := by
sorry

end NUMINAMATH_CALUDE_phase_shift_cos_l17_1750


namespace NUMINAMATH_CALUDE_max_sum_distances_l17_1719

-- Define the points A, B, and O
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the incircle of triangle AOB
def incircle (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 1 + Real.cos θ ∧ P.2 = 4/3 + Real.sin θ

-- Define the distance function
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- State the theorem
theorem max_sum_distances :
  ∀ P : ℝ × ℝ, incircle P →
    dist_squared P A + dist_squared P B + dist_squared P O ≤ 22 ∧
    ∃ P : ℝ × ℝ, incircle P ∧
      dist_squared P A + dist_squared P B + dist_squared P O = 22 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_distances_l17_1719


namespace NUMINAMATH_CALUDE_permutation_problem_l17_1764

-- Define permutation function
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then (n - r + 1).factorial / (n - r).factorial else 0

-- Theorem statement
theorem permutation_problem : 
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * Nat.factorial 0 = 4 :=
by sorry

end NUMINAMATH_CALUDE_permutation_problem_l17_1764


namespace NUMINAMATH_CALUDE_lee_soccer_game_probability_l17_1738

theorem lee_soccer_game_probability (p : ℚ) (h : p = 5/9) :
  1 - p = 4/9 := by sorry

end NUMINAMATH_CALUDE_lee_soccer_game_probability_l17_1738


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l17_1786

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ x ≠ 1 ∧ (x / (x - 3) = (x + 1) / (x - 1)) ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l17_1786


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l17_1763

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def PossibleSlopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def LineEquation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def EllipseEquation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ PossibleSlopes ↔
  ∃ x : ℝ, EllipseEquation x (LineEquation m x) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l17_1763


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l17_1733

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l17_1733


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l17_1745

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * (1 + Real.cos A * Real.cos B * Real.cos C) ∧
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l17_1745


namespace NUMINAMATH_CALUDE_ratio_of_amounts_l17_1736

theorem ratio_of_amounts (total : ℕ) (r_amount : ℕ) (h1 : total = 5000) (h2 : r_amount = 2000) :
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_amounts_l17_1736


namespace NUMINAMATH_CALUDE_system_solution_l17_1700

theorem system_solution : ∃ (x y : ℚ), (4 * x - 3 * y = -13) ∧ (5 * x + 3 * y = -14) ∧ (x = -3) ∧ (y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l17_1700


namespace NUMINAMATH_CALUDE_factoring_expression_l17_1760

theorem factoring_expression (x y : ℝ) : 
  72 * x^4 * y^2 - 180 * x^8 * y^5 = 36 * x^4 * y^2 * (2 - 5 * x^4 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l17_1760


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l17_1790

def monomial : ℤ × ℤ × ℤ → ℤ
| (a, m, n) => a * m * n^3

theorem coefficient_of_monomial :
  ∃ (m n : ℤ), monomial (-5, m, n) = -5 * m * n^3 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l17_1790


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l17_1762

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 + m*x + m - 2
  (f (-2) = 0) →
  (∃ x, x ≠ -2 ∧ f x = 0 ∧ x = 0) ∧
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l17_1762


namespace NUMINAMATH_CALUDE_equation_root_l17_1711

theorem equation_root : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l17_1711


namespace NUMINAMATH_CALUDE_min_odd_integers_l17_1718

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_3 : a + b + c = 36)
  (sum_5 : a + b + c + d + e = 59)
  (sum_6 : a + b + c + d + e + f = 78) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧
    (∀ x ∈ odds, Odd x) ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l17_1718


namespace NUMINAMATH_CALUDE_winning_scores_count_l17_1767

-- Define the number of teams and runners per team
def num_teams : Nat := 3
def runners_per_team : Nat := 3

-- Define the total number of runners
def total_runners : Nat := num_teams * runners_per_team

-- Define the sum of all positions
def total_points : Nat := (total_runners * (total_runners + 1)) / 2

-- Define the maximum possible winning score
def max_winning_score : Nat := total_points / 2

-- Define the minimum possible winning score
def min_winning_score : Nat := 1 + 2 + 3

-- Theorem statement
theorem winning_scores_count :
  (∃ (winning_scores : Finset Nat),
    (∀ s ∈ winning_scores, min_winning_score ≤ s ∧ s ≤ max_winning_score) ∧
    (∀ s ∈ winning_scores, ∃ (a b c : Nat),
      a < b ∧ b < c ∧ c ≤ total_runners ∧ s = a + b + c) ∧
    winning_scores.card = 17) :=
by sorry

end NUMINAMATH_CALUDE_winning_scores_count_l17_1767


namespace NUMINAMATH_CALUDE_triangle_sine_value_l17_1791

theorem triangle_sine_value (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  -- Given conditions
  (C = π / 6) ∧
  (a = 3) ∧
  (c = 4) →
  Real.sin A = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_value_l17_1791


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l17_1743

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l17_1743


namespace NUMINAMATH_CALUDE_textbook_savings_l17_1716

/-- Calculates the savings when buying a textbook from an external bookshop instead of the school bookshop -/
def calculate_savings (school_price : ℚ) (discount_percent : ℚ) : ℚ :=
  school_price * discount_percent / 100

/-- Represents the prices and discounts for the three textbooks -/
structure TextbookPrices where
  math_price : ℚ
  math_discount : ℚ
  science_price : ℚ
  science_discount : ℚ
  literature_price : ℚ
  literature_discount : ℚ

/-- Calculates the total savings for all three textbooks -/
def total_savings (prices : TextbookPrices) : ℚ :=
  calculate_savings prices.math_price prices.math_discount +
  calculate_savings prices.science_price prices.science_discount +
  calculate_savings prices.literature_price prices.literature_discount

/-- Theorem stating that the total savings is $29.25 -/
theorem textbook_savings :
  let prices : TextbookPrices := {
    math_price := 45,
    math_discount := 20,
    science_price := 60,
    science_discount := 25,
    literature_price := 35,
    literature_discount := 15
  }
  total_savings prices = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_textbook_savings_l17_1716


namespace NUMINAMATH_CALUDE_equation_solution_l17_1789

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 89) / 40 ∧ 
     x₂ = (3 - Real.sqrt 89) / 40) ∧ 
    (∀ x y : ℝ, y = 3 * x → 
      (4 * y^2 + y + 5 = 2 * (8 * x^2 + y + 3) ↔ 
       (x = x₁ ∨ x = x₂))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l17_1789


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_l17_1723

def is_quadratic_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i-1)| = (i : ℤ)^2

theorem quadratic_sequence_exists (h k : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = h ∧ a n = k ∧ is_quadratic_sequence a n :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_l17_1723


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l17_1773

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I + 2) :
  z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l17_1773


namespace NUMINAMATH_CALUDE_units_digit_of_17_to_2107_l17_1702

theorem units_digit_of_17_to_2107 :
  (17^2107 : ℕ) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_17_to_2107_l17_1702


namespace NUMINAMATH_CALUDE_simplify_expression_l17_1740

theorem simplify_expression : (1024 : ℝ) ^ (1/5 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l17_1740


namespace NUMINAMATH_CALUDE_angle_tangent_sum_zero_l17_1777

theorem angle_tangent_sum_zero :
  ∃ θ : Real,
    0 < θ ∧ θ < π / 6 ∧
    Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) + Real.tan (5 * θ) = 0 ∧
    Real.tan θ = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_tangent_sum_zero_l17_1777


namespace NUMINAMATH_CALUDE_factorization_problem_l17_1792

theorem factorization_problem (x y : ℝ) : (y + 2*x)^2 - (x + 2*y)^2 = 3*(x + y)*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l17_1792


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l17_1721

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > c →
  b = 3 →
  (a * c * (1 / 3) = 2) →  -- Equivalent to BA · BC = 2 and cos B = 1/3
  a + c = 5 →              -- From the solution, but derivable from given conditions
  (a = 3 ∧ c = 2) ∧ (Real.cos C = 7 / 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l17_1721


namespace NUMINAMATH_CALUDE_unique_positive_solution_l17_1747

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 7 = 5 / (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l17_1747


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l17_1796

theorem multiplication_addition_equality : 26 * 43 + 57 * 26 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l17_1796


namespace NUMINAMATH_CALUDE_composite_shape_area_l17_1751

/-- The total area of a composite shape consisting of three rectangles -/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 68 square units -/
theorem composite_shape_area : composite_area 7 6 3 2 4 5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l17_1751


namespace NUMINAMATH_CALUDE_specific_rhombus_area_l17_1739

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus with the given properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 165,
    diagonal_difference := 10,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 305 / 4 := by sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l17_1739


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l17_1704

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of a line in the form ax + by = c -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y = c

theorem perpendicular_lines_intersection (a b c : ℝ) :
  line_equation a (-2) c 1 (-5) ∧
  line_equation 2 b (-c) 1 (-5) ∧
  perpendicular (a / 2) (-2 / b) →
  c = 13 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l17_1704


namespace NUMINAMATH_CALUDE_frog_hop_probability_l17_1707

/-- Represents the possible positions on a 3x3 grid -/
inductive Position
  | Center
  | Edge
  | Corner

/-- Represents a single hop of the frog -/
def hop (pos : Position) : Position :=
  match pos with
  | Position.Center => Position.Edge
  | Position.Edge => sorry  -- Randomly choose between Center, Edge, or Corner
  | Position.Corner => Position.Corner

/-- Calculates the probability of landing on a corner exactly once in at most four hops -/
def prob_corner_once (hops : Nat) : ℚ :=
  sorry  -- Implement the probability calculation

/-- The main theorem stating the probability of landing on a corner exactly once in at most four hops -/
theorem frog_hop_probability : 
  prob_corner_once 4 = 25 / 32 := by
  sorry


end NUMINAMATH_CALUDE_frog_hop_probability_l17_1707


namespace NUMINAMATH_CALUDE_suit_cost_ratio_l17_1799

theorem suit_cost_ratio (off_rack_cost tailoring_cost total_cost : ℝ) 
  (h1 : off_rack_cost = 300)
  (h2 : tailoring_cost = 200)
  (h3 : total_cost = 1400)
  (h4 : ∃ x : ℝ, total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost)) :
  ∃ x : ℝ, x = 3 ∧ total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost) :=
by sorry

end NUMINAMATH_CALUDE_suit_cost_ratio_l17_1799


namespace NUMINAMATH_CALUDE_pencil_count_l17_1771

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 7 →
  pencils = 42 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l17_1771


namespace NUMINAMATH_CALUDE_least_integer_square_quadruple_l17_1712

theorem least_integer_square_quadruple (x : ℤ) : x^2 = 4*x + 56 → x ≥ -7 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_quadruple_l17_1712


namespace NUMINAMATH_CALUDE_square_sum_equality_l17_1714

theorem square_sum_equality (x y z : ℝ) 
  (h1 : x^2 + 4*y^2 + 16*z^2 = 48) 
  (h2 : x*y + 4*y*z + 2*z*x = 24) : 
  x^2 + y^2 + z^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l17_1714


namespace NUMINAMATH_CALUDE_packing_theorem_l17_1752

/-- Represents the types of boxes that can be packed. -/
inductive BoxType
  | Large
  | Medium
  | Small

/-- Represents the types of packing tape. -/
inductive TapeType
  | A
  | B

/-- Calculates the amount of tape needed for a given box type. -/
def tapeNeeded (b : BoxType) : ℕ :=
  match b with
  | BoxType.Large => 5
  | BoxType.Medium => 3
  | BoxType.Small => 2

/-- Calculates the total tape used for packing a list of boxes. -/
def totalTapeUsed (boxes : List (BoxType × ℕ)) : ℕ :=
  boxes.foldl (fun acc (b, n) => acc + n * tapeNeeded b) 0

/-- Represents the packing scenario for Debbie and Mike. -/
structure PackingScenario where
  debbieBoxes : List (BoxType × ℕ)
  mikeBoxes : List (BoxType × ℕ)
  tapeARollLength : ℕ
  tapeBRollLength : ℕ

/-- Calculates the remaining tape for Debbie and Mike. -/
def remainingTape (scenario : PackingScenario) : TapeType → ℕ
  | TapeType.A => scenario.tapeARollLength - totalTapeUsed scenario.debbieBoxes
  | TapeType.B => 
      let usedTapeB := totalTapeUsed scenario.mikeBoxes
      scenario.tapeBRollLength - (usedTapeB % scenario.tapeBRollLength)

/-- The main theorem stating the remaining tape for Debbie and Mike. -/
theorem packing_theorem (scenario : PackingScenario) 
    (h1 : scenario.debbieBoxes = [(BoxType.Large, 2), (BoxType.Medium, 8), (BoxType.Small, 5)])
    (h2 : scenario.mikeBoxes = [(BoxType.Large, 3), (BoxType.Medium, 6), (BoxType.Small, 10)])
    (h3 : scenario.tapeARollLength = 50)
    (h4 : scenario.tapeBRollLength = 40) :
    remainingTape scenario TapeType.A = 6 ∧ remainingTape scenario TapeType.B = 27 := by
  sorry

end NUMINAMATH_CALUDE_packing_theorem_l17_1752


namespace NUMINAMATH_CALUDE_unique_base_perfect_square_l17_1709

theorem unique_base_perfect_square : 
  ∃! n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_perfect_square_l17_1709


namespace NUMINAMATH_CALUDE_simplify_expression_l17_1725

theorem simplify_expression : (1 / ((-8^4)^2)) * (-8)^11 = -512 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l17_1725


namespace NUMINAMATH_CALUDE_probability_exactly_one_instrument_l17_1748

/-- Given a group of people, calculate the probability that a randomly selected person plays exactly one instrument. -/
theorem probability_exactly_one_instrument 
  (total_people : ℕ) 
  (at_least_one_fraction : ℚ) 
  (two_or_more : ℕ) 
  (h1 : total_people = 800)
  (h2 : at_least_one_fraction = 1 / 5)
  (h3 : two_or_more = 128) :
  (total_people : ℚ) * at_least_one_fraction - two_or_more / total_people = 1 / 25 := by
sorry

end NUMINAMATH_CALUDE_probability_exactly_one_instrument_l17_1748


namespace NUMINAMATH_CALUDE_divisibility_implies_seven_divides_l17_1731

theorem divisibility_implies_seven_divides (n : ℕ) : 
  n ≥ 2 → (n ∣ 3^n + 4^n) → (7 ∣ n) := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_seven_divides_l17_1731


namespace NUMINAMATH_CALUDE_boys_participation_fraction_l17_1788

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of participating students
def participating_students : ℕ := 550

-- Define the number of participating girls
def participating_girls : ℕ := 150

-- Define the fraction of girls who participated
def girls_participation_fraction : ℚ := 3/4

-- Theorem to prove
theorem boys_participation_fraction :
  let total_girls : ℕ := participating_girls * 4 / 3
  let total_boys : ℕ := total_students - total_girls
  let participating_boys : ℕ := participating_students - participating_girls
  (participating_boys : ℚ) / total_boys = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_boys_participation_fraction_l17_1788


namespace NUMINAMATH_CALUDE_max_value_a_l17_1720

theorem max_value_a (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a : ℝ, a ≤ 1/x + 9/y) → (∃ a : ℝ, a = 16 ∧ ∀ b : ℝ, b ≤ 1/x + 9/y → b ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l17_1720


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l17_1708

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l17_1708


namespace NUMINAMATH_CALUDE_arccos_sum_eq_pi_half_l17_1785

theorem arccos_sum_eq_pi_half (x : ℝ) :
  Real.arccos (3 * x) + Real.arccos x = π / 2 →
  x = 1 / Real.sqrt 10 ∨ x = -1 / Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_arccos_sum_eq_pi_half_l17_1785


namespace NUMINAMATH_CALUDE_trig_identity_l17_1703

theorem trig_identity : 
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l17_1703


namespace NUMINAMATH_CALUDE_not_multiple_of_121_l17_1729

theorem not_multiple_of_121 (n : ℤ) : ¬(121 ∣ (n^2 + 2*n + 12)) := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_121_l17_1729


namespace NUMINAMATH_CALUDE_distance_to_school_l17_1728

/-- Represents the travel conditions to Jeremy's school -/
structure TravelConditions where
  normal_time : ℝ  -- Normal travel time in hours
  fast_time : ℝ    -- Travel time when speed is increased in hours
  slow_time : ℝ    -- Travel time when speed is decreased in hours
  speed_increase : ℝ  -- Speed increase in mph
  speed_decrease : ℝ  -- Speed decrease in mph

/-- Calculates the distance to Jeremy's school given the travel conditions -/
def calculateDistance (tc : TravelConditions) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the distance to Jeremy's school is 15 miles -/
theorem distance_to_school :
  let tc : TravelConditions := {
    normal_time := 1/2,  -- 30 minutes in hours
    fast_time := 3/10,   -- 18 minutes in hours
    slow_time := 2/3,    -- 40 minutes in hours
    speed_increase := 15,
    speed_decrease := 10
  }
  calculateDistance tc = 15 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_school_l17_1728


namespace NUMINAMATH_CALUDE_f_range_l17_1761

/-- The function f(x) = (x^2-1)(x^2-12x+35) -/
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

/-- The graph of f(x) is symmetric about the line x=3 -/
axiom f_symmetry (x : ℝ) : f (6 - x) = f x

theorem f_range : Set.range f = Set.Ici (-36) := by sorry

end NUMINAMATH_CALUDE_f_range_l17_1761


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_for_48_l17_1757

/-- Represents a hexagonal grid structure --/
structure HexagonalGrid where
  toothpicks : ℕ
  small_hexagons : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles --/
def min_toothpicks_to_remove (grid : HexagonalGrid) : ℕ :=
  sorry

/-- Theorem stating the minimum number of toothpicks to remove for a specific grid --/
theorem min_toothpicks_removal_for_48 :
  ∀ (grid : HexagonalGrid),
    grid.toothpicks = 48 →
    min_toothpicks_to_remove grid = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_for_48_l17_1757


namespace NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l17_1758

theorem quadratic_roots_when_positive_discriminant
  (a b c : ℝ) (h_a : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l17_1758


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l17_1735

def B : Set ℕ := {n : ℕ | ∃ y : ℕ+, n = 4 * y + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l17_1735


namespace NUMINAMATH_CALUDE_number_of_grades_l17_1781

theorem number_of_grades (students_per_grade : ℕ) (total_students : ℕ) : 
  students_per_grade = 75 → total_students = 22800 → total_students / students_per_grade = 304 := by
  sorry

end NUMINAMATH_CALUDE_number_of_grades_l17_1781


namespace NUMINAMATH_CALUDE_simplify_rational_expression_l17_1795

theorem simplify_rational_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / (x^2 - 2*x)) = 1 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_rational_expression_l17_1795


namespace NUMINAMATH_CALUDE_E_equals_F_l17_1776

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_equals_F : E = F := by
  sorry

end NUMINAMATH_CALUDE_E_equals_F_l17_1776

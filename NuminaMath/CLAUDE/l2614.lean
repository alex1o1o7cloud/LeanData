import Mathlib

namespace NUMINAMATH_CALUDE_basket_problem_l2614_261483

theorem basket_problem (total : ℕ) (apples : ℕ) (oranges : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : apples = 10)
  (h3 : oranges = 8)
  (h4 : both = 5) :
  total - (apples + oranges - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_basket_problem_l2614_261483


namespace NUMINAMATH_CALUDE_population_growth_l2614_261465

/-- The initial population of the town -/
def initial_population : ℝ := 1000

/-- The growth rate in the first year -/
def first_year_growth : ℝ := 0.10

/-- The growth rate in the second year -/
def second_year_growth : ℝ := 0.20

/-- The final population after two years -/
def final_population : ℝ := 1320

/-- Theorem stating the relationship between initial and final population -/
theorem population_growth :
  initial_population * (1 + first_year_growth) * (1 + second_year_growth) = final_population := by
  sorry

#check population_growth

end NUMINAMATH_CALUDE_population_growth_l2614_261465


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2614_261491

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem rhombus_longer_diagonal (r : Rhombus) 
  (h1 : r.diagonal1 = 12)
  (h2 : r.area = 120) :
  r.diagonal2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2614_261491


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l2614_261416

theorem consecutive_integers_sum_of_powers (n : ℤ) : 
  ((n - 1)^2 + n^2 + (n + 1)^2 = 2450) → 
  ((n - 1)^5 + n^5 + (n + 1)^5 = 52070424) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_powers_l2614_261416


namespace NUMINAMATH_CALUDE_parabola_x_axis_intersection_l2614_261432

/-- The parabola defined by y = x^2 - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that (1, 0) is the only intersection point of the parabola and the x-axis -/
theorem parabola_x_axis_intersection :
  ∃! x : ℝ, parabola x = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersection_l2614_261432


namespace NUMINAMATH_CALUDE_optimal_price_l2614_261415

/-- Represents the daily sales volume as a function of price -/
def sales (x : ℝ) : ℝ := 400 - 20 * x

/-- Represents the daily profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 8) * sales x

theorem optimal_price :
  ∃ (x : ℝ), 8 ≤ x ∧ x ≤ 15 ∧ profit x = 640 :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_l2614_261415


namespace NUMINAMATH_CALUDE_candy_ratio_problem_l2614_261437

/-- Proof of candy ratio problem -/
theorem candy_ratio_problem (chocolate_bars : ℕ) (mm_multiplier : ℕ) (candies_per_basket : ℕ) (num_baskets : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : mm_multiplier = 7)
  (h3 : candies_per_basket = 10)
  (h4 : num_baskets = 25) :
  (num_baskets * candies_per_basket - chocolate_bars - mm_multiplier * chocolate_bars) / (mm_multiplier * chocolate_bars) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_problem_l2614_261437


namespace NUMINAMATH_CALUDE_cos_2x_plus_2y_l2614_261436

theorem cos_2x_plus_2y (x y : ℝ) (h : Real.cos x * Real.cos y - Real.sin x * Real.sin y = 1/4) :
  Real.cos (2*x + 2*y) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_plus_2y_l2614_261436


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2614_261423

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 100) - (Real.sqrt 288 / Real.sqrt 72) = 2 * Real.sqrt 2 - 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2614_261423


namespace NUMINAMATH_CALUDE_xy_sum_values_l2614_261480

theorem xy_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt : x < 25) (hy_lt : y < 25) 
  (h_eq : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end NUMINAMATH_CALUDE_xy_sum_values_l2614_261480


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l2614_261498

theorem greatest_k_for_inequality : 
  ∃ (k : ℤ), k = 5 ∧ 
  (∀ (j : ℤ), j > k → 
    ∃ (n : ℕ), n ≥ 2 ∧ ⌊n / Real.sqrt 3⌋ + 1 ≤ n^2 / Real.sqrt (3 * n^2 - j)) ∧
  (∀ (n : ℕ), n ≥ 2 → ⌊n / Real.sqrt 3⌋ + 1 > n^2 / Real.sqrt (3 * n^2 - k)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l2614_261498


namespace NUMINAMATH_CALUDE_seven_points_triangle_l2614_261414

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points --/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of seven points on a plane --/
def SevenPoints : Type := Fin 7 → Point

theorem seven_points_triangle (points : SevenPoints) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (angle (points i) (points j) (points k) > 2 * π / 3 ∨
     angle (points j) (points k) (points i) > 2 * π / 3 ∨
     angle (points k) (points i) (points j) > 2 * π / 3) :=
  sorry

end NUMINAMATH_CALUDE_seven_points_triangle_l2614_261414


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2614_261413

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2614_261413


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_notation_l2614_261459

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient|
  abs_coeff_lt_ten : |coefficient| < 10

/-- The given number in centimeters -/
def given_number : ℝ := 0.0000021

/-- The scientific notation representation of the given number -/
def scientific_notation : ScientificNotation := {
  coefficient := 2.1
  exponent := -6
  one_le_abs_coeff := sorry
  abs_coeff_lt_ten := sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_notation : 
  given_number = scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_notation_l2614_261459


namespace NUMINAMATH_CALUDE_intersection_M_N_l2614_261490

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2614_261490


namespace NUMINAMATH_CALUDE_dot_product_AB_AC_l2614_261453

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (3, 4)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_AB_AC : dot_product vector_AB vector_AC = -2 := by sorry

end NUMINAMATH_CALUDE_dot_product_AB_AC_l2614_261453


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2614_261407

/-- The function f(x) = sin(2x) - a*cos(x) is monotonically increasing on [0, π] iff a ≥ 2 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 Real.pi, MonotoneOn (fun x => Real.sin (2 * x) - a * Real.cos x) (Set.Icc 0 Real.pi)) ↔ 
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2614_261407


namespace NUMINAMATH_CALUDE_dog_age_64_human_years_l2614_261477

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAge (humanYears : ℕ) : ℕ :=
  if humanYears ≤ 15 then 1
  else if humanYears ≤ 24 then 2
  else 2 + (humanYears - 24) / 5

/-- Theorem stating that a dog that has lived 64 human years is 10 years old in dog years -/
theorem dog_age_64_human_years : dogAge 64 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_64_human_years_l2614_261477


namespace NUMINAMATH_CALUDE_total_rent_is_105_l2614_261445

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  share : ℚ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def calculate_total_rent (a b c : RentShare) : ℚ :=
  let total_oxen_months : ℕ := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let rent_per_oxen_month : ℚ := c.share / (c.oxen * c.months)
  rent_per_oxen_month * total_oxen_months

/-- Theorem: The total rent of the pasture is 105.00 given the specified conditions -/
theorem total_rent_is_105 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.share = 26.999999999999996) :
  calculate_total_rent a b c = 105 :=
by sorry

end NUMINAMATH_CALUDE_total_rent_is_105_l2614_261445


namespace NUMINAMATH_CALUDE_divisible_by_24_l2614_261467

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n * (n + 2) * (5 * n - 1) * (5 * n + 1) = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l2614_261467


namespace NUMINAMATH_CALUDE_no_valid_n_l2614_261411

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l2614_261411


namespace NUMINAMATH_CALUDE_intersection_M_N_l2614_261487

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2614_261487


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l2614_261403

theorem square_sum_equals_three (x y z : ℝ) 
  (h1 : x - y - z = 3) 
  (h2 : y * z - x * y - x * z = 3) : 
  x^2 + y^2 + z^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l2614_261403


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l2614_261499

/-- A function that counts the number of positive integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ) : Prop := count_divisors n = 12

/-- Theorem stating that 60 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors : 
  (has_12_divisors 60) ∧ (∀ m : ℕ, 0 < m ∧ m < 60 → ¬(has_12_divisors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l2614_261499


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l2614_261474

theorem olivia_remaining_money (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by
sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l2614_261474


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l2614_261429

theorem min_value_quadratic_sum (a b c d : ℝ) (h : a * d + b * c = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = a^2 + b^2 + c^2 + d^2 + (a + c)^2 + (b - d)^2 → u ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l2614_261429


namespace NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l2614_261443

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16) 
  (h2 : failed_chinese = 0.07) 
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l2614_261443


namespace NUMINAMATH_CALUDE_smallest_angle_60_implies_n_3_or_4_l2614_261494

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space determined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A configuration of n points in 3D space -/
def Configuration (n : ℕ) := Fin n → Point3D

/-- The smallest angle formed by any pair of lines in a configuration -/
def smallestAngle (config : Configuration n) : ℝ := sorry

theorem smallest_angle_60_implies_n_3_or_4 (n : ℕ) (h1 : n > 2) 
  (config : Configuration n) (h2 : smallestAngle config = 60) :
  n = 3 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_60_implies_n_3_or_4_l2614_261494


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l2614_261420

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l2614_261420


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l2614_261441

theorem sum_of_first_and_third (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B / C = 5 / 8 →
  B = 30 →
  A + C = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l2614_261441


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2614_261469

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2614_261469


namespace NUMINAMATH_CALUDE_expression_arrangements_l2614_261404

/-- Given three distinct real numbers, there are 96 possible ways to arrange
    the eight expressions ±x ±y ±z in increasing order. -/
theorem expression_arrangements (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (Set.ncard {l : List ℝ | 
    l.length = 8 ∧ 
    l.Nodup ∧
    (∀ a ∈ l, ∃ (s₁ s₂ s₃ : Bool), a = (if s₁ then x else -x) + (if s₂ then y else -y) + (if s₃ then z else -z)) ∧
    l.Sorted (· < ·)}) = 96 :=
by sorry

end NUMINAMATH_CALUDE_expression_arrangements_l2614_261404


namespace NUMINAMATH_CALUDE_teacher_instruction_l2614_261496

theorem teacher_instruction (x : ℝ) : ((x - 2) * 3 + 3) * 3 = 63 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_teacher_instruction_l2614_261496


namespace NUMINAMATH_CALUDE_new_person_weight_l2614_261458

/-- Given a group of people where:
  * There are initially 4 persons
  * One person weighing 70 kg is replaced by a new person
  * The average weight increases by 3 kg after the replacement
  * The total combined weight of all five people after the change is 390 kg
  Prove that the weight of the new person is 58 kg -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℕ) 
  (avg_increase : ℕ) (total_weight : ℕ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ),
    new_weight = 58 ∧
    (total_weight - new_weight + replaced_weight) / initial_count = 
    (total_weight - new_weight) / initial_count + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2614_261458


namespace NUMINAMATH_CALUDE_initial_average_production_is_50_l2614_261409

/-- Calculates the initial average daily production given the number of past days,
    today's production, and the new average including today. -/
def initialAverageProduction (n : ℕ) (todayProduction : ℕ) (newAverage : ℚ) : ℚ :=
  (newAverage * (n + 1) - todayProduction) / n

theorem initial_average_production_is_50 :
  initialAverageProduction 10 105 55 = 50 := by sorry

end NUMINAMATH_CALUDE_initial_average_production_is_50_l2614_261409


namespace NUMINAMATH_CALUDE_sum_maximized_at_14_l2614_261444

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (83 - 3 * n) / 2

/-- The theorem stating that the sum is maximized when n = 14 -/
theorem sum_maximized_at_14 :
  ∀ k : ℕ, k ≠ 0 → S 14 ≥ S k :=
sorry

end NUMINAMATH_CALUDE_sum_maximized_at_14_l2614_261444


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2614_261430

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt 3 * x) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6 ∧ y = 0) →
  (∃ x y : ℝ, y^2 = 2*x ∧ x = -6) →
  (∀ x y : ℝ, x^2 / 9 - y^2 / 27 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2614_261430


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2614_261449

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nonagon (nine-sided polygon) has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2614_261449


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l2614_261497

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Finds the smallest number greater than 10 that is a palindrome in both base 2 and base 3 -/
def smallestDualBasePalindrome : ℕ := sorry

theorem smallest_dual_base_palindrome_is_585 :
  smallestDualBasePalindrome = 585 := by sorry

theorem dual_base_palindrome_properties (n : ℕ) :
  n = smallestDualBasePalindrome →
  n > 10 ∧ isPalindrome n 2 ∧ isPalindrome n 3 := by sorry

theorem no_smaller_dual_base_palindrome (n : ℕ) :
  10 < n ∧ n < smallestDualBasePalindrome →
  ¬(isPalindrome n 2 ∧ isPalindrome n 3) := by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l2614_261497


namespace NUMINAMATH_CALUDE_equal_cupcake_distribution_l2614_261472

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_equal_cupcake_distribution_l2614_261472


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2614_261433

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (max_val : ℝ), max_val = 15 ∧ ∀ (x' y' : ℝ), 2 * x'^2 - 6 * x' + y'^2 = 0 →
    x'^2 + y'^2 + 2 * x' ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2614_261433


namespace NUMINAMATH_CALUDE_percentage_equality_l2614_261438

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2614_261438


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l2614_261412

theorem perfect_square_fraction (n : ℤ) : 
  n > 2020 → 
  (∃ m : ℤ, (n - 2020) / (2120 - n) = m^2) → 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l2614_261412


namespace NUMINAMATH_CALUDE_sequence_value_l2614_261473

theorem sequence_value (a : ℕ → ℝ) (h : ∀ n, (3 - a (n + 1)) * (6 + a n) = 18) (h0 : a 0 ≠ 3) :
  ∀ n, a n = 2^(n + 2) - n - 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_value_l2614_261473


namespace NUMINAMATH_CALUDE_friends_money_sharing_l2614_261492

theorem friends_money_sharing (A : ℝ) (h_pos : A > 0) :
  let jorge_total := 5 * A
  let jose_total := 4 * A
  let janio_total := 3 * A
  let joao_received := 3 * A
  let group_total := jorge_total + jose_total + janio_total
  (joao_received / group_total) = (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_friends_money_sharing_l2614_261492


namespace NUMINAMATH_CALUDE_temple_shop_charge_l2614_261450

/-- The charge per object at the temple shop -/
def charge_per_object : ℕ → ℕ → ℕ → ℕ → ℚ
  | num_people, shoes_per_person, socks_per_person, mobiles_per_person =>
    let total_objects := num_people * (shoes_per_person + socks_per_person + mobiles_per_person)
    let total_cost := 165
    total_cost / total_objects

theorem temple_shop_charge :
  charge_per_object 3 2 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_temple_shop_charge_l2614_261450


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2614_261435

/-- Given a square with side length a and a circle circumscribed around it,
    the area of a square inscribed in one of the resulting segments is a²/25 -/
theorem inscribed_square_area (a : ℝ) (a_pos : 0 < a) :
  ∃ (x : ℝ), x > 0 ∧ x^2 = a^2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2614_261435


namespace NUMINAMATH_CALUDE_selection_schemes_correct_l2614_261454

/-- The number of ways to select 4 students from 4 boys and 2 girls, with at least 1 girl in the group -/
def selection_schemes (num_boys num_girls group_size : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) group_size - Nat.choose num_boys group_size

theorem selection_schemes_correct :
  selection_schemes 4 2 4 = 14 := by
  sorry

#eval selection_schemes 4 2 4

end NUMINAMATH_CALUDE_selection_schemes_correct_l2614_261454


namespace NUMINAMATH_CALUDE_min_distance_intersection_l2614_261451

/-- The minimum distance between intersection points --/
theorem min_distance_intersection (m : ℝ) : 
  let f (x : ℝ) := |x - (x + Real.exp x + 3) / 2|
  ∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_intersection_l2614_261451


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2614_261408

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2614_261408


namespace NUMINAMATH_CALUDE_apple_cider_volume_l2614_261440

/-- The volume of apple cider in a cylindrical pot -/
theorem apple_cider_volume (h : Real) (d : Real) (fill_ratio : Real) (cider_ratio : Real) :
  h = 9 →
  d = 4 →
  fill_ratio = 2/3 →
  cider_ratio = 2/7 →
  (fill_ratio * h * π * (d/2)^2) * cider_ratio = 48*π/7 :=
by sorry

end NUMINAMATH_CALUDE_apple_cider_volume_l2614_261440


namespace NUMINAMATH_CALUDE_simplify_expression_l2614_261486

theorem simplify_expression : (324 : ℝ) ^ (1/4) * (98 : ℝ) ^ (1/2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2614_261486


namespace NUMINAMATH_CALUDE_die_probability_l2614_261495

/-- A fair 8-sided die -/
def Die : Finset ℕ := Finset.range 8

/-- Perfect squares from 1 to 8 -/
def PerfectSquares : Finset ℕ := {1, 4}

/-- Even numbers from 1 to 8 -/
def EvenNumbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling a number that is either a perfect square or an even number -/
theorem die_probability : 
  (Finset.card (PerfectSquares ∪ EvenNumbers) : ℚ) / Finset.card Die = 5 / 8 :=
sorry

end NUMINAMATH_CALUDE_die_probability_l2614_261495


namespace NUMINAMATH_CALUDE_bees_flew_in_l2614_261428

theorem bees_flew_in (initial_bees final_bees : ℕ) (h : initial_bees ≤ final_bees) :
  final_bees - initial_bees = final_bees - initial_bees :=
by sorry

end NUMINAMATH_CALUDE_bees_flew_in_l2614_261428


namespace NUMINAMATH_CALUDE_max_value_of_z_l2614_261431

-- Define the variables and the objective function
variables (x y : ℝ)
def z : ℝ → ℝ → ℝ := λ x y => 2 * x + y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := x + 2 * y ≤ 2
def constraint2 (x y : ℝ) : Prop := x + y ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 x →
  z x y ≤ 11 ∧ ∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 x₀ ∧ z x₀ y₀ = 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2614_261431


namespace NUMINAMATH_CALUDE_factor_expression_l2614_261488

theorem factor_expression (x : ℝ) : 75 * x + 50 = 25 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2614_261488


namespace NUMINAMATH_CALUDE_profit_percentage_cricket_bat_l2614_261405

/-- The profit percentage calculation for a cricket bat sale -/
theorem profit_percentage_cricket_bat (selling_price profit : ℝ)
  (h1 : selling_price = 850)
  (h2 : profit = 230) :
  ∃ (percentage : ℝ), abs (percentage - 37.10) < 0.01 ∧
  percentage = (profit / (selling_price - profit)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_cricket_bat_l2614_261405


namespace NUMINAMATH_CALUDE_alices_number_l2614_261489

theorem alices_number (n : ℕ) 
  (h1 : n % 243 = 0)
  (h2 : n % 36 = 0)
  (h3 : 1000 < n ∧ n < 3000) :
  n = 1944 ∨ n = 2916 := by
sorry

end NUMINAMATH_CALUDE_alices_number_l2614_261489


namespace NUMINAMATH_CALUDE_simplify_fraction_l2614_261476

theorem simplify_fraction : (144 : ℚ) / 1008 = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2614_261476


namespace NUMINAMATH_CALUDE_monday_polygons_tuesday_segments_wednesday_polygons_l2614_261481

/-- Represents the types of polygons Miky can draw -/
inductive Polygon
| Square
| Pentagon
| Hexagon

/-- Number of sides for each polygon type -/
def sides (p : Polygon) : Nat :=
  match p with
  | .Square => 4
  | .Pentagon => 5
  | .Hexagon => 6

/-- Number of diagonals for each polygon type -/
def diagonals (p : Polygon) : Nat :=
  match p with
  | .Square => 2
  | .Pentagon => 5
  | .Hexagon => 9

/-- Total number of line segments (sides + diagonals) for each polygon type -/
def totalSegments (p : Polygon) : Nat :=
  sides p + diagonals p

theorem monday_polygons :
  ∃ p : Polygon, sides p = diagonals p ∧ p = Polygon.Pentagon :=
sorry

theorem tuesday_segments (n : Nat) (h : n * sides Polygon.Hexagon = 18) :
  n * diagonals Polygon.Hexagon = 27 :=
sorry

theorem wednesday_polygons (n : Nat) (h : n * totalSegments Polygon.Pentagon = 70) :
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_monday_polygons_tuesday_segments_wednesday_polygons_l2614_261481


namespace NUMINAMATH_CALUDE_sum_of_divisors_l2614_261461

def isPrime (n : ℕ) : Prop := sorry

def numDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors (p q r : ℕ) (hp : isPrime p) (hq : isPrime q) (hr : isPrime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  let a := p^4
  let b := q * r
  let k := a^5
  let m := b^2
  numDivisors k + numDivisors m = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_l2614_261461


namespace NUMINAMATH_CALUDE_total_amount_paid_l2614_261446

/-- Calculates the discounted price for a fruit given its weight, price per kg, and discount percentage. -/
def discountedPrice (weight : Float) (pricePerKg : Float) (discountPercent : Float) : Float :=
  weight * pricePerKg * (1 - discountPercent / 100)

/-- Represents the shopping trip and calculates the total amount paid. -/
def shoppingTrip : Float :=
  discountedPrice 8 70 10 +    -- Grapes
  discountedPrice 11 55 0 +    -- Mangoes
  discountedPrice 5 45 20 +    -- Oranges
  discountedPrice 3 90 5 +     -- Apples
  discountedPrice 4.5 120 0    -- Cherries

/-- Theorem stating that the total amount paid is $2085.50 -/
theorem total_amount_paid : shoppingTrip = 2085.50 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l2614_261446


namespace NUMINAMATH_CALUDE_composition_of_convex_increasing_and_convex_is_convex_l2614_261493

def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≤ y → f x ≤ f y

theorem composition_of_convex_increasing_and_convex_is_convex
  (f g : ℝ → ℝ) (hf : IsConvex f) (hg : IsConvex g) (hf_inc : IsIncreasing f) :
  IsConvex (f ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_convex_increasing_and_convex_is_convex_l2614_261493


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l2614_261485

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ (p : ℕ),
    isFourDigit p ∧
    p % 63 = 0 ∧
    (reverseDigits p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      isFourDigit q ∧
      q % 63 = 0 ∧
      (reverseDigits q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 9779 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l2614_261485


namespace NUMINAMATH_CALUDE_factorization_of_75x_plus_45_l2614_261426

theorem factorization_of_75x_plus_45 (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_75x_plus_45_l2614_261426


namespace NUMINAMATH_CALUDE_square_and_circle_l2614_261460

theorem square_and_circle (square_area : ℝ) (side_length : ℝ) (circle_radius : ℝ) : 
  square_area = 1 →
  side_length ^ 2 = square_area →
  circle_radius * 2 = side_length →
  side_length = 1 ∧ circle_radius = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_square_and_circle_l2614_261460


namespace NUMINAMATH_CALUDE_rectangle_area_is_216_l2614_261439

/-- Represents a rectangle with given properties -/
structure Rectangle where
  length : ℝ
  breadth : ℝ
  perimeterToBreadthRatio : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: A rectangle with length 18 and perimeter to breadth ratio of 5 has an area of 216 -/
theorem rectangle_area_is_216 (r : Rectangle) 
    (h1 : r.length = 18)
    (h2 : r.perimeterToBreadthRatio = 5)
    (h3 : perimeter r / r.breadth = r.perimeterToBreadthRatio) : 
  area r = 216 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_is_216_l2614_261439


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2614_261482

theorem sandy_shopping_money (total : ℝ) (spent_percentage : ℝ) (left : ℝ) : 
  total = 320 →
  spent_percentage = 30 →
  left = total * (1 - spent_percentage / 100) →
  left = 224 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2614_261482


namespace NUMINAMATH_CALUDE_complement_of_intersection_AB_l2614_261471

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem complement_of_intersection_AB : 
  (A ∩ B)ᶜ = {x : ℝ | x < 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_AB_l2614_261471


namespace NUMINAMATH_CALUDE_days_worked_by_c_l2614_261468

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The daily wage of person c in rupees -/
def wage_c : ℕ := 115

/-- The total earnings of all three persons in rupees -/
def total_earnings : ℕ := 1702

/-- The ratio of daily wages for persons a, b, and c -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- Theorem stating that person c worked for 4 days -/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_by_c_l2614_261468


namespace NUMINAMATH_CALUDE_car_travel_time_ratio_l2614_261422

theorem car_travel_time_ratio : 
  let distance : ℝ := 540
  let original_time : ℝ := 8
  let new_speed : ℝ := 45
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 1.5 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_ratio_l2614_261422


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2614_261455

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2614_261455


namespace NUMINAMATH_CALUDE_exists_polygon_with_equal_area_division_l2614_261466

/-- A polygon in the plane --/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : ∀ (p : ℝ × ℝ), p ∈ vertices → ∃ (q : ℝ × ℝ), q ∈ vertices ∧ q ≠ p

/-- A point is on the boundary of a polygon --/
def OnBoundary (p : ℝ × ℝ) (poly : Polygon) : Prop :=
  p ∈ poly.vertices

/-- A line divides a polygon into two parts --/
def DividesPolygon (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), A ∪ B = poly.vertices ∧ A ∩ B ⊆ l

/-- The area of a set of points in the plane --/
def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- A line divides a polygon into two equal parts --/
def DividesEquallyByArea (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), 
    DividesPolygon l poly ∧
    Area A = Area B

/-- Main theorem: There exists a polygon and a point on its boundary such that 
    any line passing through this point divides the area of the polygon into two equal parts --/
theorem exists_polygon_with_equal_area_division :
  ∃ (poly : Polygon) (p : ℝ × ℝ), 
    OnBoundary p poly ∧
    ∀ (l : Set (ℝ × ℝ)), p ∈ l → DividesEquallyByArea l poly := by
  sorry

end NUMINAMATH_CALUDE_exists_polygon_with_equal_area_division_l2614_261466


namespace NUMINAMATH_CALUDE_quarterback_passes_l2614_261484

theorem quarterback_passes (total : ℕ) (left : ℕ) : 
  total = 50 → 
  left + 2 * left + (left + 2) = total → 
  left = 12 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l2614_261484


namespace NUMINAMATH_CALUDE_diana_wins_probability_l2614_261401

def standard_die := Finset.range 6

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product standard_die).filter (fun (d, a) => d > a)

theorem diana_wins_probability :
  (favorable_outcomes.card : ℚ) / (standard_die.card * standard_die.card) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l2614_261401


namespace NUMINAMATH_CALUDE_max_arithmetic_mean_of_special_pairs_l2614_261457

theorem max_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a > b ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c > d ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_max_arithmetic_mean_of_special_pairs_l2614_261457


namespace NUMINAMATH_CALUDE_expand_product_l2614_261470

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 5) = 2*x^2 + 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2614_261470


namespace NUMINAMATH_CALUDE_personal_planner_cost_proof_l2614_261464

/-- The cost of a spiral notebook -/
def spiral_notebook_cost : ℝ := 15

/-- The number of spiral notebooks -/
def num_spiral_notebooks : ℕ := 4

/-- The number of personal planners -/
def num_personal_planners : ℕ := 8

/-- The discount rate -/
def discount_rate : ℝ := 0.2

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 112

/-- The cost of a personal planner -/
def personal_planner_cost : ℝ := 10

theorem personal_planner_cost_proof :
  let total_cost := spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * num_personal_planners
  total_cost * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end NUMINAMATH_CALUDE_personal_planner_cost_proof_l2614_261464


namespace NUMINAMATH_CALUDE_sum_of_first_100_factorials_mod_100_l2614_261421

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_of_first_100_factorials_mod_100 :
  sum_of_factorials 100 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_100_factorials_mod_100_l2614_261421


namespace NUMINAMATH_CALUDE_mia_excess_over_double_darwin_l2614_261447

def darwin_money : ℕ := 45
def mia_money : ℕ := 110

theorem mia_excess_over_double_darwin : mia_money - 2 * darwin_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_mia_excess_over_double_darwin_l2614_261447


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2614_261402

/-- Given two parallel lines x - ky - k = 0 and y = k(x - 1), prove that k = -1 -/
theorem parallel_lines_k_value (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x y : ℝ, x - k * y - k = 0 ↔ y = k * (x - 1)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2614_261402


namespace NUMINAMATH_CALUDE_counterexample_exists_l2614_261400

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2614_261400


namespace NUMINAMATH_CALUDE_stating_rabbit_distribution_count_l2614_261448

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
  Calculates the number of ways to distribute rabbits to pet stores
  such that no store has both a parent and a child
--/
def distribute_rabbits : ℕ :=
  -- Definition of the function to calculate the number of ways
  -- This is left undefined as the actual implementation is not provided
  sorry

/-- 
  Theorem stating that the number of ways to distribute the rabbits
  is equal to 560
--/
theorem rabbit_distribution_count : distribute_rabbits = 560 := by
  sorry

end NUMINAMATH_CALUDE_stating_rabbit_distribution_count_l2614_261448


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2614_261410

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2614_261410


namespace NUMINAMATH_CALUDE_tobys_sharing_l2614_261452

theorem tobys_sharing (initial_amount : ℚ) (remaining_amount : ℚ) (num_brothers : ℕ) :
  initial_amount = 343 →
  remaining_amount = 245 →
  num_brothers = 2 →
  (initial_amount - remaining_amount) / (num_brothers * initial_amount) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tobys_sharing_l2614_261452


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2614_261419

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2614_261419


namespace NUMINAMATH_CALUDE_triangle_angle_range_l2614_261406

theorem triangle_angle_range (A B C : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l2614_261406


namespace NUMINAMATH_CALUDE_second_group_size_l2614_261424

theorem second_group_size (n : ℕ) : 
  (30 : ℝ) * 20 + n * 30 = (30 + n) * 24 → n = 20 := by sorry

end NUMINAMATH_CALUDE_second_group_size_l2614_261424


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l2614_261456

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l2614_261456


namespace NUMINAMATH_CALUDE_central_cell_value_l2614_261434

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are adjacent -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, adjacent (i, j) (k, l) → (g i j).val + 1 = (g k l).val ∨ (g k l).val + 1 = (g i j).val

/-- Returns the sum of corner cell values in the grid -/
def corner_sum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- The main theorem to be proved -/
theorem central_cell_value (g : Grid) 
  (h_consec : consecutive_condition g) 
  (h_corner_sum : corner_sum g = 18) :
  (g 1 1).val = 2 :=
sorry

end NUMINAMATH_CALUDE_central_cell_value_l2614_261434


namespace NUMINAMATH_CALUDE_inverse_of_A_l2614_261442

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 5]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![5/22, 1/22; -1/11, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2614_261442


namespace NUMINAMATH_CALUDE_six_quarters_around_nickel_l2614_261478

/-- Represents the arrangement of coins on a table -/
structure CoinArrangement where
  nickelDiameter : ℝ
  quarterDiameter : ℝ

/-- Calculates the maximum number of quarters that can be placed around a nickel -/
def maxQuarters (arrangement : CoinArrangement) : ℕ :=
  sorry

/-- Theorem stating that for the given coin sizes, 6 quarters can be placed around a nickel -/
theorem six_quarters_around_nickel :
  let arrangement : CoinArrangement := { nickelDiameter := 2, quarterDiameter := 2.4 }
  maxQuarters arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_quarters_around_nickel_l2614_261478


namespace NUMINAMATH_CALUDE_quadratic_roots_same_sign_a_range_l2614_261425

theorem quadratic_roots_same_sign_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) →
  0 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_same_sign_a_range_l2614_261425


namespace NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l2614_261475

theorem exactly_two_sunny_days_probability :
  let days : ℕ := 3
  let rain_probability : ℚ := 60 / 100
  let sunny_probability : ℚ := 1 - rain_probability
  let ways_to_choose_two_days : ℕ := (days.choose 2)
  let probability_two_sunny_one_rainy : ℚ := sunny_probability^2 * rain_probability
  ways_to_choose_two_days * probability_two_sunny_one_rainy = 36 / 125 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l2614_261475


namespace NUMINAMATH_CALUDE_pauls_hourly_rate_is_35_l2614_261418

/-- Paul's Plumbing's hourly labor charge -/
def pauls_hourly_rate : ℝ := 35

/-- Paul's Plumbing's site visit fee -/
def pauls_visit_fee : ℝ := 55

/-- Reliable Plumbing's site visit fee -/
def reliable_visit_fee : ℝ := 75

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly_rate : ℝ := 30

/-- The number of hours worked -/
def hours_worked : ℝ := 4

theorem pauls_hourly_rate_is_35 :
  pauls_hourly_rate = 35 ∧
  pauls_visit_fee + hours_worked * pauls_hourly_rate =
  reliable_visit_fee + hours_worked * reliable_hourly_rate :=
by sorry

end NUMINAMATH_CALUDE_pauls_hourly_rate_is_35_l2614_261418


namespace NUMINAMATH_CALUDE_intersection_locus_l2614_261462

-- Define the two lines as functions of t
def line1 (x y t : ℝ) : Prop := 2 * x + 3 * y = t
def line2 (x y t : ℝ) : Prop := 5 * x - 7 * y = t

-- Define the locus line
def locusLine (x y : ℝ) : Prop := y = 0.3 * x

-- Theorem statement
theorem intersection_locus :
  ∀ (t : ℝ), ∃ (x y : ℝ), line1 x y t ∧ line2 x y t → locusLine x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l2614_261462


namespace NUMINAMATH_CALUDE_f_derivative_l2614_261463

noncomputable def f (x : ℝ) := Real.sin x + 3^x

theorem f_derivative (x : ℝ) : 
  deriv f x = Real.cos x + 3^x * Real.log 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l2614_261463


namespace NUMINAMATH_CALUDE_quotient_change_l2614_261479

theorem quotient_change (initial_quotient : ℝ) (dividend_multiplier : ℝ) (divisor_multiplier : ℝ) :
  initial_quotient = 0.78 →
  dividend_multiplier = 10 →
  divisor_multiplier = 0.1 →
  initial_quotient * dividend_multiplier / divisor_multiplier = 78 := by
  sorry

#check quotient_change

end NUMINAMATH_CALUDE_quotient_change_l2614_261479


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_point_l2614_261427

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  per_sheet : ℚ
  sitting_fee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.per_sheet * sheets + company.sitting_fee

/-- John's Photo World pricing -/
def johns_photo_world : PhotoCompany :=
  { per_sheet := 2.75, sitting_fee := 125 }

/-- Sam's Picture Emporium pricing -/
def sams_picture_emporium : PhotoCompany :=
  { per_sheet := 1.50, sitting_fee := 140 }

/-- Theorem stating that the two companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  total_cost johns_photo_world 12 = total_cost sams_picture_emporium 12 :=
by sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_point (sheets : ℚ) :
  total_cost johns_photo_world sheets = total_cost sams_picture_emporium sheets ↔ sheets = 12 :=
by sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_point_l2614_261427


namespace NUMINAMATH_CALUDE_function_equation_solution_l2614_261417

theorem function_equation_solution (a b : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  (∀ x : ℚ, f x = x + b - a) ∨ (∀ x : ℚ, f x = -x + b - a) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2614_261417
